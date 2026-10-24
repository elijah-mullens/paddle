from typing import Tuple
import math
import torch


class OrbitalForcing(torch.nn.Module):
    """Fixed Kepler orbit illumination and an explicit radiative increment.

    Use SI units and radians. Longitudes are east-positive, latitudes are
    planetocentric, and time is seconds from the chosen epoch. The generic
    orbit has its northern spring equinox at periapsis; planet presets supply
    a J2000 orientation instead. See docs/notes/orbital/index.rst.
    """

    def __init__(
        self,
        stellar_luminosity: float,
        semi_major_axis: float,
        eccentricity: float,
        obliquity: float,
        rotation_rate: float,
        orbital_period: float,
        mean_anomaly: float = 0.0,
        prime_meridian: float = 0.0,
        energy_index: int = 4,
        hydro_u_key: str = "hydro_u",
        rt_heating_key: str = "rt_heating",
    ) -> None:
        super().__init__()
        values = (
            stellar_luminosity,
            semi_major_axis,
            eccentricity,
            obliquity,
            rotation_rate,
            orbital_period,
            mean_anomaly,
            prime_meridian,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Orbital parameters must be finite")
        if semi_major_axis <= 0 or orbital_period <= 0 or stellar_luminosity < 0:
            raise ValueError("Axis and period must be positive; luminosity nonnegative")
        if not 0 <= eccentricity < 1:
            raise ValueError("A bound elliptical orbit requires 0 <= eccentricity < 1")
        self.stellar_luminosity = float(stellar_luminosity)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.obliquity = float(obliquity)
        self.rotation_rate = float(rotation_rate)
        self.orbital_period = float(orbital_period)
        self.mean_anomaly = float(mean_anomaly)
        self.prime_meridian = float(prime_meridian)
        self.energy_index = int(energy_index)
        self.hydro_u_key = str(hydro_u_key)
        self.rt_heating_key = str(rt_heating_key)
        # Columns map (cos f, sin f) to the stellar direction in a
        # nonrotating planet-equatorial frame. Presets replace this matrix.
        self.register_buffer(
            "solar_basis",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, math.cos(obliquity)],
                    [0.0, math.sin(obliquity)],
                ],
                dtype=torch.float64,
            ),
        )

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        """Return an increment; heating is energy/volume/time, not K/s.

        ``stage`` is accepted for integrator compatibility but unused. The
        caller owns stage weights and must pass the appropriate ``dt``.
        """
        hydro_du = torch.zeros_like(variables[self.hydro_u_key])
        hydro_du[self.energy_index] = variables[self.rt_heating_key] * dt
        return {self.hydro_u_key: hydro_du}

    @torch.jit.export
    def insolation(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu0, beam, distance, substellar_lon, substellar_lat).

        Floating tensors must share a device and have broadcastable shapes.
        Use time[:, None, None] with a spatial grid for a time series.
        Beam is dayside normal irradiance (W/m²); horizontal flux is mu0*beam.
        Geometry outputs retain the time shape. Prefer float64 for long runs.
        """
        mean_anomaly = (
            torch.remainder(
                self.mean_anomaly
                + 2.0 * math.pi * time / self.orbital_period
                + math.pi,
                2.0 * math.pi,
            )
            - math.pi
        )
        eccentric_anomaly = mean_anomaly.clone()
        if self.eccentricity > 0.8:
            eccentric_anomaly = torch.where(
                mean_anomaly >= 0,
                torch.full_like(mean_anomaly, math.pi),
                torch.full_like(mean_anomaly, -math.pi),
            )

        # calculate distance
        for _ in range(32):
            eccentric_anomaly = eccentric_anomaly - (
                eccentric_anomaly
                - self.eccentricity * eccentric_anomaly.sin()
                - mean_anomaly
            ) / (1.0 - self.eccentricity * eccentric_anomaly.cos())
        distance = self.semi_major_axis * (
            1.0 - self.eccentricity * eccentric_anomaly.cos()
        )

        # calculate substellar latitude
        true_anomaly = 2.0 * torch.atan2(
            math.sqrt(1.0 + self.eccentricity) * (0.5 * eccentric_anomaly).sin(),
            math.sqrt(1.0 - self.eccentricity) * (0.5 * eccentric_anomaly).cos(),
        )
        basis = self.solar_basis.to(time)
        cf, sf = true_anomaly.cos(), true_anomaly.sin()
        sx = basis[0, 0] * cf + basis[0, 1] * sf
        sy = basis[1, 0] * cf + basis[1, 1] * sf
        sz = basis[2, 0] * cf + basis[2, 1] * sf
        substellar_lat = sz.clamp(-1.0, 1.0).asin()

        # calculate substellar longitude
        stellar_right_ascension = torch.atan2(sy, sx)
        spin_angle = torch.remainder(
            self.rotation_rate * time + self.prime_meridian, 2.0 * math.pi
        )
        substellar_lon = (
            torch.remainder(
                stellar_right_ascension - spin_angle + math.pi, 2.0 * math.pi
            )
            - math.pi
        )

        # calculate stellar flux and incidence angle
        raw_mu0 = (
            lat.sin() * substellar_lat.sin()
            + lat.cos() * substellar_lat.cos() * (lon - substellar_lon).cos()
        )
        dayside = raw_mu0 > 0.0
        mu0 = torch.where(dayside, raw_mu0, torch.zeros_like(raw_mu0))
        irradiance = self.stellar_luminosity / (4.0 * math.pi * distance * distance)
        beam = torch.where(
            dayside, torch.ones_like(raw_mu0) * irradiance, torch.zeros_like(raw_mu0)
        )
        return mu0, beam, distance, substellar_lon, substellar_lat
