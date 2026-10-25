import math
import torch
from typing import Tuple
from snapy import kIPR


class OrbitalForcing(torch.nn.Module):
    """Keplerian illumination and radiative forcing, using SI units and radians."""

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
        rt_heating_key: str = "rt_heating",
    ) -> None:
        super().__init__()
        if not 0 <= eccentricity < 1 or semi_major_axis <= 0 or orbital_period <= 0:
            raise ValueError(
                "Require 0 <= eccentricity < 1 and positive axis and period"
            )
        self.stellar_luminosity = float(stellar_luminosity)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.obliquity = float(obliquity)
        self.rotation_rate = float(rotation_rate)
        self.orbital_period = float(orbital_period)
        self.mean_anomaly = float(mean_anomaly)
        self.prime_meridian = float(prime_meridian)
        self.rt_heating_key = str(rt_heating_key)

        # mapping (cos f, sin f) to the stellar direction.
        # Planet presets replace this idealized equinox-at-periapsis basis.
        self.solar_basis = (
            (1.0, 0.0),
            (0.0, math.cos(obliquity)),
            (0.0, math.sin(obliquity)),
        )

        # torchscript needs this to compile
        self.energy_index = int(kIPR)

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        # Heating is energy per volume per time; the caller owns stage weights.
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.energy_index] = variables[self.rt_heating_key] * dt
        return {"hydro_du": hydro_du}

    @torch.jit.export
    def insolation(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """Return mu0, beam, distance, substellar longitude and latitude.

        Time is one elapsed time in seconds (from J2000 for planet presets).
        Orbit math uses CPU scalars; lon/lat are east-positive, planetocentric
        grids on the same device. Beam is normal irradiance in W/m².
        """
        time = float(time)
        mean_anomaly = self.mean_anomaly + 2.0 * math.pi * time / self.orbital_period
        mean_anomaly = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi
        eccentric_anomaly = mean_anomaly
        if self.eccentricity > 0.8:
            eccentric_anomaly = math.pi if mean_anomaly >= 0.0 else -math.pi

        # calculate distance
        for _ in range(8):
            eccentric_anomaly -= (
                eccentric_anomaly
                - self.eccentricity * math.sin(eccentric_anomaly)
                - mean_anomaly
            ) / (1.0 - self.eccentricity * math.cos(eccentric_anomaly))
        distance = self.semi_major_axis * (
            1.0 - self.eccentricity * math.cos(eccentric_anomaly)
        )

        # calculate stellar latitude
        true_anomaly = 2.0 * math.atan2(
            math.sqrt(1.0 + self.eccentricity) * math.sin(0.5 * eccentric_anomaly),
            math.sqrt(1.0 - self.eccentricity) * math.cos(0.5 * eccentric_anomaly),
        )
        cf, sf = math.cos(true_anomaly), math.sin(true_anomaly)
        sx = self.solar_basis[0][0] * cf + self.solar_basis[0][1] * sf
        sy = self.solar_basis[1][0] * cf + self.solar_basis[1][1] * sf
        sz = self.solar_basis[2][0] * cf + self.solar_basis[2][1] * sf
        stellar_lat = math.asin(max(-1.0, min(1.0, sz)))

        # calculate stellar longitude
        stellar_right_ascension = math.atan2(sy, sx)
        stellar_lon = (
            stellar_right_ascension - self.rotation_rate * time - self.prime_meridian
        )
        stellar_lon = (stellar_lon + math.pi) % (2.0 * math.pi) - math.pi

        # convert to torch tensor
        stellar_lat = torch.tensor(stellar_lat, dtype=lat.dtype, device=lat.device)
        stellar_lon = torch.tensor(stellar_lon, dtype=lon.dtype, device=lon.device)

        # calculate stellar flux and incidence angle
        raw_mu0 = (
            lat.sin() * stellar_lat.sin()
            + lat.cos() * stellar_lat.cos() * (lon - stellar_lon).cos()
        )
        dayside = raw_mu0 > 0.0
        mu0 = torch.where(dayside, raw_mu0, torch.zeros_like(raw_mu0))
        irradiance = self.stellar_luminosity / (4.0 * math.pi * distance * distance)
        beam = torch.where(
            dayside, torch.ones_like(raw_mu0) * irradiance, torch.zeros_like(raw_mu0)
        )
        return mu0, beam, distance, stellar_lon, stellar_lat
