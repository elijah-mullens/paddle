import math
import torch

from snapy import kIPR


class OrbitalRTForcing(torch.nn.Module):
    """Keplerian orbit illumination radiative forcing.

    Angles are in radians; time and orbital_period share a unit, and
    rotation_rate is in radians per that unit. solar_longitude_periapsis is
    the star's orbital longitude from the ascending equinox at periapsis;
    its default aligns periapsis with that equinox. Positive rotation moves
    the subsolar longitude westward. Luminosity and semi-major axis set the
    irradiance units.

    ``insolation`` returns the nonnegative zenith cosine, dayside irradiance
    normal to the beam, orbital distance, subsolar longitude, and subsolar
    latitude. Multiply the beam irradiance by the cosine for horizontal flux.
    The forward callback adds heating times dt to the energy component.
    """

    def __init__(
        self,
        stellar_luminosity: float,
        semi_major_axis: float,
        eccentricity: float,
        obliquity: float,
        rotation_rate: float,
        orbital_period: float,
        mean_anomaly_epoch: float = 0.0,
        prime_meridian_epoch: float = 0.0,
        key: str = "rt_heating",
        solar_longitude_periapsis: float = 0.0,
    ) -> None:
        super().__init__()
        self.stellar_luminosity = float(stellar_luminosity)
        self.semi_major_axis = float(semi_major_axis)
        self.eccentricity = float(eccentricity)
        self.obliquity = float(obliquity)
        self.rotation_rate = float(rotation_rate)
        self.orbital_period = float(orbital_period)
        self.mean_anomaly_epoch = float(mean_anomaly_epoch)
        self.prime_meridian_epoch = float(prime_meridian_epoch)
        self.solar_longitude_periapsis = float(solar_longitude_periapsis)
        for name in (
            "stellar_luminosity",
            "semi_major_axis",
            "eccentricity",
            "obliquity",
            "rotation_rate",
            "orbital_period",
            "mean_anomaly_epoch",
            "prime_meridian_epoch",
            "solar_longitude_periapsis",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.stellar_luminosity < 0.0:
            raise ValueError("stellar_luminosity must be nonnegative")
        if self.semi_major_axis <= 0.0:
            raise ValueError("semi_major_axis must be positive")
        if self.orbital_period <= 0.0:
            raise ValueError("orbital_period must be positive")
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must satisfy 0 <= eccentricity < 1")
        self.key = key
        self.energy_index = int(kIPR)

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.energy_index] = variables[self.key] * dt
        return {"hydro_du": hydro_du}

    @torch.jit.export
    def insolation(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        two_pi = 2.0 * math.pi
        mean_anomaly = torch.remainder(
            self.mean_anomaly_epoch + two_pi * time / self.orbital_period, two_pi
        )
        mean_anomaly = torch.where(
            mean_anomaly > math.pi, mean_anomaly - two_pi, mean_anomaly
        )
        eccentric_anomaly = mean_anomaly
        if self.eccentricity >= 0.8:
            absolute_mean_anomaly = mean_anomaly.abs()
            eccentric_anomaly = mean_anomaly.sign() * torch.minimum(
                absolute_mean_anomaly / (1.0 - self.eccentricity),
                (6.0 * absolute_mean_anomaly / self.eccentricity).pow(1.0 / 3.0),
            )
        for _ in range(8):
            residual = (
                (1.0 - self.eccentricity) * eccentric_anomaly
                + self.eccentricity * (eccentric_anomaly - eccentric_anomaly.sin())
                - mean_anomaly
            )
            derivative = (1.0 - self.eccentricity) + 2.0 * self.eccentricity * (
                0.5 * eccentric_anomaly
            ).sin().square()
            eccentric_anomaly = eccentric_anomaly - residual / derivative

        distance = self.semi_major_axis * (
            1.0 - self.eccentricity * torch.cos(eccentric_anomaly)
        )

        true_anomaly = 2.0 * torch.atan2(
            math.sqrt(1.0 + self.eccentricity) * torch.sin(0.5 * eccentric_anomaly),
            math.sqrt(1.0 - self.eccentricity) * torch.cos(0.5 * eccentric_anomaly),
        )

        solar_longitude = true_anomaly + self.solar_longitude_periapsis
        obliquity = torch.as_tensor(self.obliquity, dtype=lat.dtype, device=lat.device)
        subsolar_lat = (obliquity.sin() * solar_longitude.sin()).asin()
        stellar_right_ascension = torch.atan2(
            obliquity.cos() * solar_longitude.sin(), solar_longitude.cos()
        )

        subsolar_lon = (
            stellar_right_ascension
            - self.rotation_rate * time
            - self.prime_meridian_epoch
        )
        raw_mu0 = (
            lat.sin() * subsolar_lat.sin()
            + lat.cos() * subsolar_lat.cos() * (lon - subsolar_lon).cos()
        )
        dayside = raw_mu0 > 0.0
        mu0 = torch.where(dayside, raw_mu0, torch.zeros_like(raw_mu0))
        irradiance = self.stellar_luminosity / (4.0 * math.pi * distance * distance)
        beam = torch.where(
            dayside, torch.ones_like(raw_mu0) * irradiance, torch.zeros_like(raw_mu0)
        )
        return mu0, beam, distance, subsolar_lon, subsolar_lat
