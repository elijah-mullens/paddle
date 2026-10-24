"""Present-day Earth forcing with calendar-aligned initial illumination.

Parameter sources:
https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
https://ssd.jpl.nasa.gov/astro_par.html
https://aa.usno.navy.mil/faq/sun_approx
https://aa.usno.navy.mil/faq/GAST
"""

from datetime import datetime, timezone
import math

from orbital_rt_forcing import OrbitalRTForcing


ASTRONOMICAL_UNIT = 149_597_870_700.0
EARTH_ECCENTRICITY = 0.0167
EARTH_ORBITAL_PERIOD = 365.25636 * 86400.0
EARTH_SIDEREAL_DAY = 86164.09054
SOLAR_IRRADIANCE = 1361.0
DEFAULT_EPOCH = "2000-01-01T12:00:00+00:00"


class EarthOrbitalRTForcing(OrbitalRTForcing):
    """Earth defaults for the shared Newtonian orbital forcing model.

    ``time`` is elapsed seconds since ``epoch``, an ISO-8601 date/time.
    The default is 2000-01-01 12:00 UTC, a civil-time approximation to J2000
    (formally defined at 12:00 TT).
    Naive dates are interpreted as UTC; timezone offsets are converted to
    UTC. Longitudes are east-positive radians and latitudes are radians.
    Distances are metres and beam irradiances are W/m². ``solar_irradiance``
    specifies the irradiance at one astronomical unit, before night masking.

    The mean anomaly, solar longitude of perihelion, obliquity, and Greenwich
    sidereal angle are initialized from USNO approximations at the epoch.
    Earth parameters then remain fixed for a repeating Keplerian orbit.
    This is a present-day climate approximation, not a precision ephemeris:
    precession during the run, nutation, perturbations, solar variability,
    and UTC/UT1 differences are omitted. Use float64 time tensors for long
    runs to retain the precision needed for the diurnal cycle.

    The inherited callback accepts ``hydro_u`` and the heating field ``key``;
    it returns ``hydro_du`` with heating times ``dt`` in the energy channel.
    """

    def __init__(
        self,
        epoch: str = DEFAULT_EPOCH,
        solar_irradiance: float = SOLAR_IRRADIANCE,
        key: str = "rt_heating",
    ) -> None:
        epoch_datetime = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
        if epoch_datetime.tzinfo is None:
            epoch_datetime = epoch_datetime.replace(tzinfo=timezone.utc)
        epoch_datetime = epoch_datetime.astimezone(timezone.utc)
        reference = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
        days = (epoch_datetime - reference).total_seconds() / 86400.0
        midnight = epoch_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        midnight_days = (midnight - reference).total_seconds() / 86400.0
        hours = (epoch_datetime - midnight).total_seconds() / 3600.0
        centuries = days / 36525.0

        mean_anomaly = (357.529 + 0.98560028 * days) % 360.0
        mean_longitude = (280.459 + 0.98564736 * days) % 360.0
        obliquity = 23.439 - 0.00000036 * days
        sidereal_hours = (
            6.697375
            + 0.065709824279 * midnight_days
            + 1.0027379 * hours
            + 0.0000258 * centuries**2
        ) % 24.0

        super().__init__(
            stellar_luminosity=4.0
            * math.pi
            * ASTRONOMICAL_UNIT**2
            * float(solar_irradiance),
            semi_major_axis=ASTRONOMICAL_UNIT,
            eccentricity=EARTH_ECCENTRICITY,
            obliquity=math.radians(obliquity),
            rotation_rate=2.0 * math.pi / EARTH_SIDEREAL_DAY,
            orbital_period=EARTH_ORBITAL_PERIOD,
            mean_anomaly_epoch=math.radians(mean_anomaly),
            prime_meridian_epoch=math.radians(15.0 * sidereal_hours),
            key=key,
            solar_longitude_periapsis=math.radians(
                (mean_longitude - mean_anomaly) % 360.0
            ),
        )
        self.epoch = epoch_datetime.isoformat()
