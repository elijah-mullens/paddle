"""Solar-system presets: fixed J2000 elements and simplified IAU rotation.

Sources and approximation limits: docs/notes/orbital/index.rst.
"""

import math

import torch

from .orbital import OrbitalForcing

AU = 149_597_870_700.0  # m, exact astronomical unit
DAY = 86_400.0
SOLAR_LUMINOSITY = 3.828e26  # W, IAU nominal value
J2000_JD = 2451545.0  # TDB; input time is elapsed seconds from this epoch


class _J2000Planet(OrbitalForcing):
    # a [au], e, I, L, longitude of perihelion, ascending node [degrees],
    # mean-anomaly rate [degrees/Julian century]. JPL Table 1 at T=0.
    elements: tuple
    # Pole RA, Dec, W0 [degrees], W rate [degrees/day]; PCK polynomial terms.
    rotation: tuple

    def __init__(
        self,
        hydro_u_key: str = "hydro_u",
        rt_heating_key: str = "rt_heating",
    ) -> None:
        a, e, inc, longitude, perihelion, node, anomaly_rate = self.elements
        ra, dec, w0, w_rate = self.rotation
        inc, node, arg = map(math.radians, (inc, node, perihelion - node))
        ra, dec = map(math.radians, (ra, dec))
        c, s = math.cos, math.sin

        # Perihelion direction P and transverse direction Q in J2000 ecliptic.
        pq = torch.tensor(
            [
                [
                    c(node) * c(arg) - s(node) * s(arg) * c(inc),
                    -c(node) * s(arg) - s(node) * c(arg) * c(inc),
                ],
                [
                    s(node) * c(arg) + c(node) * s(arg) * c(inc),
                    -s(node) * s(arg) + c(node) * c(arg) * c(inc),
                ],
                [s(arg) * s(inc), c(arg) * s(inc)],
            ],
            dtype=torch.float64,
        )
        eps = math.radians(23.43928)
        ecliptic_to_equatorial = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, c(eps), -s(eps)],
                [0.0, s(eps), c(eps)],
            ],
            dtype=torch.float64,
        )
        pq = ecliptic_to_equatorial @ pq
        pole = torch.tensor(
            [c(dec) * c(ra), c(dec) * s(ra), s(dec)], dtype=torch.float64
        )

        # IAU equatorial origin at RA + 90 degrees; W rotates from this origin.
        origin = torch.tensor([-s(ra), c(ra), 0.0], dtype=torch.float64)
        transverse = torch.linalg.cross(pole, origin)
        orbit_normal = torch.linalg.cross(pq[:, 0], pq[:, 1])
        spin_sign = 1.0 if w_rate > 0 else -1.0
        obliquity = math.acos(
            float(torch.dot(spin_sign * pole, orbit_normal).clamp(-1, 1))
        )

        super().__init__(
            stellar_luminosity=SOLAR_LUMINOSITY,
            semi_major_axis=a * AU,
            eccentricity=e,
            obliquity=obliquity,
            rotation_rate=math.radians(w_rate) / DAY,
            orbital_period=360.0 / anomaly_rate * 36525.0 * DAY,
            mean_anomaly=math.radians(longitude - perihelion),
            prime_meridian=math.radians(w0),
            hydro_u_key=hydro_u_key,
            rt_heating_key=rt_heating_key,
        )

        # Sun direction is opposite to the heliocentric planet direction.
        basis = -torch.stack((origin, transverse, pole)) @ pq
        self.solar_basis = (
            (float(basis[0, 0]), float(basis[0, 1])),
            (float(basis[1, 0]), float(basis[1, 1])),
            (float(basis[2, 0]), float(basis[2, 1])),
        )


class Earth(_J2000Planet):
    """Earth, using the Earth-Moon barycenter orbit and simplified IAU Earth spin."""

    elements = (
        1.00000261,
        0.01671123,
        -0.00001531,
        100.46457166,
        102.93768193,
        0.0,
        35999.37244981 - 0.32327364,
    )
    rotation = (0.0, 90.0, 190.147, 360.9856235)


class Jupiter(_J2000Planet):
    """Jupiter with the IAU System III rotation convention."""

    elements = (
        5.20288700,
        0.04838624,
        1.30439695,
        34.39644051,
        14.72847983,
        100.47390909,
        3034.74612775 - 0.21252668,
    )
    rotation = (268.056595, 64.495303, 284.95, 870.5360000)


class Saturn(_J2000Planet):
    """Saturn with the IAU reference rotation (not atmospheric differential rotation)."""

    elements = (
        9.53667594,
        0.05386179,
        2.48599187,
        49.95424423,
        92.59887831,
        113.66242448,
        1222.49362201 + 0.41897216,
    )
    rotation = (40.589, 83.537, 38.90, 810.7939024)


class Uranus(_J2000Planet):
    """Uranus: IAU north latitude, east longitude, and negative (retrograde) spin."""

    elements = (
        19.18916464,
        0.04725744,
        0.77263783,
        313.23810451,
        170.95427630,
        74.01692503,
        428.48202785 - 0.40805281,
    )
    rotation = (257.311, -15.175, 203.81, -501.1600928)


class Neptune(_J2000Planet):
    """Neptune using PCK00011 polynomial rotation terms (periodic terms omitted)."""

    elements = (
        30.06992276,
        0.00859048,
        1.77004347,
        -55.12002969,
        44.96476227,
        131.78422574,
        218.45945325 + 0.32241464,
    )
    rotation = (299.36, 43.46, 249.978, 541.1397757)
