import torch
from snapy import kIDN, kIV2, kIV3


class GammaPlaneCoriolis(torch.nn.Module):
    """Polar Coriolis forcing with x1 up and x2/x3 in the tangent plane.

    f = f0 - gamma * ((x2 - x0)**2 + (x3 - y0)**2), with f0 in s^-1,
    gamma in m^-2 s^-1, and the pole location (x0, y0) in meters.
    For a spherical planet of radius R, gamma = f0 / (2*R**2);
    f0 and gamma have the same sign, positive at the north pole for
    prograde rotation. Use instead of the native Coriolis forcing.
    """

    def __init__(
        self,
        f0: float,
        gamma: float,
        x0: float = 0.0,
        y0: float = 0.0,
    ):
        super().__init__()
        self.f0 = float(f0)
        self.gamma = float(gamma)
        self.x0 = float(x0)
        self.y0 = float(y0)

        # TorchScript needs indices on the module.
        self.density_index = int(kIDN)
        self.east_index = int(kIV2)
        self.north_index = int(kIV3)

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        hydro_w = variables["hydro_w"]
        x = variables["coord.x2v"].reshape(1, -1, 1) - self.x0
        y = variables["coord.x3v"].reshape(-1, 1, 1) - self.y0
        f = self.f0 - self.gamma * (x * x + y * y)

        # du/dt = f*v, dv/dt = -f*u; the caller applies stage weights.
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.east_index] = (
            dt * f * hydro_w[self.density_index] * hydro_w[self.north_index]
        )
        hydro_du[self.north_index] = (
            -dt * f * hydro_w[self.density_index] * hydro_w[self.east_index]
        )
        return {"hydro_du": hydro_du}
