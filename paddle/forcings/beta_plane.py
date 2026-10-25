import torch
from snapy import kIDN, kIV2, kIV3


class BetaPlaneCoriolis(torch.nn.Module):
    """Cartesian Coriolis forcing with x1 up, x2 east, and x3 north.

    f = f0 + beta * (x3 - y0), with f0 in s^-1, beta in m^-1 s^-1,
    and y0 in meters. Use instead of the native Coriolis forcing.
    """

    def __init__(self, f0: float, beta: float, y0: float = 0.0):
        super().__init__()
        self.f0 = float(f0)
        self.beta = float(beta)
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
        y = variables["coord.x3v"].reshape(-1, 1, 1)
        f = self.f0 + self.beta * (y - self.y0)

        # du/dt = f*v, dv/dt = -f*u; the caller applies stage weights.
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.east_index] = (
            dt * f * hydro_w[self.density_index] * hydro_w[self.north_index]
        )
        hydro_du[self.north_index] = (
            -dt * f * hydro_w[self.density_index] * hydro_w[self.east_index]
        )
        return {"hydro_du": hydro_du}
