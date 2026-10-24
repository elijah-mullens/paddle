import math

import torch
from snapy import kIDN, kIV2, kIV3


class DeepWindForcing(torch.nn.Module):
    """Relax the bottom horizontal velocity toward a prescribed deep wind.

    deep_wind has shape (2, nc3, nc2), including horizontal ghosts, and
    contains local contravariant x2/x3 velocities in m/s. Singleton spatial
    dimensions may broadcast. tau is in seconds; bottom is the physical
    lower x1 face coordinate in meters, shared by all vertical blocks.
    Only the cell containing bottom is forced. Supply each horizontal block
    with its own wind distribution. Like Snapy's bottom-velocity relaxation,
    this supplies momentum increments without a direct total-energy source.
    """

    def __init__(self, deep_wind: torch.Tensor, tau: float, bottom: float):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if deep_wind.dim() != 3 or deep_wind.size(0) != 2:
            raise ValueError("deep_wind must have shape (2, nc3, nc2)")
        self.register_buffer("deep_wind", deep_wind)
        self.tau = float(tau)
        self.bottom = float(bottom)

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
        x1f = variables["coord.x1f"]
        bottom = (x1f[:-1] <= self.bottom) & (x1f[1:] > self.bottom)
        scale = dt / self.tau * hydro_w[self.density_index] * bottom

        # Relax local velocities; the caller applies stage weights.
        force2 = scale * (self.deep_wind[0].unsqueeze(-1) - hydro_w[self.east_index])
        force3 = scale * (self.deep_wind[1].unsqueeze(-1) - hydro_w[self.north_index])

        # Lower the horizontal force to covariant momentum (identity on Cartesian grids).
        cosine = variables["coord.cosine_cell_kj"]
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.east_index] = force2 + cosine * force3
        hydro_du[self.north_index] = force3 + cosine * force2
        return {"hydro_du": hydro_du}
