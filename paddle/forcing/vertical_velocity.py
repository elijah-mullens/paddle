import math

import torch
from snapy import kIDN, kIV1


class VerticalVelocityForcing(torch.nn.Module):
    """Relax vertical velocity throughout the domain toward a 1D profile.

    profile contains x1 velocities in m/s at each local vertical cell center,
    including ghosts, and is broadcast over all horizontal columns. tau is
    in seconds. Supply the local profile for each vertically decomposed
    block. This supplies vertical momentum without a direct energy source.
    """

    def __init__(self, profile: torch.Tensor, tau: float):
        super().__init__()
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if profile.dim() != 1 or profile.numel() == 0:
            raise ValueError("profile must be a nonempty 1D tensor")
        self.register_buffer("profile", profile)
        self.tau = float(tau)

        # TorchScript needs indices on the module.
        self.density_index = int(kIDN)
        self.vertical_index = int(kIV1)

    def forward(
        self,
        variables: dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> dict[str, torch.Tensor]:
        hydro_w = variables["hydro_w"]
        if self.profile.numel() != hydro_w.size(-1):
            raise ValueError(
                "profile must match the local vertical cells, including ghosts"
            )

        # Relax every vertical level; the caller applies stage weights.
        hydro_du = torch.zeros_like(variables["hydro_u"])
        hydro_du[self.vertical_index] = (
            dt
            / self.tau
            * hydro_w[self.density_index]
            * (self.profile - hydro_w[self.vertical_index])
        )
        return {"hydro_du": hydro_du}
