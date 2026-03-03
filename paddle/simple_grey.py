import torch


class SimpleGreyOpacity(torch.nn.Module):
    """TorchScriptable double-grey opacity module for pyharp JIT loading."""

    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        nwave: int = 1,
        nmom: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "species_weights",
            torch.tensor(species_weights, dtype=torch.float64),
            persistent=True,
        )
        self.kappa_a = float(kappa_a)
        self.kappa_b = float(kappa_b)
        self.kappa_cut = float(kappa_cut)
        self.nwave = int(nwave)
        self.nprop = 2 + int(nmom)

    def forward(
        self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        # conc: (ncol, nlyr, nspecies) [mol/m^3]
        # pres/temp: (ncol, nlyr)
        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # Match the C++ simple_grey.cpp behavior: extinction = rho * kappa(p)
        mw = self.species_weights.to(device=conc.device, dtype=conc.dtype)
        rho = (conc * mw.view(1, 1, -1)).sum(dim=-1)

        kappa = self.kappa_a * torch.pow(pres, self.kappa_b)
        kappa = torch.clamp(kappa, min=self.kappa_cut)
        extinction = rho * kappa  # [1/m]

        out = torch.zeros(
            (self.nwave, ncol, nlyr, self.nprop),
            dtype=conc.dtype,
            device=conc.device,
        )

        out[..., 0] = extinction.unsqueeze(0)
        # pure absorption: single scattering albedo = 0, g = 0
        return out
