import torch


class GreyOpacity(torch.nn.Module):
    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        w0: float = 0.0,
        g: float = 0.0,
        nwave: int = 1,
        nmom: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "species_weights",
            torch.tensor(species_weights),
            persistent=True,
        )
        self.kappa_a = float(kappa_a)
        self.kappa_b = float(kappa_b)
        self.kappa_cut = float(kappa_cut)
        self.nwave = int(nwave)
        self.w0 = float(w0)
        self.g = float(g)
        self.nprop = 2 + int(nmom)  # (extinction, w0, g)

    def forward(
        self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute grey opacity properties for each layer and column.

        Args:
            conc: (ncol, nlyr, nspecies) [mol/m^3]
            pres: (ncol, nlyr) [pa]
            temp: (ncol, nlyr) [K]

        Returns:
            prop: (nwave, ncol, nlyr, nprop) where nprop includes extinction [1/m],
            single scattering albedo, and asymmetry factor.
        """

        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # extinction = rho * kappa(pres)
        rho = (conc * self.species_weights.view(1, 1, -1)).sum(dim=-1)

        kappa = self.kappa_a * torch.pow(pres, self.kappa_b)
        kappa = torch.clamp(kappa, min=self.kappa_cut)
        extinction = rho * kappa  # [1/m]

        out = torch.zeros(
            (self.nwave, ncol, nlyr, self.nprop),
            dtype=conc.dtype,
            device=conc.device,
        )

        out[..., 0] = extinction.unsqueeze(0)
        out[..., 1] = self.w0
        out[..., 2] = self.g

        return out
