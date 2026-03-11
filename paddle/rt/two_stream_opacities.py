from typing import List, Tuple
import torch


class GreyOpacity(torch.nn.Module):
    """
    A simple grey opacity module that computes extinction, single scattering albedo, and asymmetry factor
    based on a power-law dependence on pressure.
    """

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

        kappa = min(kappa_a * pres^kappa_b, kappa_cut)

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


def create_two_stream_opacities(
    species_weights: List[float],
    kappa_a: Tuple[float, float],
    kappa_b: Tuple[float, float],
    kappa_cut: Tuple[float, float],
    w0: Tuple[float, float] = [0.0, 0.0],
    g: Tuple[float, float] = [0.0, 0.0],
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create two grey opacity modules for shortwave and longwave.

    Args:
        species_weights (List[float]): List of floats for the molecular weights of each
        species. The size must match the number of species in the concentration input to
        the opacity modules.
        kappa_a (Tuple[float, float]): List of two floats for the kappa_a parameter for shortwave and longwave.
        kappa_b (Tuple[float, float]): List of two floats for the kappa_b parameter for shortwave and longwave.
        kappa_cut (Tuple[float, float]): List of two floats for the kappa_cut parameter for shortwave and longwave.
        w0 (Tuple[float, float]): List of two floats for the single scattering albedo for shortwave and longwave.
        g (Tuple[float, float]): List of two floats for the asymmetry factor for shortwave and longwave.

    Returns:
        Tuple of two GreyOpacity modules for shortwave and longwave.
    """

    sw_opacity = GreyOpacity(
        species_weights=species_weights,
        kappa_a=kappa_a[0],
        kappa_b=kappa_b[0],
        kappa_cut=kappa_cut[0],
        w0=w0[0],
        g=g[0],
    )

    lw_opacity = GreyOpacity(
        species_weights=species_weights,
        kappa_a=kappa_a[1],
        kappa_b=kappa_b[1],
        kappa_cut=kappa_cut[1],
        w0=w0[1],
        g=g[1],
    )

    return sw_opacity, lw_opacity
