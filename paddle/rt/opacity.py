import torch


class GasOpacity(torch.nn.Module):
    """Sum pressure-power-law mass opacities for selected gas species."""

    def __init__(
        self,
        species_indices: list[int],
        molecular_weights: list[float],
        kappa_ref: list[float],
        exponent: list[float],
        p_ref: float = 1.0e5,
    ):
        super().__init__()
        if (
            not len(species_indices)
            == len(molecular_weights)
            == len(kappa_ref)
            == len(exponent)
            or not species_indices
        ):
            raise ValueError("Gas parameter lists must have the same nonzero length")
        if p_ref <= 0.0:
            raise ValueError("p_ref must be positive")
        self.register_buffer(
            "species_indices", torch.tensor(species_indices, dtype=torch.int64)
        )
        self.register_buffer(
            "molecular_weights", torch.tensor(molecular_weights, dtype=torch.float64)
        )
        self.register_buffer("kappa_ref", torch.tensor(kappa_ref, dtype=torch.float64))
        self.register_buffer("exponent", torch.tensor(exponent, dtype=torch.float64))
        self.p_ref = float(p_ref)

    def forward(
        self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        """Map conc [mol/m^3] and pres [Pa] to (extinction [1/m], albedo, g).

        Inputs have shapes (ncol, nlyr, nspecies), (ncol, nlyr), (ncol, nlyr).
        Output is (1, ncol, nlyr, 3). Temperature is unused. Move the module
        to the input device/dtype before evaluation.
        """
        # calculate density and mass opacity for each selected gas
        gas_conc = torch.index_select(conc, -1, self.species_indices)
        density = gas_conc * self.molecular_weights
        kappa = self.kappa_ref * (pres.clamp(min=0.0).unsqueeze(-1) / self.p_ref).pow(
            self.exponent
        )

        # sum gas absorption; gases do not scatter in this model
        output = torch.zeros(
            (1, conc.shape[0], conc.shape[1], 3), dtype=conc.dtype, device=conc.device
        )
        output[0, :, :, 0] = (density * kappa).sum(-1)
        return output


class CloudOpacity(torch.nn.Module):
    """Combine extinction and scattering for selected condensate species."""

    def __init__(
        self,
        species_indices: list[int],
        molecular_weights: list[float],
        mass_extinction: list[float],
        single_scattering_albedo: list[float],
        asymmetry: list[float],
    ):
        super().__init__()
        if (
            not len(species_indices)
            == len(molecular_weights)
            == len(mass_extinction)
            == len(single_scattering_albedo)
            == len(asymmetry)
            or not species_indices
        ):
            raise ValueError("Cloud parameter lists must have the same nonzero length")
        self.register_buffer(
            "species_indices", torch.tensor(species_indices, dtype=torch.int64)
        )
        self.register_buffer(
            "molecular_weights", torch.tensor(molecular_weights, dtype=torch.float64)
        )
        self.register_buffer(
            "mass_extinction", torch.tensor(mass_extinction, dtype=torch.float64)
        )
        self.register_buffer(
            "single_scattering_albedo",
            torch.tensor(single_scattering_albedo, dtype=torch.float64),
        )
        self.register_buffer("asymmetry", torch.tensor(asymmetry, dtype=torch.float64))

    def forward(
        self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor
    ) -> torch.Tensor:
        """Return (1, ncol, nlyr, 3) with extinction [1/m], albedo and g.

        Concentration is (ncol, nlyr, nspecies) in mol/m^3. Pressure and
        temperature are unused. Move the module to the input device/dtype.
        """
        # calculate extinction and scattering for each condensate
        cloud_conc = torch.index_select(conc, -1, self.species_indices)
        density = cloud_conc * self.molecular_weights
        extinction = density * self.mass_extinction
        scattering = extinction * self.single_scattering_albedo

        # combine optical properties, with zero albedo/g for empty mixtures
        total_extinction = extinction.sum(-1)
        total_scattering = scattering.sum(-1)
        safe_extinction = torch.where(
            total_extinction > 0.0, total_extinction, torch.ones_like(total_extinction)
        )
        safe_scattering = torch.where(
            total_scattering > 0.0, total_scattering, torch.ones_like(total_scattering)
        )
        output = torch.zeros(
            (1, conc.shape[0], conc.shape[1], 3), dtype=conc.dtype, device=conc.device
        )
        output[0, :, :, 0] = total_extinction
        output[0, :, :, 1] = total_scattering / safe_extinction
        output[0, :, :, 2] = (scattering * self.asymmetry).sum(-1) / safe_scattering
        return output
