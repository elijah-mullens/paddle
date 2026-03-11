from typing import Tuple, Union
import torch
import pyharp


def create_two_stream_solvers(
    wave_lo: float, wave_hi: float
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Create two-stream solvers for shortwave and longwave radiation using Pyharp.

    Args:
        wave_lo (float): Lower wavenumber limit for longwave solver.
        wave_hi (float): Upper wavenumber limit for longwave solver.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: A tuple containing the shortwave and longwave two-stream solvers.
    """
    # shortwave solver
    op_sw = pyharp.ToonMcKay89Options()
    toon_sw = pyharp.ToonMcKay89(op_sw)

    # longwave solver
    op_lw = pyharp.ToonMcKay89Options()
    op_lw.wave_lower([wave_lo])
    op_lw.wave_upper([wave_hi])
    toon_lw = pyharp.ToonMcKay89(op_lw)

    return toon_sw, toon_lw


def compute_shortwave_flux(
    grey_sw: torch.nn.Module,
    toon_sw: torch.nn.Module,
    dz: torch.Tensor,
    conc_i: torch.Tensor,
    pres_i: torch.Tensor,
    temp_i: torch.Tensor,
    *,
    cos_zenith_angle: Union[torch.Tensor, float] = 1.0,
    stellar_flux: float = 0.0,
    albedo: float = 0,
) -> torch.Tensor:
    """
    Compute shortwave fluxes using Toon's two-stream shortwave solver.

    Args:
        grey_sw (torch.nn.Module): A module that computes grey shortwave properties.
        toon_sw (torch.nn.Module): A two-stream solver for shortwave radiation.
        dz (torch.Tensor): Layer thicknesses [m] (nlyr,).
        conc_i (torch.Tensor): concentrations [mol/m^3] at cell center (ncol, nlyr, nspecies).
        pres_i (torch.Tensor): Pressures [pa] at cell center (ncol, nlyr).
        temp_i (torch.Tensor): Temperatures [K] at cell center (ncol, nlyr).
        cos_zenith_angle (torch.Tensor or float): Cosine of solar zenith angle (ncol,) or scalar for constant angle.
        stellar_flux (float) : Stellar flux at the top of the atmosphere (W/m^2).
        albedo (float): Surface albedo for shortwave radiation.

    Returns:
        torch.Tensor: Net shortwave fluxes (W/m^2) at layer interfaces (ncol, nlyr+1).
    """

    ncol = conc_i.shape[0]

    if isinstance(cos_zenith_angle, float):
        cos_zenith_dayside = cos_zenith_angle * torch.ones(
            (ncol,), device=conc_i.device
        )
    else:
        cos_zenith_dayside = cos_zenith_angel.view(ncol).clamp(
            min=0.0
        )  # ensure non-negative

    bc: dict[str, torch.Tensor] = {
        "fbeam": (
            stellar_flux * torch.ones((1, ncol), device=conc_i.device)  # (nwave, ncol)
        ),
        "umu0": cos_zenith_dayside.view(ncol),  # (ncol,)
        "albedo": (
            albedo * torch.ones((1, ncol), device=conc_i.device)  # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = grey_sw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= dz.view(1, 1, -1, 1)

    result = toon_sw(prop, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # set net flux to zero in layers with zero cos(zenith) to avoid numerical issues
    # with Toon solver output
    zero_cosz_mask = (bc["umu0"] == 0.0).view(ncol, 1, 1).expand_as(result)
    result[zero_cosz_mask] = 0.0

    # net flux = upward - downward
    return result[..., 0] - result[..., 1]


def compute_longwave_flux(
    grey_lw: torch.nn.Module,
    toon_lw: torch.nn.Module,
    dz: torch.Tensor,
    conc_i: torch.Tensor,
    pres_i: torch.Tensor,
    temp_i: torch.Tensor,
    *,
    albedo: float = 0.0,
) -> torch.Tensor:
    """
    Compute longwave fluxes using Toon's two-stream longwave solver.

    Args:
        grey_lw (torch.nn.Module): A module that computes grey longwave properties.
        toon_lw (torch.nn.Module): A two-stream solver for longwave radiation.
        dz (torch.Tensor): Layer thicknesses [m] (nlyr,).
        conc_i (torch.Tensor): concentrations [mol/m^3] at cell center (ncol, nlyr, nspecies).
        pres_i (torch.Tensor): Pressures [pa] at cell center (ncol, nlyr).
        temp_i (torch.Tensor): Temperatures [K] at cell center (ncol, nlyr).
        albedo (float): Surface albedo for longwave radiation.

    Returns:
        torch.Tensor: Net longwave fluxes (W/m^2) at layer interfaces (ncol, nlyr+1).
    """
    ncol, nlyr, _ = conc_i.shape

    bc: dict[str, torch.Tensor] = {
        "albedo": (
            albedo * torch.ones((1, ncol), device=conc_i.device)  # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = grey_lw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= dz.view(1, 1, -1, 1)

    # temperature at layer interfaces for thermal emission
    temf = torch.zeros((ncol, nlyr + 1), device=conc_i.device)

    temf[:, 0] = 2 * temp_i[:, 0] - temp_i[:, 1]  # extrapolate bottom interface
    temf[:, 1:-1] = 0.5 * (temp_i[:, :-1] + temp_i[:, 1:])  # average for interior
    temf[:, -1] = 2 * temp_i[:, -1] - temp_i[:, -2]  # extrapolate top interface

    result = toon_lw(prop, temf=temf, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # net flux = upward - downward
    return result[..., 0] - result[..., 1]
