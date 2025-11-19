from typing import Tuple

import torch
import snapy
import kintera


def integrate_neutral(
    thermo_x: kintera.ThermoX,
    temp: torch.Tensor,
    pres: torch.Tensor,
    xfrac: torch.Tensor,
    grav: float,
    dz: float,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A neutral density profile assumes no cloud and:

        (1) dP/dz = -rho*g
        (2) d(rho)/dz = ...

    In discretized form:

        rho_bar = 0.5 * (rho_old + rho_ad)
        P_new = P_old - rho_bar * g * dz
    """
    conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
    rho = thermo_x.compute("V->D", [conc])

    # make an adiabatic step first
    temp_ad = temp.clone()
    pres_ad = pres.clone()
    xfrac_ad = xfrac.clone()

    thermo_x.extrapolate_ad(temp_ad, pres_ad, xfrac_ad, grav, dz)
    conc_ad = thermo_x.compute("TPX->V", [temp_ad, pres_ad, xfrac_ad])
    rho_ad = thermo_x.compute("V->D", [conc_ad])
    rho_bar = 0.5 * (rho + rho_ad)

    pres2 = pres - rho_bar * grav * dz

    # initial guess
    temp2 = temp_ad.clone()
    count = 0
    while count < max_iter:
        xfrac2 = xfrac_ad.clone()

        # equilibrate clouds
        thermo_x.forward(temp2, pres2, xfrac2)

        # drop clouds fractions
        for cid in thermo_x.options.cloud_ids():
            xfrac2[..., cid] = 0.0
        # renormalize mole fractions
        xfrac2 /= xfrac.sum(dim=-1, keepdim=True)

        conc2 = thermo_x.compute("TPX->V", [temp2, pres2, xfrac2])
        rho2 = thermo_x.compute("V->D", [conc2])

        if torch.allclose(rho2, rho_ad):
            break

        temp2 -= temp2 * (rho_ad - rho2) / rho2
        count += 1

    if count == max_iter:
        raise RuntimeError("neutral density integration did not converge.")

    return temp2, pres2, xfrac2


def integrate_dry_adiabat(
    thermo_x: kintera.ThermoX,
    temp: torch.Tensor,
    pres: torch.Tensor,
    xfrac: torch.Tensor,
    grav: float,
    dz: float,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A dry adiabatic profile assumes no cloud and:

        (1) dT/dz = -g/cp
        (2) dP/dz = -rho*g

    In discretized form:

        cp_bar = 0.5 * (cp(T_old) + cp(T_new))
        T_new = T_old - g/bar * dz
        rho_bar = 0.5 * (rho_old + rho_new)
        P_new = P_old - rho_bar * g * dz

    """
    conc1 = thermo_x.compute("TPX->V", [temp, pres, xfrac])
    cp1 = thermo_x.compute("TV->cp", [temp, conc1]) / conc1.sum(-1)
    rho1 = thermo_x.compute("V->D", [conc1])
    mmw1 = (thermo_x.mu * xfrac).sum(-1)

    # initial guess
    temp2 = temp - grav * mmw1 * dz / cp1
    pres2 = pres - rho1 * grav * dz

    count = 0
    while count < max_iter:
        xfrac2 = xfrac.clone()

        # equilibrate clouds
        thermo_x.forward(temp2, pres2, xfrac2)

        # drop clouds fractions
        for cid in thermo_x.options.cloud_ids():
            xfrac2[..., cid] = 0.0
        # renormalize mole fractions
        xfrac2 /= xfrac.sum(dim=-1, keepdim=True)

        conc2 = thermo_x.compute("TPX->V", [temp2, pres2, xfrac2])
        cp2 = thermo_x.compute("TV->cp", [temp2, conc2]) / conc2.sum(-1)
        rho2 = thermo_x.compute("V->D", [conc2])
        mmw2 = (thermo_x.mu * xfrac2).sum(-1)

        cp_bar = 0.5 * (cp1 / mmw1 + cp2 / mmw2)
        rho_bar = 0.5 * (rho1 + rho2)

        temp_new = temp - grav * dz / cp_bar
        pres_new = pres - rho_bar * grav * dz

        if torch.allclose(temp_new, temp2) and torch.allclose(pres_new, pres2):
            break

        temp2 = temp_new
        pres2 = pres_new
        count += 1

    if count == max_iter:
        raise RuntimeError("Dry adiabat integration did not converge.")

    return temp2, pres2, xfrac2


def setup_profile(
    block: snapy.MeshBlock,
    param: dict[str, float] = {},
    method: str = "moist-adiabat",
    verbose: bool = False,
) -> torch.Tensor:
    """
    Set up an adiabatic initial condition for the mesh block.

    This function initializes the primitive variables in the mesh block
    and returns the initialized tensor.

    Args:
        block (snapy.MeshBlock): The mesh block to set up.
        param (dict[str, float], optional): Parameters for the adiabat setup. Defaults to {}.
        method (str, optional): Method for the adiabat setup. Choose between
            (1) "dry-adiabat"
            (2) "moist-adiabat"
            (3) "isothermal"
            (4) "pseudo-adiabat"
            (5) "neutral"
            Defaults to "moist-adiabat".

        Required parameters in `param`:
            Ts (float): Surface temperature in Kelvin. Default is 300 K.
            Ps (float): Surface pressure in Pascals. Default is 1e5 Pa.
            x<species> (float): Mole fraction of a specific species (e.g., xH2O for
            water vapor). Default is 0.0.
            grav (float): Gravitational acceleration in m/s^2. Default is 9.8 m/s^2.

    Returns:
        torch.Tensor: The initialized primitive variables tensor.
    """

    # check method
    valid_methods = [
        "dry-adiabat",
        "moist-adiabat",
        "isothermal",
        "pseudo-adiabat",
        "neutral",
    ]

    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from {valid_methods}.")

    Ts = param.get("Ts", 300.0)
    Ps = param.get("Ps", 1.0e5)
    grav = param.get("grav", 9.8)
    Tmin = param.get("Tmin", 0.0)

    # get handles to modules
    coord = block.module("hydro.coord")
    thermo_y = block.module("hydro.eos.thermo")

    # get coordinates
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    # handling mole fractions
    thermo_x = kintera.ThermoX(thermo_y.options)
    thermo_x.to(dtype=x1v.dtype, device=x1v.device)

    # get dimensions
    nc3, nc2, nc1 = x1v.shape
    ny = len(thermo_y.options.species()) - 1
    nvar = 5 + ny

    w = torch.zeros((nvar, nc3, nc2, nc1), dtype=x1v.dtype, device=x1v.device)

    temp = Ts * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)
    pres = Ps * torch.ones((nc3, nc2), dtype=w.dtype, device=w.device)
    xfrac = torch.zeros((nc3, nc2, ny + 1), dtype=w.dtype, device=w.device)

    for name in thermo_y.options.species():
        index = thermo_y.options.species().index(name)
        xfrac[..., index] = param.get(f"x{name}", 0.0)

    # dry air mole fraction
    xfrac[..., 0] = 1.0 - xfrac[..., 1:].sum(dim=-1)

    # start and end indices for the vertical direction
    # excluding ghost cells
    ifirst = coord.ifirst()
    ilast = coord.ilast()

    # vertical grid distance of the first cell
    dz = coord.buffer("dx1f")[ifirst]

    # half a grid to cell center
    thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz / 2.0, verbose=verbose)

    # adiabatic extrapolation
    if method == "isothermal":
        i_isothermal = ifirst
        ifirst = ilast
    else:
        i_isothermal = ilast

    for i in range(ifirst, ilast):
        # drop clouds fractions
        if method.split("-")[0] != "moist":
            for cid in thermo_x.options.cloud_ids():
                xfrac[..., cid] = 0.0
            # renormalize mole fractions
            xfrac /= xfrac.sum(dim=-1, keepdim=True)
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])

        w[snapy.index.ipr, ..., i] = pres
        w[snapy.index.idn, ..., i] = thermo_x.compute("V->D", [conc])
        w[snapy.index.icy :, ..., i] = thermo_x.compute("X->Y", [xfrac])

        dz = coord.buffer("dx1f")[i]
        if method.split("-")[0] == "dry":
            temp, pres, xfrac = integrate_dry_adiabat(
                thermo_x, temp, pres, xfrac, grav, dz
            )
        elif method.split("-")[0] == "neutral":
            temp, pres, xfrac = integrate_neutral(thermo_x, temp, pres, xfrac, grav, dz)
        else:
            thermo_x.extrapolate_ad(temp, pres, xfrac, grav, dz, verbose=verbose)

        if torch.any(temp < Tmin):
            i_isothermal = i + 1
            break

    # isothermal extrapolation
    for i in range(i_isothermal, ilast):
        # drop clouds fractions
        if method.split("-")[0] != "moist":
            for cid in thermo_x.options.cloud_ids():
                xfrac[..., cid] = 0.0
            # renormalize mole fractions
            xfrac /= xfrac.sum(dim=-1, keepdim=True)

        mu = (thermo_x.mu * xfrac).sum(-1)
        dz = coord.buffer("dx1f")[i]
        pres *= torch.exp(-grav * mu * dz / (kintera.constants.Rgas * temp))
        conc = thermo_x.compute("TPX->V", [temp, pres, xfrac])
        w[snapy.index.ipr, ..., i] = pres
        w[snapy.index.idn, ..., i] = thermo_x.compute("V->D", [conc])
        w[snapy.index.icy :, ..., i] = thermo_x.compute("X->Y", [xfrac])
    return w
