import torch
import snapy
import kintera
from snapy import kIDN, kIPR, kICY


def evolve_kinetics(
    hydro_w: torch.Tensor,
    eos: snapy.EquationOfState,
    thermo_x: kintera.ThermoX,
    thermo_y: kintera.ThermoY,
    kinet: kintera.Kinetics,
    dt,
) -> torch.Tensor:
    """
    Evolve the chemical kinetics for one time step using implicit method.

    Args:
        hydro_w (torch.Tensor): The primitive variables tensor.
        eos (snapy.EquationOfState): The equation-of-state.
        thermo_x (kintera.ThermoX): The thermodynamics module for computing properties.
        thermo_y (kintera.ThermoY): The thermodynamics module for computing properties.
        kinet (kintera.Kinetics): The kinetics module for chemical reactions.
        dt (float): The time step for evolution.

    Returns:
        torch.Tensor: The change in mass density due to chemical reactions.
    """
    # compute temperature and pressure
    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]

    # compute mole fractions from mass fractions
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))

    # compute molar concentrations
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

    # compute volumetric heat capacity
    cp_vol = thermo_x.compute("TV->cp", (temp, conc))

    # narrow species for kinetics
    # conc_kinet = kinet.options.narrow_copy(conc, thermo_y.options)
    conc_kinet = conc[:, :, :, 1:]  # exclude dry species

    # compute rate and rate jacobians
    rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc_kinet)

    # compute reaction jacobian
    jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)

    # compute concentration change
    stoich = kinet.buffer("stoich")
    del_conc = kintera.evolve_implicit(rate, stoich, jac, dt)

    # compute density change
    inv_mu = thermo_y.buffer("inv_mu")
    del_rho = del_conc / inv_mu[1:].view(1, 1, 1, -1)

    # return permutated density change for hydro
    return del_rho.permute(3, 0, 1, 2)
