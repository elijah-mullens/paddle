import torch
import snapy
import kintera
from snapy import kIDN, kIPR, kICY


def evolve_kinetics(
    hydro_w: torch.Tensor,
    block: snapy.MeshBlock,
    kinet: kintera.Kinetics,
    thermo_x: kintera.ThermoX,
    dt,
) -> torch.Tensor:
    """
    Evolve the chemical kinetics for one time step using implicit method.

    Args:
        hydro_w (torch.Tensor): The primitive variables tensor.
        block (snapy.MeshBlock): The mesh block containing the simulation data.
        kinet (kintera.Kinetics): The kinetics module for chemical reactions.
        thermo_x (kintera.ThermoX): The thermodynamics module for computing properties.
        dt (float): The time step for evolution.

    Returns:
        torch.Tensor: The change in mass density due to chemical reactions.
    """
    eos = block.hydro.get_eos()
    thermo_y = block.module("hydro.eos.thermo")

    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))
    cp_vol = thermo_x.compute("TV->cp", (temp, conc))

    conc_kinet = kinet.options.narrow_copy(conc, thermo_y.options)
    rate, rc_ddC, rc_ddT = kinet.forward_nogil(temp, pres, conc_kinet)
    jac = kinet.jacobian(temp, conc_kinet, cp_vol, rate, rc_ddC, rc_ddT)

    stoich = kinet.buffer("stoich")
    del_conc = kintera.evolve_implicit(rate, stoich, jac, dt)

    inv_mu = thermo_y.buffer("inv_mu")
    del_rho = del_conc / inv_mu[1:].view(1, 1, 1, -1)
    return del_rho.permute(3, 0, 1, 2)
