import argparse
import math
from collections.abc import Callable

import torch
import yaml
from kintera import ThermoX
from snapy import Mesh, MeshOptions, kICY, kIDN, kIPR


RD = 287.0
EPS = 0.621
GAMMA = 1.4
RCP_VAPOR = 1.166
RCP_LIQUID = 3.46
BETA = 24.845
T3 = 273.16
P3 = 611.7
DELTA = (RCP_LIQUID - RCP_VAPOR) * EPS / (1.0 - 1.0 / GAMMA)
RV = RD / EPS
CPD = GAMMA / (GAMMA - 1.0) * RD
CP_LIQUID = RCP_LIQUID * CPD


def species_offset(species: list[str], name: str) -> int:
    for index, species_name in enumerate(species[1:]):
        if species_name == name:
            return index
    return -1


def bryan_saturation_pressure(temp: torch.Tensor) -> torch.Tensor:
    reduced_temp = temp / T3
    return P3 * torch.exp(
        BETA * (1.0 - 1.0 / reduced_temp) - DELTA * torch.log(reduced_temp)
    )


def make_user_output_func(
    species: list[str], p0: float
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    ih2o = species_offset(species, "H2O")
    ih2oc = species_offset(species, "H2O(l)")

    def call_user_output(bvars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hydro_w = bvars["hydro_w"]
        zero = torch.zeros_like(hydro_w[kIDN])
        qv = hydro_w[kICY + ih2o] if ih2o >= 0 else zero
        qc = hydro_w[kICY + ih2oc] if ih2oc >= 0 else zero
        qtol = qv + qc

        qd = torch.clamp_min(1.0 - qtol, 1.0e-12)
        feps = 1.0 + qv * (1.0 / EPS - 1.0) - qc
        temp = hydro_w[kIPR] / (hydro_w[kIDN] * RD * feps)

        eta = qv / (qd * EPS)
        xgas = 1.0 + eta
        pd = hydro_w[kIPR] / xgas
        pv = hydro_w[kIPR] * eta / xgas
        rh = torch.clamp_min(pv / bryan_saturation_pressure(temp), 1.0e-12)

        cpt = CPD * qd + CP_LIQUID * qtol
        lv = RV * (BETA * T3 - DELTA * temp)
        theta_e = (
            temp
            * torch.pow(p0 / pd, RD * qd / cpt)
            * torch.pow(rh, -RV * qv / cpt)
            * torch.exp(lv * qv / (cpt * temp))
        )
        return {"qtol": qtol, "theta_e": theta_e}

    return call_user_output


def surface_mass_fractions(
    species: list[str],
    nc3: int,
    nc2: int,
    qt: float,
    options: dict,
) -> torch.Tensor:
    yfrac = torch.zeros((len(species) - 1, nc3, nc2), **options)
    ih2o = species_offset(species, "H2O")
    if ih2o >= 0:
        yfrac[ih2o].fill_(qt)
    return yfrac


def solve_virtual_temperature_perturbation(
    thermo_x: ThermoX,
    temp0: torch.Tensor,
    pres: torch.Tensor,
    xfrac0: torch.Tensor,
    target_tv: torch.Tensor,
    mask: torch.Tensor,
    dtemp: float,
    rd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    temp_lo = temp0.clone()
    temp_hi = temp0 + max(5.0, 2.0 * abs(dtemp))

    for _ in range(32):
        temp_mid = 0.5 * (temp_lo + temp_hi)
        xtrial = xfrac0.clone()
        thermo_x.forward(temp_mid, pres, xtrial)

        conc = thermo_x.compute("TPX->V", [temp_mid, pres, xtrial])
        dens = thermo_x.compute("V->D", [conc])
        tv_mid = pres / (dens * rd)
        too_cold = torch.logical_and(mask, tv_mid < target_tv)

        temp_lo = torch.where(too_cold, temp_mid, temp_lo)
        temp_hi = torch.where(torch.logical_and(mask, ~too_cold), temp_mid, temp_hi)

    temp_out = torch.where(mask, 0.5 * (temp_lo + temp_hi), temp0)
    xfrac_out = xfrac0.clone()
    thermo_x.forward(temp_out, pres, xfrac_out)
    return temp_out, xfrac_out


def initialize_block(
    block,
    config: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    coord = block.module("coord")
    eos = block.module("hydro.eos")
    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    species = thermo_y.options.species()
    ny = len(species) - 1
    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    il = coord.il()
    iu = coord.iu()
    dtype = torch.float64
    tensor_options = {"dtype": dtype, "device": device}

    hydro_w = torch.zeros((eos.nvar(), nc3, nc2, nc1), **tensor_options)

    problem = config["problem"]
    ps = float(problem["p0"])
    ts = float(problem["Ts"])
    xc = float(problem["xc"])
    zc = float(problem["zc"])
    xr = float(problem["xr"])
    zr = float(problem["zr"])
    dtemp = float(problem["dT"])
    qt = float(problem["qt"])
    grav = -float(config["forcing"]["const-gravity"]["grav1"])

    temp_state = torch.zeros((nc3, nc2, nc1), **tensor_options)
    pres_state = torch.zeros((nc3, nc2, nc1), **tensor_options)
    xfrac_state = torch.zeros((nc3, nc2, nc1, len(species)), **tensor_options)

    temp = torch.full((nc3, nc2), ts, **tensor_options)
    pres = torch.full((nc3, nc2), ps, **tensor_options)
    yfrac = surface_mass_fractions(species, nc3, nc2, qt, tensor_options)
    xfrac = thermo_y.compute("Y->X", [yfrac])
    thermo_x.forward(temp, pres, xfrac)

    dx1f = coord.buffer("dx1f").to(device=device, dtype=dtype)
    dz = dx1f[il].item()
    thermo_x.extrapolate_dz(temp, pres, xfrac, 0.5 * dz, grav, 0.0, False)

    for i in range(il, iu + 1):
        temp_state[:, :, i].copy_(temp)
        pres_state[:, :, i].copy_(pres)
        xfrac_state[:, :, i].copy_(xfrac)

        if i < iu:
            dz = dx1f[i].item()
            thermo_x.extrapolate_dz(temp, pres, xfrac, dz, grav, 0.0, False)

    x2 = (
        coord.buffer("x2v").to(device=device, dtype=dtype).view(1, nc2).expand(nc3, nc2)
    )
    rd = float(8.31446261815324 / thermo_x.mu[0].item())

    for i in range(il, iu + 1):
        x1 = coord.buffer("x1v")[i].to(device=device, dtype=dtype)
        length = torch.sqrt(((x2 - xc) / xr) ** 2 + ((x1 - zc) / zr) ** 2)
        mask = length < 1.0
        amp = dtemp * torch.cos(0.5 * math.pi * length).square() / 300.0

        temp_i = temp_state[:, :, i]
        pres_i = pres_state[:, :, i]
        xfrac_i = xfrac_state[:, :, i]
        conc_i = thermo_x.compute("TPX->V", [temp_i, pres_i, xfrac_i])
        dens_i = thermo_x.compute("V->D", [conc_i])
        target_tv = pres_i / (dens_i * rd) * (1.0 + amp)

        temp_new, xfrac_new = solve_virtual_temperature_perturbation(
            thermo_x, temp_i, pres_i, xfrac_i, target_tv, mask, dtemp, rd
        )
        temp_i.copy_(temp_new)
        xfrac_i.copy_(xfrac_new)

    for i in range(il, iu + 1):
        temp_i = temp_state[:, :, i]
        pres_i = pres_state[:, :, i]
        xfrac_i = xfrac_state[:, :, i]
        conc_i = thermo_x.compute("TPX->V", [temp_i, pres_i, xfrac_i])

        hydro_w[kIPR, :, :, i].copy_(pres_i)
        hydro_w[kIDN, :, :, i].copy_(thermo_x.compute("V->D", [conc_i]))
        hydro_w[kICY : kICY + ny, :, :, i].copy_(thermo_x.compute("X->Y", [xfrac_i]))

    return {"hydro_w": hydro_w}


def run_with(infile: str, restart_file: str = "") -> None:
    with open(infile, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    options = MeshOptions.from_yaml(infile)
    mesh = Mesh(options)
    device = torch.device(options.device_str())
    mesh.to(device)
    blocks = list(mesh.blocks)

    for block in blocks:
        species = block.module("hydro.eos.thermo").options.species()
        block.set_user_output_func(
            make_user_output_func(species, float(config["problem"]["p0"]))
        )

    if restart_file:
        mesh_vars, current_time = mesh.initialize_from_restart(restart_file)
    else:
        mesh_vars = [initialize_block(block, config, device) for block in blocks]
        mesh_vars, current_time = mesh.initialize(mesh_vars)

    mesh.make_outputs(mesh_vars, current_time)

    intg = blocks[0].module("intg")
    cycle = blocks[0].cycle()
    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)

        dt = mesh.max_time_step(mesh_vars)
        mesh.print_cycle_info(mesh_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)

        err = mesh.check_redo(mesh_vars)
        if err > 0:
            cycle = blocks[0].cycle()
            continue
        if err < 0:
            break

        current_time += dt
        mesh.make_outputs(mesh_vars, current_time)

    mesh.finalize(mesh_vars, current_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="bryan.yaml")
    parser.add_argument("-r", "--restart", default="")
    args = parser.parse_args()
    run_with(args.input, args.restart)


if __name__ == "__main__":
    main()
