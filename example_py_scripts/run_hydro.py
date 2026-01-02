import torch

import yaml

import argparse
from snapy import MeshBlockOptions, MeshBlock, kICY
from kintera import ThermoX, KineticsOptions, Kinetics
from paddle import (
    setup_profile,
    evolve_kinetics,
)


def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    return out


def run_with(infile: str):
    with open(infile, "r") as f:
        config = yaml.safe_load(f)

    # use cuda if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op)
    block.to(device)

    # get handles to modules
    thermo_y = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")
    # thermo_y.options.max_iter(100)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    param = {}
    param["Ts"] = float(config["problem"]["Ts"])
    param["Ps"] = float(config["problem"]["Ps"])
    param["grav"] = -float(config["forcing"]["const-gravity"]["grav1"])
    param["Tmin"] = float(config["problem"]["Tmin"])
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))

    block_vars = {}
    block_vars["hydro_w"] = setup_profile(block, param, method="pseudo-adiabat")
    block_vars, current_time = block.initialize(block_vars)

    block.set_user_output_func(call_user_output)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(infile)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    # integration
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        del_rho = evolve_kinetics(
            block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt
        )
        block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-i", "--infile", type=str, required=True, help="Input YAML configuration file."
    )
    args = parser.parse_args()
    run_with(args.infile)


if __name__ == "__main__":
    main()
