import torch

import yaml

import argparse
from snapy import MeshBlockOptions, MeshBlock, kICY, kIV1
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


def run_with(infile: str, restart_file:str):
    with open(infile, "r") as f:
        config = yaml.safe_load(f)

    # set hydrodynamic options
    op = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(op)

    # use cuda if available
    if torch.cuda.is_available() and op.layout().backend() == "nccl":
        device = torch.device(block.device())
        print("device = ", device)
    else:
        device = torch.device("cpu")

    block.to(device)

    # get handles to modules
    coord = block.module("coord")
    thermo_y = block.module("hydro.eos.thermo")
    eos = block.module("hydro.eos")
    # thermo_y.options.max_iter(100)

    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    block_vars = {}

    if restart_file != '':
        module = torch.jit.load(restart_file)
        for name, data in module.named_buffers():
            block_vars[name] = data.to(device)
    else:
        param = {}
        param["Ts"] = float(config["problem"]["Ts"])
        param["Ps"] = float(config["problem"]["Ps"])
        param["grav"] = -float(config["forcing"]["const-gravity"]["grav1"])
        param["Tmin"] = float(config["problem"]["Tmin"])
        for name in thermo_y.options.species():
            param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))

        block_vars["hydro_w"] = setup_profile(block, param, method="pseudo-adiabat")

        # add random vertical velocity
        block_vars["hydro_w"][kIV1] += 0.1 * torch.rand_like(block_vars["hydro_w"][kIV1])
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
        "-i", "--infile", type=str, 
        required=True, help="Input YAML configuration file."
    )
    parser.add_argument(
        "-r", "--restart", type=str, 
        required=False, help="Restart from restart dump.",
        default=""
    )
    args = parser.parse_args()
    run_with(args.infile, args.restart)


if __name__ == "__main__":
    main()
