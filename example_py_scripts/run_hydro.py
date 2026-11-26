import argparse

import torch
import yaml
from kintera import KineticsOptions
from snapy import Mesh, MeshOptions, kICY, kIV1
from paddle import setup_profile


def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    return out


def make_params(config: dict, species: list[str]) -> dict[str, float]:
    param = {
        "Ts": float(config["problem"]["Ts"]),
        "Ps": float(config["problem"]["Ps"]),
        "grav": -float(config["forcing"]["const-gravity"]["grav1"]),
        "Tmin": float(config["problem"]["Tmin"]),
    }
    for name in species:
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))
    return param


def run_with(infile: str, restart_file: str):
    with open(infile, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    options = MeshOptions.from_yaml(infile)
    mesh = Mesh(options)
    blocks = list(mesh.blocks)

    device = torch.device(options.device_str())
    print("device = ", device)

    mesh.to(device)

    if config.get("reactions"):
        mesh.set_chemistry(KineticsOptions.from_yaml(infile))

    if restart_file != "":
        block_vars, current_time = mesh.initialize_from_restart(restart_file)
    else:
        species = blocks[0].module("hydro.eos.thermo").options.species()
        param = make_params(config, species)
        block_vars = []
        for block in blocks:
            hydro_w = setup_profile(block, param, method="pseudo-adiabat")
            hydro_w[kIV1] += 0.1 * torch.rand_like(hydro_w[kIV1])
            block_vars.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(block_vars)

    for block in blocks:
        block.set_user_output_func(call_user_output)

    intg = mesh.module("block0.intg")
    cycle = blocks[0].cycle()
    if restart_file == "":
        mesh.make_outputs(block_vars, current_time)

    while not intg.stop(cycle, current_time):
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        err = mesh.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        cycle += 1
        mesh.set_cycle(cycle)
        current_time += dt
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-i", "--infile", type=str, required=True, help="Input YAML configuration file."
    )
    parser.add_argument(
        "-r",
        "--restart",
        type=str,
        required=False,
        help="Restart from restart dump.",
        default="",
    )
    args = parser.parse_args()
    run_with(args.infile, args.restart)


if __name__ == "__main__":
    main()
