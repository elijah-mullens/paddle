import argparse
import yaml
import torch
from dataclasses import dataclass

from snapy import Mesh, MeshOptions, kICY, kIV1
from snapy import EquationOfState
from kintera import ThermoX, ThermoY, KineticsOptions, Kinetics
from paddle import (
    start_dist,
    close_dist,
    setup_profile,
    evolve_kinetics,
)
from spherical_random import randomize_initial_velocity


@dataclass(frozen=True)
class ThermoConfigs:
    eos: EquationOfState
    thermo_x: ThermoX
    thermo_y: ThermoY


def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    out = {}
    out["qtol"] = hydro_w[kICY:].sum(dim=0)
    return out


def make_problem_params(config: dict, species: list[str]) -> dict[str, float]:
    param = {
        "Ts": float(config["problem"]["Ts"]),
        "Ps": float(config["problem"]["Ps"]),
        "grav": -float(config["forcing"]["const-gravity"]["grav1"]),
        "Tmin": float(config["problem"]["Tmin"]),
    }
    for name in species:
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))
    return param


def run_with(args: argparse.Namespace):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = start_dist(config["distribute"].get("backend", "gloo"))

    options = MeshOptions.from_yaml(args.config)
    options.block().output_dir(args.output_dir)

    mesh = Mesh(options)
    mesh.to(device)

    thermo_configs: list[ThermoConfigs] = []
    for block in mesh.blocks:
        thermo_y = block.module("hydro.eos.thermo")
        thermo_x = ThermoX(thermo_y.options)
        thermo_x.to(device)
        thermo_configs.append(
            ThermoConfigs(
                eos=block.module("hydro.eos"), thermo_y=thermo_y, thermo_x=thermo_x
            )
        )

    if args.restart != "":
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        species = thermo_configs[0].thermo_y.options.species()
        param: dict[str, float] = make_problem_params(config, species)
        block_vars = []
        for block in mesh.blocks:
            hydro_w = setup_profile(block, param, method="pseudo-adiabat")
            hydro_w[kIV1] += randomize_initial_velocity(block, hydro_w[kIV1])
            block_vars.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(block_vars)

    for block in mesh.blocks:
        block.set_user_output_func(call_user_output)

    # kinetics model
    op_kinet = KineticsOptions.from_yaml(args.config)
    kinet = Kinetics(op_kinet)
    kinet.to(device)

    intg = mesh.module("block0.intg")
    cycle = 0
    if args.restart == "":
        mesh.make_outputs(block_vars, current_time)

    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)

        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        for bvars, thermo in zip(block_vars, thermo_configs):
            del_rho = evolve_kinetics(
                bvars["hydro_w"],
                thermo.eos,
                thermo.thermo_x,
                thermo.thermo_y,
                kinet,
                dt,
            )
            bvars["hydro_u"][kICY:] += del_rho

        err = mesh.check_redo(block_vars)
        if err > 0:
            continue  # redo current step
        if err < 0:
            break  # terminate

        current_time += dt
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)
    close_dist()


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Run hydrodynamic simulation.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        default="jupiter_crm.yaml",
        help="Input YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "-r",
        "--restart",
        type=str,
        required=False,
        help="Restart from restart dump.",
        default="",
    )
    run_with(parser.parse_args())


if __name__ == "__main__":
    main()
