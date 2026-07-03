import argparse

import torch
import yaml
from snapy import MeshBlock, MeshBlockOptions, kICY, kIV1
from paddle import setup_profile


def select_device(options: MeshBlockOptions) -> torch.device:
    if torch.cuda.is_available() and options.layout().backend() == "ucx":
        return torch.device(options.device_str())
    return torch.device("cpu")


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


def run_with(infile: str) -> None:
    with open(infile, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    options = MeshBlockOptions.from_yaml(infile, verbose=False)
    block = MeshBlock(options)
    device = select_device(options)
    block.to(device)

    thermo_y = block.module("hydro.eos.thermo")
    intg = block.module("intg")
    param = make_params(config, thermo_y.options.species())

    def call_user_output(bvars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hydro_w = bvars["hydro_w"]
        return {"qtol": hydro_w[kICY:].sum(dim=0)}

    block.set_user_output_func(call_user_output)

    hydro_w = setup_profile(block, param, method="pseudo-adiabat")
    hydro_w[kIV1] += 0.01 * torch.rand_like(hydro_w[kIV1])

    block_vars, current_time = block.initialize({"hydro_w": hydro_w})
    block.make_outputs(block_vars, current_time)

    if config.get("dynamics", {}).get("disable", False):
        block.finalize(block_vars, current_time)
        return

    while not intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            block.forward(block_vars, dt, stage)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        current_time += dt
        block.make_outputs(block_vars, current_time)

    block.finalize(block_vars, current_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="earth_moist.yaml")
    args = parser.parse_args()
    run_with(args.input)


if __name__ == "__main__":
    main()
