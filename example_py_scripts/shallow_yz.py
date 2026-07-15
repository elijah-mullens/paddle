import argparse

import torch
import yaml
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIV2, kIV3


def run_with(infile: str) -> None:
    with open(infile, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    phi = float(config["problem"]["phi"])
    uphi = float(config["problem"]["uphi"])
    dphi = float(config["problem"]["dphi"])

    options = MeshBlockOptions.from_yaml(infile, verbose=False)
    block = MeshBlock(options)
    device = torch.device(options.device_str())
    block.to(device)

    coord = block.module("coord")
    intg = block.module("intg")

    x3v, x2v, _ = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    hydro_w = torch.zeros((4, nc3, nc2, nc1), device=device)

    hydro_w[kIDN] = phi
    hydro_w[kIDN][torch.logical_and(x3v > 0.0, x3v < 5.0)] += dphi
    hydro_w[kIV3] = torch.where(x2v > 0.0, -uphi / hydro_w[kIDN], uphi / hydro_w[kIDN])
    hydro_w[kIV2] = 0.0

    block_vars, current_time = block.initialize({"hydro_w": hydro_w})
    block.make_outputs(block_vars, current_time)

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
    parser.add_argument("--input", default="shallow_yz.yaml")
    args = parser.parse_args()
    run_with(args.input)


if __name__ == "__main__":
    main()
