import argparse

import torch
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR, kIV1, kIV2, kIV3


def select_device(block: MeshBlock, options: MeshBlockOptions) -> torch.device:
    if torch.cuda.is_available() and options.layout().backend() == "ucx":
        return torch.device(block.device())
    return torch.device("cpu")


def run_with(infile: str) -> None:
    options = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(options)
    device = select_device(block, options)
    block.to(device)

    coord = block.module("coord")
    intg = block.module("intg")

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]

    hydro_w = torch.zeros((5, nc3, nc2, nc1), device=device)
    hydro_w[kIDN] = torch.where(x1v < 0.0, 1.0, 0.125)
    hydro_w[kIPR] = torch.where(x1v < 0.0, 1.0, 0.1)
    hydro_w[kIV1] = 0.0
    hydro_w[kIV2] = 0.0
    hydro_w[kIV3] = 0.0

    radius = torch.sqrt(x1v * x1v + x2v * x2v + x3v * x3v)
    solid = radius < 0.1

    block_vars, current_time = block.initialize({"hydro_w": hydro_w, "solid": solid})
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
    parser.add_argument("--input", default="shock.yaml")
    args = parser.parse_args()
    run_with(args.input)


if __name__ == "__main__":
    main()
