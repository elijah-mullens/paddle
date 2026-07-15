import argparse
import math

import kintera
import torch
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR


P0 = 1.0e5
TS = 300.0
XC = 0.0
XR = 4.0e3
ZC = 3.0e3
ZR = 2.0e3
DT = -15.0


def run_with(infile: str) -> None:
    options = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(options)
    device = torch.device(options.device_str())
    block.to(device)

    coord = block.module("coord")
    eos = block.module("hydro.eos")
    intg = block.module("intg")
    grav = -block.options.hydro().grav().grav1()

    rd = kintera.constants.Rgas / eos.options.weight()
    gamma = eos.options.gammad()
    cp = gamma / (gamma - 1.0) * rd

    def call_user_output(bvars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hydro_w = bvars["hydro_w"]
        temp = hydro_w[kIPR] / (rd * hydro_w[kIDN])
        return {
            "temp": temp,
            "theta": temp * (P0 / hydro_w[kIPR]).pow(rd / cp),
        }

    block.set_user_output_func(call_user_output)

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    hydro_w = torch.zeros((5, nc3, nc2, nc1), device=device)

    temp = TS - grav * x1v / cp
    hydro_w[kIPR] = P0 * torch.pow(temp / TS, cp / rd)
    length = torch.sqrt(((x2v - XC) / XR) ** 2 + ((x1v - ZC) / ZR) ** 2)
    temp += torch.where(
        length <= 1.0, DT * (torch.cos(length * math.pi) + 1.0) / 2.0, 0.0
    )
    hydro_w[kIDN] = hydro_w[kIPR] / (rd * temp)

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
    parser.add_argument("--input", default="straka.yaml")
    args = parser.parse_args()
    run_with(args.input)


if __name__ == "__main__":
    main()
