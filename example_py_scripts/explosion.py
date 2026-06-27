import argparse

import kintera
import torch
import yaml
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR


def select_device(block: MeshBlock, options: MeshBlockOptions) -> torch.device:
    if torch.cuda.is_available() and options.layout().backend() == "ucx":
        return torch.device(block.device())
    return torch.device("cpu")


def run_with(infile: str) -> None:
    with open(infile, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    options = MeshBlockOptions.from_yaml(infile)
    block = MeshBlock(options)
    device = select_device(block, options)
    block.to(device)

    coord = block.module("coord")
    eos = block.module("hydro.eos")
    intg = block.module("intg")

    gamma = eos.options.gammad()
    rd = kintera.constants.Rgas / eos.options.weight()
    cp = gamma / (gamma - 1.0) * rd
    p0 = float(config["problem"]["p0"])

    def call_user_output(bvars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hydro_w = bvars["hydro_w"]
        temp = hydro_w[kIPR] / (rd * hydro_w[kIDN])
        return {
            "temp": temp,
            "theta": temp * (p0 / hydro_w[kIPR]).pow(rd / cp),
        }

    block.set_user_output_func(call_user_output)

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    hydro_w = torch.zeros((eos.nvar(), nc3, nc2, nc1), device=device)

    ts = float(config["problem"]["Ts"])
    radius = float(config["problem"]["radius"])
    dt_burst = float(config["problem"]["dT"])
    dp_burst = float(config["problem"]["dP"])
    count = int(config["problem"].get("count", 5))

    temp = torch.full((nc3, nc2, nc1), ts, device=device)
    hydro_w[kIPR] = p0

    for _ in range(count):
        zc = 0.04 * torch.rand((), device=device) - 0.02
        xc = (
            0.04 * torch.rand((), device=device) - 0.02
            if nc2 > 1
            else torch.tensor(0.0, device=device)
        )
        yc = (
            0.04 * torch.rand((), device=device) - 0.02
            if nc3 > 1
            else torch.tensor(0.0, device=device)
        )
        dist = torch.sqrt((x1v - zc) ** 2 + (x2v - xc) ** 2 + (x3v - yc) ** 2)
        hot = dist < radius
        temp[hot] = dt_burst
        hydro_w[kIPR][hot] = dp_burst

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
    parser.add_argument("--input", default="explosion.yaml")
    args = parser.parse_args()
    run_with(args.input)


if __name__ == "__main__":
    main()
