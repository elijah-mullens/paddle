from __future__ import annotations

import argparse
import math

import kintera
import torch
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

from paddle import conservative_refine, refine_meshblock


P0 = 1.0e5
TS = 300.0
XC = 0.0
XR = 4.0e3
ZC = 3.0e3
ZR = 2.0e3
DT = -15.0


def select_device(block: MeshBlock, options: MeshBlockOptions) -> torch.device:
    if torch.cuda.is_available() and options.layout().backend() == "nccl":
        return torch.device(block.device())
    return torch.device("cpu")


def set_user_output(block: MeshBlock) -> None:
    eos = block.module("hydro.eos")
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


def initialize_coarse_state(
    block: MeshBlock,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    coord = block.module("coord")
    eos = block.module("hydro.eos")
    grav = -block.options.hydro().grav().grav1()
    rd = kintera.constants.Rgas / eos.options.weight()
    gamma = eos.options.gammad()
    cp = gamma / (gamma - 1.0) * rd

    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )
    hydro_w = torch.zeros((5, x3v.shape[0], x2v.shape[1], x1v.shape[2]), device=device)

    temp = TS - grav * x1v / cp
    hydro_w[kIPR] = P0 * torch.pow(temp / TS, cp / rd)
    length = torch.sqrt(((x2v - XC) / XR) ** 2 + ((x1v - ZC) / ZR) ** 2)
    temp += torch.where(
        length <= 1.0, DT * (torch.cos(length * math.pi) + 1.0) / 2.0, 0.0
    )
    hydro_w[kIDN] = hydro_w[kIPR] / (rd * temp)
    return block.initialize({"hydro_w": hydro_w})


def run_until(
    block: MeshBlock,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    end_time: float,
) -> tuple[dict[str, torch.Tensor], float]:
    intg = block.module("intg")
    while current_time < end_time and not intg.stop(block.inc_cycle(), current_time):
        dt = min(block.max_time_step(block_vars), end_time - current_time)
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

    return block_vars, current_time


def refine_state(
    block: MeshBlock,
    block_vars: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[MeshBlock, dict[str, torch.Tensor]]:
    nghost = block.options.coord().nghost()
    refined_hydro_u = conservative_refine(block_vars["hydro_u"], nghost)
    coarse_nx2 = block.options.coord().nx2()

    refined_block = refine_meshblock(block)
    refined_block.to(device)
    set_user_output(refined_block)

    eos = refined_block.module("hydro.eos")
    hydro_w = eos.compute("U->W", (refined_hydro_u,))
    refined_vars, _ = refined_block.initialize({"hydro_w": hydro_w})

    print(
        f"Refined Straka mesh: nx2 {coarse_nx2} -> "
        f"{refined_block.options.coord().nx2()}",
        flush=True,
    )
    return refined_block, refined_vars


def run_with(infile: str, refine_time: float | None = None) -> None:
    options = MeshBlockOptions.from_yaml(infile)
    final_time = options.intg().tlim()
    refine_time = final_time / 2.0 if refine_time is None else refine_time
    if not 0.0 < refine_time < final_time:
        raise ValueError(
            f"refine_time must be between 0 and tlim={final_time}, got {refine_time}"
        )

    block = MeshBlock(options)
    device = select_device(block, options)
    block.to(device)
    set_user_output(block)

    block_vars, current_time = initialize_coarse_state(block, device)
    block.make_outputs(block_vars, current_time)
    block_vars, current_time = run_until(block, block_vars, current_time, refine_time)

    block, block_vars = refine_state(block, block_vars, device)
    block_vars, current_time = run_until(block, block_vars, current_time, final_time)
    block.finalize(block_vars, current_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Straka density-current case with one refinement."
    )
    parser.add_argument("--input", default="straka.yaml")
    parser.add_argument(
        "--refine-time",
        type=float,
        default=None,
        help="Simulation time for refinement; defaults to half of integration.tlim.",
    )
    args = parser.parse_args()
    run_with(args.input, args.refine_time)


if __name__ == "__main__":
    main()
