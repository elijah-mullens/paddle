import argparse
import yaml
import torch
from dataclasses import dataclass

import kintera
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1
from paddle import (
    start_dist,
    close_dist,
)


@dataclass(frozen=True)
class AtmParams:
    Ts: float
    Ps: float
    Tmin: float
    grav: float
    gamma: float
    weight: float


def make_problem_params(config: dict) -> AtmParams:
    return AtmParams(
        Ts=float(config["problem"]["Ts"]),
        Ps=float(config["problem"]["Ps"]),
        Tmin=float(config["problem"]["Tmin"]),
        grav=-float(config["forcing"]["const-gravity"]["grav1"]),
        gamma=float(config["dynamics"]["equation-of-state"]["gammad"]),
        weight=float(config["dynamics"]["equation-of-state"]["weight"]),
    )


def build_user_output(rd: float, cp: float, ps: float):
    def call_user_output(
        bvars: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hydro_w = bvars["hydro_w"]
        temp = hydro_w[kIPR] / (rd * hydro_w[kIDN])
        out = {
            "temp": temp,
            "theta": temp * (ps / hydro_w[kIPR]).pow(rd / cp),
        }
        if "scalar_r" in bvars:
            out["tracer"] = bvars["scalar_r"][0]
        return out

    return call_user_output


def initialize_dry_adiabat(
    block,
    params: AtmParams,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    coord = block.module("coord")
    x3v, x2v, x1v = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )

    del x3v, x2v

    nc3, nc2, nc1 = x1v.shape
    hydro_w = torch.zeros((5, nc3, nc2, nc1), dtype=x1v.dtype, device=x1v.device)

    rd = kintera.constants.Rgas / params.weight
    cp = params.gamma / (params.gamma - 1.0) * rd

    temp_ad = params.Ts - params.grav * x1v / cp
    temp = torch.maximum(temp_ad, torch.full_like(temp_ad, params.Tmin))

    if params.Tmin < params.Ts:
        z_iso = cp * (params.Ts - params.Tmin) / params.grav
        pres_iso = params.Ps * (params.Tmin / params.Ts) ** (cp / rd)
        pres_ad = params.Ps * torch.pow(
            torch.clamp_min(temp_ad, params.Tmin) / params.Ts,
            cp / rd,
        )
        pres = torch.where(
            temp_ad > params.Tmin,
            pres_ad,
            pres_iso * torch.exp(-params.grav * (x1v - z_iso) / (rd * params.Tmin)),
        )
    else:
        pres = params.Ps * torch.exp(-params.grav * x1v / (rd * params.Tmin))

    hydro_w[kIPR] = pres
    hydro_w[kIDN] = pres / (rd * temp)
    x1min = coord.buffer("x1v")[0]
    x1max = coord.buffer("x1v")[-1]
    tracer_r = (1.0 - (x1v - x1min) / (x1max - x1min)).clamp(0.0, 1.0)
    tracer_r = tracer_r.unsqueeze(0)
    return hydro_w, tracer_r, rd, cp


def build_tracer_forcing(nghost: int):
    if nghost > 0:
        bottom_index = nghost
        top_index = -nghost - 1
    else:
        bottom_index = 0
        top_index = -1

    def call_user_forcing(
        bvars: dict[str, torch.Tensor], dt: float, stage: int
    ) -> dict[str, torch.Tensor]:
        del stage

        scalar_s = bvars["scalar_s"]
        rho = bvars["hydro_u"][kIDN].unsqueeze(0)
        target = scalar_s.new_zeros(scalar_s.shape)

        target[..., bottom_index] = rho[..., bottom_index]
        target[..., top_index] = 0.0

        scalar_ds = torch.zeros_like(scalar_s)
        tau = max(1.0e-12, dt)
        scalar_ds[..., bottom_index] = (
            target[..., bottom_index] - scalar_s[..., bottom_index]
        ) / tau
        scalar_ds[..., top_index] = (
            target[..., top_index] - scalar_s[..., top_index]
        ) / tau
        return {"scalar_ds": scalar_ds}

    return call_user_forcing


def run_with(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = start_dist(config["distribute"].get("backend", "gloo"))

    options = MeshOptions.from_yaml(args.config)
    options.block().output_dir(args.output_dir)

    mesh = Mesh(options)
    mesh.to(device)

    params = make_problem_params(config)
    nghost = int(config["geometry"]["cells"]["nghost"])
    rd = kintera.constants.Rgas / params.weight
    cp = params.gamma / (params.gamma - 1.0) * rd
    call_user_output = build_user_output(rd, cp, params.Ps)
    call_user_forcing = build_tracer_forcing(nghost)

    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        block_vars = []
        for block in mesh.blocks:
            hydro_w, tracer_r, _, _ = initialize_dry_adiabat(block, params)
            hydro_w[kIV1] += 0.1 * torch.rand_like(hydro_w[kIV1])
            block_vars.append({"hydro_w": hydro_w, "scalar_r": tracer_r})
        block_vars, current_time = mesh.initialize(block_vars)

    for block in mesh.blocks:
        block.set_user_output_func(call_user_output)
        block.set_user_forcing_func(call_user_forcing)

    intg = mesh.module("block0.intg")
    cycle = 0
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)

    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)

        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        err = mesh.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        current_time += dt
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)
    close_dist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dry Jupiter hydrodynamics.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="jupiter_dry.yaml",
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
        default="",
        help="Restart from restart dump.",
    )
    run_with(parser.parse_args())


if __name__ == "__main__":
    main()
