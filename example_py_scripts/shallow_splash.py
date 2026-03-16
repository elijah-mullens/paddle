import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as dist_c10d
import yaml
import snapy
from snapy import Mesh, MeshOptions
from snapy import kIDN, kIV2, kIV3
from snapy.coord import cs_ab_to_lonlat, get_cs_face_name


def init_dist(backend: str) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", os.environ["RANK"])

    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "gloo":
        dist.init_process_group(backend="gloo", init_method="env://")
        device = torch.device("cpu")
    elif backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", device_id=device, init_method="env://"
        )
    else:
        raise ValueError("Unsupported backend")

    snapy.distributed.set_process_group(dist_c10d._get_default_group())
    return device


def initialize_block(
    block, config: dict, device: torch.device
) -> dict[str, torch.Tensor]:
    phi = float(config["problem"]["phi"])
    dphi = float(config["problem"]["dphi"])
    radius = float(config["problem"]["radius"])

    coord = block.module("coord")
    layout = block.get_layout()
    _, _, face_id = layout.loc_of(layout.options.rank())
    face = get_cs_face_name(face_id)

    beta, alpha, r_planet = torch.meshgrid(
        coord.buffer("x3v"), coord.buffer("x2v"), coord.buffer("x1v"), indexing="ij"
    )
    _, lat = cs_ab_to_lonlat(face, alpha, beta)

    nc3 = coord.buffer("x3v").shape[0]
    nc2 = coord.buffer("x2v").shape[0]
    nc1 = coord.buffer("x1v").shape[0]
    nvar = 4

    w = torch.zeros((nvar, nc3, nc2, nc1), device=device)
    gc_dist = r_planet * (np.pi / 2.0 - lat)

    w[kIDN] = phi
    w[kIDN][torch.logical_and(gc_dist < radius, lat > np.pi / 4.0)] += dphi
    w[kIV2] = 0.0
    w[kIV3] = 0.0

    return {"hydro_w": w}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="shallow_splash.yaml")
    parser.add_argument("--output-dir", default="/data")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    device = init_dist(config["distribute"].get("backend", "gloo"))

    options = MeshOptions.from_yaml(args.input, verbose=False)
    mesh = Mesh(options)
    mesh.to(device)

    block_vars = [initialize_block(block, config, device) for block in mesh.blocks]
    block_vars, current_time = mesh.initialize(block_vars)
    mesh.make_outputs(block_vars, current_time)

    intg = mesh.module("block0.intg")
    cycle = 0
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

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
