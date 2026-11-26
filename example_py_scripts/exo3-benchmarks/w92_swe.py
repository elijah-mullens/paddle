#!/usr/bin/env python3
"""
Williamson (1992) shallow-water test case 6 — Rossby-Haurwitz wave — in snapy,
reproducing cshsgy/ExoCubed examples/2023-Chen-exo3/W92.{cpp,inp}.

Shallow water on the gnomonic-equiangle cubed sphere (snapy `shallow-water` EOS,
`shallow-roe` Riemann solver), Coriolis via the `coriolis` forcing. The
Rossby-Haurwitz initial condition (geopotential gh + winds U,V) is set per cell
using snapy's exact (face,xi,eta)->(lon,lat) map; physical (U,V) east/north are
converted to panel velocities with Snapy's spherical-to-contravariant transform.
"""
import argparse
import os

import torch
from snapy import Mesh, MeshOptions

from snapy.coord import cs_ab_to_lonlat, cs_sph_to_contra_, get_cs_face_name

# --- Rossby-Haurwitz parameters (from W92.cpp) ---
h0 = 8000.0
G = 9.80616
OMG = 7.848e-6
A = 6.37122e6
K = 7.848e-6
R = 4.0
OM_EARTH = 7.292e-5


def rh_wave(lon, lat):
    """Rossby-Haurwitz geopotential gh and physical winds U (east), V (north)."""
    cl = torch.cos(lat)
    sl = torch.sin(lat)
    Aa = 0.5 * OMG * (2 * OM_EARTH + OMG) * cl**2 + 0.25 * K * K * (
        (R + 1) * cl ** (2 * R + 2)
        + (2 * R * R - R - 2) * cl ** (2 * R)
        - 2 * R * R * cl ** (2 * R - 2)
    )
    Bb = (
        2
        * (OM_EARTH + OMG)
        * K
        / ((R + 1) * (R + 2))
        * cl**R
        * ((R * R + 2 * R + 2) - (R + 1) ** 2 * cl**2)
    )
    Cc = 0.25 * K * K * cl ** (2 * R) * ((R + 1) * cl**2 - (R + 2))
    gh = (
        G * h0
        + A * A * Aa
        + A * A * Bb * torch.cos(R * lon)
        + A * A * Cc * torch.cos(2 * R * lon)
    )
    U = A * OMG * cl + A * K * cl ** (R - 1) * torch.cos(R * lon) * (
        R * sl * sl - cl * cl
    )
    V = -A * K * R * cl ** (R - 1) * torch.sin(R * lon) * sl
    return gh, U, V


def run(args):
    opt = MeshOptions.from_yaml(args.config)
    device = torch.device(opt.device_str())
    opt.block().output_dir(args.output_dir)
    mesh = Mesh(opt)
    mesh.to(device)

    block_vars = []
    for block in mesh.blocks:
        layout = block.get_layout()
        _, _, face_id = layout.loc_of(layout.options.rank())
        coord = block.module("coord")
        beta, alpha = torch.meshgrid(
            coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij"
        )
        lon, lat = cs_ab_to_lonlat(get_cs_face_name(face_id), alpha, beta)
        gh, east, north = rh_wave(lon, lat)
        # Spherical velocity components point radially outward, south and east.
        w = torch.stack((gh, torch.zeros_like(gh), -north, east))
        cs_sph_to_contra_(w[1:], alpha, beta, face_id)
        block_vars.append({"hydro_w": w.unsqueeze(-1)})
    block_vars, current_time = mesh.initialize(block_vars)

    intg = mesh.module("block0.intg")
    cycle = 0
    mesh.make_outputs(block_vars, current_time)
    while not intg.stop(cycle, current_time):
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)
        err = mesh.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break
        cycle += 1
        mesh.set_cycle(cycle)
        current_time += dt
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c", "--config", default=os.path.join(os.path.dirname(__file__), "w92.yaml")
    )
    p.add_argument("--output-dir", default="out_w92")
    run(p.parse_args())
