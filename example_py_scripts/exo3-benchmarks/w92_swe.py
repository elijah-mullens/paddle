#!/usr/bin/env python3
"""
Williamson (1992) shallow-water test case 6 — Rossby-Haurwitz wave — in snapy,
reproducing cshsgy/ExoCubed examples/2023-Chen-exo3/W92.{cpp,inp}.

Shallow water on the gnomonic-equiangle cubed sphere (snapy `shallow-water` EOS,
`shallow-roe` Riemann solver), Coriolis via the `coriolis` forcing. The
Rossby-Haurwitz initial condition (geopotential gh + winds U,V) is set per cell
using snapy's exact (face,xi,eta)->(lon,lat) map; physical (U,V) east/north are
converted to snapy contravariant (vel2,vel3) by inverting the validated
elijah-mullens/paddle contra->geographic rotation per cell.
"""
import argparse
import os

import yaml
import numpy as np
import torch
from snapy import Mesh, MeshOptions

# paddle velocity rotation (contravariant -> global Cartesian)
from paddle import cubed_sphere_remap as csr

FACE_NAMES = [
    "+X",
    "+Y",
    "-X",
    "+Z",
    "-Y",
    "-Z",
]  # snapy CS_FACE_NAMES (face-major order)

# --- Rossby-Haurwitz parameters (from W92.cpp) ---
h0 = 8000.0
G = 9.80616
OMG = 7.848e-6
A = 6.37122e6
K = 7.848e-6
R = 4.0
OM_EARTH = 7.292e-5


def ab_to_lonlat(face, alpha, beta):
    """snapy cs_ab_to_lonlat: equiangular (xi=alpha, eta=beta) -> (lon[0,2pi], lat)."""
    x = np.tan(alpha)
    y = np.tan(beta)
    r = np.sqrt(x * x + y * y + 1.0)
    if face == "+X":
        lon = alpha.copy()
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Y":
        lon = alpha + 0.5 * np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-X":
        lon = alpha + np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "-Y":
        lon = alpha + 1.5 * np.pi
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Z":
        lon = np.arctan2(x, -y)
        lat = np.arcsin(1.0 / r)
    elif face == "-Z":
        lon = np.arctan2(x, y)
        lat = -np.arcsin(1.0 / r)
    else:
        raise ValueError(face)
    lon = np.where(lon < 0.0, lon + 2 * np.pi, lon)
    return lon, lat


def rh_wave(lon, lat):
    """Rossby-Haurwitz geopotential gh and physical winds U (east), V (north)."""
    cl = np.cos(lat)
    sl = np.sin(lat)
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
        + A * A * Bb * np.cos(R * lon)
        + A * A * Cc * np.cos(2 * R * lon)
    )
    U = A * OMG * cl + A * K * cl ** (R - 1) * np.cos(R * lon) * (R * sl * sl - cl * cl)
    V = -A * K * R * cl ** (R - 1) * np.sin(R * lon) * sl
    return gh, U, V


def uv_to_contra(face_id, alpha, beta, lon, lat, U, V):
    """(U east, V north) -> snapy contravariant (vel2, vel3) by inverting the
    per-cell 2x2 contra->east/north map (vel1=0, tangential flow)."""

    def east_north(v2, v3):
        gx, gy, gz = csr._local_contra_to_global_xyz(
            face_id, np.zeros_like(v2), v2, v3, alpha, beta
        )
        east = -np.sin(lon) * gx + np.cos(lon) * gy
        north = (
            -np.sin(lat) * np.cos(lon) * gx
            - np.sin(lat) * np.sin(lon) * gy
            + np.cos(lat) * gz
        )
        return east, north

    one = np.ones_like(alpha)
    zero = np.zeros_like(alpha)
    e1, n1 = east_north(one, zero)  # response to vel2=1
    e2, n2 = east_north(zero, one)  # response to vel3=1
    det = e1 * n2 - e2 * n1
    vel2 = (n2 * U - e2 * V) / det
    vel3 = (-n1 * U + e1 * V) / det
    return vel2, vel3


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
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
        x2v = coord.buffer("x2v").cpu().numpy()  # xi (nc2,)
        x3v = coord.buffer("x3v").cpu().numpy()  # eta (nc3,)
        nc2, nc3 = x2v.size, x3v.size
        alpha, beta = np.meshgrid(x2v, x3v)  # shape (nc3, nc2): [k,j]
        lon, lat = ab_to_lonlat(FACE_NAMES[face_id], alpha, beta)
        gh, U, V = rh_wave(lon, lat)
        vel2, vel3 = uv_to_contra(face_id, alpha, beta, lon, lat, U, V)
        w = torch.zeros((4, nc3, nc2, 1), dtype=torch.float64)
        w[0, :, :, 0] = torch.from_numpy(gh)
        w[1, :, :, 0] = 0.0
        w[2, :, :, 0] = torch.from_numpy(vel2)
        w[3, :, :, 0] = torch.from_numpy(vel3)
        block_vars.append({"hydro_w": w.to(device)})
    block_vars, current_time = mesh.initialize(block_vars)

    intg = mesh.module("block0.intg")
    cycle = 0
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c", "--config", default=os.path.join(os.path.dirname(__file__), "w92.yaml")
    )
    p.add_argument("--output-dir", default="out_w92")
    run(p.parse_args())
