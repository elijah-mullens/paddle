#!/usr/bin/env python3
"""
Hot-Jupiter dry GCM benchmark in snapy, reproducing cshsgy/ExoCubed
examples/2023-Chen-exo3/hot_jupiter.{cpp,inp}.

Dry primitive equations on the gnomonic-equiangle cubed sphere (ideal-moist dry
H2 species, hllc, vertical-implicit) + Coriolis. Forced by Newtonian relaxation
(fixed timescale Kt) of T toward a day-night equilibrium
    T_eq = T_vert(z, sigma) + beta(sigma) * dT_e2p * cos(lat) * cos(lon)
(hottest at the substellar point lon=lat=0) plus a Rayleigh sponge near the top.
IC: isothermal atmosphere at Ts. Expected: a strong prograde EQUATORIAL
SUPER-ROTATING jet.
"""
import argparse
import os

import yaml
import numpy as np
import torch
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3
from paddle import setup_profile

FACE_NAMES = ["+X", "+Y", "-X", "+Z", "-Y", "-Z"]

# --- hot-Jupiter parameters (hot_jupiter.cpp / .inp) ---
G = 8.0
M_MOL = 2.216e-3
RD = 8.31446 / M_MOL
CP = 3.5 * RD
CV = CP - RD
P0 = 1.0e5
TS = 1600.0
RP = 1.0e8
DT_E2P = 300.0
DT_STRA = 10.0
Z_STRA = 2.0e6
GAMMA = 2.0e-4
KT = 1.5e5
SIGMA_STRA = 0.12
SPONGE_TAU = 1.0e4
SPONGE_Z = 2.5e6


def ab_to_lonlat(face, alpha, beta):
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


def hj_forcing(hw, hu, coslat, coslon, z, dt):
    """Operator-split hot-Jupiter source on conserved hu (in place).
    coslat,coslon: (nc3,nc2,1); z: (1,1,nc1)."""
    rho = hw[kIDN]
    p = hw[kIPR]
    T = p / (rho * RD)
    sigma = p / P0
    below = (z <= Z_STRA).double()
    T_vert = below * (
        TS
        - GAMMA * (Z_STRA + z) / 2.0
        + torch.sqrt((GAMMA * (z - Z_STRA) / 2.0) ** 2 + DT_STRA**2)
    ) + (1.0 - below) * (TS - GAMMA * Z_STRA + DT_STRA)
    beta = below * torch.sin(np.pi * (sigma - SIGMA_STRA) / (2.0 * (1.0 - SIGMA_STRA)))
    Teq = T_vert + beta * DT_E2P * coslat * coslon
    # Newtonian relaxation (fixed timescale Kt) on total energy (internal = rho*cv*T)
    hu[kIPR] += -dt * CV * rho * (T - Teq) / KT
    # top Rayleigh sponge on (covariant) momentum
    spng = (z > SPONGE_Z).double()
    damp = 1.0 / (1.0 + spng * dt / SPONGE_TAU)
    hu[kIV1] *= damp
    hu[kIV2] *= damp
    hu[kIV3] *= damp


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    opt = MeshOptions.from_yaml(args.config)
    device = torch.device(opt.device_str())
    opt.block().output_dir(args.output_dir)
    mesh = Mesh(opt)
    mesh.to(device)

    global RD, CV, CP
    geom = []  # per-block (coslat, coslon, z)
    block_vars = []
    for local_index, block in enumerate(mesh.blocks):
        layout = block.get_layout()
        _, _, face_id = layout.loc_of(layout.options.rank())
        coord = block.module("coord")
        x1v = coord.buffer("x1v").cpu().numpy()
        x2v = coord.buffer("x2v").cpu().numpy()
        x3v = coord.buffer("x3v").cpu().numpy()
        alpha, beta = np.meshgrid(x2v, x3v)  # (nc3,nc2)
        lon, lat = ab_to_lonlat(FACE_NAMES[face_id], alpha, beta)
        coslat = torch.from_numpy(np.cos(lat)[..., None]).to(device, torch.float64)
        coslon = torch.from_numpy(np.cos(lon)[..., None]).to(device, torch.float64)
        z = torch.from_numpy((x1v - RP)[None, None, :]).to(device, torch.float64)
        geom.append((coslat, coslon, z))
        w = setup_profile(
            block, {"Ts": TS, "Ps": P0, "grav": G, "Tmin": 1200.0}, method="isothermal"
        )
        if local_index == 0:  # calibrate once per process from the isothermal IC
            il, jl, kl = coord.il(), coord.jl(), coord.kl()
            RD = float(w[kIPR][kl, jl, il] / w[kIDN][kl, jl, il]) / TS
            CV = 2.5 * RD
            CP = 3.5 * RD
            print(
                f"calibrated Rd={RD:.1f} J/kg/K  cp={CP:.1f}  cv={CV:.1f}", flush=True
            )
        w[kIV1] = 0.0
        w[kIV2] = 0.0
        w[kIV3] = 0.0
        block_vars.append({"hydro_w": w})
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
        for bv, block, (cla, clo, z) in zip(block_vars, mesh.blocks, geom):
            hj_forcing(bv["hydro_w"], bv["hydro_u"], cla, clo, z, dt)
            bv["hydro_w"] = block.module("hydro.eos").compute("U->W", [bv["hydro_u"]])
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
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "hjupiter.yaml"),
    )
    p.add_argument("--output-dir", default="out_hjupiter")
    run(p.parse_args())
