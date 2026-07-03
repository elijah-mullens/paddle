#!/usr/bin/env python3
"""
Held-Suarez 1994 dry dynamical-core benchmark in snapy, reproducing
cshsgy/ExoCubed examples/2023-Chen-exo3/hs94.{cpp,inp}.

Dry primitive equations on the gnomonic-equiangle cubed sphere (snapy `ideal-gas`
EOS, lmars, vertical-implicit), Coriolis via the `coriolis` forcing. The HS94
forcing — Newtonian relaxation of T toward Teq(lat,sigma) and low-level Rayleigh
friction — is not a built-in snapy module, so it is applied as an operator-split
source on the conserved state each step (matching the ExoCubed `Forcing`).
IC: dry-adiabatic hydrostatic profile (theta=Ts up to z_iso, then isothermal),
seeded with small random horizontal velocity.
"""
import argparse
import os

import yaml
import numpy as np
import torch
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3
from paddle import start_dist, close_dist, setup_profile

FACE_NAMES = ["+X", "+Y", "-X", "+Z", "-Y", "-Z"]

# --- HS94 + Earth dry-air parameters (from hs94.cpp / hs94.inp) ---
# Rd/cp from the snapy "dry" species (M=29 g/mol, cv_R=2.5 -> gamma=1.4)
G = 9.81
RD = 8.31446 / 0.029
CP = 3.5 * RD
CV = CP - RD
KAPPA = RD / CP
P0 = 1.0e5
TS = 315.0
DT_h = 60.0
DTHETA = 10.0
SIGMAB = 0.7
Z_ISO = 2.0e4
RP = 6.371e6
DAY = 86400.0
KF = 1.0 / DAY
KA = 0.025 / DAY
KS = 0.25 / DAY
TEQ_FLOOR = 200.0


def ab_to_lat(face, alpha, beta):
    x = np.tan(alpha)
    y = np.tan(beta)
    r = np.sqrt(x * x + y * y + 1.0)
    if face in ("+X", "+Y", "-X", "-Y"):
        lat = np.arctan(y / np.sqrt(1 + x * x))
    elif face == "+Z":
        lat = np.arcsin(1.0 / r)
    elif face == "-Z":
        lat = -np.arcsin(1.0 / r)
    else:
        raise ValueError(face)
    return lat


def dry_adiabat_profile(z):
    """T(z), p(z), rho(z): dry adiabat (theta=Ts) up to z_iso, isothermal above."""
    z = np.asarray(z, dtype=np.float64)
    T_iso = TS - G * Z_ISO / CP
    p_iso = P0 * (T_iso / TS) ** (CP / RD)
    T = np.where(z <= Z_ISO, TS - (G / CP) * z, T_iso)
    p = np.where(
        z <= Z_ISO,
        P0 * (np.clip(TS - (G / CP) * z, 1.0, None) / TS) ** (CP / RD),
        p_iso * np.exp(-G * (z - Z_ISO) / (RD * T_iso)),
    )
    rho = p / (RD * T)
    return T, p, rho


def hs_forcing(hw, hu, lat_col, dt):
    """Operator-split HS94 source on conserved state hu (in place).
    hw,hu: (5, nc3, nc2, nc1); lat_col: (nc3, nc2, 1) latitude broadcast over height."""
    rho = hw[kIDN]
    p = hw[kIPR]
    T = p / (rho * RD)
    sigma = p / P0
    coslat = torch.cos(lat_col)
    sinlat = torch.sin(lat_col)
    Teq = (
        TS
        - DT_h * sinlat**2
        - DTHETA * torch.log(torch.clamp(sigma, min=1e-12)) * coslat**2
    ) * sigma**KAPPA
    Teq = torch.clamp(Teq, min=TEQ_FLOOR)
    sigma_p = torch.clamp((sigma - SIGMAB) / (1.0 - SIGMAB), min=0.0)
    Kv = sigma_p * KF
    Kt = KA + (KS - KA) * sigma_p * coslat**4
    # Newtonian cooling on total energy (internal energy = rho*cv*T)
    hu[kIPR] += -dt * rho * CV * Kt * (T - Teq)
    # low-level Rayleigh friction on (covariant) momentum, implicit/stable
    damp = 1.0 / (1.0 + dt * Kv)
    hu[kIV1] *= damp
    hu[kIV2] *= damp
    hu[kIV3] *= damp


def run(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = start_dist(config["distribute"].get("backend", "gloo"))
    opt = MeshOptions.from_yaml(args.config)
    opt.block().output_dir(args.output_dir)
    mesh = Mesh(opt)
    mesh.to(device)
    rng = np.random.default_rng(0)

    lats = []  # per-block (nc3,nc2,1) latitude
    block_vars = []
    for f, block in enumerate(mesh.blocks):
        coord = block.module("coord")
        x1v = coord.buffer("x1v").cpu().numpy()
        x2v = coord.buffer("x2v").cpu().numpy()
        x3v = coord.buffer("x3v").cpu().numpy()
        alpha, beta = np.meshgrid(x2v, x3v)  # (nc3,nc2)
        lat2d = ab_to_lat(FACE_NAMES[f], alpha, beta)  # (nc3,nc2)
        lats.append(torch.from_numpy(lat2d[..., None]).to(device, torch.float64))
        # dry-adiabatic hydrostatic IC (theta=Ts up to where T hits Tmin, then isothermal)
        w = setup_profile(
            block, {"Ts": TS, "Ps": P0, "grav": G, "Tmin": 120.0}, method="dry-adiabat"
        )
        w[kIV1] = 0.0
        # start from rest: grid-scale white-noise seeding drives negative density at the
        # cube corners; the cubed-sphere grid's zonal asymmetry seeds the baroclinic eddies.
        _ = x1v  # (kept for reference; no explicit perturbation)
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
        # HS94 Newtonian cooling + Rayleigh drag (operator split), then refresh prim
        for bv, block, lat_col in zip(block_vars, mesh.blocks, lats):
            hs_forcing(bv["hydro_w"], bv["hydro_u"], lat_col, dt)
            bv["hydro_w"] = block.module("hydro.eos").compute("U->W", [bv["hydro_u"]])
        err = mesh.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break
        current_time += dt
        mesh.make_outputs(block_vars, current_time)
    mesh.finalize(block_vars, current_time)
    close_dist()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c", "--config", default=os.path.join(os.path.dirname(__file__), "hs94.yaml")
    )
    p.add_argument("--output-dir", default="out_hs94")
    run(p.parse_args())
