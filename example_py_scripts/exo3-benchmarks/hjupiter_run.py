#!/usr/bin/env python3
"""
Hot-Jupiter dry GCM benchmark in snapy based on
cshsgy/ExoCubed/examples/2023-Chen-exo3/hot_jupiter.{cpp,inp}.

Dry primitive equations on the gnomonic-equiangle cubed sphere (ideal-moist dry
H2 species, hllc, vertical-implicit) + Coriolis.

The Hot-Jupiter forcings
    (1) Newtonian relaxation (fixed timescale Kt) of T toward a day-night equilibrium
        T_eq = T_vert(z, sigma) + beta(sigma) * dT_e2p * cos(lat) * cos(lon)
        (hottest at the substellar point lon=lat=0)
    (2) Rayleigh sponge near the top.

are applied by a saved TorchScript module during each Runge-Kutta
stage (matching the ExoCubed `Forcing`) without calling back into Python.

IC: isothermal atmosphere at Ts.
Expected: a strong prograde equatorial super-rotating jet.
"""
import argparse
import os
from typing import Dict

import torch
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3
from paddle import setup_profile

# --- hot-Jupiter parameters (based on hot_jupiter.cpp / .inp) ---
G = 8.0
P0 = 1.0e5
TS = 1600.0
PERTURBATION_SEED = 0
VERTICAL_VELOCITY_PERTURBATION = 1.0e-2


class HotJupiterForcing(torch.nn.Module):
    """Hot-Jupiter Newtonian and sponge forcing."""

    def __init__(self, rd: float, cv: float):
        super().__init__()
        self.rd = rd
        self.cv = cv
        self.p0 = P0
        self.ts = TS
        self.rp = 1.0e8
        self.dt_e2p = 300.0
        self.dt_stra = 10.0
        self.z_stra = 2.0e6
        self.gamma = 2.0e-4
        self.kt = 1.5e5
        self.sigma_stra = 0.12
        self.sponge_tau = 1.0e4
        self.sponge_z = 2.5e6
        self.pi = torch.pi
        self.idn = kIDN
        self.ipr = kIPR
        self.iv1 = kIV1
        self.iv3 = kIV3

    def forward(
        self,
        var: Dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> Dict[str, torch.Tensor]:
        hydro_w = var["hydro_w"]
        latitude = var["coord.latitude"]
        longitude = var["coord.longitude"]
        z = var["coord.x1v"].unsqueeze(0).unsqueeze(0) - self.rp

        rho = hydro_w[self.idn]
        pressure = hydro_w[self.ipr]
        temperature = pressure / (rho * self.rd)
        sigma = pressure / self.p0
        below = (z <= self.z_stra).to(dtype=hydro_w.dtype)

        t_vert = below * (
            self.ts
            - self.gamma * (self.z_stra + z) / 2.0
            + torch.sqrt((self.gamma * (z - self.z_stra) / 2.0) ** 2 + self.dt_stra**2)
        ) + (1.0 - below) * (self.ts - self.gamma * self.z_stra + self.dt_stra)

        beta = below * torch.sin(
            self.pi * (sigma - self.sigma_stra) / (2.0 * (1.0 - self.sigma_stra))
        )

        teq = t_vert + beta * self.dt_e2p * torch.cos(latitude) * torch.cos(longitude)

        du = torch.zeros_like(hydro_w)

        # Newtonian relaxation on total energy (internal = rho*cv*T).
        du[self.ipr] = -dt * self.cv * rho * (temperature - teq) / self.kt

        # Express the implicit top-sponge update as an additive increment.
        sponge = (z > self.sponge_z).to(dtype=hydro_w.dtype)
        damping = 1.0 / (1.0 + sponge * dt / self.sponge_tau)
        du[self.iv1 : self.iv3 + 1] = (
            (damping - 1.0) * rho * hydro_w[self.iv1 : self.iv3 + 1]
        )
        return {"hydro_du": du}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c",
        "--config",
        default=os.path.join(os.path.dirname(__file__), "hjupiter.yaml"),
    )
    p.add_argument("--output-dir", default="out_hjupiter")
    args = p.parse_args()

    opt = MeshOptions.from_yaml(args.config)
    opt.block().output_dir(args.output_dir)
    mesh = Mesh(opt)
    mesh.to(torch.device(opt.device_str()))

    os.makedirs(args.output_dir, exist_ok=True)

    block_vars = []
    for local_index, block in enumerate(mesh.blocks):
        layout = block.get_layout()
        _, _, face_id = layout.loc_of(layout.options.rank())
        coord = block.module("coord")
        w = setup_profile(
            block, {"Ts": TS, "Ps": P0, "grav": G, "Tmin": 1200.0}, method="isothermal"
        )
        if local_index == 0:  # calibrate once per process from the isothermal IC
            il, jl, kl = coord.il(), coord.jl(), coord.kl()
            rd = float(w[kIPR][kl, jl, il] / w[kIDN][kl, jl, il]) / TS
            cv = 2.5 * rd
            cp = 3.5 * rd
            print(
                f"calibrated Rd={rd:.1f} J/kg/K  cp={cp:.1f}  cv={cv:.1f}",
                flush=True,
            )
        generator = torch.Generator(device=w.device).manual_seed(
            PERTURBATION_SEED + face_id
        )
        w[kIV1].uniform_(
            -VERTICAL_VELOCITY_PERTURBATION,
            VERTICAL_VELOCITY_PERTURBATION,
            generator=generator,
        )
        w[kIV2] = 0.0
        w[kIV3] = 0.0
        block_vars.append({"hydro_w": w})

    # Save the Hot-Jupiter forcing as TorchScript
    forcing_path = os.path.join(args.output_dir, "hot_jupiter_forcing.pt")
    torch.jit.script(HotJupiterForcing(rd, cv).eval()).save(forcing_path)
    mesh.set_user_stage_forcings([forcing_path])

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
    main()
