#!/usr/bin/env python3
"""
Held-Suarez 1994 dry dynamical-core benchmark in snapy based on
cshsgy/ExoCubed/examples/2023-Chen-exo3/hs94.{cpp,inp}.

Dry primitive equations on the gnomonic-equiangle cubed sphere (snapy `ideal-gas`
EOS, lmars, vertical-implicit), Coriolis via the `coriolis` forcing.

The HS94 forcings
    (1) Newtonian relaxation of T toward Teq(lat,sigma)
    (2) low-level Rayleigh friction

are applied by a saved TorchScript module during each Runge-Kutta
stage (matching the ExoCubed `Forcing`) without calling back into Python.

IC: dry-adiabatic hydrostatic profile (theta=Ts up to z_iso, then isothermal),
seeded with small random vertical velocity.
"""
import argparse
import os
from typing import Dict

import torch
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3
from paddle import setup_profile

# --- HS94 + Earth dry-air parameters (from hs94.cpp / hs94.inp) ---
# Rd/cp from the snapy "dry" species (M=29 g/mol, cv_R=2.5 -> gamma=1.4)
G = 9.81
P0 = 1.0e5
TS = 315.0
DAY = 86400.0
PERTURBATION_SEED = 0
VERTICAL_VELOCITY_PERTURBATION = 1.0e-2


class HS94Forcing(torch.nn.Module):
    """scriptable HS94 forcing."""

    def __init__(self):
        super().__init__()
        self.rd = 8.31446 / 0.029
        self.cv = 2.5 * self.rd
        self.kappa = 2.0 / 7.0
        self.p0 = P0
        self.ts = TS
        self.dt_h = 60.0
        self.dtheta = 10.0
        self.sigmab = 0.7
        self.kf = 1.0 / DAY
        self.ka = 0.025 / DAY
        self.ks = 0.25 / DAY
        self.teq_floor = 200.0
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
        """Return the additive conserved-state increment for one RK stage."""
        hydro_w = var["hydro_w"]
        latitude = var["coord.latitude"]

        rho = hydro_w[self.idn]
        pressure = hydro_w[self.ipr]
        temperature = pressure / (rho * self.rd)
        sigma = pressure / self.p0
        coslat = torch.cos(latitude)
        sinlat = torch.sin(latitude)
        teq = (
            self.ts
            - self.dt_h * sinlat**2
            - self.dtheta * torch.log(torch.clamp(sigma, min=1e-12)) * coslat**2
        ) * sigma**self.kappa
        teq = torch.clamp(teq, min=self.teq_floor)
        sigma_p = torch.clamp((sigma - self.sigmab) / (1.0 - self.sigmab), min=0.0)
        kv = sigma_p * self.kf
        kt = self.ka + (self.ks - self.ka) * sigma_p * coslat**4

        du = torch.zeros_like(hydro_w)

        # Newtonian cooling on total energy (internal energy = rho*cv*T).
        du[self.ipr] = -dt * rho * self.cv * kt * (temperature - teq)

        # Low-level Rayleigh friction on covariant momentum.
        du[self.iv1 : self.iv3 + 1] = (
            -dt * kv * hydro_w[self.idn] * hydro_w[self.iv1 : self.iv3 + 1]
        )
        return {"hydro_du": du}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-c", "--config", default=os.path.join(os.path.dirname(__file__), "hs94.yaml")
    )
    p.add_argument("--output-dir", default="out_hs94")
    args = p.parse_args()

    opt = MeshOptions.from_yaml(args.config)
    opt.block().output_dir(args.output_dir)
    mesh = Mesh(opt)
    mesh.to(torch.device(opt.device_str()))

    os.makedirs(args.output_dir, exist_ok=True)

    # Save the HS94 forcing as TorchScript
    forcing_path = os.path.join(args.output_dir, "hs94_forcing.pt")
    torch.jit.script(HS94Forcing().eval()).save(forcing_path)
    mesh.set_user_stage_forcings([forcing_path])

    block_vars = []
    for block in mesh.blocks:
        layout = block.get_layout()
        _, _, face_id = layout.loc_of(layout.options.rank())
        w = setup_profile(
            block, {"Ts": TS, "Ps": P0, "grav": G, "Tmin": 120.0}, method="dry-adiabat"
        )
        generator = torch.Generator(device=w.device).manual_seed(
            PERTURBATION_SEED + face_id
        )
        w[kIV1].uniform_(
            -VERTICAL_VELOCITY_PERTURBATION,
            VERTICAL_VELOCITY_PERTURBATION,
            generator=generator,
        )
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
