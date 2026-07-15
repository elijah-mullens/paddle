# Cubed-sphere dynamical-core benchmarks (exo3)

Three standard dynamical-core benchmarks on the gnomonic-equiangle cubed sphere,
reproducing the cases from
[`cshsgy/ExoCubed` `examples/2023-Chen-exo3`](https://github.com/chengcli/ExoCubed)
(Chen et al. 2023) in `snapy`. Each case is a self-contained Python driver plus a
YAML config.

| case | driver | config | physics |
|------|--------|--------|---------|
| **W92** Williamson (1992) shallow-water test 6 (Rossby–Haurwitz wave) | `w92_swe.py` | `w92.yaml` | shallow-water EOS, `shallow-roe` Riemann, Coriolis |
| **HS94** Held–Suarez (1994) dry dynamical-core benchmark | `hs94_run.py` | `hs94.yaml` | dry ideal-gas primitive equations, `lmars`, vertical-implicit, Coriolis + HS94 Newtonian-relaxation/Rayleigh forcing (operator-split) |
| **Hot Jupiter** dry GCM (day–night forced) | `hjupiter_run.py` | `hjupiter.yaml` | dry primitive equations + Newtonian relaxation toward a substellar-hot equilibrium + top Rayleigh sponge |

The HS94 and hot-Jupiter forcings are not built-in `snapy` modules; each driver
applies them as an operator-split source on the conserved state every step,
matching the corresponding ExoCubed `Forcing`.

## Running

Each driver starts a distributed run via `paddle.start_dist` using the
`distribute` block in its config (default `blocks_per_process: 6`, i.e. the six
cube faces). Launch with `torchrun`:

```bash
# W92 shallow water (one process holding all 6 faces)
torchrun --nproc_per_node=1 w92_swe.py --output-dir out_w92

# HS94 dry dynamical core
torchrun --nproc_per_node=1 hs94_run.py --output-dir out_hs94

# Hot Jupiter
torchrun --nproc_per_node=1 hjupiter_run.py --output-dir out_hjupiter
```

Pass `-c <config.yaml>` to override the default config (each driver defaults to
the YAML next to it). Adjust `distribute` (`blocks_per_process`, `backend`) and
`CUDA_VISIBLE_DEVICES` for multi-GPU.

The process count and `blocks_per_process` must describe the six cube faces:
use one process with `blocks_per_process: 6` (the supplied configs), or six
processes with `blocks_per_process: 1`. In the latter case, launch with
`torchrun --nproc_per_node=6 ...`. The `ucx` backend requires `commux`; use
`gloo` for a CPU-only run.

## Expected results

- **W92** — the Rossby–Haurwitz wave-4 pattern propagates eastward at about
  12.2 degrees/day (roughly 171 degrees by day 14). It is an exact solution of
  the barotropic-vorticity equation, not of the shallow-water equations, so
  some deformation and numerical damping are expected. Geopotential and winds
  should remain bounded and smooth across panel boundaries.
- **HS94** — relaxes to the classic Held–Suarez climate: midlatitude eddy-driven
  westerly jets and a realistic zonal-mean temperature structure.
- **Hot Jupiter** — develops a strong prograde **equatorial superrotating jet**.

## Dependencies

`snapy`, `paddle` (distributed/profile helpers and `paddle.cubed_sphere_remap`
for the W92 contravariant↔geographic velocity rotation), `torch`, `numpy`,
`pyyaml`.

W92 requires `snapy>=2.7.4`. Earlier 2.7 releases lack either the spherical
cross-panel velocity exchange or the corrected spherical Coriolis treatment;
they may start successfully but produce incorrect dynamics.
