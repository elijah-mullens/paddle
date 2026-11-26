# Cubed-sphere dynamical-core benchmarks (exo3)

Three standard dynamical-core benchmarks on the gnomonic-equiangle cubed sphere,
reproducing the cases from
[`cshsgy/ExoCubed` `examples/2023-Chen-exo3`](https://github.com/chengcli/ExoCubed)
(Chen et al. 2023) in `snapy`. Each case is a self-contained Python driver plus a
YAML config.

| case | driver | config | physics |
|------|--------|--------|---------|
| **W92** Williamson (1992) shallow-water test 6 (Rossby–Haurwitz wave) | `w92_swe.py` | `w92.yaml` | shallow-water EOS, `shallow-roe` Riemann, Coriolis |
| **HS94** Held–Suarez (1994) dry dynamical-core benchmark | `hs94_run.py` | `hs94.yaml` | dry ideal-gas primitive equations, `lmars`, vertical-implicit, Coriolis + TorchScript HS94 Newtonian-relaxation/Rayleigh forcing |
| **Hot Jupiter** dry GCM (day–night forced) | `hjupiter_run.py` | `hjupiter.yaml` | dry primitive equations + TorchScript Newtonian relaxation toward a substellar-hot equilibrium and top Rayleigh sponge |

The HS94 and hot-Jupiter forcings are not built-in `snapy` modules. Each driver
scripts one block-independent forcing, saves it under the output directory, and
registers it with `mesh.set_user_stage_forcings([forcing_path])`. Their scripted
`forward` methods have the required interface:

```python
def forward(
    self,
    variables: Dict[str, torch.Tensor],
    dt: float,
    stage: int,
) -> Dict[str, torch.Tensor]:
    ...
```

For each block, `variables` contains the evolving model state and the block's
named buffers. HS94 reads `coord.latitude`; hot Jupiter additionally reads
`coord.longitude` and `coord.x1v`. These tensors share their underlying storage.
The saved module therefore contains no block geometry or `MeshBlock` reference.
Snapy executes every module in the registered list sequentially at each stage in
LibTorch without acquiring the Python GIL.

## Running

The supplied W92 deck assigns one cubed-sphere face to each of six processes.
The HS94 and hot-Jupiter decks assign all six faces to a single process.
Run from this directory with the launch command for the selected deck.

```bash
BACKEND=gloo DEVICE=cpu torchrun --standalone --nproc-per-node=6 w92_swe.py --output-dir out_w92
DEVICE=cpu python hs94_run.py --output-dir out_hs94
DEVICE=cpu python hjupiter_run.py --output-dir out_hjupiter
```

Use `-c <config.yaml>` for a different deck. With one block per face, the
process count times `distribute.blocks_per_process` must equal six. Set
`DEVICE=cuda` to run the single-process examples on a GPU.

## Expected results

- **W92** — the Rossby–Haurwitz wave-4 pattern propagates eastward at about
  12.2 degrees/day (roughly 171 degrees by day 14). It is an exact solution of
  the barotropic-vorticity equation, not of the shallow-water equations, so
  some deformation and numerical damping are expected. Geopotential and winds
  should remain bounded and smooth across panel boundaries.
- **HS94** — relaxes to the classic Held–Suarez climate: midlatitude eddy-driven
  westerly jets and a realistic zonal-mean temperature structure. Typical run speed is:
  1. 300 SDPH (simulation day per wall clock hour) using 1x NVIDIA RTX 5090 cards
  1. 86 SDPH using 2x NVIDIA RTX 4000 cards.
  1. 4 SDPH using 6 Intel(R) Xeon(R) E-2236 CPU @ 3.40GHz
- **Hot Jupiter** — develops a strong prograde **equatorial superrotating jet**.
  Typical run speed is:
  1. 540 SDPH (simulation day per wall clock hour) using 1x NVIDIA RTX 5090 cards
  1. 150 SDPH using 2x NVIDIA RTX 4000 cards.
  1. 7 SDPH using 6 Intel(R) Xeon(R) E-2236 CPU @ 3.40GHz

## Dependencies

`snapy`, `paddle` (atmospheric profile setup), and `torch`. W92 uses the
cubed-sphere coordinate and velocity transforms in `snapy.coord`.

Use Paddle's Snapy 3 dependency for W92. The benchmark relies on spherical
cross-panel velocity exchange and spherical Coriolis forcing.

# Visualizations

There are scripts render zonal-wind spin-up animations from Snapy cubed-sphere
NetCDF output. They transform the native contravariant velocity to spherical
components and regrid all six faces as a single point cloud on the unit sphere.

The two scripts are intentionally kept side by side:

- `plot_hs94.py` renders the Held-Suarez benchmark.
- `plot_hjupiter.py` renders the forced dry hot-Jupiter benchmark.

Set the input and output paths with environment variables:

```shell
HS94_INPUT_PATTERN='/path/to/hs94_prod.out0.{:05d}.nc' \
HS94_OUTPUT=hs94_animation.mp4 python plot_hs94.py

HJUPITER_DATA_DIR=/path/to/hjupiter/output \
HJUPITER_OUTPUT=hjupiter_animation.mp4 python plot_hjupiter.py
```

The scripts require NumPy, xarray, PyTorch, SciPy, Matplotlib, imageio, and an
FFmpeg installation available to imageio.
