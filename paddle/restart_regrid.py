#!/usr/bin/env python3
"""Move a restart onto a different grid, chiefly a taller x1 domain.

A restart stores raw cell arrays; the grid itself lives in the YAML, not in the
restart. Change x1max or nx1 and the saved tensors no longer fit, so a run can
only ever be continued on the geometry it was started with. That is a real
constraint for a long integration: the lid sits at a fixed height, so it drifts
in pressure as the column's temperature and mean molar mass evolve, and the only
way to raise it is otherwise to start over from t = 0.

This reads a restart, maps every column onto the grid a second YAML describes,
and writes a restart the new configuration accepts. last_time, last_cycle,
file_number and next_time are carried across unchanged, so the continued run
keeps its clock and its output numbering.

What is mapped, and how:

  density        log-linear in height. On a hydrostatic column log(rho) is
                 nearly straight in z, so this is accurate inside the old domain
                 and, continued above it along a fitted slope, is an isothermal
                 constant-mu extension.
  pressure       carried as P/rho, so the extension stays isothermal.
  velocity       linear in height, held at the edge value outside.
  specific       linear in height, held at the edge value outside. Constant
  energy         specific internal energy at fixed composition is isothermal,
                 which is what the density extension assumes.
  mass fractions linear in height, held at the edge value outside, except for
                 condensates, which start empty above the old top: holding
                 H2O(l) at its old top value filled 150 km of new column with
                 cloud that was never there.

Conserved variables are rebuilt as rho times the per-unit-mass quantity, so the
new state is consistent by construction rather than interpolated component by
component.

The x1 ghost zones and the top --edge-skip active cells are dropped from the
source first. The ghosts only mirror the wall, and the cells against it are not
trustworthy: on one 100-day restart the top two active cells had density rising
with height and water mass fraction jumping 0.165 -> 0.021 -> 0.404. Extending
off a two-cell difference there gives a negative scale height, which is why the
slope is fitted over --edge-fit cells instead.

Horizontal resolution must match unless --x2-mode stretch is given, which
resamples x2 periodically. Stretching maps the old domain width onto the new
one, so existing eddies are rescaled with it: that keeps a run going across a
width change, it does not preserve the flow.

Examples:
  paddle-regrid --old-config old.yaml --new-config new.yaml \
      --restart run.final.restart --output run.taller.restart
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from typing import Sequence

import numpy as np
import torch
import yaml

RESTART_BUNDLE_MAGIC = b"SNAPY_RESTART_BUNDLE_V1"

# Variables holding rho times a per-unit-mass quantity, and variables holding
# the per-unit-mass quantity itself.
CONSERVED_KEYS = ("hydro_u", "fill_solid_hydro_u")
PRIMITIVE_KEYS = ("hydro_w", "fill_solid_hydro_w")

# Per-variable mapping rules.
RHO, VEL, EOP, FRAC = "rho", "vel", "eop", "frac"


def condensate_slots(cfg: dict, nvar: int, scheme: str) -> list:
    """Variable slots holding a condensate, by Snapy's species naming.

    Snapy writes condensed phases with a parenthesised suffix, `H2O(l)` and
    `H2O(l,p)`, so the config names them without the tool needing a list.
    """
    species = [str(s.get("name", "")) for s in cfg.get("species", [])]
    if len(species) < 2:
        return []
    nmass = nvar - 5
    first = 5 if scheme == "default" else 1
    slots = []
    for k, name in enumerate(species[1 : 1 + nmass]):
        if "(" in name:
            slots.append(first + k)
    return slots


def variable_roles(nvar: int, scheme: str) -> list:
    """Assign a mapping rule to each slot of a hydro array.

    Two index schemes exist (snap.h): the default puts the species after
    pressure, the legacy Athena++ one puts them straight after density.
    """
    if scheme == "legacy":
        nmass = nvar - 5
        roles = [RHO] + [FRAC] * nmass + [VEL, VEL, VEL, EOP]
    else:
        nmass = nvar - 5
        roles = [RHO, VEL, VEL, VEL, EOP] + [FRAC] * nmass
    if nmass < 0:
        raise ValueError(f"hydro array has {nvar} variables, expected at least 5")
    return roles


class Grid:
    """The cell layout a configuration implies."""

    def __init__(self, cfg: dict):
        geom = cfg["geometry"]
        b, c = geom["bounds"], geom["cells"]
        self.nghost = int(c.get("nghost", 3))
        self.nx1 = int(c["nx1"])
        self.nx2 = int(c.get("nx2", 1))
        self.nx3 = int(c.get("nx3", 1))
        self.x1min, self.x1max = float(b["x1min"]), float(b["x1max"])
        self.x2min, self.x2max = float(b.get("x2min", 0.0)), float(b.get("x2max", 1.0))
        self.dx1 = (self.x1max - self.x1min) / self.nx1

    def _nc(self, nx: int) -> int:
        # A dimension of one cell carries no ghost zones.
        return nx + 2 * self.nghost if nx > 1 else 1

    @property
    def nc1(self) -> int:
        return self._nc(self.nx1)

    @property
    def nc2(self) -> int:
        return self._nc(self.nx2)

    @property
    def nc3(self) -> int:
        return self._nc(self.nx3)

    def x1v(self) -> np.ndarray:
        """Cell centres of the full x1 array, ghost zones included."""
        i = np.arange(self.nc1, dtype=np.float64)
        return self.x1min + (i - self.nghost + 0.5) * self.dx1

    def describe(self) -> str:
        return (
            f"x1 [{self.x1min:.6g}, {self.x1max:.6g}] nx1={self.nx1} "
            f"dx1={self.dx1:.3f} | nx2={self.nx2} nx3={self.nx3} "
            f"nghost={self.nghost} -> ({self.nc3}, {self.nc2}, {self.nc1})"
        )


def fit_top_slope(z: np.ndarray, f: np.ndarray, nfit: int) -> float:
    """Least-squares slope over the top `nfit` source cells."""
    n = int(max(2, min(nfit, z.size)))
    return float(np.polyfit(z[-n:], f[-n:], 1)[0])


def interp_extend(
    z_old: np.ndarray, f: np.ndarray, z_new: np.ndarray, nfit: int
) -> np.ndarray:
    """Linear interpolation continued past both ends with a fitted slope."""
    out = np.interp(z_new, z_old, f)
    lo, hi = z_new < z_old[0], z_new > z_old[-1]
    if lo.any():
        s = (f[1] - f[0]) / (z_old[1] - z_old[0])
        out[lo] = f[0] + s * (z_new[lo] - z_old[0])
    if hi.any():
        out[hi] = f[-1] + fit_top_slope(z_old, f, nfit) * (z_new[hi] - z_old[-1])
    return out


def map_x1(
    arr: np.ndarray, z_old: np.ndarray, z_new: np.ndarray, log: bool, nfit: int
) -> np.ndarray:
    """Map (..., n_src) along its last axis."""
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty((flat.shape[0], z_new.size), dtype=np.float64)
    for n in range(flat.shape[0]):
        col = flat[n]
        if log and np.all(col > 0.0):
            out[n] = np.exp(interp_extend(z_old, np.log(col), z_new, nfit))
        else:
            # Held at the edge value outside the old range; np.interp clamps.
            out[n] = np.interp(z_new, z_old, col)
    return out.reshape(arr.shape[:-1] + (z_new.size,))


def resample_x2_periodic(arr: np.ndarray, nc2_new: int, ng: int) -> np.ndarray:
    """Periodically resample the active x2 range, then refill the x2 ghosts."""
    nx2_old = arr.shape[-2] - 2 * ng
    nx2_new = nc2_new - 2 * ng
    active = arr[..., ng : ng + nx2_old, :]
    # Cell centres as a fraction of the domain width, comparable across grids.
    s_old = (np.arange(nx2_old) + 0.5) / nx2_old
    s_new = (np.arange(nx2_new) + 0.5) / nx2_new
    # Wrap one cell around each end so the interpolation stays periodic.
    s_pad = np.concatenate(([s_old[-1] - 1.0], s_old, [s_old[0] + 1.0]))
    moved = np.moveaxis(active, -2, 0)
    padded = np.concatenate((moved[-1:], moved, moved[:1]), axis=0)
    tail = padded.shape[1:]
    flat = padded.reshape(padded.shape[0], -1)
    res = np.empty((nx2_new, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        res[:, c] = np.interp(s_new, s_pad, flat[:, c])
    res = np.moveaxis(res.reshape((nx2_new,) + tail), 0, -2)
    out = np.empty(arr.shape[:-2] + (nc2_new, arr.shape[-1]), dtype=np.float64)
    out[..., ng : ng + nx2_new, :] = res
    # x2 is periodic, so its ghosts are the opposite edge of the active range.
    out[..., :ng, :] = res[..., -ng:, :]
    out[..., ng + nx2_new :, :] = res[..., :ng, :]
    return out


def fill_x1_ghosts(
    out: np.ndarray, ng: int, roles: list, inner: str, outer: str
) -> None:
    """Rebuild the x1 ghost zones the way the boundary condition defines them.

    Snapy does not refill ghosts when it loads a restart, so the first
    reconstruction uses whatever the file holds. A reflecting wall stores the
    mirror of the interior with the wall-normal velocity negated; leaving the
    smooth extrapolation there instead put the bottom ghost 12% too dense with
    the velocity pointing the wrong way, which reads as inflow through a solid
    wall and threw a six-fold kinetic energy spike into the first day.

    Anything other than a reflecting wall keeps the extrapolation, which is the
    reasonable neutral choice for outflow and for a periodic seam that the
    exchange will overwrite anyway.
    """
    try:
        v1 = roles.index(VEL)
    except ValueError:
        v1 = None
    if inner == "reflecting":
        out[..., :ng] = out[..., ng : 2 * ng][..., ::-1]
        if v1 is not None:
            out[v1][..., :ng] *= -1.0
    if outer == "reflecting":
        out[..., -ng:] = out[..., -2 * ng : -ng][..., ::-1]
        if v1 is not None:
            out[v1][..., -ng:] *= -1.0


def regrid_state(
    arr: np.ndarray,
    old: Grid,
    new: Grid,
    roles: list,
    conserved: bool,
    skip: int,
    nfit: int,
    zero_above: list = (),
    bc: tuple = ("", ""),
) -> np.ndarray:
    """Map one (nvar, nc3, nc2, nc1) array onto the new grid.

    `zero_above` names variable slots that must not be carried into the
    extension: holding a condensate's mass fraction at the old top value fills
    the whole new column with cloud that was never there.
    """
    ng = old.nghost
    nsrc = old.nx1 - skip
    if nsrc < 2:
        raise ValueError(f"--edge-skip {skip} leaves {nsrc} source cells in x1")
    src = arr[..., ng : ng + nsrc]
    z_old = old.x1v()[ng : ng + nsrc]
    z_new = new.x1v()

    rho = src[0]
    if np.any(rho <= 0.0):
        raise ValueError("restart holds non-positive density; refusing to regrid")

    rho_new = map_x1(rho, z_old, z_new, True, nfit)
    out = np.empty(src.shape[:-1] + (new.nc1,), dtype=np.float64)
    out[0] = rho_new

    for v in range(1, src.shape[0]):
        role = roles[v]
        # Momentum, total energy and tracer densities are all rho times a
        # per-unit-mass quantity, and that quantity is what stays smooth.
        # Pressure is the same case: carrying P/rho rather than P keeps the
        # extension isothermal, where extrapolating log(P) and log(rho) along
        # separately fitted slopes lets their small mismatch drift the
        # temperature (194 K instead of 255 K over 160 km, on one test).
        if role == EOP or (conserved and role in (VEL, FRAC)):
            out[v] = map_x1(src[v] / rho, z_old, z_new, False, nfit) * rho_new
        else:
            out[v] = map_x1(src[v], z_old, z_new, False, nfit)

    # Condensate is a local product of the state it was lifted from, not a
    # background the extension can inherit. Above the old top it starts empty
    # and the microphysics makes its own.
    if zero_above:
        above = z_new > z_old[-1]
        for v in zero_above:
            out[v][..., above] = 0.0

    fill_x1_ghosts(out, new.nghost, roles, bc[0], bc[1])

    if new.nc2 != old.nc2:
        out = resample_x2_periodic(out, new.nc2, new.nghost)
    return out


def read_part(src) -> dict:
    mod = torch.jit.load(src, map_location="cpu")
    # named_parameters and named_buffers deduplicate by storage, keeping only
    # the alphabetically first name. Snapy aliases fill_solid_hydro_u onto
    # hydro_u when no immersed boundary is present, so the default view drops
    # hydro_u -- the one variable a restart cannot do without -- without a
    # word. state_dict keeps both names; remove_duplicate=False is the belt to
    # its braces.
    tensors = dict(mod.state_dict())
    for name, value in list(
        mod.named_parameters(recurse=True, remove_duplicate=False)
    ) + list(mod.named_buffers(recurse=True, remove_duplicate=False)):
        tensors.setdefault(name, value)
    return tensors


def write_part(tensors: dict, dst) -> None:
    mod = torch.nn.Module()
    for name, value in tensors.items():
        mod.register_buffer(name, value)
    torch.jit.save(torch.jit.script(mod), dst)


def read_restart(path: str) -> list:
    """Return [(entry_name, tensors)]; a single-rank restart has one entry."""
    with open(path, "rb") as f:
        if f.read(len(RESTART_BUNDLE_MAGIC)) != RESTART_BUNDLE_MAGIC:
            return [(None, read_part(path))]
        f.seek(0)
        f.readline()  # magic
        count = int(f.readline())
        index = []
        for _ in range(count):
            name, size = f.readline().decode().rstrip("\n").split("\t", 1)
            index.append((name, int(size)))
        f.readline()  # blank terminator
        return [(name, read_part(io.BytesIO(f.read(size)))) for name, size in index]


def write_restart(blocks: list, path: str) -> None:
    if len(blocks) == 1 and blocks[0][0] is None:
        write_part(blocks[0][1], path)
        return
    payloads = []
    for name, tensors in blocks:
        buf = io.BytesIO()
        write_part(tensors, buf)
        payloads.append((name, buf.getvalue()))
    with open(path, "wb") as f:
        f.write(RESTART_BUNDLE_MAGIC + b"\n")
        f.write(f"{len(payloads)}\n".encode())
        for name, blob in payloads:
            f.write(f"{name}\t{len(blob)}\n".encode())
        f.write(b"\n")
        for _, blob in payloads:
            f.write(blob)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--old-config", required=True, help="YAML the restart was written with"
    )
    parser.add_argument(
        "--new-config", required=True, help="YAML the run will continue with"
    )
    parser.add_argument("--restart", required=True, help="restart file to read")
    parser.add_argument("--output", required=True, help="restart file to write")
    parser.add_argument(
        "--x2-mode",
        choices=["strict", "stretch"],
        default="strict",
        help="strict requires nx2 to match; stretch resamples x2 periodically",
    )
    parser.add_argument(
        "--index-scheme",
        choices=["default", "legacy"],
        default="default",
        help="hydro variable order; legacy is the Athena++ order used when NMASS > 0",
    )
    parser.add_argument(
        "--edge-skip",
        type=int,
        default=2,
        help="active x1 cells dropped from the top of the source before extending, "
        "because they sit against the wall (default 2)",
    )
    parser.add_argument(
        "--edge-fit",
        type=int,
        default=12,
        help="source cells the extension slope is fitted over (default 12)",
    )
    parser.add_argument(
        "--keep-condensate",
        action="store_true",
        help="carry condensate into the extension instead of starting it empty",
    )
    parser.add_argument(
        "--reset-time", action="store_true", help="resume at t = 0, cycle 0"
    )
    args = parser.parse_args(argv)

    with open(args.old_config) as f:
        old = Grid(yaml.safe_load(f))
    with open(args.new_config) as f:
        new_cfg = yaml.safe_load(f)
    new = Grid(new_cfg)
    print(f"old grid: {old.describe()}")
    print(f"new grid: {new.describe()}")

    if old.x1min != new.x1min:
        raise SystemExit(
            "x1min must be unchanged: the lower boundary anchors the column"
        )
    if old.nx3 != new.nx3:
        raise SystemExit(f"nx3 must be unchanged ({old.nx3} -> {new.nx3})")
    if old.nx2 != new.nx2 and args.x2_mode != "stretch":
        raise SystemExit(
            f"nx2 changes {old.nx2} -> {new.nx2}; pass --x2-mode stretch to resample x2, "
            "understanding that it rescales the existing eddies along with the domain"
        )
    if new.x1max < old.x1max:
        print(
            f"note: the new lid is lower ({new.x1max:.6g} < {old.x1max:.6g}); "
            "the column above it is discarded"
        )

    ext = (new_cfg.get("boundary-condition") or {}).get("external") or {}
    bc = (str(ext.get("x1-inner", "")), str(ext.get("x1-outer", "")))
    print(f"x1 boundaries: inner {bc[0] or 'unset'}, outer {bc[1] or 'unset'}")

    blocks = read_restart(args.restart)
    print(f"read {len(blocks)} block(s) from {args.restart}")
    for name, tensors in blocks:
        missing = [
            k for k in ("hydro_u", "last_time", "last_cycle") if k not in tensors
        ]
        if missing:
            raise SystemExit(
                f"{name or args.restart} is missing {', '.join(missing)}; refusing to "
                "write a restart Snapy cannot load"
            )
    if len(blocks) > 1:
        print(
            "note: several blocks. x1 is regridded per block; x2 and x3 must "
            "be unchanged, since the configuration describes the whole mesh "
            "and not one block's share of it."
        )
        if args.x2_mode == "stretch":
            raise SystemExit(
                "--x2-mode stretch cannot be used on a decomposed restart: the "
                "resample treats x2 as periodic, but one block holds a slice of "
                "x2 and its two ends are interior seams, not the same point. "
                "Regrid x1 only, or recombine to a single block first."
            )

    for _, tensors in blocks:
        for key in sorted(tensors):
            if key not in CONSERVED_KEYS and key not in PRIMITIVE_KEYS:
                continue
            tensor = tensors[key]
            arr = tensor.to(torch.float64).cpu().numpy()
            if arr.shape[-1] != old.nc1 or (
                len(blocks) == 1 and arr.shape[1:] != (old.nc3, old.nc2, old.nc1)
            ):
                raise SystemExit(
                    f"{key} has shape {tuple(arr.shape)}, which does not match the old "
                    f"configuration ({old.nc3}, {old.nc2}, {old.nc1}); --old-config is "
                    "probably not the YAML this restart came from"
                )
            conserved = key in CONSERVED_KEYS
            roles = variable_roles(arr.shape[0], args.index_scheme)
            zero = (
                []
                if args.keep_condensate
                else condensate_slots(new_cfg, arr.shape[0], args.index_scheme)
            )
            out = regrid_state(
                arr, old, new, roles, conserved, args.edge_skip, args.edge_fit, zero, bc
            )
            tensors[key] = torch.from_numpy(out).to(tensor.dtype)
            print(
                f"  {key}: {tuple(arr.shape)} -> {tuple(out.shape)} "
                f"({'conserved' if conserved else 'primitive'})"
            )
        if args.reset_time:
            for key in ("last_time", "last_cycle", "file_number", "next_time"):
                if key in tensors:
                    tensors[key] = torch.zeros_like(tensors[key])
        print(
            f"  resume at t={tensors['last_time'].item():.6g}, "
            f"cycle={tensors['last_cycle'].item()}"
        )

    write_restart(blocks, args.output)
    print(f"wrote {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
