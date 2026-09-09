"""
HS94 animation, v5 — final regridder.

History of regrid bugs (all diagnosed against the raw data scattered at its
own true lon/lat, which is smooth across every face border):
  v1: single global Delaunay -> bridge triangles across cube corners.
  v2/v3: hand-derived face formulas -> wrong alpha/beta convention.
  v4: per-face interpolants blended by angular distance to face CENTERS with
      radius = circumscribed radius (53.8 deg, out to the corners). Near edge
      midpoints a pole face still received ~full weight in a band where it has
      NO data, so its EXTRAPOLATED values were blended with the neighbor's
      real ones -> creases locked to cube edges and spurious easterly patches
      below the field's true minimum ("grid imprinting").

v5: no face logic at all. All 6 faces' cells form one seamless point cloud on
the unit sphere (verified smooth across borders). Regrid = KD-tree lookup of
the k=4 nearest data points per target pixel with a narrow Gaussian kernel
(h = 0.35 x median point spacing). Nothing in the interpolator knows the cube
exists, so it cannot imprint it. Verified on day 380: output range equals the
raw data range exactly; borders and corners seamless.

The neighbor indices/distances are static across frames (the grid never
moves), so they are computed once and each frame is just a weighted gather.
"""

import os
import time
import numpy as np
import xarray as xr
import torch
from scipy.spatial import cKDTree
from snapy.coord import cs_contra_to_sph_

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

NX = 48
LEVEL = 20
BASE = os.environ.get("HS94_INPUT_PATTERN", "hs94_prod.out0.{:05d}.nc")
DAY_STRIDE = 2
DAY_MAX = 380
NLON, NLAT = 360, 180
VLIM = 40.0
OUT_MP4 = os.environ.get("HS94_OUTPUT", "hs94_animation.mp4")

local_angle = -np.pi / 4 + (np.arange(NX) + 0.5) * (np.pi / 2) / NX
ROW, COL = np.meshgrid(local_angle, local_angle, indexing="ij")
# snapy convention: alpha varies with the column index (x2), beta with the row
# index (x3) -- validated to exact agreement with the simulation's own lon/lat.
ALPHA_T = torch.tensor(COL, dtype=torch.float64)
BETA_T = torch.tensor(ROW, dtype=torch.float64)


def unit(lon, lat):
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


def load_frame(path, level):
    """Return (vals, lons, lats, t) concatenated over all 6 faces."""
    ds = xr.open_dataset(path)
    t = float(ds.time.values[0])
    vel1 = ds.vel1.isel(time=0, x1=level).values
    vel2 = ds.vel2.isel(time=0, x1=level).values
    vel3 = ds.vel3.isel(time=0, x1=level).values
    lon_full = ds.lon.isel(time=0).values
    lat_full = ds.lat.isel(time=0).values
    ds.close()
    vals, lons, lats = [], [], []
    for face in range(6):
        row, col = divmod(face, 3)
        v1 = vel1[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX]
        v2 = vel2[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX]
        v3 = vel3[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX]
        vel = torch.stack(
            [
                torch.tensor(v1, dtype=torch.float64),
                torch.tensor(v2, dtype=torch.float64),
                torch.tensor(v3, dtype=torch.float64),
            ],
            dim=0,
        )
        cs_contra_to_sph_(vel, ALPHA_T, BETA_T, face)
        vals.append(vel[2].numpy().ravel())
        lons.append(
            lon_full[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX].ravel()
        )
        lats.append(
            lat_full[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX].ravel()
        )
    return (np.concatenate(vals), np.concatenate(lons), np.concatenate(lats), t)


# ---- static geometry: KD-tree neighbor indices/weights, computed once ----
_, lons0, lats0, _ = load_frame(BASE.format(0), LEVEL)
tree = cKDTree(unit(lons0, lats0))
dnn, _ = tree.query(unit(lons0, lats0), k=2)
H = 0.35 * np.median(dnn[:, 1])

lon_t = np.linspace(0, 2 * np.pi, NLON, endpoint=False)
lat_t = np.linspace(-np.pi / 2, np.pi / 2, NLAT)
LON, LAT = np.meshgrid(lon_t, lat_t)
lat_deg = np.degrees(lat_t)

D, IDX = tree.query(unit(LON, LAT).reshape(-1, 3), k=4)
W = np.maximum(np.exp(-((D / H) ** 2)), 1e-12)
WSUM = W.sum(axis=1)


def regrid(vals):
    return ((W * vals[IDX]).sum(axis=1) / WSUM).reshape(NLAT, NLON)


days = list(range(0, DAY_MAX + 1, DAY_STRIDE))
nframes = len(days)
print(
    f"Precomputing {nframes} frames (KD-tree Gaussian regrid, h={np.degrees(H):.2f} deg)..."
)

maps = np.empty((nframes, NLAT, NLON), dtype=np.float32)
profiles = np.empty((nframes, NLAT), dtype=np.float32)

t0 = time.time()
for i, day in enumerate(days):
    vals, _, _, t = load_frame(BASE.format(day), LEVEL)
    grid = regrid(vals)
    maps[i] = grid
    profiles[i] = grid.mean(axis=1)
    if i % 20 == 0:
        print(f"  frame {i}/{nframes} (day {day})  elapsed {time.time()-t0:.1f}s")
print(f"Precompute done in {time.time()-t0:.1f}s")
print(f"Data range over all frames: min={maps.min():.1f} max={maps.max():.1f} m/s")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(11, 8), dpi=120)
gs = fig.add_gridspec(
    2, 3, height_ratios=[2.2, 1], width_ratios=[3, 1.1, 0.08], hspace=0.35, wspace=0.35
)
ax_map = fig.add_subplot(gs[0, 0])
ax_prof = fig.add_subplot(gs[0, 1], sharey=ax_map)
ax_cb = fig.add_subplot(gs[0, 2])
ax_hov = fig.add_subplot(gs[1, :])

im = ax_map.imshow(
    maps[0],
    origin="lower",
    extent=(0, 360, -90, 90),
    aspect="auto",
    cmap="RdBu_r",
    vmin=-VLIM,
    vmax=VLIM,
)
ax_map.set_xlabel("Longitude (deg)")
ax_map.set_ylabel("Latitude (deg)")
ax_map.set_yticks([-90, -60, -30, 0, 30, 60, 90])
title = ax_map.set_title(
    f"HS94 zonal wind $u_\\lambda$ (level {LEVEL}) -- Day {days[0]:3d}", fontsize=12
)

cb = fig.colorbar(im, cax=ax_cb)
cb.set_label("$u_\\lambda$ (m/s)")

(line_prof,) = ax_prof.plot(profiles[0], lat_deg, color="k", lw=1.5)
ax_prof.axvline(0, color="0.6", lw=0.8, ls="--")
ax_prof.set_xlim(-15, 35)
ax_prof.set_ylim(-90, 90)
ax_prof.set_xlabel("zonal-mean\n$u_\\lambda$ (m/s)")
plt.setp(ax_prof.get_yticklabels(), visible=False)
ax_prof.set_title("jet profile", fontsize=10)

hov_display = np.full((nframes, NLAT), np.nan, dtype=np.float32)
hov_im = ax_hov.imshow(
    hov_display.T,
    origin="lower",
    extent=(days[0], days[-1], -90, 90),
    aspect="auto",
    cmap="RdBu_r",
    vmin=-VLIM / 1.5,
    vmax=VLIM / 1.5,
)
ax_hov.set_xlabel("Simulation day")
ax_hov.set_ylabel("Latitude (deg)")
ax_hov.set_yticks([-90, -45, 0, 45, 90])
ax_hov.set_title("Hovmoller: zonal-mean $u_\\lambda$ spin-up", fontsize=10)
day_marker = ax_hov.axvline(days[0], color="k", lw=1.0)


def update(i):
    im.set_data(maps[i])
    title.set_text(f"HS94 zonal wind $u_\\lambda$ (level {LEVEL}) -- Day {days[i]:3d}")
    line_prof.set_data(profiles[i], lat_deg)
    hov_display[: i + 1] = profiles[: i + 1]
    hov_im.set_data(hov_display.T)
    day_marker.set_xdata([days[i], days[i]])


print("Rendering frames to mp4 via imageio-ffmpeg...")
writer = imageio.get_writer(
    OUT_MP4,
    fps=15,
    codec="libx264",
    quality=None,
    pixelformat="yuv420p",
    output_params=["-crf", "18", "-preset", "medium"],
)
t0 = time.time()
for i in range(nframes):
    update(i)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3]
    h, w, _ = frame.shape
    writer.append_data(frame[: h - (h % 2), : w - (w % 2)])
    if i % 40 == 0:
        print(f"  rendered {i}/{nframes}  elapsed {time.time()-t0:.1f}s")
writer.close()
print(f"mp4 render done in {time.time()-t0:.1f}s -> {OUT_MP4}")
plt.close(fig)
