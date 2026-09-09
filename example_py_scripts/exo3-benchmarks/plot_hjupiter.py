"""
Build an animation of the equatorial superrotating jet spinning up over a
700-day Hot Jupiter forced dry GCM benchmark run on a cubed-sphere grid.

v3: uses a single-point-cloud KD-tree Gaussian regridder (see below).
The per-face partition-of-unity blend used in v2 weighted each face by
angular distance to its CENTER out to the circumscribed radius, which let a
pole face contribute EXTRAPOLATED values (at ~full weight) in a band where it
has no data -- producing creases locked to the cube edges ("grid imprinting",
found on the HS94 case). The v1 version regridded by pooling all 6 faces' scattered
(lon,lat) points into one Delaunay triangulation via scipy.interpolate.griddata
-- bridge triangles across face corners. v3 fix: all 6 faces' cells form one
seamless point cloud on the unit sphere (data verified smooth across borders);
regrid = k=4 nearest neighbors with a narrow Gaussian kernel (h = 0.35 x
median point spacing). No face logic anywhere, so cube geometry cannot
imprint. Neighbor indices are static across frames and precomputed once.

The (alpha, beta) convention used to compute u_lon from the raw contravariant
velocities was independently verified correct in this script already (alpha
varies with the column/x2 index, beta with the row/x3 index -- confirmed
against the simulation's own lon/lat output, max error ~1e-5 deg), so it is
kept as-is here.

Top panel: u_lon (zonal wind) reprojected to an equirectangular lat-lon map,
           with the substellar point (lon=0, lat=0) marked.
Bottom panel: equatorial-mean (|lat|<15 deg) zonal wind vs time, showing the
              jet spin-up, with a moving marker at the current frame's day.
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
DATADIR = os.environ.get("HJUPITER_DATA_DIR", ".")
NDAYS = 701  # files 00000..00700
LEVEL = 8  # chosen vertical level (see level-selection scan)
FRAME_STRIDE = 4  # animate every 4th day -> 176 frames
NLON, NLAT = 360, 180
MARGIN = np.radians(8.0)
OUT_MP4 = os.environ.get("HJUPITER_OUTPUT", "hjupiter_animation.mp4")

local_angle = -np.pi / 4 + (np.arange(NX) + 0.5) * (np.pi / 2) / NX
_beta0, _alpha0 = np.meshgrid(local_angle, local_angle, indexing="ij")
ALPHA_T = torch.tensor(_alpha0, dtype=torch.float64)
BETA_T = torch.tensor(_beta0, dtype=torch.float64)


def face_ulon_lonlat(path, level):
    ds = xr.open_dataset(path)
    t = float(ds.time.values[0])
    vel1 = ds.vel1.isel(time=0, x1=level).values
    vel2 = ds.vel2.isel(time=0, x1=level).values
    vel3 = ds.vel3.isel(time=0, x1=level).values
    lon_full = ds.lon.isel(time=0).values
    lat_full = ds.lat.isel(time=0).values
    ds.close()
    val, lon_f, lat_f = {}, {}, {}
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
        val[face] = vel[2].numpy()
        lon_f[face] = lon_full[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX]
        lat_f[face] = lat_full[row * NX : (row + 1) * NX, col * NX : (col + 1) * NX]
    return val, lon_f, lat_f, t


# ---- static geometry: KD-tree neighbor indices/weights, computed once ----
def unit3(lon, lat):
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1
    )


_val0, lon0, lat0, _ = face_ulon_lonlat(f"{DATADIR}/hjupiter_prod.out0.00000.nc", LEVEL)
lons0 = np.concatenate([lon0[f].ravel() for f in range(6)])
lats0 = np.concatenate([lat0[f].ravel() for f in range(6)])
tree = cKDTree(unit3(lons0, lats0))
_dnn, _ = tree.query(unit3(lons0, lats0), k=2)
H = 0.35 * np.median(_dnn[:, 1])

lon_t = np.linspace(0, 2 * np.pi, NLON, endpoint=False)
lat_t = np.linspace(-np.pi / 2, np.pi / 2, NLAT)
LONG, LATG = np.meshgrid(lon_t, lat_t)
lon_deg = np.degrees(lon_t)
lat_deg = np.degrees(lat_t)

D, IDX = tree.query(unit3(LONG, LATG).reshape(-1, 3), k=4)
W = np.maximum(np.exp(-((D / H) ** 2)), 1e-12)
WSUM = W.sum(axis=1)


def regrid_datadriven(val):
    vals = np.concatenate([val[f].ravel() for f in range(6)])
    return ((W * vals[IDX]).sum(axis=1) / WSUM).reshape(NLAT, NLON)


def main():
    days = np.arange(NDAYS)
    eq_mean_series = np.full(NDAYS, np.nan)

    frame_days = list(range(0, NDAYS, FRAME_STRIDE))
    if frame_days[-1] != NDAYS - 1:
        frame_days.append(NDAYS - 1)

    frame_grids = []

    t0 = time.time()
    for i, day in enumerate(days):
        fname = f"{DATADIR}/hjupiter_prod.out0.{day:05d}.nc"
        val, lon_f, lat_f, t = face_ulon_lonlat(fname, LEVEL)

        # equatorial mean directly from native data (no regrid needed)
        all_lat = np.concatenate([lat_f[f].ravel() for f in range(6)])
        all_val = np.concatenate([val[f].ravel() for f in range(6)])
        mask = np.abs(all_lat) < np.radians(15)
        eq_mean_series[day] = all_val[mask].mean()

        if day in frame_days:
            grid = regrid_datadriven(val)
            frame_grids.append((day, grid))

        if i % 100 == 0:
            print(f"  processed day {day}/{NDAYS-1}  ({time.time()-t0:.1f}s elapsed)")

    print(
        f"Done processing {NDAYS} days in {time.time()-t0:.1f}s; "
        f"{len(frame_grids)} animation frames."
    )

    late_vals = np.concatenate(
        [g[~np.isnan(g)].ravel() for d, g in frame_grids if d >= NDAYS - 100]
    )
    vmax = np.ceil(np.percentile(np.abs(late_vals), 99.5) / 50) * 50
    print(f"Using symmetric color limits +/- {vmax:.0f} m/s")
    peak_speed = np.nanmax([np.nanmax(np.abs(g)) for _, g in frame_grids])
    print(f"Peak |u_lon| over all sampled frames: {peak_speed:.1f} m/s")

    # ---- Build the animation ----
    fig, (ax_map, ax_ts) = plt.subplots(
        2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [3, 1.1]}
    )

    grid0 = frame_grids[0][1]
    im = ax_map.pcolormesh(
        lon_deg, lat_deg, grid0, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    cb = fig.colorbar(im, ax=ax_map, pad=0.02, extend="both")
    cb.set_label("Zonal wind $u_{lon}$ (m/s)")
    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    ax_map.set_ylim(-90, 90)
    ax_map.set_xlim(0, 360)
    (star,) = ax_map.plot(
        [0],
        [0],
        marker="*",
        color="gold",
        markersize=18,
        markeredgecolor="k",
        markeredgewidth=0.8,
        label="substellar point",
        zorder=5,
    )
    ax_map.legend(loc="upper right", fontsize=8, framealpha=0.7)
    title = ax_map.set_title("")

    ax_ts.plot(days, eq_mean_series, color="0.3", lw=1.2)
    ax_ts.axhline(0, color="k", lw=0.5)
    ax_ts.set_xlabel("Day")
    ax_ts.set_ylabel(r"Eq-mean $u_{lon}$ (m/s)")
    ax_ts.set_xlim(0, NDAYS - 1)
    (marker,) = ax_ts.plot([0], [eq_mean_series[0]], "o", color="crimson", zorder=5)
    fig.tight_layout()

    print("Rendering frames to mp4 via imageio-ffmpeg...")
    writer = imageio.get_writer(
        OUT_MP4,
        fps=12,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        output_params=["-crf", "18", "-preset", "medium"],
    )
    t0 = time.time()
    for i, (day, grid) in enumerate(frame_grids):
        im.set_array(grid.ravel())
        title.set_text(
            f"Hot Jupiter forced dry GCM: day {day} "
            f"(level {LEVEL}, ~{100*LEVEL/32:.0f}% up the 3000 km shell)"
        )
        marker.set_data([day], [eq_mean_series[day]])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3]
        h, w, _ = frame.shape
        h2, w2 = h - (h % 2), w - (w % 2)
        writer.append_data(frame[:h2, :w2])
        if i % 40 == 0:
            print(f"  rendered {i}/{len(frame_grids)}  elapsed {time.time()-t0:.1f}s")
    writer.close()
    plt.close(fig)
    print(f"mp4 render done in {time.time()-t0:.1f}s -> {OUT_MP4}")

    series_path = os.path.splitext(OUT_MP4)[0] + "_equatorial_mean.npy"
    np.save(series_path, eq_mean_series)


if __name__ == "__main__":
    main()
