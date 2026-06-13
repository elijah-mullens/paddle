from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from netCDF4 import Dataset
from scipy import sparse


FACE_NAMES = ("+X", "+Y", "-X", "+Z", "-Y", "-Z")
MOSAIC_FACE_GRID = ((0, 1, 2), (3, 4, 5))
SNAP_TO_TEMPEST_FACE_ORDER = (0, 1, 2, 4, 5, 3)

# Snapy stores global Cartesian components in the order (Z, X, Y).
_CS_L2G_VEL = (
    ((1, +1), (2, +1), (0, +1)),
    ((2, +1), (1, -1), (0, +1)),
    ((1, -1), (2, -1), (0, +1)),
    ((0, +1), (2, +1), (1, -1)),
    ((2, -1), (1, +1), (0, +1)),
    ((0, -1), (2, +1), (1, +1)),
)

_COORDINATE_VARS = {"time", "x1", "x1f", "x2", "x2f", "x3", "x3f", "lon", "lat"}
_DEFAULT_VECTOR_TRIPLETS = (("vel1", "vel2", "vel3"),)
_REMAP_METHODS = ("bilinear", "conservative")


@dataclass(frozen=True)
class MosaicLayout:
    face_nx: int
    face_ny: int


@dataclass(frozen=True)
class TempestPaths:
    generate_cs_mesh: str
    generate_rll_mesh: str
    generate_overlap_mesh: str
    generate_offline_map: str
    apply_offline_map: str | None = None


def infer_mosaic_layout(x2_size: int, x3_size: int) -> MosaicLayout:
    if x2_size % 3 != 0 or x3_size % 2 != 0:
        raise ValueError(
            f"Expected stitched cubed-sphere mosaic with x2 divisible by 3 "
            f"and x3 divisible by 2, got x2={x2_size}, x3={x3_size}"
        )

    face_nx = x2_size // 3
    face_ny = x3_size // 2
    if face_nx != face_ny:
        raise ValueError(
            f"Expected square cubed-sphere faces, got face_nx={face_nx}, "
            f"face_ny={face_ny}"
        )

    return MosaicLayout(face_nx=face_nx, face_ny=face_ny)


def split_stitched_mosaic(array: np.ndarray, layout: MosaicLayout) -> np.ndarray:
    """Split the Snapy 3x2 stitched mosaic into a face-major array."""
    if array.shape[-2:] != (layout.face_ny * 2, layout.face_nx * 3):
        raise ValueError(
            "Unexpected stitched mosaic shape "
            f"{array.shape[-2:]} for layout {layout}"
        )

    faces = []
    for row in range(2):
        for col in range(3):
            y0 = row * layout.face_ny
            y1 = y0 + layout.face_ny
            x0 = col * layout.face_nx
            x1 = x0 + layout.face_nx
            faces.append(array[..., y0:y1, x0:x1])
    return np.stack(faces, axis=-3)


def flatten_faces(face_data: np.ndarray) -> np.ndarray:
    """Flatten face-major data into Tempest face-major row-major ordering."""
    return face_data.reshape(*face_data.shape[:-3], -1)


def reorder_faces_to_tempest(face_data: np.ndarray) -> np.ndarray:
    """Reorder Snapy's face order (+X,+Y,-X,+Z,-Y,-Z) to Tempest's order."""
    return np.take(face_data, SNAP_TO_TEMPEST_FACE_ORDER, axis=-3)


def lonlat_to_face_ab(
    face: int, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xx = np.cos(lon) * np.cos(lat)
    yy = np.sin(lon) * np.cos(lat)
    zz = np.sin(lat)

    if face == 0:  # +X
        sx, sy, sz = yy, zz, xx
    elif face == 1:  # +Y
        sx, sy, sz = -xx, zz, yy
    elif face == 2:  # -X
        sx, sy, sz = -yy, zz, -xx
    elif face == 3:  # +Z
        sx, sy, sz = yy, -xx, zz
    elif face == 4:  # -Y
        sx, sy, sz = xx, zz, -yy
    elif face == 5:  # -Z
        sx, sy, sz = yy, xx, -zz
    else:
        raise ValueError(f"Invalid face index {face}")

    alpha = np.arctan2(sx, sz)
    beta = np.arctan2(sy, sz)
    return alpha, beta


def _local_contra_to_global_xyz(
    face: int,
    vel1: np.ndarray,
    vel2: np.ndarray,
    vel3: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.tan(alpha)
    y = np.tan(beta)

    delta = np.sqrt(x * x + y * y + 1.0)
    c_val = np.sqrt(1.0 + x * x)
    d_val = np.sqrt(1.0 + y * y)

    local_z = (vel1 - vel2 * x / d_val - vel3 * y / c_val) / delta
    local_x = (vel1 * x + vel2 * d_val - (vel3 * x * y) / c_val) / delta
    local_y = (vel1 * y + vel3 * c_val - (vel2 * x * y) / d_val) / delta

    global_zyx = [None, None, None]
    for local_idx, values in enumerate((local_z, local_x, local_y)):
        global_idx, sign = _CS_L2G_VEL[face][local_idx]
        global_zyx[global_idx] = sign * values

    global_x = global_zyx[1]
    global_y = global_zyx[2]
    global_z = global_zyx[0]
    return global_x, global_y, global_z


def rotate_vector_faces_to_geographic(
    vel1_faces: np.ndarray,
    vel2_faces: np.ndarray,
    vel3_faces: np.ndarray,
    lon_faces: np.ndarray,
    lat_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    east_faces = []
    north_faces = []
    up_faces = []

    for face in range(6):
        alpha, beta = lonlat_to_face_ab(face, lon_faces[face], lat_faces[face])
        gx, gy, gz = _local_contra_to_global_xyz(
            face,
            vel1_faces[face],
            vel2_faces[face],
            vel3_faces[face],
            alpha,
            beta,
        )

        lon = lon_faces[face]
        lat = lat_faces[face]

        east = -np.sin(lon) * gx + np.cos(lon) * gy
        north = (
            -np.sin(lat) * np.cos(lon) * gx
            - np.sin(lat) * np.sin(lon) * gy
            + np.cos(lat) * gz
        )
        up = (
            np.cos(lat) * np.cos(lon) * gx
            + np.cos(lat) * np.sin(lon) * gy
            + np.sin(lat) * gz
        )

        east_faces.append(east)
        north_faces.append(north)
        up_faces.append(up)

    return (
        np.stack(east_faces, axis=0),
        np.stack(north_faces, axis=0),
        np.stack(up_faces, axis=0),
    )


def _which(program: str) -> str | None:
    return shutil.which(program)


def ensure_tempestremap_available(
    tempest_bin_dir: str | os.PathLike[str] | None = None,
) -> TempestPaths:
    def resolve(name: str) -> str:
        if tempest_bin_dir is not None:
            candidate = Path(tempest_bin_dir) / name
            if candidate.exists():
                return str(candidate)

        found = _which(name)
        if found is not None:
            return found

        raise FileNotFoundError(
            f"Could not find TempestRemap executable '{name}'. "
            "Install TempestRemap and add it to PATH, or pass "
            "--tempest-bin-dir."
        )

    apply_path = None
    try:
        apply_path = resolve("ApplyOfflineMap")
    except FileNotFoundError:
        apply_path = None

    return TempestPaths(
        generate_cs_mesh=resolve("GenerateCSMesh"),
        generate_rll_mesh=resolve("GenerateRLLMesh"),
        generate_overlap_mesh=resolve("GenerateOverlapMesh"),
        generate_offline_map=resolve("GenerateOfflineMap"),
        apply_offline_map=apply_path,
    )


def _run_command(cmd: Sequence[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _generate_tempest_map(
    source_res: int,
    nlat: int,
    nlon: int,
    map_path: Path,
    tempest_paths: TempestPaths,
    remap_method: str = "bilinear",
) -> Path:
    _validate_remap_method(remap_method)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if map_path.exists():
        return map_path

    source_mesh = map_path.with_name(f"cs_res{source_res}.g")
    target_mesh = map_path.with_name(f"rll_{nlat}x{nlon}.g")
    overlap_mesh = map_path.with_name(f"ov_cs{source_res}_rll_{nlat}x{nlon}.g")

    if not source_mesh.exists():
        _run_command(
            [
                tempest_paths.generate_cs_mesh,
                "--res",
                str(source_res),
                "--file",
                str(source_mesh),
            ]
        )

    if not target_mesh.exists():
        _run_command(
            [
                tempest_paths.generate_rll_mesh,
                "--lat",
                str(nlat),
                "--lon",
                str(nlon),
                "--file",
                str(target_mesh),
            ]
        )

    if not overlap_mesh.exists():
        _run_command(
            [
                tempest_paths.generate_overlap_mesh,
                "--a",
                str(source_mesh),
                "--b",
                str(target_mesh),
                "--out",
                str(overlap_mesh),
            ]
        )

    map_command = [
        tempest_paths.generate_offline_map,
        "--in_mesh",
        str(source_mesh),
        "--out_mesh",
        str(target_mesh),
        "--ov_mesh",
        str(overlap_mesh),
        "--in_type",
        "fv",
        "--out_type",
        "fv",
        "--in_np",
        "1",
    ]
    if remap_method == "bilinear":
        map_command.extend(["--method", "bilin"])
    map_command.extend(["--out_map", str(map_path)])
    _run_command(map_command)
    return map_path


def _validate_remap_method(remap_method: str) -> None:
    if remap_method not in _REMAP_METHODS:
        methods = ", ".join(_REMAP_METHODS)
        raise ValueError(
            f"Unknown remap method '{remap_method}'; expected one of {methods}"
        )


def _map_cache_path(
    cache_dir: Path,
    layout: MosaicLayout,
    nlat: int,
    nlon: int,
    remap_method: str,
) -> Path:
    _validate_remap_method(remap_method)
    return cache_dir / f"cs_res{layout.face_nx}_to_rll_{nlat}x{nlon}_{remap_method}.nc"


def _load_offline_map_matrix(map_path: str | os.PathLike[str]) -> sparse.csr_matrix:
    with Dataset(map_path) as nc:
        n_source = _pick_dim_size(nc, ("n_a", "src_grid_size", "n_src"))
        n_target = _pick_dim_size(nc, ("n_b", "dst_grid_size", "n_dst"))

        row = _pick_var(nc, ("row", "row_idx", "dst_address"))[:].astype(np.int64)
        col = _pick_var(nc, ("col", "col_idx", "src_address"))[:].astype(np.int64)
        weights = _pick_var(nc, ("S", "weight", "weights", "remap_matrix"))[:]

    if row.min() == 1 and col.min() == 1:
        row = row - 1
        col = col - 1

    matrix = sparse.coo_matrix(
        (weights, (row, col)),
        shape=(n_target, n_source),
    )
    return matrix.tocsr()


def _pick_dim_size(nc: Dataset, candidates: Sequence[str]) -> int:
    for name in candidates:
        if name in nc.dimensions:
            return len(nc.dimensions[name])
    raise KeyError(f"None of the expected dimensions {candidates} found in map file")


def _pick_var(nc: Dataset, candidates: Sequence[str]):
    for name in candidates:
        if name in nc.variables:
            return nc.variables[name]
    raise KeyError(f"None of the expected variables {candidates} found in map file")


def _default_scalar_vars(
    variable_names: Iterable[str],
    vector_triplets: Sequence[tuple[str, str, str]],
) -> list[str]:
    blocked = set(_COORDINATE_VARS)
    for triplet in vector_triplets:
        blocked.update(triplet)
    return [name for name in variable_names if name not in blocked]


def _normalize_vector_triplets(
    vector_triplets: Sequence[Sequence[str]] | None,
    variable_names: set[str],
) -> list[tuple[str, str, str]]:
    if vector_triplets is None:
        default = []
        for triplet in _DEFAULT_VECTOR_TRIPLETS:
            if all(name in variable_names for name in triplet):
                default.append(tuple(triplet))
        return default

    normalized = []
    for triplet in vector_triplets:
        if len(triplet) != 3:
            raise ValueError(f"Vector triplet must have length 3, got {triplet}")
        normalized.append((triplet[0], triplet[1], triplet[2]))
    return normalized


def _load_inputs(
    inputs: Sequence[str | os.PathLike[str]],
) -> tuple[list[Dataset], dict[str, tuple[Dataset, str]]]:
    datasets = [Dataset(path) for path in inputs]
    var_sources: dict[str, tuple[Dataset, str]] = {}
    try:
        for ds in datasets:
            for name in ds.variables:
                if name in _COORDINATE_VARS:
                    continue
                if name in var_sources:
                    raise ValueError(
                        f"Variable '{name}' appears in multiple input files"
                    )
                var_sources[name] = (ds, name)
        return datasets, var_sources
    except Exception:
        for ds in datasets:
            ds.close()
        raise


def _read_coordinate_faces(
    ds: Dataset, layout: MosaicLayout
) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(ds.variables["lon"][0], dtype=np.float64)
    lat = np.asarray(ds.variables["lat"][0], dtype=np.float64)
    return split_stitched_mosaic(lon, layout), split_stitched_mosaic(lat, layout)


def remap_scalar_fields(
    source_arrays: dict[str, np.ndarray],
    weights: sparse.csr_matrix,
    nlat: int,
    nlon: int,
) -> dict[str, np.ndarray]:
    remapped = {}
    for name, array in source_arrays.items():
        leading_shape = array.shape[:-1]
        flat = array.reshape(-1, array.shape[-1])
        mapped = weights.dot(flat.T).T
        remapped[name] = mapped.reshape(*leading_shape, nlat, nlon)
    return remapped


def remap_vector_fields(
    vector_triplets: Sequence[tuple[str, str, str]],
    var_sources: dict[str, tuple[Dataset, str]],
    lon_faces: np.ndarray,
    lat_faces: np.ndarray,
    layout: MosaicLayout,
    weights: sparse.csr_matrix,
    nlat: int,
    nlon: int,
) -> dict[str, np.ndarray]:
    remapped: dict[str, np.ndarray] = {}

    for vel1_name, vel2_name, vel3_name in vector_triplets:
        east_name, north_name, up_name = _vector_output_names(
            vel1_name,
            vel2_name,
            vel3_name,
        )
        ds1, _ = var_sources[vel1_name]
        ds2, _ = var_sources[vel2_name]
        ds3, _ = var_sources[vel3_name]

        vel1 = np.asarray(ds1.variables[vel1_name][:], dtype=np.float64)
        vel2 = np.asarray(ds2.variables[vel2_name][:], dtype=np.float64)
        vel3 = np.asarray(ds3.variables[vel3_name][:], dtype=np.float64)

        time_size, x1_size = vel1.shape[:2]
        east = np.empty(
            (time_size, x1_size, 6, layout.face_ny, layout.face_nx), dtype=np.float64
        )
        north = np.empty_like(east)
        up = np.empty_like(east)

        for t_index in range(time_size):
            for x1_index in range(x1_size):
                v1_faces = split_stitched_mosaic(vel1[t_index, x1_index], layout)
                v2_faces = split_stitched_mosaic(vel2[t_index, x1_index], layout)
                v3_faces = split_stitched_mosaic(vel3[t_index, x1_index], layout)
                east_faces, north_faces, up_faces = rotate_vector_faces_to_geographic(
                    v1_faces,
                    v2_faces,
                    v3_faces,
                    lon_faces,
                    lat_faces,
                )
                east[t_index, x1_index] = east_faces
                north[t_index, x1_index] = north_faces
                up[t_index, x1_index] = up_faces

        east_flat = flatten_faces(reorder_faces_to_tempest(east))
        north_flat = flatten_faces(reorder_faces_to_tempest(north))
        up_flat = flatten_faces(reorder_faces_to_tempest(up))

        remapped[east_name] = remap_scalar_fields(
            {east_name: east_flat},
            weights,
            nlat,
            nlon,
        )[east_name]
        remapped[north_name] = remap_scalar_fields(
            {north_name: north_flat},
            weights,
            nlat,
            nlon,
        )[north_name]
        remapped[up_name] = remap_scalar_fields(
            {up_name: up_flat},
            weights,
            nlat,
            nlon,
        )[up_name]

    return remapped


def _vector_output_names(
    vel1_name: str,
    vel2_name: str,
    vel3_name: str,
) -> tuple[str, str, str]:
    if (vel1_name, vel2_name, vel3_name) == ("vel1", "vel2", "vel3"):
        return ("vel_east", "vel_north", "vel_up")

    base = vel1_name.rstrip("123") or "vector"
    return (f"{base}_east", f"{base}_north", f"{base}_up")


def _build_target_coordinates(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    lat_edges = np.linspace(-90.0, 90.0, nlat + 1)
    lon_edges = np.linspace(0.0, 360.0, nlon + 1)
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return lat.astype(np.float32), lon.astype(np.float32)


def _write_output_file(
    output: str | os.PathLike[str],
    template_ds: Dataset,
    coordinate_ds: Dataset,
    var_sources: dict[str, tuple[Dataset, str]],
    vector_triplets: Sequence[tuple[str, str, str]],
    remapped_scalars: dict[str, np.ndarray],
    remapped_vectors: dict[str, np.ndarray],
    nlat: int,
    nlon: int,
) -> None:
    lat, lon = _build_target_coordinates(nlat, nlon)
    time_vals = np.asarray(template_ds.variables["time"][:], dtype=np.float32)
    x1_vals = np.asarray(template_ds.variables["x1"][:], dtype=np.float32)
    x1f_vals = (
        np.asarray(template_ds.variables["x1f"][:], dtype=np.float32)
        if "x1f" in template_ds.variables
        else None
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(output_path, "w") as nc:
        nc.Conventions = "CF-1.10"
        nc.createDimension("time", time_vals.shape[0])
        nc.createDimension("altitude", x1_vals.shape[0])
        if x1f_vals is not None:
            nc.createDimension("bnds", 2)
        nc.createDimension("lat", nlat)
        nc.createDimension("lon", nlon)

        for attr in template_ds.ncattrs():
            nc.setncattr(attr, template_ds.getncattr(attr))

        time_var = nc.createVariable("time", "f4", ("time",))
        time_var.standard_name = "time"
        time_var.axis = "T"
        if "time" in template_ds.variables:
            src_time = template_ds.variables["time"]
            _copy_variable_attrs(src_time, time_var)
        if "long_name" not in time_var.ncattrs():
            time_var.long_name = "time"
        time_var[:] = time_vals

        altitude_var = nc.createVariable("altitude", "f4", ("altitude",))
        altitude_var.standard_name = "altitude"
        altitude_var.axis = "Z"
        altitude_var.positive = "up"
        if "x1" in template_ds.variables:
            _copy_variable_attrs(template_ds.variables["x1"], altitude_var)
        if "units" not in altitude_var.ncattrs():
            altitude_var.units = "m"
        altitude_var.long_name = "altitude"
        if x1f_vals is not None:
            altitude_var.bounds = "altitude_bounds"
        altitude_var[:] = x1_vals

        if x1f_vals is not None:
            altitude_bounds_var = nc.createVariable(
                "altitude_bounds",
                "f4",
                ("altitude", "bnds"),
            )
            if "x1f" in template_ds.variables:
                _copy_variable_attrs(template_ds.variables["x1f"], altitude_bounds_var)
            altitude_bounds_var[:] = np.stack([x1f_vals[:-1], x1f_vals[1:]], axis=1)

        lat_var = nc.createVariable("lat", "f4", ("lat",))
        lat_var.standard_name = "latitude"
        lat_var.axis = "Y"
        if "lat" in coordinate_ds.variables:
            _copy_variable_attrs(coordinate_ds.variables["lat"], lat_var)
        lat_var.units = "degrees_north"
        lat_var.long_name = "latitude"
        lat_var[:] = lat
        lon_var = nc.createVariable("lon", "f4", ("lon",))
        lon_var.standard_name = "longitude"
        lon_var.axis = "X"
        if "lon" in coordinate_ds.variables:
            _copy_variable_attrs(coordinate_ds.variables["lon"], lon_var)
        lon_var.units = "degrees_east"
        lon_var.long_name = "longitude"
        lon_var[:] = lon

        for name, data in {**remapped_scalars, **remapped_vectors}.items():
            var = nc.createVariable(name, "f4", ("time", "altitude", "lat", "lon"))
            if name in var_sources:
                _copy_variable_attrs(
                    var_sources[name][0].variables[var_sources[name][1]], var
                )
            else:
                _apply_vector_attrs(name, vector_triplets, var_sources, var)
            var[:] = data.astype(np.float32)


def _copy_variable_attrs(src_var, dst_var) -> None:
    for attr_name in src_var.ncattrs():
        if attr_name == "_FillValue":
            continue
        dst_var.setncattr(attr_name, src_var.getncattr(attr_name))


def _apply_vector_attrs(
    output_name: str,
    vector_triplets: Sequence[tuple[str, str, str]],
    var_sources: dict[str, tuple[Dataset, str]],
    dst_var,
) -> None:
    for vel1_name, vel2_name, vel3_name in vector_triplets:
        east_name, north_name, up_name = _vector_output_names(
            vel1_name, vel2_name, vel3_name
        )
        if output_name == east_name:
            src = var_sources[vel2_name][0].variables[vel2_name]
            _copy_variable_attrs(src, dst_var)
            if "long_name" in dst_var.ncattrs():
                dst_var.long_name = "eastward velocity"
            return
        if output_name == north_name:
            src = var_sources[vel3_name][0].variables[vel3_name]
            _copy_variable_attrs(src, dst_var)
            if "long_name" in dst_var.ncattrs():
                dst_var.long_name = "northward velocity"
            return
        if output_name == up_name:
            src = var_sources[vel1_name][0].variables[vel1_name]
            _copy_variable_attrs(src, dst_var)
            if "long_name" in dst_var.ncattrs():
                dst_var.long_name = "upward velocity"
            return


def remap_cubed_sphere_files(
    inputs: Sequence[str | os.PathLike[str]],
    output: str | os.PathLike[str],
    nlat: int,
    nlon: int,
    scalar_vars: Sequence[str] | None = None,
    vector_triplets: Sequence[Sequence[str]] | None = None,
    map_cache_dir: str | os.PathLike[str] | None = None,
    tempest_bin_dir: str | os.PathLike[str] | None = None,
    remap_method: str = "bilinear",
) -> Path:
    if not inputs:
        raise ValueError("At least one input file is required")
    _validate_remap_method(remap_method)

    datasets, var_sources = _load_inputs(inputs)
    try:
        template_ds = datasets[0]
        coordinate_ds = next(
            (ds for ds in datasets if "lon" in ds.variables and "lat" in ds.variables),
            None,
        )
        if coordinate_ds is None:
            raise KeyError("No input file contains the required lon/lat coordinates")
        layout = infer_mosaic_layout(
            len(template_ds.dimensions["x2"]),
            len(template_ds.dimensions["x3"]),
        )
        lon_faces, lat_faces = _read_coordinate_faces(coordinate_ds, layout)

        vector_triplets_norm = _normalize_vector_triplets(
            vector_triplets, set(var_sources)
        )
        if scalar_vars is None:
            scalar_vars_norm = _default_scalar_vars(var_sources, vector_triplets_norm)
        else:
            scalar_vars_norm = list(scalar_vars)

        for scalar_name in scalar_vars_norm:
            if scalar_name not in var_sources:
                raise KeyError(f"Scalar variable '{scalar_name}' not found in inputs")
        for triplet in vector_triplets_norm:
            for name in triplet:
                if name not in var_sources:
                    raise KeyError(f"Vector variable '{name}' not found in inputs")

        cache_dir = (
            Path(map_cache_dir)
            if map_cache_dir is not None
            else Path(output).resolve().parent / ".tempestremap"
        )
        map_path = _map_cache_path(cache_dir, layout, nlat, nlon, remap_method)
        if not map_path.exists():
            tempest_paths = ensure_tempestremap_available(tempest_bin_dir)
            _generate_tempest_map(
                layout.face_nx,
                nlat,
                nlon,
                map_path,
                tempest_paths,
                remap_method=remap_method,
            )
        weights = _load_offline_map_matrix(map_path)

        scalar_source_arrays = {}
        for name in scalar_vars_norm:
            ds, var_name = var_sources[name]
            data = np.asarray(ds.variables[var_name][:], dtype=np.float64)
            faces = split_stitched_mosaic(data, layout)
            scalar_source_arrays[name] = flatten_faces(reorder_faces_to_tempest(faces))

        remapped_scalars = remap_scalar_fields(
            scalar_source_arrays, weights, nlat, nlon
        )
        remapped_vectors = remap_vector_fields(
            vector_triplets_norm,
            var_sources,
            lon_faces,
            lat_faces,
            layout,
            weights,
            nlat,
            nlon,
        )
        _write_output_file(
            output,
            template_ds,
            coordinate_ds,
            var_sources,
            vector_triplets_norm,
            remapped_scalars,
            remapped_vectors,
            nlat,
            nlon,
        )
        return Path(output)
    finally:
        for ds in datasets:
            ds.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remap Snapy stitched cubed-sphere NetCDF outputs to a lat-lon grid."
    )
    parser.add_argument("inputs", nargs="+", help="Input Snapy NetCDF files")
    parser.add_argument("output", help="Output lat-lon NetCDF file")
    parser.add_argument(
        "--nlat", type=int, required=True, help="Number of latitude cells"
    )
    parser.add_argument(
        "--nlon", type=int, required=True, help="Number of longitude cells"
    )
    parser.add_argument(
        "--scalar",
        action="append",
        default=None,
        help="Scalar variable to remap. Repeat to specify multiple variables.",
    )
    parser.add_argument(
        "--vector",
        action="append",
        default=None,
        help="Vector triplet in the form vel1,vel2,vel3",
    )
    parser.add_argument(
        "--method",
        choices=_REMAP_METHODS,
        default="bilinear",
        help=(
            "Remapping method. Bilinear treats Snapy fields as cell-centered "
            "samples; conservative treats them as finite-volume cell averages."
        ),
    )
    parser.add_argument(
        "--map-cache-dir",
        default=None,
        help="Directory to cache TempestRemap mesh and map files.",
    )
    parser.add_argument(
        "--tempest-bin-dir",
        default=None,
        help="Directory containing TempestRemap executables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    vector_triplets = None
    if args.vector:
        vector_triplets = [tuple(item.split(",")) for item in args.vector]
    remap_cubed_sphere_files(
        inputs=args.inputs,
        output=args.output,
        nlat=args.nlat,
        nlon=args.nlon,
        scalar_vars=args.scalar,
        vector_triplets=vector_triplets,
        map_cache_dir=args.map_cache_dir,
        tempest_bin_dir=args.tempest_bin_dir,
        remap_method=args.method,
    )
    return 0
