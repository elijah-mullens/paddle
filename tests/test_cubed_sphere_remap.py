from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paddle import __main__ as paddle_main
from paddle.cubed_sphere_remap import (
    _CS_L2G_VEL,
    _map_cache_path,
    _run_command,
    _write_output_file,
    _generate_tempest_map,
    _load_offline_map_matrix,
    build_parser,
    infer_mosaic_layout,
    remap_scalar_fields,
    reorder_faces_to_tempest,
    rotate_vector_faces_to_geographic,
    split_stitched_mosaic,
)


def _snapy_face_coordinates(n: int) -> tuple[np.ndarray, ...]:
    angles = np.linspace(-np.pi / 4.0, np.pi / 4.0, n, endpoint=False)
    angles += np.pi / (4.0 * n)
    alpha, beta = np.meshgrid(angles, angles)
    aa = np.tan(alpha)
    bb = np.tan(beta)
    raw_xyz = (
        (np.ones_like(aa), aa, bb),
        (-aa, np.ones_like(aa), bb),
        (-np.ones_like(aa), -aa, bb),
        (-bb, aa, np.ones_like(aa)),
        (aa, -np.ones_like(aa), bb),
        (bb, aa, -np.ones_like(aa)),
    )
    xyz = np.stack([np.stack(values, axis=-1) for values in raw_xyz])
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    lon = np.mod(np.arctan2(xyz[..., 1], xyz[..., 0]), 2.0 * np.pi)
    lat = np.arcsin(xyz[..., 2])
    return alpha, beta, xyz, lon, lat


def _global_cartesian_to_snapy_contravariant(
    global_xyz: tuple[float, float, float],
    alpha: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_zyx = (global_xyz[2], global_xyz[0], global_xyz[1])
    vel1 = np.empty((6, *alpha.shape))
    vel2 = np.empty_like(vel1)
    vel3 = np.empty_like(vel1)
    x = np.tan(alpha)
    y = np.tan(beta)
    delta = np.sqrt(x * x + y * y + 1.0)
    c_val = np.sqrt(1.0 + x * x)
    d_val = np.sqrt(1.0 + y * y)

    for face in range(6):
        local_z, local_x, local_y = (
            sign * global_zyx[global_idx] for global_idx, sign in _CS_L2G_VEL[face]
        )
        vel1[face] = (local_z + x * local_x + y * local_y) / delta
        vel2[face] = d_val / delta * (local_x - x * local_z)
        vel3[face] = c_val / delta * (local_y - y * local_z)

    return vel1, vel2, vel3


def test_split_stitched_mosaic_preserves_snapy_face_order() -> None:
    mosaic = np.arange(24).reshape(4, 6)
    layout = infer_mosaic_layout(x2_size=6, x3_size=4)
    faces = split_stitched_mosaic(mosaic, layout)
    assert faces.shape == (6, 2, 2)
    expected = [
        np.array([[0, 1], [6, 7]]),
        np.array([[2, 3], [8, 9]]),
        np.array([[4, 5], [10, 11]]),
        np.array([[12, 13], [18, 19]]),
        np.array([[14, 15], [20, 21]]),
        np.array([[16, 17], [22, 23]]),
    ]
    for face, expected_face in enumerate(expected):
        np.testing.assert_array_equal(faces[face], expected_face)


def test_rotate_vector_faces_to_geographic_at_face_centers() -> None:
    lon_faces = np.zeros((6, 1, 1))
    lat_faces = np.zeros((6, 1, 1))
    lon_faces[0, 0, 0] = 0.0
    lat_faces[0, 0, 0] = 0.0

    vel1 = np.zeros((6, 1, 1))
    vel2 = np.zeros((6, 1, 1))
    vel3 = np.zeros((6, 1, 1))

    vel1[0, 0, 0] = 2.0
    east, north, up = rotate_vector_faces_to_geographic(
        vel1, vel2, vel3, lon_faces, lat_faces
    )
    assert np.isclose(up[0, 0, 0], 2.0)

    vel1[:] = 0.0
    vel2[0, 0, 0] = 3.0
    east, north, up = rotate_vector_faces_to_geographic(
        vel1, vel2, vel3, lon_faces, lat_faces
    )
    assert np.isclose(east[0, 0, 0], 3.0)

    vel2[:] = 0.0
    vel3[0, 0, 0] = 4.0
    east, north, up = rotate_vector_faces_to_geographic(
        vel1, vel2, vel3, lon_faces, lat_faces
    )
    assert np.isclose(north[0, 0, 0], 4.0)


def test_rotate_vector_faces_to_geographic_recovers_global_cartesian_vector() -> None:
    alpha, beta, _, lon, lat = _snapy_face_coordinates(5)
    global_x, global_y, global_z = (2.0, -3.0, 4.0)
    vel1, vel2, vel3 = _global_cartesian_to_snapy_contravariant(
        (global_x, global_y, global_z),
        alpha,
        beta,
    )

    east, north, up = rotate_vector_faces_to_geographic(vel1, vel2, vel3, lon, lat)

    expected_east = -np.sin(lon) * global_x + np.cos(lon) * global_y
    expected_north = (
        -np.sin(lat) * np.cos(lon) * global_x
        - np.sin(lat) * np.sin(lon) * global_y
        + np.cos(lat) * global_z
    )
    expected_up = (
        np.cos(lat) * np.cos(lon) * global_x
        + np.cos(lat) * np.sin(lon) * global_y
        + np.sin(lat) * global_z
    )
    np.testing.assert_allclose(east, expected_east, atol=1.0e-12)
    np.testing.assert_allclose(north, expected_north, atol=1.0e-12)
    np.testing.assert_allclose(up, expected_up, atol=1.0e-12)


def test_reorder_faces_to_tempest_matches_verified_face_order() -> None:
    faces = np.arange(6).reshape(6, 1, 1)
    reordered = reorder_faces_to_tempest(faces)
    np.testing.assert_array_equal(reordered[:, 0, 0], np.array([0, 1, 2, 4, 5, 3]))


def test_reorder_faces_to_tempest_preserves_geographic_scalar_alignment() -> None:
    _, _, xyz, _, _ = _snapy_face_coordinates(5)
    scalar = xyz[..., 0] + 2.0 * xyz[..., 1] - 0.5 * xyz[..., 2]

    reordered_xyz = reorder_faces_to_tempest(np.moveaxis(xyz, -1, 0))
    reordered_scalar = reorder_faces_to_tempest(scalar)

    expected = reordered_xyz[0] + 2.0 * reordered_xyz[1] - 0.5 * reordered_xyz[2]
    np.testing.assert_allclose(reordered_scalar, expected)


def test_load_offline_map_matrix_supports_standard_weight_file(tmp_path: Path) -> None:
    path = tmp_path / "map.nc"
    with Dataset(path, "w") as nc:
        nc.createDimension("n_a", 4)
        nc.createDimension("n_b", 2)
        nc.createDimension("n_s", 4)
        row = nc.createVariable("row", "i4", ("n_s",))
        col = nc.createVariable("col", "i4", ("n_s",))
        weights = nc.createVariable("S", "f8", ("n_s",))
        row[:] = [1, 1, 2, 2]
        col[:] = [1, 2, 3, 4]
        weights[:] = [0.25, 0.75, 0.1, 0.9]

    matrix = _load_offline_map_matrix(path)
    vector = np.array([[1.0, 2.0, 3.0, 4.0]])
    mapped = matrix.dot(vector.T).T
    np.testing.assert_allclose(mapped, np.array([[1.75, 3.9]]))


def test_remap_scalar_fields_reshapes_back_to_lat_lon() -> None:
    weights = sparse.csr_matrix(np.eye(4))
    source = {"rho": np.array([[[1.0, 2.0, 3.0, 4.0]]])}
    remapped = remap_scalar_fields(source, weights, nlat=2, nlon=2)
    np.testing.assert_allclose(remapped["rho"], np.array([[[[1.0, 2.0], [3.0, 4.0]]]]))


@pytest.mark.parametrize(
    ("remap_method", "expected_method"),
    [("bilinear", "bilin"), ("conservative", None)],
)
def test_generate_tempest_map_builds_expected_commands(
    tmp_path: Path,
    monkeypatch,
    remap_method: str,
    expected_method: str | None,
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None, check=True):
        calls.append(list(cmd))
        if "--file" in cmd:
            Path(cmd[cmd.index("--file") + 1]).touch()
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).touch()
        if "--out_map" in cmd:
            Path(cmd[cmd.index("--out_map") + 1]).touch()

    monkeypatch.setattr("paddle.cubed_sphere_remap._run_command", fake_run)
    paths = type(
        "TempestPathsLike",
        (),
        {
            "generate_cs_mesh": "GenerateCSMesh",
            "generate_rll_mesh": "GenerateRLLMesh",
            "generate_overlap_mesh": "GenerateOverlapMesh",
            "generate_offline_map": "GenerateOfflineMap",
        },
    )()

    out_map = tmp_path / "map.nc"
    _generate_tempest_map(4, 8, 16, out_map, paths, remap_method=remap_method)
    assert len(calls) == 4
    assert calls[0][:3] == ["GenerateCSMesh", "--res", "4"]
    assert calls[1][:3] == ["GenerateRLLMesh", "--lat", "8"]
    assert calls[2][0] == "GenerateOverlapMesh"
    assert calls[3][0] == "GenerateOfflineMap"
    if expected_method is None:
        assert "--method" not in calls[3]
    else:
        method_index = calls[3].index("--method")
        assert calls[3][method_index + 1] == expected_method


def test_map_cache_path_separates_remap_methods(tmp_path: Path) -> None:
    layout = infer_mosaic_layout(x2_size=12, x3_size=8)
    bilinear = _map_cache_path(tmp_path, layout, 181, 360, "bilinear")
    conservative = _map_cache_path(tmp_path, layout, 181, 360, "conservative")

    assert bilinear.name == "cs_res4_to_rll_181x360_bilinear.nc"
    assert conservative.name == "cs_res4_to_rll_181x360_conservative.nc"
    assert bilinear != conservative


def test_run_command_executes_subprocess_once(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, cwd=None, check=True):
        calls.append((cmd, cwd, check))

    monkeypatch.setattr("paddle.cubed_sphere_remap.subprocess.run", fake_run)
    _run_command(["GenerateCSMesh", "--help"])
    assert calls == [(["GenerateCSMesh", "--help"], None, True)]


def test_parser_defaults_to_bilinear_and_accepts_conservative() -> None:
    parser = build_parser()
    default = parser.parse_args(["in.nc", "out.nc", "--nlat", "181", "--nlon", "360"])
    conservative = parser.parse_args(
        [
            "in.nc",
            "out.nc",
            "--nlat",
            "181",
            "--nlon",
            "360",
            "--method",
            "conservative",
        ]
    )
    assert default.method == "bilinear"
    assert conservative.method == "conservative"


def test_paddle_cs_remap_help_is_forwarded(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        paddle_main.main(["cs-remap", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--nlat" in help_text
    assert "--nlon" in help_text
    assert "--method" in help_text


def test_write_output_file_omits_coordinates_attribute(tmp_path: Path) -> None:
    template_path = tmp_path / "template.nc"
    output_path = tmp_path / "out.nc"

    with Dataset(template_path, "w") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("x1", 2)
        nc.createDimension("x1f", 3)
        nc.createDimension("x2", 6)
        nc.createDimension("x3", 4)
        time = nc.createVariable("time", "f4", ("time",))
        time.units = "s"
        time[:] = [0.0]
        x1 = nc.createVariable("x1", "f4", ("x1",))
        x1.units = "m"
        x1[:] = [1.0, 2.0]
        x1f = nc.createVariable("x1f", "f4", ("x1f",))
        x1f[:] = [0.0, 1.5, 3.0]
        lon = nc.createVariable("lon", "f4", ("time", "x3", "x2"))
        lat = nc.createVariable("lat", "f4", ("time", "x3", "x2"))
        lon[:] = 0.0
        lat[:] = 0.0
        rho = nc.createVariable("rho", "f4", ("time", "x1", "x3", "x2"))
        rho.units = "kg/m^3"
        rho.long_name = "density"
        rho[:] = 1.0

    with Dataset(template_path) as template_ds, Dataset(template_path) as coordinate_ds:
        _write_output_file(
            output_path,
            template_ds,
            coordinate_ds,
            {"rho": (template_ds, "rho")},
            (),
            {"rho": np.ones((1, 2, 2, 2), dtype=np.float64)},
            {},
            2,
            2,
        )

    with Dataset(output_path) as out_nc:
        assert "coordinates" not in out_nc.variables["rho"].ncattrs()
