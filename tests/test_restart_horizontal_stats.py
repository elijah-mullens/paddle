from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from paddle.cubed_sphere_shrink import (
    _load_entry,
    _save_tensors,
    _write_bundle,
    read_restart_bundle_index,
)
from paddle.restart_horizontal_stats import create_restart
from paddle.restart_horizontal_stats import _block_count_from_layout


class _FakeValue:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is not None:
            self.value = value
            return self
        return self.value


class _FakeLayout:
    type = _FakeValue("cubed-sphere")
    px = _FakeValue(1)
    py = _FakeValue(1)
    pz = _FakeValue(1)


class _FakeCoord:
    nghost = _FakeValue(1)
    nx1 = _FakeValue(2)
    nx2 = _FakeValue(3)
    nx3 = _FakeValue(2)


class _FakeBlockOptions:
    def layout(self):
        return _FakeLayout()

    def coord(self):
        return _FakeCoord()


class _FakeMeshOptions:
    block_count: int | None = None
    local_cells: tuple[int, int] | None = None

    @staticmethod
    def from_yaml(_path):
        return _FakeMeshOptions()

    def block(self):
        return _FakeBlockOptions()

    def blocks_per_process(self, value):
        self.block_count = value

    def set_local_horizontal_cells(self, nx2, nx3):
        self.local_cells = (nx2, nx3)


class _FakeMesh:
    calls: list[int] = []

    def __init__(self, options):
        self.options = options

    def exchange_ghost_zones(self, variables, field_type):
        self.calls.append(field_type)


def _fake_snapy():
    return SimpleNamespace(
        MeshOptions=_FakeMeshOptions,
        Mesh=_FakeMesh,
        kPrimitive=0,
        kConserved=1,
        kScalar=2,
        kIV1=1,
    )


def _source_tensor(offset: float) -> torch.Tensor:
    tensor = torch.zeros((3, 4, 5, 4), dtype=torch.float64)
    for component in range(3):
        for x1 in range(4):
            tensor[component, :, :, x1] = offset + component * 100 + x1 * 10
    tensor[:, 1:3, 1:4, :] += torch.arange(2 * 3, dtype=torch.float64).reshape(
        1, 2, 3, 1
    )
    return tensor


def _make_source_restart(path: Path) -> None:
    part_dir = path.parent / "parts"
    part_dir.mkdir()
    parts = []
    for rank, offset in enumerate((0.0, 1000.0)):
        part = part_dir / f"block{rank}.part"
        hydro_u = _source_tensor(offset)
        hydro_w = _source_tensor(offset + 10_000.0)
        _save_tensors(
            {
                "hydro_u": hydro_u,
                "hydro_w": hydro_w,
                "fill_solid_hydro_u": hydro_u + 20_000.0,
                "fill_solid_hydro_w": hydro_w + 30_000.0,
                "last_cycle": torch.tensor([123], dtype=torch.int64),
                "last_time": torch.tensor([45.0], dtype=torch.float64),
            },
            part,
        )
        parts.append(part)
    _write_bundle(path, parts)


def _load_output(path: Path) -> list[dict[str, torch.Tensor]]:
    return [_load_entry(path, entry) for entry in read_restart_bundle_index(path)]


def test_restart_parser_uses_public_command_name() -> None:
    from paddle.restart_horizontal_stats import build_parser

    assert build_parser().prog == "paddle restart"


def test_create_restart_uses_target_shape_and_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    output = tmp_path / "output.restart"
    _make_source_restart(source)
    monkeypatch.setitem(sys.modules, "snapy", _fake_snapy())
    _FakeMesh.calls = []

    result = create_restart(config, source, output, seed=10)

    assert result == output.resolve()
    blocks = _load_output(output)
    assert len(blocks) == 6
    for block in blocks:
        assert tuple(block["hydro_u"].shape) == (3, 4, 5, 4)
        assert tuple(block["hydro_w"].shape) == (3, 4, 5, 4)
        torch.testing.assert_close(block["last_cycle"], torch.tensor([123]))
        torch.testing.assert_close(block["last_time"], torch.tensor([45.0]))
    assert _FakeMesh.calls == [1, 0, 1, 0]


def test_only_vertical_momentum_gets_standard_deviation_perturbation(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    output = tmp_path / "output.restart"
    _make_source_restart(source)
    monkeypatch.setitem(sys.modules, "snapy", _fake_snapy())

    create_restart(config, source, output, seed=1)

    block = _load_output(output)[0]
    hydro_u = block["hydro_u"]
    hydro_w = block["hydro_w"]
    source_blocks = _load_output(source)
    samples = torch.stack(
        [source_block["hydro_u"][..., 1:-1, 1:-1, :] for source_block in source_blocks],
        dim=0,
    )
    hydro_u_mean = samples.mean(dim=(0, 2, 3))

    torch.testing.assert_close(
        hydro_u[0],
        hydro_u_mean[0][None, None, :].expand_as(hydro_u[0]),
    )
    torch.testing.assert_close(
        hydro_u[2],
        hydro_u_mean[2][None, None, :].expand_as(hydro_u[2]),
    )
    assert not torch.allclose(
        hydro_u[1],
        hydro_u_mean[1][None, None, :].expand_as(hydro_u[1]),
    )

    hydro_w_samples = torch.stack(
        [source_block["hydro_w"][..., 1:-1, 1:-1, :] for source_block in source_blocks],
        dim=0,
    )
    hydro_w_mean = hydro_w_samples.mean(dim=(0, 2, 3))
    torch.testing.assert_close(
        hydro_w,
        hydro_w_mean[:, None, None, :].expand_as(hydro_w),
    )


def test_seed_makes_vertical_momentum_deterministic(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    first = tmp_path / "first.restart"
    second = tmp_path / "second.restart"
    third = tmp_path / "third.restart"
    _make_source_restart(source)
    monkeypatch.setitem(sys.modules, "snapy", _fake_snapy())

    create_restart(config, source, first, seed=7)
    create_restart(config, source, second, seed=7)
    create_restart(config, source, third, seed=8)

    first_u = _load_output(first)[0]["hydro_u"][1]
    second_u = _load_output(second)[0]["hydro_u"][1]
    third_u = _load_output(third)[0]["hydro_u"][1]
    torch.testing.assert_close(first_u, second_u)
    assert not torch.allclose(first_u, third_u)


def test_rejects_vertical_dimension_mismatch(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    output = tmp_path / "output.restart"
    _make_source_restart(source)
    monkeypatch.setitem(sys.modules, "snapy", _fake_snapy())
    _FakeCoord.nx1.value = 3
    try:
        with pytest.raises(ValueError, match="vertical dimension"):
            create_restart(config, source, output)
    finally:
        _FakeCoord.nx1.value = 2


def test_supports_cartesian_target_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    output = tmp_path / "output.restart"
    _make_source_restart(source)
    monkeypatch.setitem(sys.modules, "snapy", _fake_snapy())
    _FakeLayout.type.value = "slab"
    _FakeLayout.px.value = 2
    try:
        create_restart(config, source, output)

        blocks = _load_output(output)
        assert len(blocks) == 2
        for block in blocks:
            assert tuple(block["hydro_u"].shape) == (3, 4, 5, 4)
    finally:
        _FakeLayout.type.value = "cubed-sphere"
        _FakeLayout.px.value = 1


def test_real_jupiter_gcm_config_has_six_ghosted_face_blocks() -> None:
    snapy = pytest.importorskip("snapy")
    config = Path(
        "/home/chengcli/data/2026.JupiterCRM/"
        "jup_gcm_H2O-NH3_F100/jup_gcm_H2O-NH3_F100.yaml"
    )
    if not config.exists():
        pytest.skip("Jupiter GCM reference config is not available")

    options = snapy.MeshOptions.from_yaml(str(config))
    block = options.block()
    coord = block.coord()

    assert block.layout().type() == "cubed-sphere"
    assert _block_count_from_layout(block.layout()) == 6
    assert (coord.nx3() + 2 * coord.nghost()) == 150
    assert (coord.nx2() + 2 * coord.nghost()) == 150
    assert (coord.nx1() + 2 * coord.nghost()) == 106
