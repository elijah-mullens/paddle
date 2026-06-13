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
from paddle.restart_resize import build_parser, resize_restart, resize_spatial_tensor


def test_refine_doubles_only_horizontal_interior() -> None:
    source = torch.ones((2, 6, 8, 3), dtype=torch.float64)

    result = resize_spatial_tensor(source, mode="refine", nghost=1)

    assert result.shape == (2, 10, 14, 3)
    torch.testing.assert_close(result, torch.ones_like(result))


def test_coarsen_averages_two_by_two_interior() -> None:
    source = torch.zeros((1, 6, 6, 1), dtype=torch.float64)
    source[..., 1:-1, 1:-1, :] = torch.arange(16, dtype=torch.float64).reshape(
        1, 4, 4, 1
    )

    result = resize_spatial_tensor(source, mode="coarsen", nghost=1)

    expected = torch.tensor([2.5, 4.5, 10.5, 12.5], dtype=torch.float64).reshape(
        1, 2, 2, 1
    )
    torch.testing.assert_close(result[..., 1:-1, 1:-1, :], expected)


def test_coarsen_rejects_odd_interior() -> None:
    with pytest.raises(ValueError, match="even horizontal"):
        resize_spatial_tensor(torch.ones((1, 7, 6, 1)), mode="coarsen", nghost=1)


@pytest.mark.parametrize(
    ("mode", "expected_shape"),
    [("refine", (1, 8, 12, 1)), ("coarsen", (1, 2, 3, 1))],
)
def test_resize_without_ghost_zones(mode, expected_shape) -> None:
    source = torch.ones((1, 4, 6, 1), dtype=torch.float64)

    result = resize_spatial_tensor(source, mode=mode, nghost=0)

    assert result.shape == expected_shape
    torch.testing.assert_close(result, torch.ones_like(result))


def test_resize_parser_uses_entry_point_name(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["paddle-refine"])

    assert build_parser("refine").prog == "paddle-refine"


class _FakeValue:
    def __init__(self, value):
        self.value = value

    def __call__(self, value=None):
        if value is not None:
            self.value = value
            return self
        return self.value


class _FakeLayout:
    type = _FakeValue("slab")
    px = _FakeValue(1)
    py = _FakeValue(1)
    pz = _FakeValue(1)


class _FakeCoord:
    nghost = _FakeValue(1)
    nx2 = _FakeValue(4)
    nx3 = _FakeValue(4)


class _FakeBlockOptions:
    def layout(self):
        return _FakeLayout()

    def coord(self):
        return _FakeCoord()


class _FakeMeshOptions:
    local_cells: tuple[int, int] | None = None
    block_count: int | None = None

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
        for block in variables:
            tensor = next(iter(block.values()))
            tensor[:, 0, :, :] = 99.0


def test_resize_restart_uses_configured_snapy_mesh(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("geometry: {}\n")
    source = tmp_path / "source.restart"
    output = tmp_path / "output.restart"
    part = tmp_path / "block0.part"
    _save_tensors(
        {
            "hydro_u": torch.ones((1, 6, 6, 1), dtype=torch.float64),
            "last_cycle": torch.tensor([12]),
        },
        part,
    )
    _write_bundle(source, [part])

    fake_snapy = SimpleNamespace(
        MeshOptions=_FakeMeshOptions,
        Mesh=_FakeMesh,
        kPrimitive=0,
        kConserved=1,
        kScalar=2,
    )
    _FakeMesh.calls = []
    monkeypatch.setitem(sys.modules, "snapy", fake_snapy)

    resize_restart(config, source, output, mode="refine")

    entry = read_restart_bundle_index(output)[0]
    tensors = _load_entry(output, entry)
    assert tensors["hydro_u"].shape == (1, 10, 10, 1)
    assert torch.all(tensors["hydro_u"][:, 0] == 99.0)
    torch.testing.assert_close(tensors["last_cycle"], torch.tensor([12]))
    assert _FakeMesh.calls == [1]
