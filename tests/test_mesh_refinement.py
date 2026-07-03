from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import paddle.mesh_refinement as refinement
from paddle.mesh_refinement import (
    coarsen_meshblock,
    coarsen_spatial,
    conservative_coarsen,
    conservative_refine,
    refine_boundary_state_to_match,
    refine_meshblock,
    refine_spatial,
    refined_global_horizontal_cells,
)


def test_refine_and_coarsen_spatial_preserve_shape_and_averages() -> None:
    state = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)

    refined = refine_spatial(state, method="area")

    assert refined.shape == (2, 6, 8, 2)
    assert torch.equal(coarsen_spatial(refined), state)


def test_spatial_refinement_preserves_singleton_dimensions() -> None:
    state = torch.arange(2 * 1 * 4 * 3, dtype=torch.float64).reshape(2, 1, 4, 3)

    refined = refine_spatial(state, method="area")

    assert refined.shape == (2, 1, 8, 3)
    assert torch.equal(coarsen_spatial(refined), state)


def test_conservative_refinement_preserves_singleton_dimensions() -> None:
    state = torch.randn((2, 1, 8, 3), dtype=torch.float64)

    refined = conservative_refine(state, nghost=1)
    coarsened = conservative_coarsen(refined, nghost=1)

    assert refined.shape == (2, 1, 14, 3)
    assert torch.allclose(coarsened[..., 1:-1, :], state[..., 1:-1, :])


@pytest.mark.parametrize("nghost", [0, 1])
def test_conservative_refinement_round_trip(nghost: int) -> None:
    shape = (2, 4 + 2 * nghost, 6 + 2 * nghost, 3)
    state = torch.randn(shape, dtype=torch.float64)

    refined = conservative_refine(state, nghost)
    coarsened = conservative_coarsen(refined, nghost)

    expected = state.clone()
    if nghost:
        expected[..., :nghost, :, :] = 0
        expected[..., -nghost:, :, :] = 0
        expected[..., :, :nghost, :] = 0
        expected[..., :, -nghost:, :] = 0
    assert torch.allclose(coarsened, expected)


def test_refine_boundary_state_to_match_reaches_multiple_levels() -> None:
    boundary = torch.arange(2 * 8 * 8 * 3, dtype=torch.float64).reshape(2, 8, 8, 3)

    refined = refine_boundary_state_to_match(boundary, (2, 20, 20, 3), nghost=2)

    assert refined.shape == (2, 20, 20, 3)
    assert torch.isfinite(refined).all()


def test_refine_boundary_state_rejects_unreachable_target() -> None:
    boundary = torch.zeros((2, 8, 8, 3))

    with pytest.raises(ValueError, match="exceeds target"):
        refine_boundary_state_to_match(boundary, (2, 16, 16, 3), nghost=2)


def test_refined_global_horizontal_cells_uses_layout_counts() -> None:
    assert refined_global_horizontal_cells(28, 46, px=2, py=1) == (112, 92)
    assert refined_global_horizontal_cells(1, 46, px=1, py=1) == (1, 92)


class FakeCoord:
    def __init__(self, nx2: int, nx3: int, nghost: int = 2):
        self._nx2 = nx2
        self._nx3 = nx3
        self._nghost = nghost

    def nx2(self, value=None):
        if value is not None:
            self._nx2 = value
        return self._nx2

    def nx3(self, value=None):
        if value is not None:
            self._nx3 = value
        return self._nx3

    def nghost(self):
        return self._nghost


class FakeBlock:
    def __init__(self, options):
        self.options = options
        self._outputs = [
            SimpleNamespace(file_number=3, next_time=4.5),
            SimpleNamespace(file_number=7, next_time=8.5),
        ]

    def get_outputs(self):
        return self._outputs


class FakeOptions:
    def __init__(self, coord):
        self._coord = coord

    def coord(self):
        return self._coord


def test_meshblock_rebuild_changes_cells_and_preserves_output_schedule(
    monkeypatch,
) -> None:
    monkeypatch.setattr(refinement, "MeshBlock", FakeBlock)
    options = FakeOptions(FakeCoord(8, 1))
    block = FakeBlock(options)

    refined = refine_meshblock(block)
    assert refined.options.coord().nx2() == 16
    assert refined.options.coord().nx3() == 1
    assert [(out.file_number, out.next_time) for out in refined.get_outputs()] == [
        (3, 4.5),
        (7, 8.5),
    ]

    coarsened = coarsen_meshblock(refined)
    assert coarsened.options.coord().nx2() == 8
