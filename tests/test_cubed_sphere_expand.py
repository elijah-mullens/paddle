from __future__ import annotations

from pathlib import Path

import pytest
import torch

from paddle.cubed_sphere_shrink import (
    _load_entry,
    _save_tensors,
    shrink_cubed_sphere_restart,
    read_restart_bundle_index,
)
from paddle.cubed_sphere_expand import expand_cubed_sphere_restart
from test_cubed_sphere_shrink import _write_bundle


def _make_face_bundle(path: Path, *, size: int = 14) -> None:
    part_dir = path.parent / f"{path.stem}-parts"
    part_dir.mkdir()
    parts = []
    for face in range(6):
        values = torch.arange(1 * size * size * 2, dtype=torch.float64).reshape(
            1, size, size, 2
        )
        part = part_dir / f"block{face}.part"
        _save_tensors(
            {
                "hydro_u": values + face * 10000,
                "last_cycle": torch.tensor([42], dtype=torch.int64),
            },
            part,
        )
        parts.append(part)
    _write_bundle(path, parts)


@pytest.mark.parametrize("n", [2, 3])
def test_expand_then_shrink_round_trip(tmp_path: Path, n: int) -> None:
    source = tmp_path / "source.restart"
    expanded = tmp_path / "expanded.restart"
    shrunk = tmp_path / "shrunk.restart"
    _make_face_bundle(source)

    result = expand_cubed_sphere_restart(source, expanded, n=n, nghost=1)
    shrink_cubed_sphere_restart(expanded, shrunk, nghost=1)

    assert result == expanded.resolve()
    expanded_entries = read_restart_bundle_index(expanded)
    assert len(expanded_entries) == 6 * n * n
    source_entries = read_restart_bundle_index(source)
    shrunk_entries = read_restart_bundle_index(shrunk)
    for source_entry, shrunk_entry in zip(source_entries, shrunk_entries):
        source_tensors = _load_entry(source, source_entry)
        shrunk_tensors = _load_entry(shrunk, shrunk_entry)
        assert source_tensors.keys() == shrunk_tensors.keys()
        for name in source_tensors:
            torch.testing.assert_close(source_tensors[name], shrunk_tensors[name])


def test_expand_uses_snapy_zorder_for_output_ranks(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    expanded = tmp_path / "expanded.restart"
    _make_face_bundle(source, size=8)

    expand_cubed_sphere_restart(source, expanded, n=3, nghost=1)

    entries = read_restart_bundle_index(expanded)
    rank4 = _load_entry(expanded, entries[4])["hydro_u"]
    face = _load_entry(source, read_restart_bundle_index(source)[0])["hydro_u"]
    torch.testing.assert_close(rank4, face[..., 0:4, 4:8, :])


def test_expand_rejects_non_six_block_input(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "expanded.restart"
    part_dir = tmp_path / "five-parts"
    part_dir.mkdir()
    parts = []
    for rank in range(5):
        part = part_dir / f"block{rank}.part"
        _save_tensors({"last_cycle": torch.tensor([rank])}, part)
        parts.append(part)
    _write_bundle(source, parts)

    with pytest.raises(ValueError, match="0 through 5"):
        expand_cubed_sphere_restart(source, output, n=2, nghost=1)

    assert not output.exists()


def test_expand_rejects_indivisible_spatial_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "expanded.restart"
    _make_face_bundle(source, size=9)

    with pytest.raises(ValueError, match="must be divisible"):
        expand_cubed_sphere_restart(source, output, n=2, nghost=1)

    assert not output.exists()


def test_expand_rejects_nonpositive_n(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "expanded.restart"
    _make_face_bundle(source)

    with pytest.raises(ValueError, match="N must be positive"):
        expand_cubed_sphere_restart(source, output, n=0, nghost=1)

    assert not output.exists()
