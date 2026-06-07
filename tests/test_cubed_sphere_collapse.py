from __future__ import annotations

from pathlib import Path

import pytest
import torch

from paddle.cubed_sphere_collapse import (
    RESTART_BUNDLE_MAGIC,
    _load_entry,
    _save_tensors,
    collapse_cubed_sphere_restart,
    read_restart_bundle_index,
)


def _write_bundle(path: Path, parts: list[Path]) -> None:
    with path.open("xb") as output:
        output.write(f"{RESTART_BUNDLE_MAGIC}\n".encode())
        output.write(f"{len(parts)}\n".encode())
        for rank, part in enumerate(parts):
            output.write(
                f"source.block{rank}.final.part\t{part.stat().st_size}\n".encode()
            )
        output.write(b"\n")
        for part in parts:
            output.write(part.read_bytes())


def _make_source_bundle(path: Path, *, inconsistent_metadata: bool = False) -> None:
    part_dir = path.parent / "parts"
    part_dir.mkdir()
    parts = []
    for rank in range(24):
        face = rank // 4
        local_rank = rank % 4
        values = torch.arange(1 * 6 * 6 * 2, dtype=torch.float64).reshape(
            1, 6, 6, 2
        )
        values = values + face * 10000 + local_rank * 1000
        metadata_value = rank if inconsistent_metadata and rank == 23 else 42
        part = part_dir / f"block{rank}.part"
        _save_tensors(
            {
                "hydro_u": values,
                "last_cycle": torch.tensor([metadata_value], dtype=torch.int64),
                "last_time": torch.tensor([12.5], dtype=torch.float64),
            },
            part,
        )
        parts.append(part)
    _write_bundle(path, parts)


def _expected_face(face: int) -> torch.Tensor:
    tiles = []
    for local_rank in range(4):
        values = torch.arange(1 * 6 * 6 * 2, dtype=torch.float64).reshape(
            1, 6, 6, 2
        )
        tiles.append(values + face * 10000 + local_rank * 1000)
    bottom = torch.cat(
        (tiles[0][..., :-1, :-1, :], tiles[1][..., :-1, 1:, :]), dim=-2
    )
    top = torch.cat(
        (tiles[2][..., 1:, :-1, :], tiles[3][..., 1:, 1:, :]), dim=-2
    )
    return torch.cat((bottom, top), dim=-3)


def test_collapse_cubed_sphere_restart_stitches_six_faces(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "collapsed.restart"
    _make_source_bundle(source)
    source_bytes = source.read_bytes()

    result = collapse_cubed_sphere_restart(source, output, nghost=1)

    assert result == output.resolve()
    assert source.read_bytes() == source_bytes
    entries = read_restart_bundle_index(output)
    assert len(entries) == 6
    for face, entry in enumerate(entries):
        assert f".block{face}." in entry.name
        tensors = _load_entry(output, entry)
        assert tuple(tensors["hydro_u"].shape) == (1, 10, 10, 2)
        torch.testing.assert_close(tensors["hydro_u"], _expected_face(face))
        torch.testing.assert_close(tensors["last_cycle"], torch.tensor([42]))
        torch.testing.assert_close(tensors["last_time"], torch.tensor([12.5]))


def test_collapse_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "collapsed.restart"
    _make_source_bundle(source)
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        collapse_cubed_sphere_restart(source, output, nghost=1)

    assert output.read_bytes() == b"existing"


def test_collapse_refuses_input_as_output(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    _make_source_bundle(source)

    with pytest.raises(ValueError, match="must differ"):
        collapse_cubed_sphere_restart(source, source, nghost=1)


def test_collapse_rejects_metadata_that_differs_between_faces(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "collapsed.restart"
    _make_source_bundle(source, inconsistent_metadata=True)

    with pytest.raises(ValueError, match="metadata tensor last_cycle differs"):
        collapse_cubed_sphere_restart(source, output, nghost=1)

    assert not output.exists()
