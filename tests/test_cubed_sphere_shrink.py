from __future__ import annotations

from pathlib import Path

import pytest
import torch

from paddle.cubed_sphere_shrink import (
    RESTART_BUNDLE_MAGIC,
    _shrink_spatial_tensor,
    _load_entry,
    _save_tensors,
    _zorder_coords,
    shrink_cubed_sphere_restart,
    read_restart_bundle_index,
)
from paddle.cubed_sphere_expand import _expand_spatial_tensor


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


def _make_source_bundle(
    path: Path, *, n: int = 2, inconsistent_metadata: bool = False
) -> None:
    part_dir = path.parent / f"{path.stem}-parts"
    part_dir.mkdir()
    parts = []
    face_values = torch.arange(
        1 * (4 * n + 2) * (4 * n + 2) * 2, dtype=torch.float64
    ).reshape(1, 4 * n + 2, 4 * n + 2, 2)
    for face in range(6):
        tiles = _expand_spatial_tensor(face_values + face * 10000, n, 1)
        for local_rank, values in enumerate(tiles):
            rank = face * n * n + local_rank
            metadata_value = (
                rank if inconsistent_metadata and rank == 6 * n * n - 1 else 42
            )
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


def _expected_face(face: int, n: int) -> torch.Tensor:
    values = torch.arange(
        1 * (4 * n + 2) * (4 * n + 2) * 2, dtype=torch.float64
    ).reshape(1, 4 * n + 2, 4 * n + 2, 2)
    return values + face * 10000


@pytest.mark.parametrize("n", [1, 2, 3])
def test_shrink_cubed_sphere_restart_stitches_six_faces(tmp_path: Path, n: int) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "shrunk.restart"
    _make_source_bundle(source, n=n)
    source_bytes = source.read_bytes()

    result = shrink_cubed_sphere_restart(source, output, nghost=1)

    assert result == output.resolve()
    assert source.read_bytes() == source_bytes
    entries = read_restart_bundle_index(output)
    assert len(entries) == 6
    for face, entry in enumerate(entries):
        assert f".block{face}." in entry.name
        tensors = _load_entry(output, entry)
        assert tuple(tensors["hydro_u"].shape) == (1, 4 * n + 2, 4 * n + 2, 2)
        torch.testing.assert_close(tensors["hydro_u"], _expected_face(face, n))
        torch.testing.assert_close(tensors["last_cycle"], torch.tensor([42]))
        torch.testing.assert_close(tensors["last_time"], torch.tensor([12.5]))


def test_shrink_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "shrunk.restart"
    _make_source_bundle(source)
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        shrink_cubed_sphere_restart(source, output, nghost=1)

    assert output.read_bytes() == b"existing"


def test_shrink_refuses_input_as_output(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    _make_source_bundle(source)

    with pytest.raises(ValueError, match="must differ"):
        shrink_cubed_sphere_restart(source, source, nghost=1)


def test_shrink_rejects_metadata_that_differs_between_faces(tmp_path: Path) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "shrunk.restart"
    _make_source_bundle(source, inconsistent_metadata=True)

    with pytest.raises(ValueError, match="metadata tensor last_cycle differs"):
        shrink_cubed_sphere_restart(source, output, nghost=1)

    assert not output.exists()


def test_zorder_coords_match_snapy_filtered_morton_order() -> None:
    assert _zorder_coords(3) == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (2, 1),
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert _zorder_coords(4)[4:8] == [(2, 0), (3, 0), (2, 1), (3, 1)]


def test_shrink_spatial_tensor_removes_internal_ghosts_in_logical_order() -> None:
    tiles = [
        torch.full((1, 4, 4, 1), float(rank), dtype=torch.float64) for rank in range(9)
    ]

    result = _shrink_spatial_tensor(tiles, n=3, nghost=1)

    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 4, 4, 4],
            [0, 0, 0, 1, 1, 4, 4, 4],
            [0, 0, 0, 1, 1, 4, 4, 4],
            [2, 2, 2, 3, 3, 5, 5, 5],
            [2, 2, 2, 3, 3, 5, 5, 5],
            [6, 6, 6, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 8, 8, 8],
            [6, 6, 6, 7, 7, 8, 8, 8],
        ],
        dtype=torch.float64,
    ).reshape(1, 8, 8, 1)
    torch.testing.assert_close(result, expected)


def test_shrink_rejects_block_count_that_is_not_six_times_a_square(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.restart"
    output = tmp_path / "shrunk.restart"
    part_dir = tmp_path / "invalid-parts"
    part_dir.mkdir()
    parts = []
    for rank in range(12):
        part = part_dir / f"block{rank}.part"
        _save_tensors({"last_cycle": torch.tensor([rank])}, part)
        parts.append(part)
    _write_bundle(source, parts)

    with pytest.raises(ValueError, match=r"6\*N\^2"):
        shrink_cubed_sphere_restart(source, output, nghost=1)

    assert not output.exists()
