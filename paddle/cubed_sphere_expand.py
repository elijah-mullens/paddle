from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Sequence

import torch

from .cubed_sphere_shrink import (
    FACE_COUNT,
    RestartBundleEntry,
    _entries_by_rank,
    _load_entry,
    _publish_bundle,
    _save_tensors,
    _validate_paths,
    _validate_worker_results,
    _write_bundle,
    _zorder_coords,
    read_restart_bundle_index,
)


def _expand_spatial_tensor(
    face_tensor: torch.Tensor, n: int, nghost: int
) -> list[torch.Tensor]:
    interior_x3 = face_tensor.shape[-3] - 2 * nghost
    interior_x2 = face_tensor.shape[-2] - 2 * nghost
    if interior_x3 <= 0 or interior_x2 <= 0:
        raise ValueError(
            f"spatial tensor shape {tuple(face_tensor.shape)} is too small for "
            f"nghost={nghost}"
        )
    if interior_x3 % n or interior_x2 % n:
        raise ValueError(
            f"spatial tensor shape {tuple(face_tensor.shape)} interior dimensions "
            f"must be divisible by N={n}"
        )

    tile_x3 = interior_x3 // n
    tile_x2 = interior_x2 // n
    return [
        face_tensor[
            ...,
            y * tile_x3 : (y + 1) * tile_x3 + 2 * nghost,
            x * tile_x2 : (x + 1) * tile_x2 + 2 * nghost,
            :,
        ].contiguous()
        for x, y in _zorder_coords(n)
    ]


def _expand_face(
    bundle_path: str,
    entry: RestartBundleEntry,
    face: int,
    n: int,
    nghost: int,
    part_paths: Sequence[str],
) -> tuple[int, dict[str, str], dict[str, tuple[tuple[int, ...], str]]]:
    torch.set_num_threads(1)
    source = _load_entry(bundle_path, entry)
    outputs = [dict() for _ in range(n * n)]
    metadata: dict[str, str] = {}
    schema: dict[str, tuple[tuple[int, ...], str]] = {}

    for name in sorted(source):
        tensor = source[name]
        if tensor.ndim >= 3:
            for output, tile in zip(outputs, _expand_spatial_tensor(tensor, n, nghost)):
                output[name] = tile
        else:
            for output in outputs:
                output[name] = tensor
            metadata[name] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        schema[name] = (tuple(tensor.shape), str(tensor.dtype))

    for output, part_path in zip(outputs, part_paths):
        _save_tensors(output, Path(part_path))
    return face, metadata, schema


def expand_cubed_sphere_restart(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    n: int,
    nghost: int = 3,
) -> Path:
    source, output = _validate_paths(input_path, output_path, nghost)
    if n <= 0:
        raise ValueError("N must be positive")

    entries = read_restart_bundle_index(source)
    entries_by_rank = _entries_by_rank(entries, FACE_COUNT)
    ordered_entries = [entries_by_rank[rank] for rank in range(FACE_COUNT)]
    block_count = FACE_COUNT * n * n

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.expand-", dir=output.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        part_paths = [temporary / f"block{rank}.part" for rank in range(block_count)]
        with ProcessPoolExecutor(max_workers=FACE_COUNT) as executor:
            futures = [
                executor.submit(
                    _expand_face,
                    str(source),
                    ordered_entries[face],
                    face,
                    n,
                    nghost,
                    [
                        str(path)
                        for path in part_paths[face * n * n : (face + 1) * n * n]
                    ],
                )
                for face in range(FACE_COUNT)
            ]
            results = sorted(
                (future.result() for future in futures), key=lambda item: item[0]
            )

        _validate_worker_results(results)
        temporary_bundle = temporary / output.name
        _write_bundle(temporary_bundle, part_paths)
        _publish_bundle(temporary_bundle, output)

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle cs-expand",
        description=(
            "Expand a six-block Snapy cubed-sphere restart into 6*N^2 blocks "
            "using six worker processes."
        ),
    )
    parser.add_argument("input", help="Input six-block Snapy restart bundle")
    parser.add_argument("output", help="New 6*N^2-block Snapy restart bundle")
    parser.add_argument("--n", type=int, required=True, help="Blocks per face edge")
    parser.add_argument(
        "--nghost",
        type=int,
        default=3,
        help="Number of ghost cells on each tile edge (default: 3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expand_cubed_sphere_restart(args.input, args.output, n=args.n, nghost=args.nghost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
