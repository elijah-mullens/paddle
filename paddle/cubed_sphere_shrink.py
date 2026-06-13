from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Sequence

import torch


RESTART_BUNDLE_MAGIC = "SNAPY_RESTART_BUNDLE_V1"
FACE_COUNT = 6
_BLOCK_RANK_RE = re.compile(r"\.block(\d+)\.")


@dataclass(frozen=True)
class RestartBundleEntry:
    name: str
    size: int
    offset: int


class _TensorModule(torch.nn.Module):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        super().__init__()
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor)


def read_restart_bundle_index(path: str | os.PathLike[str]) -> list[RestartBundleEntry]:
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.readline().decode("utf-8", errors="strict").rstrip("\n")
        if magic != RESTART_BUNDLE_MAGIC:
            raise ValueError(f"{path}: not a {RESTART_BUNDLE_MAGIC} restart bundle")

        count_line = stream.readline().decode("utf-8", errors="strict").strip()
        if not count_line:
            raise ValueError(f"{path}: restart bundle is missing its entry count")
        entry_count = int(count_line)

        entries: list[RestartBundleEntry] = []
        for _ in range(entry_count):
            line = stream.readline().decode("utf-8", errors="strict").rstrip("\n")
            try:
                name, size_text = line.split("\t", 1)
                size = int(size_text)
            except ValueError as exc:
                raise ValueError(f"{path}: invalid restart bundle index line") from exc
            if size < 0:
                raise ValueError(f"{path}: restart bundle entry has a negative size")
            entries.append(RestartBundleEntry(name=name, size=size, offset=0))

        if stream.readline() != b"\n":
            raise ValueError(f"{path}: restart bundle header is missing its terminator")

        payload_offset = stream.tell()

    indexed_entries = []
    running_offset = payload_offset
    for entry in entries:
        indexed_entries.append(
            RestartBundleEntry(entry.name, entry.size, running_offset)
        )
        running_offset += entry.size

    if path.stat().st_size != running_offset:
        raise ValueError(
            f"{path}: restart bundle payload size does not match its index"
        )
    return indexed_entries


def _entries_by_rank(
    entries: Sequence[RestartBundleEntry], expected_count: int
) -> dict[int, RestartBundleEntry]:
    result: dict[int, RestartBundleEntry] = {}
    for entry in entries:
        match = _BLOCK_RANK_RE.search(entry.name)
        if match is None:
            raise ValueError(f"restart bundle entry has no block rank: {entry.name}")
        rank = int(match.group(1))
        if rank in result:
            raise ValueError(f"restart bundle contains duplicate block rank {rank}")
        result[rank] = entry

    expected = set(range(expected_count))
    if set(result) != expected:
        raise ValueError(
            f"expected source block ranks 0 through {expected_count - 1}, got "
            f"{sorted(result)}"
        )
    return result


def _infer_subdivision(block_count: int) -> int:
    if block_count % FACE_COUNT:
        raise ValueError(
            f"cubed-sphere restart block count must equal 6*N^2, got {block_count}"
        )
    n = math.isqrt(block_count // FACE_COUNT)
    if n <= 0 or FACE_COUNT * n * n != block_count:
        raise ValueError(
            f"cubed-sphere restart block count must equal 6*N^2, got {block_count}"
        )
    return n


def _compact1by1(value: int) -> int:
    value &= 0x55555555
    value = (value | (value >> 1)) & 0x33333333
    value = (value | (value >> 2)) & 0x0F0F0F0F
    value = (value | (value >> 4)) & 0x00FF00FF
    value = (value | (value >> 8)) & 0x0000FFFF
    return value


def _zorder_coords(n: int) -> list[tuple[int, int]]:
    if n <= 0:
        raise ValueError("N must be positive")
    coords: list[tuple[int, int]] = []
    code = 0
    while len(coords) < n * n:
        x = _compact1by1(code)
        y = _compact1by1(code >> 1)
        if x < n and y < n:
            coords.append((x, y))
        code += 1
    return coords


def _load_entry(
    bundle_path: str | os.PathLike[str], entry: RestartBundleEntry
) -> dict[str, torch.Tensor]:
    with tempfile.NamedTemporaryFile(suffix=".part") as part:
        with Path(bundle_path).open("rb") as source:
            source.seek(entry.offset)
            remaining = entry.size
            while remaining:
                chunk = source.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"truncated restart bundle entry: {entry.name}")
                part.write(chunk)
                remaining -= len(chunk)
        part.flush()
        module = torch.jit.load(part.name, map_location="cpu")
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in module.named_buffers(recurse=True)
        }
        tensors.update(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in module.named_parameters(recurse=True)
            }
        )
    return tensors


def _save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    module = torch.jit.script(_TensorModule(tensors))
    module.save(str(path))


def _validate_schema(blocks: Sequence[dict[str, torch.Tensor]]) -> list[str]:
    names = sorted(blocks[0])
    expected_names = set(names)
    for block in blocks[1:]:
        if set(block) != expected_names:
            raise ValueError("source blocks have inconsistent tensor names")
        for name in names:
            if block[name].shape != blocks[0][name].shape:
                raise ValueError(f"source blocks have inconsistent shape for {name}")
            if block[name].dtype != blocks[0][name].dtype:
                raise ValueError(f"source blocks have inconsistent dtype for {name}")
    return names


def _shrink_spatial_tensor(
    tiles: Sequence[torch.Tensor], n: int, nghost: int
) -> torch.Tensor:
    tile_x3 = tiles[0].shape[-3]
    tile_x2 = tiles[0].shape[-2]
    if tile_x3 <= 2 * nghost or tile_x2 <= 2 * nghost:
        raise ValueError(
            f"spatial tensor shape {tuple(tiles[0].shape)} is too small for "
            f"nghost={nghost}"
        )

    tiles_by_coord = dict(zip(_zorder_coords(n), tiles))
    rows = []
    for y in range(n):
        row = []
        for x in range(n):
            x3_start = 0 if y == 0 else nghost
            x3_stop = None if y == n - 1 else -nghost
            x2_start = 0 if x == 0 else nghost
            x2_stop = None if x == n - 1 else -nghost
            row.append(tiles_by_coord[x, y][..., x3_start:x3_stop, x2_start:x2_stop, :])
        rows.append(torch.cat(row, dim=-2))
    return torch.cat(rows, dim=-3).contiguous()


def _shrink_face(
    bundle_path: str,
    entries: Sequence[RestartBundleEntry],
    face: int,
    n: int,
    nghost: int,
    part_path: str,
) -> tuple[int, dict[str, str], dict[str, tuple[tuple[int, ...], str]]]:
    torch.set_num_threads(1)
    blocks_per_face = n * n
    blocks = [
        _load_entry(bundle_path, entries[face * blocks_per_face + i])
        for i in range(blocks_per_face)
    ]
    names = _validate_schema(blocks)

    output: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    schema: dict[str, tuple[tuple[int, ...], str]] = {}
    for name in names:
        source_tensors = [block[name] for block in blocks]
        if source_tensors[0].ndim >= 3:
            output[name] = _shrink_spatial_tensor(source_tensors, n, nghost)
        else:
            if not all(
                torch.equal(source_tensors[0], tensor) for tensor in source_tensors[1:]
            ):
                raise ValueError(f"metadata tensor {name} differs within face {face}")
            output[name] = source_tensors[0]
            metadata[name] = hashlib.sha256(
                source_tensors[0].numpy().tobytes()
            ).hexdigest()
        schema[name] = (tuple(source_tensors[0].shape), str(source_tensors[0].dtype))

    _save_tensors(output, Path(part_path))
    return face, metadata, schema


def _validate_worker_results(
    results: Sequence[
        tuple[int, dict[str, str], dict[str, tuple[tuple[int, ...], str]]]
    ],
) -> None:
    reference_face, reference_metadata, reference_schema = results[0]
    for face, metadata, schema in results[1:]:
        if schema != reference_schema:
            raise ValueError(
                f"source tensor schema differs between faces {reference_face} and {face}"
            )
        if set(metadata) != set(reference_metadata):
            raise ValueError(f"metadata tensor names differ on face {face}")
        for name, fingerprint in metadata.items():
            if reference_metadata[name] != fingerprint:
                raise ValueError(
                    f"metadata tensor {name} differs between faces "
                    f"{reference_face} and {face}"
                )


def _write_bundle(path: Path, part_paths: Sequence[Path]) -> None:
    entries = [
        (f"{path.stem}.block{rank}.final.part", part_path.stat().st_size)
        for rank, part_path in enumerate(part_paths)
    ]
    with path.open("xb") as output:
        output.write(f"{RESTART_BUNDLE_MAGIC}\n".encode())
        output.write(f"{len(entries)}\n".encode())
        for name, size in entries:
            output.write(f"{name}\t{size}\n".encode())
        output.write(b"\n")
        for part_path in part_paths:
            with part_path.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def _validate_paths(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    nghost: int,
) -> tuple[Path, Path]:
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("input and output restart paths must differ")
    if not source.is_file():
        raise FileNotFoundError(f"input restart does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output restart already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    if nghost <= 0:
        raise ValueError("nghost must be positive")
    return source, output


def _publish_bundle(temporary_bundle: Path, output: Path) -> None:
    try:
        os.link(temporary_bundle, output)
    except FileExistsError:
        raise FileExistsError(f"output restart already exists: {output}") from None


def shrink_cubed_sphere_restart(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    nghost: int = 3,
) -> Path:
    source, output = _validate_paths(input_path, output_path, nghost)
    entries = read_restart_bundle_index(source)
    n = _infer_subdivision(len(entries))
    entries_by_rank = _entries_by_rank(entries, len(entries))
    ordered_entries = [entries_by_rank[rank] for rank in range(len(entries))]

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.shrink-", dir=output.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        part_paths = [temporary / f"block{face}.part" for face in range(FACE_COUNT)]
        with ProcessPoolExecutor(max_workers=FACE_COUNT) as executor:
            futures = [
                executor.submit(
                    _shrink_face,
                    str(source),
                    ordered_entries,
                    face,
                    n,
                    nghost,
                    str(part_paths[face]),
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
        prog="paddle cs-shrink",
        description=(
            "Shrink a Snapy 6*N^2-block cubed-sphere restart into six "
            "face blocks using six worker processes."
        ),
    )
    parser.add_argument("input", help="Input 6*N^2-block Snapy restart bundle")
    parser.add_argument("output", help="New six-block Snapy restart bundle")
    parser.add_argument(
        "--nghost",
        type=int,
        default=3,
        help="Number of ghost cells on each tile edge (default: 3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shrink_cubed_sphere_restart(args.input, args.output, nghost=args.nghost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
