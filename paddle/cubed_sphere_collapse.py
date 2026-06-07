from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Sequence

import torch


RESTART_BUNDLE_MAGIC = "SNAPY_RESTART_BUNDLE_V1"
SOURCE_BLOCK_COUNT = 24
TARGET_BLOCK_COUNT = 6
BLOCKS_PER_FACE = 4
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
        raise ValueError(f"{path}: restart bundle payload size does not match its index")
    return indexed_entries


def _entries_by_rank(entries: Sequence[RestartBundleEntry]) -> dict[int, RestartBundleEntry]:
    result: dict[int, RestartBundleEntry] = {}
    for entry in entries:
        match = _BLOCK_RANK_RE.search(entry.name)
        if match is None:
            raise ValueError(f"restart bundle entry has no block rank: {entry.name}")
        rank = int(match.group(1))
        if rank in result:
            raise ValueError(f"restart bundle contains duplicate block rank {rank}")
        result[rank] = entry

    expected = set(range(SOURCE_BLOCK_COUNT))
    if set(result) != expected:
        raise ValueError(
            "expected source block ranks 0 through 23, got "
            f"{sorted(result)}"
        )
    return result


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


def _collapse_spatial_tensor(tiles: Sequence[torch.Tensor], nghost: int) -> torch.Tensor:
    tile_x3 = tiles[0].shape[-3]
    tile_x2 = tiles[0].shape[-2]
    if tile_x3 <= 2 * nghost or tile_x2 <= 2 * nghost:
        raise ValueError(
            f"spatial tensor shape {tuple(tiles[0].shape)} is too small for "
            f"nghost={nghost}"
        )

    bottom = torch.cat(
        (
            tiles[0][..., :-nghost, :-nghost, :],
            tiles[1][..., :-nghost, nghost:, :],
        ),
        dim=-2,
    )
    top = torch.cat(
        (
            tiles[2][..., nghost:, :-nghost, :],
            tiles[3][..., nghost:, nghost:, :],
        ),
        dim=-2,
    )
    return torch.cat((bottom, top), dim=-3).contiguous()


def _collapse_face(
    bundle_path: str,
    entries: Sequence[RestartBundleEntry],
    face: int,
    nghost: int,
    part_path: str,
) -> tuple[int, dict[str, str], dict[str, tuple[tuple[int, ...], str]]]:
    torch.set_num_threads(1)
    blocks = [
        _load_entry(bundle_path, entries[face * BLOCKS_PER_FACE + i])
        for i in range(BLOCKS_PER_FACE)
    ]
    names = _validate_schema(blocks)

    output: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    schema: dict[str, tuple[tuple[int, ...], str]] = {}
    for name in names:
        source_tensors = [block[name] for block in blocks]
        if source_tensors[0].ndim >= 3:
            output[name] = _collapse_spatial_tensor(source_tensors, nghost)
        else:
            if not all(
                torch.equal(source_tensors[0], tensor)
                for tensor in source_tensors[1:]
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


def collapse_cubed_sphere_restart(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    nghost: int = 3,
) -> Path:
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

    entries_by_rank = _entries_by_rank(read_restart_bundle_index(source))
    ordered_entries = [entries_by_rank[rank] for rank in range(SOURCE_BLOCK_COUNT)]

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.collapse-", dir=output.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        part_paths = [
            temporary / f"block{face}.part" for face in range(TARGET_BLOCK_COUNT)
        ]
        with ProcessPoolExecutor(max_workers=TARGET_BLOCK_COUNT) as executor:
            futures = [
                executor.submit(
                    _collapse_face,
                    str(source),
                    ordered_entries,
                    face,
                    nghost,
                    str(part_paths[face]),
                )
                for face in range(TARGET_BLOCK_COUNT)
            ]
            results = sorted(
                (future.result() for future in futures), key=lambda item: item[0]
            )

        _validate_worker_results(results)
        temporary_bundle = temporary / output.name
        _write_bundle(temporary_bundle, part_paths)
        try:
            os.link(temporary_bundle, output)
        except FileExistsError:
            raise FileExistsError(f"output restart already exists: {output}") from None

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle cs-collapse",
        description=(
            "Collapse a Snapy nb2=nb3=2 cubed-sphere restart into six "
            "nb2=nb3=1 face blocks using six worker processes."
        ),
    )
    parser.add_argument("input", help="Input 24-block Snapy restart bundle")
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
    collapse_cubed_sphere_restart(args.input, args.output, nghost=args.nghost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
