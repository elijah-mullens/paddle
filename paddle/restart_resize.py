from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from .cubed_sphere_shrink import (
    _entries_by_rank,
    _load_entry,
    _publish_bundle,
    _save_tensors,
    _write_bundle,
    read_restart_bundle_index,
)


ResizeMode = Literal["refine", "coarsen"]


def _resize_plane(
    tensor: torch.Tensor, size: tuple[int, int], mode: str
) -> torch.Tensor:
    height, width = tensor.shape[-3:-1]
    leading = tensor.shape[:-3]
    x1 = tensor.shape[-1]
    planes = tensor.movedim(-1, -3).reshape(-1, 1, height, width)
    resized = F.interpolate(
        planes,
        size=size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return resized.reshape(*leading, x1, *size).movedim(-3, -1).contiguous()


def resize_spatial_tensor(
    tensor: torch.Tensor, *, mode: ResizeMode, nghost: int
) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError("spatial tensors must have at least three dimensions")
    if not tensor.is_floating_point():
        raise TypeError(f"spatial tensor must be floating point, got {tensor.dtype}")

    source_x3, source_x2 = tensor.shape[-3:-1]
    interior_x3 = source_x3 - 2 * nghost
    interior_x2 = source_x2 - 2 * nghost
    if interior_x3 <= 0 or interior_x2 <= 0:
        raise ValueError(
            f"spatial tensor shape {tuple(tensor.shape)} is too small for "
            f"nghost={nghost}"
        )

    if mode == "refine":
        output_interior = (2 * interior_x3, 2 * interior_x2)
        interpolation_mode = "bilinear"
    else:
        if interior_x3 % 2 or interior_x2 % 2:
            raise ValueError(
                "coarsening requires even horizontal interior dimensions, got "
                f"{interior_x3}x{interior_x2}"
            )
        output_interior = (interior_x3 // 2, interior_x2 // 2)
        interpolation_mode = "area"

    if nghost == 0:
        return _resize_plane(tensor, output_interior, interpolation_mode)

    output_size = (
        output_interior[0] + 2 * nghost,
        output_interior[1] + 2 * nghost,
    )
    output = _resize_plane(tensor, output_size, interpolation_mode)
    interior = tensor[..., nghost:-nghost, nghost:-nghost, :]
    resized_interior = _resize_plane(interior, output_interior, interpolation_mode)
    output[..., nghost:-nghost, nghost:-nghost, :] = resized_interior
    return output


def _field_type(name: str, snapy_module) -> int:
    if name.endswith("hydro_u"):
        return snapy_module.kConserved
    if name.endswith("hydro_w"):
        return snapy_module.kPrimitive
    return snapy_module.kScalar


def _flatten_for_snapy(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(tensor.shape)
    return tensor.reshape(-1, *tensor.shape[-3:]), original_shape


def _restore_from_snapy(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return tensor.reshape(shape).contiguous()


def _make_mesh(
    config: Path,
    *,
    block_count: int,
    local_nx2: int,
    local_nx3: int,
):
    import snapy

    options = snapy.MeshOptions.from_yaml(str(config))
    block_options = options.block()
    layout = block_options.layout()
    layout_type = layout.type()
    if layout_type not in {"slab", "cubed", "cubed-sphere"}:
        raise ValueError(f"unsupported Snapy layout type: {layout_type}")
    expected_blocks = layout.px() * layout.py() * layout.pz()
    if layout_type == "cubed-sphere":
        expected_blocks *= 6
    if block_count != expected_blocks:
        raise ValueError(
            f"restart has {block_count} blocks, but CONFIG layout {layout_type} "
            f"requires {expected_blocks}"
        )
    coord = block_options.coord()
    nghost = coord.nghost()
    options.blocks_per_process(block_count)
    options.set_local_horizontal_cells(local_nx2, local_nx3)
    return snapy.Mesh(options), nghost


def _validate_paths(
    config_path: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path]:
    config = Path(config_path).expanduser().resolve()
    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"CONFIG does not exist: {config}")
    if not source.is_file():
        raise FileNotFoundError(f"input restart does not exist: {source}")
    if source == output:
        raise ValueError("input and output restart paths must differ")
    if output.exists():
        raise FileExistsError(f"output restart already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    return config, source, output


def resize_restart(
    config_path: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    mode: ResizeMode,
) -> Path:
    import snapy

    config, source, output = _validate_paths(config_path, input_path, output_path)
    entries = read_restart_bundle_index(source)
    by_rank = _entries_by_rank(entries, len(entries))
    ordered_entries = [by_rank[rank] for rank in range(len(entries))]

    mesh_options = snapy.MeshOptions.from_yaml(str(config))
    block_options = mesh_options.block()
    coord = block_options.coord()
    nghost = coord.nghost()
    configured_local = (coord.nx2(), coord.nx3())

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.{mode}-", dir=output.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        tensor_paths: list[list[Path]] = []
        names: list[str] | None = None
        schema: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        output_local: tuple[int, int] | None = None

        for rank, entry in enumerate(ordered_entries):
            block = _load_entry(source, entry)
            block_names = sorted(block)
            if names is None:
                names = block_names
            elif block_names != names:
                raise ValueError("source blocks have inconsistent tensor names")

            rank_dir = temporary / f"block{rank}"
            rank_dir.mkdir()
            rank_paths: list[Path] = []
            for index, name in enumerate(block_names):
                tensor = block[name]
                current_schema = (tuple(tensor.shape), tensor.dtype)
                if name in schema and schema[name] != current_schema:
                    raise ValueError(
                        f"source blocks have inconsistent schema for {name}"
                    )
                schema[name] = current_schema

                if tensor.ndim >= 3:
                    source_local = (
                        tensor.shape[-2] - 2 * nghost,
                        tensor.shape[-3] - 2 * nghost,
                    )
                    if source_local != configured_local:
                        raise ValueError(
                            f"CONFIG local horizontal cells are "
                            f"{configured_local[0]}x{configured_local[1]}, but "
                            f"{name} has {source_local[0]}x{source_local[1]}"
                        )
                    tensor = resize_spatial_tensor(tensor, mode=mode, nghost=nghost)
                    local = (
                        tensor.shape[-2] - 2 * nghost,
                        tensor.shape[-3] - 2 * nghost,
                    )
                    if output_local is None:
                        output_local = local
                    elif output_local != local:
                        raise ValueError(
                            "spatial tensors have inconsistent horizontal dimensions"
                        )
                path = rank_dir / f"tensor{index}.pt"
                torch.save(tensor, path)
                rank_paths.append(path)
            tensor_paths.append(rank_paths)
            del block

        if names is None or output_local is None:
            raise ValueError("restart contains no spatial tensors")

        mesh, mesh_nghost = _make_mesh(
            config,
            block_count=len(entries),
            local_nx2=output_local[0],
            local_nx3=output_local[1],
        )
        if mesh_nghost != nghost:
            raise ValueError("CONFIG ghost width changed while constructing mesh")

        for index, name in enumerate(names):
            if len(schema[name][0]) < 3:
                continue
            tensors = [
                torch.load(paths[index], map_location="cpu", weights_only=True)
                for paths in tensor_paths
            ]
            snapy_tensors = []
            shapes = []
            for tensor in tensors:
                flattened, shape = _flatten_for_snapy(tensor)
                snapy_tensors.append(flattened)
                shapes.append(shape)
            variables = [{name: tensor} for tensor in snapy_tensors]
            mesh.exchange_ghost_zones(variables, _field_type(name, snapy))
            for paths, tensor, shape in zip(tensor_paths, snapy_tensors, shapes):
                torch.save(_restore_from_snapy(tensor, shape), paths[index])
            del variables, tensors, snapy_tensors

        part_paths = []
        for rank, paths in enumerate(tensor_paths):
            tensors = {
                name: torch.load(path, map_location="cpu", weights_only=True)
                for name, path in zip(names, paths)
            }
            part_path = temporary / f"block{rank}.part"
            _save_tensors(tensors, part_path)
            part_paths.append(part_path)
            del tensors
            for path in paths:
                path.unlink()

        temporary_bundle = temporary / output.name
        _write_bundle(temporary_bundle, part_paths)
        _publish_bundle(temporary_bundle, output)
    return output


def build_parser(mode: ResizeMode) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{mode.capitalize()} a Snapy restart horizontally by a factor of two.",
    )
    parser.add_argument("config", help="Snapy YAML configuration for the restart")
    parser.add_argument("input", help="Input Snapy restart bundle")
    parser.add_argument("output", help="New resized Snapy restart bundle")
    return parser


def main(mode: ResizeMode, argv: Sequence[str] | None = None) -> int:
    args = build_parser(mode).parse_args(argv)
    resize_restart(args.config, args.input, args.output, mode=mode)
    return 0


def refine_main(argv: Sequence[str] | None = None) -> int:
    return main("refine", argv)


def coarsen_main(argv: Sequence[str] | None = None) -> int:
    return main("coarsen", argv)
