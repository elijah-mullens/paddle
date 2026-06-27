from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Sequence

import torch

from .cubed_sphere_shrink import (
    _entries_by_rank,
    _load_entry,
    _publish_bundle,
    _save_tensors,
    _write_bundle,
    read_restart_bundle_index,
)
from .restart_resize import _field_type


DEFAULT_TENSOR_NAMES = (
    "hydro_u",
    "hydro_w",
    "fill_solid_hydro_u",
    "fill_solid_hydro_w",
)


def _block_count_from_layout(layout) -> int:
    block_count = layout.px() * layout.py() * layout.pz()
    if layout.type() == "cubed-sphere":
        block_count *= 6
    return block_count


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


def _ordered_entries(source: Path):
    entries = read_restart_bundle_index(source)
    by_rank = _entries_by_rank(entries, len(entries))
    return [by_rank[rank] for rank in range(len(entries))]


def _compute_column_stats(
    tensors: Sequence[torch.Tensor], *, nghost: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensors:
        raise ValueError("cannot compute statistics from no tensors")
    reference = tensors[0]
    if reference.ndim < 3:
        raise ValueError("spatial tensors must have at least three dimensions")
    if not reference.is_floating_point():
        raise TypeError(f"spatial tensor must be floating point, got {reference.dtype}")
    for tensor in tensors[1:]:
        if tensor.shape != reference.shape:
            raise ValueError("source blocks have inconsistent tensor shapes")
        if tensor.dtype != reference.dtype:
            raise ValueError("source blocks have inconsistent tensor dtypes")
    if reference.shape[-3] <= 2 * nghost or reference.shape[-2] <= 2 * nghost:
        raise ValueError(
            f"spatial tensor shape {tuple(reference.shape)} is too small for "
            f"nghost={nghost}"
        )

    samples = torch.stack(
        [tensor[..., nghost:-nghost, nghost:-nghost, :] for tensor in tensors],
        dim=0,
    )
    reduce_dims = (0, samples.ndim - 3, samples.ndim - 2)
    mean = samples.mean(dim=reduce_dims)
    std = samples.std(dim=reduce_dims, unbiased=False)
    return mean.contiguous(), std.contiguous()


def _broadcast_column_profile(
    profile: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    output = torch.empty(shape, dtype=profile.dtype, device=profile.device)
    output[...] = profile[..., None, None, :]
    return output


def _resolve_vertical_momentum_index(snapy_module, override: int | None) -> int:
    if override is not None:
        if override < 0:
            raise ValueError("vertical_momentum_index must be non-negative")
        return override
    return int(getattr(snapy_module, "kIV1", 1))


def _build_output_mesh(
    snapy_module,
    config: Path,
    *,
    block_count: int,
) -> tuple[object, int, tuple[int, int, int]]:
    options = snapy_module.MeshOptions.from_yaml(str(config))
    block_options = options.block()
    coord = block_options.coord()
    layout = block_options.layout()
    expected_blocks = _block_count_from_layout(layout)
    if block_count != expected_blocks:
        raise ValueError(
            f"CONFIG layout {layout.type()} requires {expected_blocks} blocks"
        )
    nghost = coord.nghost()
    options.blocks_per_process(block_count)
    options.set_local_horizontal_cells(coord.nx2(), coord.nx3())
    return snapy_module.Mesh(options), nghost, (coord.nx3(), coord.nx2(), coord.nx1())


def create_restart(
    config_path: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    seed: int | None = None,
    vertical_momentum_index: int | None = None,
    tensor_names: Sequence[str] = DEFAULT_TENSOR_NAMES,
) -> Path:
    import snapy

    config, source, output = _validate_paths(config_path, input_path, output_path)
    source_entries = _ordered_entries(source)
    mesh_options = snapy.MeshOptions.from_yaml(str(config))
    target_block_count = _block_count_from_layout(mesh_options.block().layout())
    mesh, nghost, target_local = _build_output_mesh(
        snapy, config, block_count=target_block_count
    )
    target_x3, target_x2, target_x1 = target_local
    target_shape_tail = (
        target_x3 + 2 * nghost,
        target_x2 + 2 * nghost,
        target_x1 + 2 * nghost,
    )
    vertical_index = _resolve_vertical_momentum_index(snapy, vertical_momentum_index)

    source_blocks = [_load_entry(source, entry) for entry in source_entries]
    if not source_blocks:
        raise ValueError("restart contains no blocks")
    source_names = sorted(source_blocks[0])
    expected_names = set(source_names)
    for block in source_blocks[1:]:
        if set(block) != expected_names:
            raise ValueError("source blocks have inconsistent tensor names")

    missing = [name for name in tensor_names if name not in expected_names]
    if missing:
        raise ValueError(f"source restart is missing tensors: {', '.join(missing)}")

    stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in tensor_names:
        stats[name] = _compute_column_stats(
            [block[name] for block in source_blocks],
            nghost=nghost,
        )

    reference_shape = tuple(source_blocks[0][tensor_names[0]].shape)
    if reference_shape[-1] != target_shape_tail[-1]:
        raise ValueError(
            f"source vertical dimension {reference_shape[-1]} does not match "
            f"target {target_shape_tail[-1]}"
        )

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.restart-", dir=output.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        tensor_paths: dict[str, list[Path]] = {name: [] for name in tensor_names}
        vertical_momentum_noises: list[torch.Tensor | None] = [
            None for _ in range(target_block_count)
        ]

        def vertical_noise(rank: int, dtype: torch.dtype) -> torch.Tensor:
            noise = vertical_momentum_noises[rank]
            if noise is None:
                noise = torch.randn(
                    target_shape_tail,
                    generator=generator,
                    dtype=dtype,
                    device="cpu",
                )
                vertical_momentum_noises[rank] = noise
            return noise.to(dtype=dtype)

        for name in tensor_names:
            mean, std = stats[name]
            shape = (*mean.shape[:-1], *target_shape_tail)
            variables = []
            paths = []
            for rank in range(target_block_count):
                tensor = _broadcast_column_profile(mean, shape)
                if name in {"hydro_u", "fill_solid_hydro_u"}:
                    if vertical_index >= tensor.shape[0]:
                        raise ValueError(
                            f"vertical_momentum_index {vertical_index} is outside "
                            f"{name} component dimension {tensor.shape[0]}"
                        )
                    noise = vertical_noise(rank, tensor.dtype)
                    tensor[vertical_index] = (
                        mean[vertical_index][None, None, :]
                        + std[vertical_index][None, None, :] * noise
                    )
                variables.append({name: tensor.contiguous()})
                paths.append(temporary / f"block{rank}.{name}.pt")

            mesh.exchange_ghost_zones(variables, _field_type(name, snapy))
            for path, variable in zip(paths, variables):
                torch.save(variable[name], path)
            tensor_paths[name] = paths
            del variables

        template = source_blocks[0]
        part_paths = []
        for rank in range(target_block_count):
            tensors: dict[str, torch.Tensor] = {}
            for name in source_names:
                if name not in tensor_paths:
                    tensors[name] = template[name].detach().cpu().clone().contiguous()
                else:
                    tensors[name] = torch.load(
                        tensor_paths[name][rank],
                        map_location="cpu",
                        weights_only=True,
                    )

            part_path = temporary / f"block{rank}.part"
            _save_tensors(tensors, part_path)
            part_paths.append(part_path)

        temporary_bundle = temporary / output.name
        _write_bundle(temporary_bundle, part_paths)
        _publish_bundle(temporary_bundle, output)

    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paddle restart",
        description=(
            "Create a restart from horizontal statistics of a source final state."
        ),
    )
    parser.add_argument("config", help="Output Snapy YAML configuration")
    parser.add_argument("input", help="Input Snapy restart bundle")
    parser.add_argument("output", help="New generated Snapy restart bundle")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--vertical-momentum-index",
        type=int,
        default=None,
        help="Component index for vertical momentum (default: snapy.kIV1)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    create_restart(
        args.config,
        args.input,
        args.output,
        seed=args.seed,
        vertical_momentum_index=args.vertical_momentum_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
