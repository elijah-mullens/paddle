from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from snapy import MeshBlock


_LINEAR_MODES = {"linear", "bilinear", "bicubic", "trilinear"}


def _horizontal_size(shape: Sequence[int], factor: int) -> tuple[int, int, int]:
    if len(shape) < 3:
        raise ValueError(
            f"Expected at least three dimensions, got shape {tuple(shape)}"
        )

    n3, n2, n1 = shape[-3:]
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    if factor == 1:
        return n3, n2, n1

    for size, name in ((n3, "n3"), (n2, "n2")):
        if size > 1 and size % factor:
            raise ValueError(
                f"Cannot coarsen {name}={size} by factor {factor}; "
                "the size must be divisible by the factor"
            )
    return (
        n3 // factor if n3 > 1 else 1,
        n2 // factor if n2 > 1 else 1,
        n1,
    )


def _interpolate_spatial(
    tensor: torch.Tensor,
    size: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected a tensor with at least three dimensions, got {tensor.ndim}"
        )

    trailing_shape = tuple(tensor.shape[-3:])
    flattened = tensor.reshape(-1, 1, *trailing_shape)
    kwargs = {"align_corners": False} if mode in _LINEAR_MODES else {}
    sampled = F.interpolate(flattened, size=size, mode=mode, **kwargs)
    return sampled.reshape(*tensor.shape[:-3], *size)


def refine_spatial(
    tensor: torch.Tensor,
    method: str = "trilinear",
    factor: int = 2,
) -> torch.Tensor:
    """Refine non-singleton horizontal dimensions while preserving x1."""
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected a tensor with at least three dimensions, got {tensor.ndim}"
        )
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    n3, n2, n1 = tensor.shape[-3:]
    size = (
        n3 * factor if n3 > 1 else 1,
        n2 * factor if n2 > 1 else 1,
        n1,
    )
    return _interpolate_spatial(tensor, size, method)


def coarsen_spatial(tensor: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Conservatively average non-singleton horizontal dimensions."""
    size = _horizontal_size(tensor.shape, factor)
    return _interpolate_spatial(tensor, size, "area")


def _interior(tensor: torch.Tensor, nghost: int) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(
            f"Expected a tensor with at least three dimensions, got {tensor.ndim}"
        )
    if nghost < 0:
        raise ValueError(f"nghost must be non-negative, got {nghost}")
    if nghost == 0:
        return tensor
    if any(size > 1 and size <= 2 * nghost for size in tensor.shape[-3:-1]):
        raise ValueError(
            f"Horizontal shape {tuple(tensor.shape[-3:-1])} has no interior "
            f"for nghost={nghost}"
        )
    n3_slice = slice(None) if tensor.shape[-3] <= 1 else slice(nghost, -nghost)
    n2_slice = slice(None) if tensor.shape[-2] <= 1 else slice(nghost, -nghost)
    return tensor[..., n3_slice, n2_slice, :]


def _shape_with_refined_interior(
    shape: Sequence[int],
    nghost: int,
    factor: int,
    *,
    coarsen: bool,
) -> tuple[int, ...]:
    result = list(shape)
    for dim in (-3, -2):
        size = result[dim]
        if size <= 1:
            continue
        interior = size - 2 * nghost
        if coarsen and interior % factor:
            raise ValueError(
                f"Interior size {interior} is not divisible by factor {factor}"
            )
        result[dim] = (
            interior // factor + 2 * nghost
            if coarsen
            else interior * factor + 2 * nghost
        )
    return tuple(result)


def conservative_refine(
    tensor: torch.Tensor,
    nghost: int,
    factor: int = 2,
) -> torch.Tensor:
    """Refine an interior state while preserving its coarse-cell averages."""
    interior = _interior(tensor, nghost)
    refined = refine_spatial(interior, "trilinear", factor)
    coarse = coarsen_spatial(refined, factor)
    correction = refine_spatial(interior - coarse, "area", factor)

    output = tensor.new_zeros(
        _shape_with_refined_interior(tensor.shape, nghost, factor, coarsen=False)
    )
    _interior(output, nghost).copy_(refined + correction)
    return output


def conservative_coarsen(
    tensor: torch.Tensor,
    nghost: int,
    factor: int = 2,
) -> torch.Tensor:
    """Average a refined interior state onto a coarser horizontal mesh."""
    interior = _interior(tensor, nghost)
    output = tensor.new_zeros(
        _shape_with_refined_interior(tensor.shape, nghost, factor, coarsen=True)
    )
    _interior(output, nghost).copy_(coarsen_spatial(interior, factor))
    return output


def refine_boundary_state_to_match(
    boundary_state: torch.Tensor,
    target_shape: Sequence[int],
    nghost: int,
) -> torch.Tensor:
    """Repeatedly double horizontal interior resolution to reach target_shape."""
    target_shape = tuple(target_shape)
    if len(target_shape) != boundary_state.ndim:
        raise ValueError(
            f"Target rank {len(target_shape)} does not match state rank "
            f"{boundary_state.ndim}"
        )
    _interior(boundary_state, nghost)
    if tuple(boundary_state.shape) == target_shape:
        return boundary_state

    refined = boundary_state
    while tuple(refined.shape) != target_shape:
        next_shape = _shape_with_refined_interior(
            refined.shape, nghost, 2, coarsen=False
        )
        if next_shape == tuple(refined.shape):
            raise ValueError(
                f"Cannot refine state from {tuple(refined.shape)} to {target_shape}"
            )
        if any(current > target for current, target in zip(next_shape, target_shape)):
            raise ValueError(
                f"Refinement step {next_shape} exceeds target shape {target_shape}"
            )
        refined = _interpolate_spatial(refined, next_shape[-3:], "trilinear")

    return refined


def refined_global_horizontal_cells(
    current_local_nx2: int,
    current_local_nx3: int,
    px: int = 1,
    py: int = 1,
    factor: int = 2,
) -> tuple[int, int]:
    """Return refined global nx2/nx3 counts for a decomposed local mesh."""
    if min(current_local_nx2, current_local_nx3, px, py, factor) < 1:
        raise ValueError("Cell counts, layout counts, and factor must be positive")
    global_nx2 = current_local_nx2 * px
    global_nx3 = current_local_nx3 * py
    return (
        global_nx2 * factor if global_nx2 > 1 else 1,
        global_nx3 * factor if global_nx3 > 1 else 1,
    )


def _rebuild_meshblock(block: MeshBlock, factor: int, *, coarsen: bool) -> MeshBlock:
    if factor < 1:
        raise ValueError(f"factor must be positive, got {factor}")
    options = block.options
    outputs = [(out.file_number, out.next_time) for out in block.get_outputs()]
    coord = options.coord()
    nghost = coord.nghost()

    for getter in (coord.nx2, coord.nx3):
        size = getter()
        if size <= 1:
            continue
        next_size = size // factor if coarsen else size * factor
        if coarsen and (size % factor or next_size <= nghost):
            raise ValueError(
                f"Cannot coarsen horizontal size {size} by factor {factor} "
                f"with nghost={nghost}"
            )
        getter(next_size)

    rebuilt = MeshBlock(options)
    rebuilt_outputs = rebuilt.get_outputs()
    if len(rebuilt_outputs) != len(outputs):
        raise ValueError("Rebuilt MeshBlock has a different number of outputs")
    for output, (file_number, next_time) in zip(rebuilt_outputs, outputs):
        output.file_number = file_number
        output.next_time = next_time
    return rebuilt


def refine_meshblock(block: MeshBlock, factor: int = 2) -> MeshBlock:
    """Rebuild a MeshBlock with refined horizontal cell counts."""
    return _rebuild_meshblock(block, factor, coarsen=False)


def coarsen_meshblock(block: MeshBlock, factor: int = 2) -> MeshBlock:
    """Rebuild a MeshBlock with coarsened horizontal cell counts."""
    return _rebuild_meshblock(block, factor, coarsen=True)
