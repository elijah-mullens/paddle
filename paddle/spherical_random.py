from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from snapy.coord import cs_ab_to_lonlat, get_cs_face_name


def randomize_initial_velocity(
    block,
    target: torch.Tensor,
    amplitude: float = 0.1,
    seed: int = 0,
) -> torch.Tensor:
    if not _is_cubed_sphere_block(block):
        return amplitude * torch.rand_like(target)

    coord = block.module("coord")
    x1v = coord.buffer("x1v").to(device=target.device, dtype=target.dtype)
    x2v = coord.buffer("x2v").to(device=target.device, dtype=target.dtype)
    x3v = coord.buffer("x3v").to(device=target.device, dtype=target.dtype)
    nc3, nc2, nc1 = target.shape
    nlon = 4 * nc2
    nlat = 2 * nc3
    nheight = nc1

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_field = torch.rand(
        (1, 1, nheight, nlat, nlon),
        generator=generator,
        dtype=target.dtype,
        device="cpu",
    ).to(target.device)
    random_field = torch.cat((random_field, random_field[..., :1]), dim=-1)

    beta, alpha, height = torch.meshgrid(x3v, x2v, x1v, indexing="ij")
    layout = block.get_layout()
    _, _, face_id = layout.loc_of(layout.options.rank())
    lon, lat = cs_ab_to_lonlat(get_cs_face_name(face_id), alpha, beta)

    two_pi = 2.0 * math.pi
    lon = torch.remainder(lon, two_pi)
    lon_coord = lon / two_pi * 2.0 - 1.0
    lat_coord = (lat + 0.5 * math.pi) / math.pi * 2.0 - 1.0

    hmin = x1v[0]
    hmax = x1v[-1]
    if torch.isclose(hmin, hmax):
        height_coord = torch.zeros_like(height)
    else:
        height_coord = (height - hmin) / (hmax - hmin) * 2.0 - 1.0

    sample_grid = torch.stack((lon_coord, lat_coord, height_coord), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        random_field,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return amplitude * sampled[0, 0]


def _is_cubed_sphere_block(block) -> bool:
    try:
        layout = block.get_layout()
        loc = layout.loc_of(layout.options.rank())
    except AttributeError:
        return False
    return len(loc) == 3 and 0 <= int(loc[2]) < 6
