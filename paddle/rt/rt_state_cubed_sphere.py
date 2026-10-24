import math
from pathlib import Path

import torch
import yaml
from snapy import MeshBlock
from snapy.coord import cs_ab_to_lonlat, get_cs_face_name

from .rt_state import RTState


class RTStateCubedSphere(RTState):
    """One orbital illumination model per shortwave band, in YAML band order."""

    def __init__(self, stellar_flux: list, block: MeshBlock, config_path: Path):
        super().__init__(block, config_path)
        self.layout = block.get_layout()
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        rt_config = config["radiative-transfer"]
        self.band_names = [band["name"] for band in config["bands"]]
        if len(set(self.band_names)) != len(self.band_names):
            raise ValueError("Radiation band names must be unique")
        self.nwave = [int(band["nwave"]) for band in config["bands"]]
        self.shortwave_names = []
        for band in config["bands"]:
            flags = band.get("toon", {}).get("flags", band.get("flags", ""))
            if "planck" not in flags.replace(",", " ").split():
                self.shortwave_names.append(band["name"])
                if not math.isclose(sum(band.get("weights", [1.0])), 1.0):
                    raise ValueError(
                        "Shortwave weights must sum to one for band-integrated beams"
                    )
        if len(stellar_flux) != len(self.shortwave_names):
            raise ValueError("Provide one orbital model per shortwave band")
        self.stellar_flux = list(stellar_flux)
        self.stellar_fractions = [
            float(rt_config["stellar_fractions"][name]) for name in self.shortwave_names
        ]
        if any(f < 0.0 or not math.isfinite(f) for f in self.stellar_fractions):
            raise ValueError("Stellar fractions must be finite and nonnegative")
        if sum(self.stellar_fractions) > 1.0 + 1.0e-12:
            raise ValueError("Stellar fractions must sum to at most one")
        self.albedo = [
            float(rt_config.get(f"{name}_albedo", 0.0)) for name in self.band_names
        ]
        if any(not 0.0 <= value <= 1.0 for value in self.albedo):
            raise ValueError("Albedos must be between zero and one")
        self.time_offset = float(config.get("orbit", {}).get("time_offset", 0.0))
        if not math.isfinite(self.time_offset):
            raise ValueError("Orbit time_offset must be finite")

    def compute_rt_bc(self, current_time: float) -> dict[str, torch.Tensor]:
        # each local block has its own global block rank and cube face
        layout = self.layout
        face_id = int(layout.loc_of(layout.options.rank())[2])
        face = get_cs_face_name(face_id)
        beta, alpha = torch.meshgrid(
            self.coord.buffer("x3v"),
            self.coord.buffer("x2v"),
            indexing="ij",
        )
        lon, lat = cs_ab_to_lonlat(face, alpha, beta)
        ncol = lon.numel()
        boundary = {}
        for name, orbit, fraction in zip(
            self.shortwave_names, self.stellar_flux, self.stellar_fractions
        ):
            mu0, beam, _, _, _ = orbit.insolation(
                lon, lat, current_time + self.time_offset
            )
            nwave = self.nwave[self.band_names.index(name)]
            # The night beam is zero. A positive dummy cosine avoids 0/0 in Toon.
            boundary[f"{name}/umu0"] = torch.where(
                mu0 > 0.0, mu0, torch.ones_like(mu0)
            ).reshape(ncol)
            boundary[f"{name}/fbeam"] = (
                (fraction * beam).reshape(1, ncol).expand(nwave, -1)
            )
        for name, nwave, albedo in zip(self.band_names, self.nwave, self.albedo):
            boundary[f"{name}/albedo"] = torch.full(
                (nwave, ncol), albedo, dtype=lon.dtype, device=lon.device
            )
        return boundary
