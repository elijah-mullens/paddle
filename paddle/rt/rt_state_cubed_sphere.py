import math
from pathlib import Path

import torch
import yaml
from snapy import MeshBlock
from snapy.coord import cs_ab_to_lonlat, get_cs_face_name

from .rt_state import RTState


class RTStateCubedSphere(RTState):
    """One illumination model returning actual flux per shortwave band, in YAML order."""

    def __init__(self, stellar_flux: list, block: MeshBlock, config_path: Path):
        super().__init__(block, config_path)
        self.layout = block.get_layout()
        with open(config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        rt_config = config["radiative-transfer"]

        # use planck to decide if shortwave band
        bands = self.radiation.options.bands()
        self.shortwave_indices = []
        for i, band in enumerate(config["bands"]):
            flags = band.get("toon", {}).get("flags", band.get("flags", ""))
            if "planck" not in flags.replace(",", " ").split():
                self.shortwave_indices.append(i)
                if not math.isclose(sum(band.get("weights", [1.0])), 1.0):
                    raise ValueError(
                        "Shortwave weights must sum to one for band-integrated beams"
                    )
        if len(stellar_flux) != len(self.shortwave_indices):
            raise ValueError("Provide one orbital model per shortwave band")
        self.stellar_flux = list(stellar_flux)

        # save albedo
        self.albedo = [
            float(rt_config.get(f"{band.name()}_albedo", 0.0)) for band in bands
        ]
        if any(not 0.0 <= value <= 1.0 for value in self.albedo):
            raise ValueError("Albedos must be between zero and one")

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
        bands = self.radiation.options.bands()
        rt_bc = {}
        for i, orbit in zip(self.shortwave_indices, self.stellar_flux):
            band = bands[i]
            mu0, beam, _, _, _ = orbit.insolation(lon, lat, current_time)
            # The night beam is zero. A positive dummy cosine avoids 0/0 in Toon.
            rt_bc[f"{band.name()}/umu0"] = torch.where(
                mu0 > 0.0, mu0, torch.ones_like(mu0)
            ).reshape(ncol)
            rt_bc[f"{band.name()}/fbeam"] = beam.reshape(1, ncol).expand(
                band.nwave(), -1
            )

        for band, albedo in zip(bands, self.albedo):
            rt_bc[f"{band.name()}/albedo"] = torch.full(
                (band.nwave(), ncol), albedo, dtype=lon.dtype, device=lon.device
            )
        return rt_bc
