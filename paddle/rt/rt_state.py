import math
from pathlib import Path

import torch
import yaml
from kintera import ThermoX
from pyharp import Radiation, RadiationOptions
from snapy import MeshBlock, kICY, kIPR

RT_HEATING_KEY = "rt_heating"


class RTState:
    """Hold a block's radiative heating fixed between radiation updates."""

    def __init__(self, block: MeshBlock, config_path: Path):
        config_path = Path(config_path).resolve()
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        rt_config = config["radiative-transfer"]
        self.update_dt = float(rt_config["update_dt"])
        if not math.isfinite(self.update_dt) or self.update_dt < 0.0:
            raise ValueError("update_dt must be finite and nonnegative")

        self.next_update = -math.inf
        self.net_flux = None
        self.coord = block.module("coord")
        self.eos = block.module("hydro.eos")
        self.thermo_y = block.module("hydro.eos.thermo")

        x1v = self.coord.buffer("x1v")
        x2v = self.coord.buffer("x2v")
        x3v = self.coord.buffer("x3v")
        self.thermo_x = ThermoX(self.thermo_y.options)
        self.thermo_x.to(device=x1v.device, dtype=x1v.dtype)

        # configure local column sizes and band quadrature (needed for domain composition)
        options = RadiationOptions.from_yaml(str(config_path))
        for band, band_config in zip(options.bands(), config["bands"]):
            band.ncol(x2v.numel() * x3v.numel())
            band.nlyr(self.coord.iu() - self.coord.il() + 1)
            band.weight(band_config.get("weights", [1.0]))

        self.radiation = Radiation(options)
        self.radiation.to(device=x1v.device, dtype=x1v.dtype)

    def compute_rt_bc(self, current_time: float) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must provide radiation boundaries")

    @torch.no_grad()
    def update_heating(
        self, variables: dict[str, torch.Tensor], current_time: float
    ) -> None:
        """Store q_rad [W/m^3], recomputing when missing or due."""
        if not math.isfinite(current_time):
            raise ValueError("current_time must be finite")
        if RT_HEATING_KEY in variables and current_time < self.next_update:
            return

        # use current geometry, including horizontal ghost columns
        il, iu = self.coord.il(), self.coord.iu()
        hydro_w = variables["hydro_w"][..., il : iu + 1]
        ncol = hydro_w.shape[1] * hydro_w.shape[2]
        nlyr = iu - il + 1
        area = self.coord.face_area1()[..., il : iu + 2].reshape(ncol, nlyr + 1)
        volume = self.coord.cell_volume()[..., il : iu + 1].reshape(ncol, nlyr)
        dz = self.coord.buffer("dx1f")[il : iu + 1]

        # compute thermodynamics, with species last for pyharp
        pres = hydro_w[kIPR]
        temp = self.eos.compute("W->T", (hydro_w,))
        xfrac = self.thermo_y.compute("Y->X", (hydro_w[kICY:],))
        conc = self.thermo_x.compute("TPX->V", (temp, pres, xfrac))
        atm = {
            "pres": pres.reshape(ncol, nlyr),
            "temp": temp.reshape(ncol, nlyr),
        }
        conc = conc.reshape(ncol, nlyr, conc.shape[-1])

        # net flux is upward-positive, ordered from bottom to top
        boundary = self.compute_rt_bc(current_time)
        net_flux, _, _ = self.radiation.forward(conc, dz, boundary, atm)
        heating = (
            -(area[:, 1:] * net_flux[:, 1:] - area[:, :-1] * net_flux[:, :-1]) / volume
        )
        if not torch.isfinite(heating).all():
            raise RuntimeError("Radiation produced nonfinite heating")

        # store net flux for diagnostics; radial ghost layers remain zero
        variables[RT_HEATING_KEY] = torch.zeros_like(variables["hydro_w"][kIPR])
        variables[RT_HEATING_KEY][..., il : iu + 1] = heating.reshape(pres.shape)
        self.net_flux = net_flux.detach().clone()
        self.next_update = current_time + self.update_dt
