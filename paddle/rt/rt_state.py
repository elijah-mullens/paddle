import math
from pathlib import Path

import torch
import yaml
from kintera import ThermoX
from pyharp import Radiation, RadiationOptions
from snapy import MeshBlock, kICY, kIPR


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
        self.last_update = -math.inf
        self.heating = None
        self.net_flux = None
        self.rt_heating_key = rt_config.get("heating_key", "rt_heating")
        self.block = block
        self.coord = block.module("coord")
        self.eos = block.module("hydro.eos")
        self.thermo_y = block.module("hydro.eos.thermo")
        grid = self.coord.buffer("x1v")
        self.thermo_x = ThermoX(self.thermo_y.options)
        self.thermo_x.to(device=grid.device, dtype=grid.dtype)

        # restrict radiation to active columns and layers, not halo cells
        self.il, self.iu = self.coord.il(), self.coord.iu()
        self.jl, self.ju = self.coord.jl(), self.coord.ju()
        self.kl, self.ku = self.coord.kl(), self.coord.ku()
        self.cells = (
            slice(self.kl, self.ku + 1),
            slice(self.jl, self.ju + 1),
            slice(self.il, self.iu + 1),
        )
        self.ncol = (self.ku - self.kl + 1) * (self.ju - self.jl + 1)
        self.nlyr = self.iu - self.il + 1
        self.area = self.coord.face_area1()[
            self.kl : self.ku + 1, self.jl : self.ju + 1, self.il : self.iu + 2
        ].reshape(self.ncol, self.nlyr + 1)
        self.volume = self.coord.cell_volume()[self.cells].reshape(self.ncol, self.nlyr)
        self.dz = self.coord.buffer("dx1f")[self.il : self.iu + 1]

        # PyHarp reads band definitions but needs quadrature and local sizes
        options = RadiationOptions.from_yaml(str(config_path))
        if not options.bands():
            raise ValueError("Radiation requires at least one band")
        for band, band_config in zip(options.bands(), config["bands"]):
            band.ncol(self.ncol).nlyr(self.nlyr)
            weights = band_config.get("weights", [1.0] if band.nwave() == 1 else [])
            if (
                len(weights) != band.nwave()
                or any(not math.isfinite(float(w)) or float(w) < 0.0 for w in weights)
                or sum(float(w) for w in weights) <= 0.0
            ):
                raise ValueError(
                    "Each band needs nonnegative finite weights with positive sum"
                )
            band.weight([float(weight) for weight in weights])
            for opacity in band.opacities().values():
                opacity.opacity_files(
                    [
                        str((config_path.parent / path).resolve())
                        for path in opacity.opacity_files()
                    ]
                )
        self.radiation = Radiation(options)
        self.radiation.to(device=grid.device, dtype=grid.dtype)

    def compute_rt_bc(self, current_time: float) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must provide radiation boundaries")

    @torch.no_grad()
    def update_heating(
        self, variables: dict[str, torch.Tensor], current_time: float
    ) -> None:
        """Store q_rad [W/m³], recomputing on first call, expiry, or time rewind."""
        if not math.isfinite(current_time):
            raise ValueError("current_time must be finite")
        if (
            self.heating is not None
            and self.last_update <= current_time < self.next_update
        ):
            variables[self.rt_heating_key] = self.heating
            return

        # recover thermodynamics on active cells, with species last for PyHarp
        hydro_w = variables["hydro_w"][(slice(None),) + self.cells]
        pres = hydro_w[kIPR]
        temp = self.eos.compute("W->T", (hydro_w,))
        xfrac = self.thermo_y.compute("Y->X", (hydro_w[kICY:],))
        conc = self.thermo_x.compute("TPX->V", (temp, pres, xfrac))
        atm = {
            "pres": pres.reshape(self.ncol, self.nlyr),
            "temp": temp.reshape(self.ncol, self.nlyr),
        }
        conc = conc.reshape(self.ncol, self.nlyr, conc.shape[-1])

        # net flux is upward-positive, ordered from bottom to top
        boundary = self.compute_rt_bc(current_time)
        net_flux, _, _ = self.radiation.forward(conc, self.dz, boundary, atm)
        heating = (
            -(self.area[:, 1:] * net_flux[:, 1:] - self.area[:, :-1] * net_flux[:, :-1])
            / self.volume
        )
        if not torch.isfinite(heating).all():
            raise RuntimeError("Radiation produced nonfinite heating")

        # use a spatial field, with halos zero; publish only a successful solve
        self.heating = torch.zeros_like(variables["hydro_w"][kIPR])
        self.heating[self.cells] = heating.reshape(pres.shape)
        self.net_flux = net_flux.detach().clone()
        variables[self.rt_heating_key] = self.heating
        self.last_update = float(current_time)
        self.next_update = current_time + self.update_dt
