"""Uniform-column Earth RT with opaque terrain and restartable cached heating."""

from datetime import datetime
import math

import torch

from earth_orbital_rt_forcing import DEFAULT_EPOCH, EarthOrbitalRTForcing
from earth_radiative_transfer import EarthTwoStream, GRAVITY, SW_SURFACE_ALBEDO


class RTStateCartesian:
    """Host-side Radiation driver; the orbital .pt callback applies rt_heating."""

    key = "rt_heating"

    def __init__(
        self,
        block,
        thermo_x,
        config_path,
        center_longitude,
        center_latitude,
        initialization_utc,
        update_dt=60.0,
        terrain_optical_depth=50.0,
        upper_layers=32,
        top_pressure=100.0,
        ocean_albedo=SW_SURFACE_ALBEDO,
        land_albedo=SW_SURFACE_ALBEDO,
        freeze_solar_time=False,
    ):
        if not math.isfinite(update_dt) or update_dt < 0:
            raise ValueError("RT interval must be finite and nonnegative")
        if upper_layers < 1 or not math.isfinite(top_pressure) or top_pressure <= 0:
            raise ValueError("Positive upper-layer count and top pressure required")
        if not all(
            math.isfinite(value) and 0 <= value < 1
            for value in (ocean_albedo, land_albedo)
        ):
            raise ValueError("Surface albedo must be finite in [0,1)")
        if (
            not math.isfinite(center_longitude)
            or not math.isfinite(center_latitude)
            or abs(center_latitude) > 90
        ):
            raise ValueError("Invalid geographic domain center")
        self.block, self.thermo_x = block, thermo_x
        self.update_dt = update_dt
        self.freeze_solar_time = bool(freeze_solar_time)
        self.upper_layers, self.top_pressure = upper_layers, top_pressure
        self.albedos = {"ocean": ocean_albedo, "land": land_albedo}
        epoch = datetime.fromisoformat(DEFAULT_EPOCH)
        initial = datetime.fromisoformat(initialization_utc.replace("Z", "+00:00"))
        if initial.tzinfo is None:
            raise ValueError("Initialization time must include its UTC offset")
        self.epoch_seconds = (initial - epoch).total_seconds()
        self.longitude, self.latitude = math.radians(center_longitude), math.radians(
            center_latitude
        )
        self.orbit = torch.jit.script(EarthOrbitalRTForcing())
        self.model = EarthTwoStream.from_config(config_path)
        self.model.enable_terrain(terrain_optical_depth)
        self.reflector = EarthTwoStream.from_config(config_path)
        self.reflector.enable_terrain(terrain_optical_depth)
        self._reflection_cache = {}
        coord = block.module("coord")
        if (
            str(coord) != "snap::CartesianImpl"
            or block.get_layout().options.type() != "slab"
        ):
            raise ValueError("RTStateCartesian requires Cartesian slab layout")
        if block.get_layout().options.pz() != 1:
            raise ValueError("RTStateCartesian requires one vertical block")
        self.interior = block.part((0, 0, 0), exterior=False)[1:]
        self.il, self.iu = coord.il(), coord.iu()
        self.nlyr = self.iu - self.il + 1
        self.dz = coord.buffer("dx1f")[self.il : self.iu + 1]
        self.area = coord.face_area1()[
            self.interior[:2] + (slice(self.il, self.iu + 2),)
        ]
        self.volume = coord.cell_volume()[self.interior]
        self.shape = self.volume.shape
        self.ncol = self.shape[0] * self.shape[1]
        names = [spec["name"] for spec in self.model.config["species"]]
        native = block.module("hydro.eos.thermo").options
        if list(thermo_x.options.species()) != list(native.species()):
            raise ValueError("ThermoX and native species must agree")
        active = [
            native.species()[index]
            for index in list(native.vapor_ids()) + list(native.cloud_ids())
        ]
        if len(names) != len(set(names)) or set(active) - set(names):
            raise ValueError("Radiation species must cover native species uniquely")
        self.indices = torch.tensor(
            [names.index(name) for name in active], device=self.dz.device
        )
        self.nspecies = len(names)
        self.cloud_ids = list(range(len(native.vapor_ids()), len(active)))
        self.rd = 8.31446261815324 / float(thermo_x.mu[0])
        self.cv = (
            torch.tensor(
                [
                    native.cref_R()[index]
                    for index in list(native.vapor_ids()) + list(native.cloud_ids())
                ],
                device=self.dz.device,
                dtype=self.dz.dtype,
            )
            * 8.31446261815324
            / thermo_x.mu.to(self.dz)
        )

    def initialize(self, variables):
        """Allocate only on a fresh run; restart tensors are already authoritative."""
        density = variables["hydro_w"][0]
        surface = density[..., 0]
        for key in ("rt_heating", "rt_temperature_tendency"):
            variables[key] = torch.zeros_like(density)
        for key in ("rt_surface_net", "rt_mu0", "rt_beam", "rt_zenith_angle"):
            variables[key] = torch.zeros_like(surface)
        for key in ("rt_net_flux", "rt_sw_net_flux", "rt_lw_net_flux"):
            variables[key] = torch.zeros(
                (*surface.shape, self.nlyr + 1),
                device=density.device,
                dtype=density.dtype,
            )
        variables["rt_clock"] = torch.tensor(
            [-math.inf, -math.inf], device=density.device, dtype=torch.float64
        )

    def next_update(self, variables):
        return float(variables["rt_clock"][1])

    def terrain_scattering(self, cosine, albedo):
        """Match direct-beam reflectance of a finite opaque column by bisection."""
        if albedo == 0 or cosine <= 0:
            return 0.0
        key = (cosine, albedo)
        if key in self._reflection_cache:
            return self._reflection_cache[key]
        pres = torch.full((1, 2), 1.0e5, dtype=torch.float64)
        temp = torch.full_like(pres, 288.0)
        conc = torch.zeros((1, 2, self.nspecies), dtype=torch.float64)
        cosine_tensor = torch.tensor([cosine], dtype=torch.float64)
        lower, upper = 0.0, 1.0 - 1.0e-12
        for _ in range(36):
            midpoint = (lower + upper) / 2
            result = self.reflector.forward(
                conc,
                pres,
                temp,
                torch.ones_like(pres),
                torch.full((1, 3), 288.0, dtype=torch.float64),
                cosine_tensor,
                torch.ones(1, dtype=torch.float64),
                sw_albedo=torch.zeros((1, 1), dtype=torch.float64),
                terrain=torch.ones_like(pres, dtype=torch.bool),
                terrain_ssa=torch.full_like(pres, midpoint),
            )
            reflection = float(result["sw_flux"][0, -1, 0]) / cosine
            if not math.isfinite(reflection):
                raise ValueError("Nonfinite terrain reflection calibration")
            if reflection < albedo:
                lower = midpoint
            else:
                upper = midpoint
        if abs(reflection - albedo) > 1.0e-6:
            raise ValueError(
                "Requested albedo not achievable with configured terrain optical depth"
            )
        self._reflection_cache = {key: (lower + upper) / 2}
        return self._reflection_cache[key]

    def extend_atmosphere(self, conc, pres, temp, dz, faces):
        top_face_pressure = pres[:, -1] * torch.exp(
            -GRAVITY * dz[:, -1] / (2 * self.rd * temp[:, -1])
        )
        if bool(torch.any(top_face_pressure <= self.top_pressure)):
            raise ValueError(
                "Regional top must lie below the reference-atmosphere top pressure"
            )
        fraction = torch.linspace(
            0, 1, self.upper_layers + 1, device=pres.device, dtype=pres.dtype
        )
        edges = top_face_pressure[:, None] * (
            self.top_pressure / top_face_pressure[:, None]
        ).pow(fraction)
        cap = torch.minimum(faces[:, -1], faces.new_full((self.ncol,), 216.65))
        edge_temp = torch.maximum(
            faces[:, -1:] * (edges / edges[:, :1]).pow(0.1903), cap[:, None]
        )
        upper_temp = (edge_temp[:, :-1] + edge_temp[:, 1:]) / 2
        upper_pres = torch.sqrt(edges[:, :-1] * edges[:, 1:])
        upper_density = upper_pres / (self.rd * upper_temp)
        upper_dz = (edges[:, :-1] - edges[:, 1:]) / (GRAVITY * upper_density)
        upper_conc = conc.new_zeros((self.ncol, self.upper_layers, self.nspecies))
        upper_conc[..., 0] = upper_pres / (8.31446261815324 * upper_temp)
        return (
            torch.cat((conc, upper_conc), 1),
            torch.cat((pres, upper_pres), 1),
            torch.cat((temp, upper_temp), 1),
            torch.cat((dz, upper_dz), 1),
            torch.cat((faces, edge_temp[:, 1:]), 1),
        )

    @torch.no_grad()
    def update_heating(self, variables, current_time, force=False):
        if not math.isfinite(current_time):
            raise ValueError("Simulation time must be finite")
        last, deadline = (float(value) for value in variables["rt_clock"])
        if not force and last <= current_time < deadline:
            return variables[self.key]
        hydro = variables["hydro_w"]
        thermo = self.block.module("hydro.eos.thermo")
        eos = self.block.module("hydro.eos")
        temperature = eos.compute("W->T", (hydro,))
        pressure = hydro[4][self.interior].reshape(self.ncol, self.nlyr)
        temp = temperature[self.interior].reshape(self.ncol, self.nlyr)
        solid = variables["solid"][self.interior].reshape(self.ncol, self.nlyr).bool()
        if bool(torch.any(solid[:, -1])) or bool(
            torch.any(solid[:, 1:] & ~solid[:, :-1])
        ):
            raise ValueError(
                "Terrain must be bottom-contiguous with air above every column"
            )
        bottom = solid.sum(1).long()
        first = bottom[:, None]
        safe_temp = torch.where(solid, temp.gather(1, first), temp)
        safe_pres = torch.where(solid, pressure.gather(1, first), pressure)
        fractions = hydro[5:][(slice(None),) + self.interior]
        xfrac = thermo.compute("Y->X", (fractions,))
        active_conc = self.thermo_x.compute(
            "TPX->V",
            (safe_temp.reshape(self.shape), safe_pres.reshape(self.shape), xfrac),
        )
        conc = hydro.new_zeros((self.ncol, self.nlyr, self.nspecies))
        conc.index_copy_(
            -1, self.indices, active_conc.reshape(self.ncol, self.nlyr, -1)
        )
        conc[solid] = 0
        dz = self.dz.to(hydro).expand(self.ncol, -1)
        faces = torch.cat(
            (
                safe_temp[:, :1],
                (safe_temp[:, :-1] + safe_temp[:, 1:]) / 2,
                safe_temp[:, -1:],
            ),
            1,
        )
        conc, pressure, temp, dz, faces = self.extend_atmosphere(
            conc, safe_pres, safe_temp, dz, faces
        )
        terrain = torch.cat(
            (
                solid,
                torch.zeros(
                    (self.ncol, self.upper_layers),
                    device=solid.device,
                    dtype=torch.bool,
                ),
            ),
            1,
        )
        orbital_elapsed = 0.0 if self.freeze_solar_time else current_time
        time = torch.tensor(
            orbital_elapsed + self.epoch_seconds,
            device=hydro.device,
            dtype=torch.float64,
        )
        _, beam, _, solar_longitude, solar_latitude = self.orbit.insolation(
            time.new_tensor(self.longitude), time.new_tensor(self.latitude), time
        )
        cosine = math.sin(self.latitude) * math.sin(float(solar_latitude)) + math.cos(
            self.latitude
        ) * math.cos(float(solar_latitude)) * math.cos(
            self.longitude - float(solar_longitude)
        )
        beam = float(beam)
        illumination = hydro.new_full((self.ncol,), max(cosine, 0.0))
        beam_tensor = hydro.new_full((self.ncol,), beam if cosine > 0 else 0.0)
        land = variables["surface_land_fraction"][self.interior[:2]].reshape(self.ncol)
        total = hydro.new_zeros((self.ncol, self.nlyr + 1))
        band_net = {name: torch.zeros_like(total) for name in ("sw", "lw")}
        surface_net = hydro.new_zeros(self.ncol)
        level = torch.arange(faces.shape[1], device=hydro.device)[None, :]
        for kind, weight in (("ocean", 1 - land), ("land", land)):
            target = variables[kind + "_net_radiation"]
            target.zero_()
            if not bool(torch.any(weight > 0)):
                continue
            skin = variables[kind + "_temperature"][self.interior[:2]].reshape(
                self.ncol
            )
            tile_faces = torch.where(level <= bottom[:, None], skin[:, None], faces)
            tile_temp = torch.where(terrain, skin[:, None], temp)
            ssa = self.terrain_scattering(max(cosine, 0.0), self.albedos[kind])
            result = self.model.forward(
                conc,
                pressure,
                tile_temp,
                dz,
                tile_faces,
                illumination,
                beam_tensor,
                sw_albedo=hydro.new_full((1, self.ncol), self.albedos[kind]),
                terrain=terrain,
                terrain_ssa=hydro.new_full(pressure.shape, ssa),
            )
            buried_face = (
                torch.arange(self.nlyr + 1, device=hydro.device)[None, :]
                < bottom[:, None]
            )
            net = result["net_flux"][:, : self.nlyr + 1].masked_fill(buried_face, 0)
            radiation = -net.gather(1, first).squeeze(1)
            target[self.interior[:2]] = radiation.reshape(self.shape[:2])
            surface_net += weight * radiation
            total += weight[:, None] * net
            for band in band_net:
                flux = result[band + "_flux"][:, : self.nlyr + 1]
                band_net[band] += weight[:, None] * (
                    flux[..., 0] - flux[..., 1]
                ).masked_fill(buried_face, 0)
        weighted_flux = total * self.area.reshape(self.ncol, self.nlyr + 1)
        heating = -(weighted_flux[:, 1:] - weighted_flux[:, :-1]) / self.volume.reshape(
            self.ncol, self.nlyr
        )
        heating.masked_fill_(solid, 0)
        if not bool(torch.isfinite(heating).all()) or not bool(
            torch.isfinite(total).all()
        ):
            raise ValueError("Nonfinite radiative flux or heating")
        variables[self.key].zero_()
        variables[self.key][self.interior] = heating.reshape(self.shape)
        mixture = torch.cat(((1 - fractions.sum(0)).unsqueeze(0), fractions), 0)
        capacity = (mixture * self.cv[:, None, None, None]).sum(0)
        variables["rt_temperature_tendency"].zero_()
        variables["rt_temperature_tendency"][self.interior] = heating.reshape(
            self.shape
        ) / (hydro[0][self.interior] * capacity)
        for name, flux in [
            ("rt_net_flux", total),
            ("rt_sw_net_flux", band_net["sw"]),
            ("rt_lw_net_flux", band_net["lw"]),
        ]:
            variables[name].zero_()
            variables[name][self.interior[:2]] = flux.reshape(
                *self.shape[:2], self.nlyr + 1
            )
        for name, value in [
            ("rt_surface_net", surface_net.reshape(self.shape[:2])),
            ("rt_mu0", cosine),
            ("rt_beam", beam if cosine > 0 else 0.0),
            ("rt_zenith_angle", math.acos(max(-1.0, min(1.0, cosine)))),
        ]:
            variables[name].zero_()
            variables[name][self.interior[:2]] = value
        next_time = (
            (math.floor(current_time / self.update_dt) + 1) * self.update_dt
            if self.update_dt
            else current_time
        )
        variables["rt_clock"].copy_(time.new_tensor([current_time, next_time]))
        return variables[self.key]
