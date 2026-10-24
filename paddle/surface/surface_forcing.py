"""Restartable neutral-bulk surface exchange for Snapy's stage-forcing interface."""

from typing import Dict, List

import torch


class BulkSurfaceForcing(torch.nn.Module):
    """Fixed-mass surface reservoir, with state held in each block's variables."""

    def __init__(
        self,
        kind: str,
        heat_capacity: float,
        availability: float = 1.0,
        initial_temperature: float = 288.0,
        nghost: int = 3,
        integrator: str = "rk3",
        sensible_coefficient: float = 0.0012,
        moisture_coefficient: float = 0.0012,
        drag_coefficient: float = 0.0013,
        net_radiation: float = 0.0,
        latent_heat: float = 2.5e6,
        gas_constants: List[float] = [287.0025, 461.5228, 0.0, 0.0],
        heat_capacities: List[float] = [717.50625, 1153.807, 4153.7052, 4153.7052],
        vapor_reference_energy: float = 0.0,
    ):
        super().__init__()
        if kind not in ("ocean", "land"):
            raise ValueError("kind must be ocean or land")
        if heat_capacity <= 0 or initial_temperature <= 0 or latent_heat <= 0:
            raise ValueError(
                "Heat capacity, temperature and latent heat must be positive"
            )
        if not 0 <= availability <= 1 or nghost < 1:
            raise ValueError("Availability must be in [0,1]; nghost must be positive")
        if min(sensible_coefficient, moisture_coefficient, drag_coefficient) < 0:
            raise ValueError("Exchange coefficients must be nonnegative")
        if integrator not in ("rk1", "rk2", "rk3"):
            raise ValueError("Supported integrators: rk1, rk2, rk3")
        if len(gas_constants) != len(heat_capacities) or len(gas_constants) < 2:
            raise ValueError("EOS arrays must include dry gas and first vapor H2O")
        if gas_constants[0] <= 0 or gas_constants[1] <= 0 or min(heat_capacities) <= 0:
            raise ValueError("Invalid EOS constants")

        self.kind = kind
        self.capacity = float(heat_capacity)
        self.availability = float(availability)
        self.initial_temperature = float(initial_temperature)
        self.nghost = nghost
        self.integrator = integrator
        self.ch = float(sensible_coefficient)
        self.ce = float(moisture_coefficient)
        self.cd = float(drag_coefficient)
        self.net_radiation = float(net_radiation)
        self.latent_heat = float(latent_heat)
        self.vapor_reference_energy = float(vapor_reference_energy)
        self.register_buffer(
            "gas_constants", torch.tensor(gas_constants, dtype=torch.float64)
        )
        self.register_buffer(
            "heat_capacities", torch.tensor(heat_capacities, dtype=torch.float64)
        )

    @torch.jit.export
    def initialize(
        self, hydro_w: torch.Tensor, land_fraction: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Create ghost-inclusive per-column state; do not call on a restart."""
        if land_fraction.shape != hydro_w.shape[1:3]:
            raise ValueError(
                "Land fraction must have shape (nc3,nc2), including ghosts"
            )
        if bool(
            torch.any(
                (land_fraction < 0)
                | (land_fraction > 1)
                | ~torch.isfinite(land_fraction)
            )
        ):
            raise ValueError("Land fraction must be finite in [0,1]")
        fraction = land_fraction.to(hydro_w)
        temperature = torch.full_like(fraction, self.initial_temperature)
        return {
            "surface_land_fraction": fraction,
            self.kind + "_temperature": temperature,
            self.kind + "_temperature_start": temperature.clone(),
            self.kind + "_net_radiation": torch.full_like(fraction, self.net_radiation),
            self.kind
            + "_fluxes": torch.zeros(
                (8, fraction.size(0), fraction.size(1)),
                device=hydro_w.device,
                dtype=hydro_w.dtype,
            ),
        }

    def forward(
        self, var: Dict[str, torch.Tensor], dt: float, stage: int
    ) -> Dict[str, torch.Tensor]:
        """Return conserved increments; update existing surface state tensors in place."""
        stages = (
            3 if self.integrator == "rk3" else (2 if self.integrator == "rk2" else 1)
        )
        if dt < 0 or stage < 0 or stage >= stages:
            raise ValueError("Invalid timestep or RK stage")
        hydro = var["hydro_w"]
        if hydro.size(0) != self.gas_constants.numel() + 4:
            raise ValueError("Forcing EOS species count does not match hydro_w")
        temperature = var[self.kind + "_temperature"]
        start = var[self.kind + "_temperature_start"]
        if stage == 0:
            start.copy_(temperature)
        active = torch.ones_like(hydro[0], dtype=torch.bool)
        active[:, :, : self.nghost] = False
        active[:, :, -self.nghost :] = False
        if hydro.size(1) > 1:
            active[: self.nghost] = False
            active[-self.nghost :] = False
        if hydro.size(2) > 1:
            active[:, : self.nghost] = False
            active[:, -self.nghost :] = False
        if "solid" in var:
            active = active & ~var["solid"].to(torch.bool)
        lowest = active & (active.to(torch.int64).cumsum(-1) == 1)
        column_active = lowest.any(-1)
        index = lowest.to(torch.int64).argmax(-1).unsqueeze(-1)
        air = hydro.gather(
            3, index.unsqueeze(0).expand(hydro.size(0), -1, -1, -1)
        ).squeeze(-1)
        density = air[0]
        fractions = torch.cat(((1.0 - air[5:].sum(0)).unsqueeze(0), air[5:]), 0)
        gas = (fractions * self.gas_constants[:, None, None]).sum(0)
        cv = (fractions * self.heat_capacities[:, None, None]).sum(0)
        air_temperature = air[4] / (density * gas)
        speed = torch.sqrt(air[2].square() + air[3].square())
        vapor_pressure = 611.2 * torch.exp(
            self.latent_heat
            / self.gas_constants[1]
            * (1.0 / 273.15 - 1.0 / temperature)
        )
        if self.kind == "ocean":
            vapor_pressure = 0.98 * vapor_pressure
        vapor_pressure = torch.minimum(vapor_pressure, 0.99 * air[4])
        epsilon = self.gas_constants[0] / self.gas_constants[1]
        saturated = (
            epsilon * vapor_pressure / (air[4] - (1.0 - epsilon) * vapor_pressure)
        )
        evaporation = density * self.ce * speed * (saturated - air[5])
        evaporation = torch.where(
            evaporation > 0, evaporation * self.availability, evaporation
        )
        thickness = var["coord.dx1f"].reshape(-1)[index.squeeze(-1)]
        evaporation = torch.maximum(
            evaporation, -0.5 * density * air[5] * thickness / max(dt, 1.0e-30)
        )
        sensible = (
            density * (cv + gas) * self.ch * speed * (temperature - air_temperature)
        )
        latent = self.latent_heat * evaporation
        stress_x = -density * self.cd * speed * air[2]
        stress_y = -density * self.cd * speed * air[3]
        fraction = var["surface_land_fraction"]
        if self.kind == "ocean":
            fraction = 1.0 - fraction
        weight = fraction * column_active.to(fraction.dtype)
        radiation = var[self.kind + "_net_radiation"]
        increment = dt * (radiation - sensible - latent) / self.capacity
        initial_weight = 0.0
        current_weight = 1.0
        if self.integrator == "rk2" and stage == 1:
            initial_weight, current_weight = 0.5, 0.5
        elif self.integrator == "rk3" and stage == 1:
            initial_weight, current_weight = 0.75, 0.25
        elif self.integrator == "rk3" and stage == 2:
            initial_weight, current_weight = 1.0 / 3.0, 2.0 / 3.0
        next_temperature = initial_weight * start + current_weight * (
            temperature + increment
        )
        next_temperature = torch.where(weight > 0, next_temperature, temperature)
        if bool(torch.any(~torch.isfinite(next_temperature) | (next_temperature <= 0))):
            raise ValueError("Invalid surface temperature; reduce the timestep")
        source_temperature = torch.where(evaporation >= 0, temperature, air_temperature)
        vapor_enthalpy = (
            self.heat_capacities[1] + self.gas_constants[1]
        ) * source_temperature + self.vapor_reference_energy
        condensation = torch.minimum(evaporation, torch.zeros_like(evaporation))
        kinetic = 0.5 * (air[1].square() + air[2].square() + air[3].square())
        energy = sensible + evaporation * vapor_enthalpy + condensation * kinetic
        du = torch.zeros_like(hydro)
        scale = (
            dt * weight.unsqueeze(-1) * lowest.to(hydro.dtype) / thickness.unsqueeze(-1)
        )
        du[1] = (condensation * air[1]).unsqueeze(-1) * scale
        du[2] = (stress_x + condensation * air[2]).unsqueeze(-1) * scale
        du[3] = (stress_y + condensation * air[3]).unsqueeze(-1) * scale
        du[4] = energy.unsqueeze(-1) * scale
        du[5] = evaporation.unsqueeze(-1) * scale
        var[self.kind + "_fluxes"].copy_(
            torch.stack(
                (
                    sensible,
                    latent,
                    evaporation,
                    stress_x,
                    stress_y,
                    energy,
                    energy - sensible - latent,
                    radiation,
                )
            )
            * weight.unsqueeze(0)
        )
        temperature.copy_(next_temperature)
        return {"hydro_du": du}


class SlabOceanForcing(BulkSurfaceForcing):
    """Ocean mixing depth in metres; fixed-depth water reservoir."""

    def __init__(
        self,
        mixing_depth: float = 20.0,
        water_density: float = 1025.0,
        water_heat_capacity: float = 3990.0,
        **kwargs,
    ):
        if min(mixing_depth, water_density, water_heat_capacity) <= 0:
            raise ValueError("Ocean depth, density and heat capacity must be positive")
        super().__init__(
            "ocean", mixing_depth * water_density * water_heat_capacity, **kwargs
        )


class LandSurfaceForcing(BulkSurfaceForcing):
    """Areal heat capacity in J m^-2 K^-1 and fixed moisture availability."""

    def __init__(
        self, heat_capacity: float = 2.0e6, soil_moisture: float = 1.0, **kwargs
    ):
        super().__init__("land", heat_capacity, availability=soil_moisture, **kwargs)


class SurfaceStateTransaction:
    """Snapshot before an attempted full RK step; restore on retry or exception."""

    def __init__(self, blocks):
        self.blocks = blocks
        self.saved = [
            {
                key: value.clone()
                for key, value in block.items()
                if key.startswith(("ocean_", "land_", "surface_", "rt_"))
            }
            for block in blocks
        ]

    def restore(self):
        for block, saved in zip(self.blocks, self.saved):
            for key, value in saved.items():
                block[key].copy_(value)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is not None:
            self.restore()
