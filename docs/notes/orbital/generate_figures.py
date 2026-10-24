"""Reproduce the orbital technical note's figures from the public planet API."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paddle.orbital import Earth, Jupiter, Saturn, Uranus, Neptune


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = torch.linspace(0, 1, 721, dtype=torch.float64)
    lat = torch.deg2rad(torch.linspace(-89.5, 89.5, 180, dtype=torch.float64))[:, None]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    seasonal, panels = plt.subplots(
        5, 1, figsize=(10, 12), sharex=True, sharey=True, layout="constrained"
    )
    for planet_type, panel in zip((Earth, Jupiter, Saturn, Uranus, Neptune), panels):
        planet = planet_type()
        t = phase * planet.orbital_period
        _, _, r, _, dec = planet.insolation(torch.zeros_like(t), torch.zeros_like(t), t)
        flux = planet.stellar_luminosity / (4 * np.pi * r.square())
        reference = planet.stellar_luminosity / (4 * np.pi * planet.semi_major_axis**2)
        label = planet_type.__name__
        axes[0].plot(phase, r / planet.semi_major_axis, label=label)
        axes[1].plot(phase, flux, label=label)
        axes[2].plot(phase, dec.rad2deg(), label=label)
        h0 = torch.acos(torch.clamp(-lat.tan() * dec.tan(), -1, 1))
        daily = (
            flux
            / np.pi
            * (h0 * lat.sin() * dec.sin() + lat.cos() * dec.cos() * h0.sin())
        )
        mesh = panel.pcolormesh(
            phase,
            lat[:, 0].rad2deg(),
            daily / reference,
            shading="auto",
            vmin=0,
            vmax=1,
            cmap="magma",
        )
        panel.set_title(f"{label}   S(a) = {reference:.2f} W/m²", loc="left")
        panel.set_ylabel("Latitude (°)")
        panel.set_yticks([-60, 0, 60])
    axes[0].set_ylabel("Distance / semi-major axis")
    axes[0].legend(ncol=5, loc="upper center")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Normal irradiance (W/m²)")
    axes[2].set_ylabel("Substellar latitude (°)")
    axes[2].set_xlabel("Elapsed orbital periods since J2000")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
    panels[-1].set_xlabel("Elapsed orbital periods since J2000")
    seasonal.colorbar(
        mesh, ax=panels, label="Daily-mean horizontal flux / S(a)", shrink=0.75
    )
    for figure, name in ((fig, "orbital_cycles"), (seasonal, "daily_mean_insolation")):
        for extension in ("png", "svg"):
            # Rasterize the dense mesh in SVG to keep the artifact compact.
            if figure is seasonal:
                for panel in panels:
                    panel.collections[0].set_rasterized(True)
            figure.savefig(output_dir / f"{name}.{extension}", dpi=160)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    generate(parser.parse_args().output_dir)
