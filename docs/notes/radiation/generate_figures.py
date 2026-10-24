"""Plot the radiation data flow and diagnostics from uranus_radiation.py."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch


def generate(diagnostics: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flow, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    boxes = [
        (0.3, 3.5, "Orbit\nbeam, zenith cosine"),
        (3.7, 3.5, "Atmospheric state\npressure, temperature, species"),
        (7.1, 3.5, "Gas + cloud opacity\nextinction, albedo, asymmetry"),
        (3.7, 1.8, "PyHarp radiation\nnet face fluxes"),
        (3.7, 0.1, "RTState → stage forcing\nflux divergence → energy increment"),
    ]
    for x, y, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                2.6,
                0.9,
                boxstyle="round,pad=0.1",
                facecolor="#e9f1f7",
                edgecolor="#356486",
            )
        )
        ax.text(x + 1.3, y + 0.45, label, ha="center", va="center", fontsize=9)
    for start, end in [
        ((1.6, 3.4), (3.6, 2.35)),
        ((6.4, 3.95), (7.0, 3.95)),
        ((5.0, 3.4), (5.0, 2.8)),
        ((8.4, 3.4), (6.4, 2.35)),
        ((5.0, 1.7), (5.0, 1.1)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=14, color="#356486"
            )
        )
    ax.set_title("From orbital illumination to atmospheric heating")

    data = torch.load(diagnostics, map_location="cpu", weights_only=True)
    blocks = data["blocks"]
    lon = torch.cat([b["lon"].flatten() for b in blocks])
    lat = torch.cat([b["lat"].flatten() for b in blocks])
    incident = torch.cat([(b["fbeam"].flatten() * b["umu0"]).flatten() for b in blocks])
    heating = torch.cat([b["heating"] for b in blocks])
    lon = (lon + torch.pi) % (2 * torch.pi) - torch.pi
    figure, axes = plt.subplots(2, 1, figsize=(9, 9), layout="constrained")
    scatter = axes[0].scatter(
        lon.rad2deg(), lat.rad2deg(), c=incident, cmap="magma", s=24
    )
    axes[0].set(
        xlabel="East longitude (°)",
        ylabel="Planetocentric latitude (°)",
        title=f"Uranus illumination at t = {data['time']:g} s from J2000",
        xlim=(-180, 180),
        ylim=(-90, 90),
    )
    figure.colorbar(scatter, ax=axes[0], label="Horizontal incident flux (W/m²)")
    for index, label in (
        (int(incident.argmax()), "Most illuminated column"),
        (int(incident.argmin()), "Night-side column"),
    ):
        axes[1].plot(heating[index], torch.arange(heating.shape[1]), label=label)
    axes[1].set(
        xlabel="Volumetric radiative heating (W/m³)",
        ylabel="Layer index (bottom → top)",
        title="Final radiative heating",
    )
    axes[1].axvline(0.0, color="grey", linewidth=0.7)
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    for fig, name in ((flow, "coupling"), (figure, "uranus_heating")):
        for ext in ("png", "svg"):
            path = output_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=160)
            if ext == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    args = parser.parse_args()
    generate(args.diagnostics, args.output_dir)
