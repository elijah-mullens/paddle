"""Reproduce the batched opacity note's illustrative gas and cloud mixtures."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paddle.rt.opacity import GasOpacity, CloudOpacity


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # pressure laws at fixed molar concentrations
    pres = torch.logspace(3, 6, 301, dtype=torch.float64)[None, :]
    temp = torch.full_like(pres, 300.0)
    conc = torch.tensor([1.0, 0.5, 0.2], dtype=torch.float64).expand(
        1, pres.shape[1], 3
    )
    masses, coefficients, exponents = (
        [0.002, 0.018, 0.044],
        [0.01, 0.1, 0.05],
        [0.0, 1.0, 2.0],
    )
    gas = GasOpacity([0, 1, 2], masses, coefficients, exponents)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for i in range(3):
        component = GasOpacity([i], [masses[i]], [coefficients[i]], [exponents[i]])
        ax.loglog(
            pres[0],
            component(conc, pres, temp)[0, 0, :, 0],
            label=f"Gas {i + 1}: exponent {exponents[i]:g}",
        )
    ax.loglog(
        pres[0],
        gas(conc, pres, temp)[0, 0, :, 0],
        color="black",
        linewidth=2,
        label="Mixture",
    )
    ax.axvline(1.0e5, color="grey", linestyle=":", label="Reference pressure")
    ax.set_xlabel("Total pressure (Pa)")
    ax.set_ylabel("Absorption coefficient (m⁻¹)")
    ax.set_title("Prescribed gas laws at fixed concentrations")
    ax.legend()
    ax.grid(alpha=0.2)

    # vary composition at fixed total condensate mass density
    fraction = torch.linspace(0.0, 1.0, 301, dtype=torch.float64)
    density = 1.0e-4
    cloud_conc = torch.stack(
        (density * (1 - fraction) / 0.018, density * fraction / 0.017), dim=-1
    )[None, :, :]
    cloud = CloudOpacity([0, 1], [0.018, 0.017], [10.0, 40.0], [0.4, 0.9], [0.2, 0.8])
    prop = cloud(cloud_conc, pres, temp)[0, 0]
    cloud_fig, axes = plt.subplots(
        3, 1, figsize=(8, 9), sharex=True, layout="constrained"
    )
    labels = ["Extinction (m⁻¹)", "Single-scattering albedo", "Asymmetry"]
    for i, axis in enumerate(axes):
        axis.plot(fraction, prop[:, i], linewidth=2, label="Optical mixture")
        axis.set_ylabel(labels[i])
        axis.grid(alpha=0.2)
    axes[1].plot(
        fraction,
        0.4 * (1 - fraction) + 0.9 * fraction,
        "--",
        color="grey",
        label="Mass-fraction interpolation",
    )
    axes[2].plot(fraction, 0.2 * (1 - fraction) + 0.8 * fraction, "--", color="grey")
    axes[1].legend()
    axes[0].set_title("Two clouds at fixed total condensate density: 10⁻⁴ kg/m³")
    axes[-1].set_xlabel("Mass fraction of condensate B")

    for figure, name in ((fig, "gas_pressure_laws"), (cloud_fig, "cloud_mixture")):
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"
            figure.savefig(path, dpi=160)
            if extension == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines())
                    + "\n"
                )
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures"
    )
    generate(parser.parse_args().output_dir)
