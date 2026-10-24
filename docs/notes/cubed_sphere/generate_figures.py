"""Draw cubed-sphere geometry and restart layouts from Paddle's functions."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paddle.cubed_sphere.cubed_sphere_utils import draw_panel_seam, plot_on_face
from paddle.cubed_sphere.cubed_sphere_remap import FACE_NAMES
from paddle.cubed_sphere.cubed_sphere_shrink import _zorder_coords


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    view = np.array([1.0, 1.0, 0.7])
    view /= np.linalg.norm(view)
    s = np.linspace(-np.pi / 4, np.pi / 4, 200)
    for i, face in enumerate(FACE_NAMES):
        for angle in np.linspace(-np.pi / 4, np.pi / 4, 9):
            plot_on_face(
                axes[0],
                np.full_like(s, angle),
                s,
                face,
                view,
                color=f"C{i}",
                linewidth=0.8,
            )
            plot_on_face(
                axes[0],
                s,
                np.full_like(s, angle),
                face,
                view,
                color=f"C{i}",
                linewidth=0.8,
            )
    axes[0].set_title("Equiangular grid on the visible hemisphere")
    draw_panel_seam(axes[1], N=8, nghost=2)
    axes[1].set_title("Adjacent faces and horizontal ghost cells")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlabel("Orthographic U")
        ax.set_ylabel("Orthographic W")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"geometry.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    n, m, g = 3, 4, 1
    ax = axes[0]
    for rank, (x, y) in enumerate(_zorder_coords(n)):
        ax.add_patch(
            Rectangle(
                (x * m, y * m),
                m,
                m,
                facecolor=f"C{rank%10}",
                alpha=0.35,
                edgecolor="black",
            )
        )
        ax.text((x + 0.5) * m, (y + 0.5) * m, str(rank), ha="center", va="center")
    x, y = 1, 1
    ax.add_patch(
        Rectangle(
            (x * m - g, y * m - g),
            m + 2 * g,
            m + 2 * g,
            fill=False,
            linestyle="--",
            linewidth=2,
        )
    )
    ax.add_patch(
        Rectangle((-g, -g), n * m + 2 * g, n * m + 2 * g, fill=False, linewidth=2)
    )
    ax.set(
        xlim=(-2, 14),
        ylim=(-2, 14),
        xlabel="Face x2 cell index",
        ylabel="Face x3 cell index",
        title="Restart tiles: filtered Morton order",
    )
    ax.set_aspect("equal")
    ax = axes[1]
    for i, face in enumerate(FACE_NAMES):
        row, col = divmod(i, 3)
        ax.add_patch(
            Rectangle(
                (col, row), 1, 1, facecolor=f"C{i}", alpha=0.35, edgecolor="black"
            )
        )
        ax.text(
            col + 0.5, row + 0.5, f"{i}: {face}", ha="center", va="center", fontsize=13
        )
    ax.set(
        xlim=(0, 3),
        ylim=(0, 2),
        xticks=[],
        yticks=[],
        xlabel="x2 →",
        ylabel="x3 →",
        title="Snapy NetCDF: 3 × 2 mosaic",
    )
    ax.set_aspect("equal")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"layouts.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("figures")
    )
    generate(parser.parse_args().output_dir)
