import matplotlib.pyplot as plt
from cubed_sphere_utils import (
    make_poly_patch,
    draw_single_panel,
    draw_panel_seam,
    draw_panel_corner,
)

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))

    # draw_single_panel(ax, N=6, nghost=3)
    draw_panel_seam(ax, N=6, nghost=2)
    # draw_panel_corner(ax, N=8, nghost=3)

    ax.set_aspect("equal")
    ax.set_xlabel("X (orthographic)")
    ax.set_ylabel("Y (orthographic)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    plt.show()
