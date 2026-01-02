import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from cubed_sphere_utils import draw_panel_corner

plt.close("all")
fig = plt.figure(figsize=(9, 7))
ax = fig.add_axes([0.08, 0.18, 0.82, 0.75])
ax.set_aspect("equal")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_title(
    "Interactive orthographic view — drag sliders to rotate view_dir\n(+X solid, +Y dotted, +Z dash-dot; ghosts dashed)"
)

ax_az = fig.add_axes([0.08, 0.10, 0.6, 0.03])
ax_el = fig.add_axes([0.08, 0.06, 0.6, 0.03])
s_az = Slider(ax_az, "Azimuth (°)", 0, 360, valinit=45, valstep=1)
s_el = Slider(ax_el, "Elevation (°)", -85, 85, valinit=10, valstep=1)

ax_chk = fig.add_axes([0.72, 0.05, 0.18, 0.10])
checks = CheckButtons(ax_chk, labels=["Show +Z"], actives=[True])

ax_reset = fig.add_axes([0.65, 0.12, 0.1, 0.04])
btn_reset = Button(ax_reset, "Reset")


def viewdir_from_angles(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    c = np.cos(el)
    return np.array([c * np.cos(az), c * np.sin(az), np.sin(el)])


def redraw(_=None):
    ax.cla()
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(
        "Interactive orthographic view — drag sliders to rotate view_dir\n(+X solid, +Y dotted, +Z dash-dot; ghosts dashed)"
    )
    V = viewdir_from_angles(s_az.val, s_el.val)

    draw_panel_corner(ax, N=8, nghost=3, view_dir=V)

    fig.canvas.draw_idle()


def on_reset(event):
    s_az.reset()
    s_el.reset()
    if not checks.get_status()[0]:
        checks.set_active(0)


s_az.on_changed(redraw)
s_el.on_changed(redraw)
checks.on_clicked(redraw)
btn_reset.on_clicked(on_reset)

redraw()
plt.show()
