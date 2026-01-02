import numpy as np
from matplotlib.patches import Polygon
from typing import Tuple


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def gnomonic_equiangular_to_xyz(alpha, beta, face="+X"):
    """
    Map equiangular gnomonic coordinates (alpha, beta) in radians to
    Cartesian coordinates (x, y, z) on the unit sphere for a given cube face.

    (alpha, beta) are the equiangular angles from the face center:
        t = tan(alpha), u = tan(beta)
    Faces: "+X", "-X", "+Y", "-Y", "+Z", "-Z"
    """
    t = np.tan(alpha)
    u = np.tan(beta)

    if face == "+X":
        X, Y, Z = np.ones_like(t), t, u
    elif face == "-X":
        X, Y, Z = -np.ones_like(t), -t, u
    elif face == "+Y":
        X, Y, Z = -t, np.ones_like(t), u
    elif face == "-Y":
        X, Y, Z = t, -np.ones_like(t), u
    elif face == "+Z":
        X, Y, Z = -u, t, np.ones_like(t)
    elif face == "-Z":
        X, Y, Z = u, t, -np.ones_like(t)
    else:
        raise ValueError("Invalid face specifier")

    inv_norm = 1.0 / np.sqrt(X * X + Y * Y + Z * Z)
    return X * inv_norm, Y * inv_norm, Z * inv_norm


def orthographic_project(face_xyz, view_dir=np.array([1, 0, 0])):
    """Orthographic projection onto the plane normal to view_dir."""
    x, y, z = face_xyz
    V = normalize(np.array(view_dir))
    # Construct orthonormal basis (e1,e2) spanning plane perpendicular to V
    # Choose arbitrary "up" vector not parallel to V
    up_guess = np.array([0, 0, 1]) if abs(V[2]) < 0.9 else np.array([0, 1, 0])
    e1 = normalize(np.cross(up_guess, V))
    e2 = np.cross(V, e1)
    # Project coordinates
    U = x * e1[0] + y * e1[1] + z * e1[2]
    W = x * e2[0] + y * e2[1] + z * e2[2]
    return U, W


def visible_segments(u, v, depth, vis_mask):
    segs = []
    start = None
    for i in range(len(u)):
        if vis_mask[i] and start is None:
            start = i
        elif (not vis_mask[i]) and (start is not None):
            segs.append((u[start:i], v[start:i], depth[start:i]))
            start = None
    if start is not None:
        segs.append((u[start:], v[start:], depth[start:]))
    return segs


def plot_on_face(ax, alpha, beta, face="+X", view_dir=np.array([1, 0, 0]), **kwargs):
    x, y, z = gnomonic_equiangular_to_xyz(alpha, beta, face=face)
    # r·V (nearness for ortho)
    depth = x * view_dir[0] + y * view_dir[1] + z * view_dir[2]

    vis = depth > 0  # front hemisphere
    u, v = orthographic_project((x, y, z), view_dir=view_dir)
    segs = visible_segments(u, v, depth, vis)
    segs.sort(key=lambda seg: np.max(seg[2]) if seg[2].size else -1)
    for uu, vv, dd in segs:
        ax.plot(uu, vv, zorder=np.max(dd) if dd.size else 0, **kwargs)


def scatter_on_face(ax, alpha, beta, face="+X", view_dir=np.array([1, 0, 0]), **kwargs):
    x, y, z = gnomonic_equiangular_to_xyz(alpha, beta, face=face)
    # r·V (nearness for ortho)
    depth = x * view_dir[0] + y * view_dir[1] + z * view_dir[2]

    vis = depth > 0  # front hemisphere
    u, v = orthographic_project((x, y, z), view_dir=view_dir)
    ax.scatter(u[vis], v[vis], zorder=np.max(depth[vis]), **kwargs)


def panel_ab_limits(
    dxy, N, nghost, exterior=True
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return ((a_min, b_min), (a_max, b_max)) in equiangular gnomonic coords (radians)
    for a single cubed-sphere panel.

    Panel interior spans α,β ∈ [-π/4, +π/4]. Each cell has angular size
    dθ = (π/2) / N. Ghost zones extend outward by 'nghost' cells.

    Offsets (dx, dy) ∈ {-1, 0, 1} select a slab/corner in each dimension:
      - dx = -1: left
      - dx =  0: center
      - dx = +1: right
      (analogous for dy: bottom/center/top)

    exterior=True:
      - (0, 0) returns FULL exterior limits including ghost zones
      - (-1, 0) returns the LEFT ghost region only (width = nghost * dθ), etc.

    exterior=False:
      - (0, 0) returns interior limits ONLY
      - (-1, 0) returns the interior boundary slab adjacent to the left edge,
        with the SAME thickness as the ghost zone (nghost * dθ) but inside.
        (Analogous for other offsets and for dy.)

    Returns:
      ((a_min, b_min), (a_max, b_max))
    """
    dx, dy = dxy
    if dx not in (-1, 0, 1) or dy not in (-1, 0, 1):
        raise ValueError("dx, dy must be in {-1, 0, 1}")
    if N <= 0 or nghost < 0:
        raise ValueError("N must be > 0 and nghost >= 0")

    dtheta = (np.pi / 2.0) / N
    aL_int, aR_int = -np.pi / 4, +np.pi / 4
    bB_int, bT_int = -np.pi / 4, +np.pi / 4

    # External (interior + ghosts) bounds along each dimension
    aL_ext = aL_int - nghost * dtheta
    aR_ext = aR_int + nghost * dtheta
    bB_ext = bB_int - nghost * dtheta
    bT_ext = bT_int + nghost * dtheta

    def one_dim_limits(
        offset: int, L_int: float, R_int: float, L_ext: float, R_ext: float
    ):
        if offset == 0:  # full span incl. ghosts
            return (L_int, R_int)
        if exterior:
            if offset == -1:  # left ghost slab
                return (L_ext, L_int)
            else:  # +1: right ghost slab
                return (R_int, R_ext)
        else:
            if offset == -1:  # interior boundary slab (left), same thickness as ghosts
                return (L_int, L_int + nghost * dtheta)
            else:  # +1: interior boundary slab (right)
                return (R_int - nghost * dtheta, R_int)

    a_min, a_max = one_dim_limits(dx, aL_int, aR_int, aL_ext, aR_ext)
    b_min, b_max = one_dim_limits(dy, bB_int, bT_int, bB_ext, bT_ext)

    # set to whole domain
    if dx == 0 and dy == 0 and exterior:
        a_min, a_max = aL_ext, aR_ext
        b_min, b_max = bB_ext, bT_ext

    # Ensure (lower-left, upper-right) ordering
    a0, a1 = (min(a_min, a_max), max(a_min, a_max))
    b0, b1 = (min(b_min, b_max), max(b_min, b_max))
    return (a0, b0), (a1, b1)


def sample_edge(a0, b0, a1, b1, n_pts=64):
    t = np.linspace(0.0, 1.0, n_pts)
    return (a0 + (a1 - a0) * t), (b0 + (b1 - b0) * t)


def make_poly_patch(verts_ab, face="+X", view_dir=(1, 0, 0), n_pts=64, **kwargs):
    # Sample each edge, map to sphere, cull back hemisphere
    boundary_u, boundary_v, boundary_d = [], [], []
    for (a0, b0), (a1, b1) in zip(verts_ab, verts_ab[1:] + verts_ab[:1]):
        aa, bb = sample_edge(a0, b0, a1, b1, n_pts=n_pts)
        xyz = gnomonic_equiangular_to_xyz(aa, bb, face=face)
        u, v = orthographic_project(xyz, view_dir=view_dir)
        d = xyz[0] * view_dir[0] + xyz[1] * view_dir[1] + xyz[2] * view_dir[2]
        mask = d > 0
        boundary_u.append(u[mask])
        boundary_v.append(v[mask])
        boundary_d.append(d[mask])

    if not any(len(u) for u in boundary_u):
        return None  # fully occluded

    U = np.concatenate(boundary_u)
    V = np.concatenate(boundary_v)
    D = (
        np.concatenate(boundary_d)
        if any(len(d) for d in boundary_d)
        else np.array([0.0])
    )

    poly = Polygon(np.c_[U, V], closed=True, zorder=1 + float(np.max(D)), **kwargs)
    return poly


def draw_panel_grid(
    ax,
    face="+X",
    N=8,
    nghost=3,
    n_pts=800,
    view_dir=np.array([1, 0, 0]),
    color="C0",
    linestyle="--",
    linewidth=0.8,
    facecolor="none",
):
    """
    Plot an equiangular gnomonic grid for a single cubed-sphere panel with ghost zones.
      - N: number of interior cells per direction (there are N+1 interior grid lines).
      - nghost: number of extra ghost cells beyond each edge (drawn dashed).
    """
    # Uniform grid in equiangular coordinates; panel spans [-pi/4, +pi/4].
    dtheta = (np.pi / 2) / N  # cell size in angle

    # Grid-line indices: i = -N/2 ... N/2 for interior; extend by nghost
    halfN = N // 2
    idx = np.arange(-halfN - nghost, halfN + nghost + 1)
    alphas = idx * dtheta  # positions of grid lines
    centers = 0.5 * (alphas[1:] + alphas[:-1])  # cell centers

    # Limit plotting domain to a slightly larger band so dashed ghost lines are visible
    s = np.linspace(-np.pi / 4 - nghost * dtheta, np.pi / 4 + nghost * dtheta, n_pts)

    # alpha = const lines
    for i, alpha in zip(idx, alphas):
        # skip the first and the last lines
        if i == idx[0] or i == idx[-1]:
            continue
        plot_on_face(
            ax,
            np.full_like(s, alpha),
            s,
            face=face,
            view_dir=view_dir,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
        )

    # beta = const lines
    for j, beta in zip(idx, alphas):
        # skip the first and the last lines
        if j == idx[0] or j == idx[-1]:
            continue
        plot_on_face(
            ax,
            s,
            np.full_like(s, beta),
            face=face,
            view_dir=view_dir,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
        )

    # cell centers (solid dots)
    center_a, center_b = np.meshgrid(centers, centers)
    scatter_on_face(
        ax,
        center_a.flatten(),
        center_b.flatten(),
        face=face,
        view_dir=view_dir,
        s=10,
        facecolors=facecolor,
        edgecolors=color,
    )

    # Interior panel boundary (bold): |alpha|=pi/4 and |beta|=pi/4
    for alpha in [-np.pi / 4, np.pi / 4]:
        plot_on_face(
            ax,
            np.full_like(s, alpha),
            s,
            face=face,
            view_dir=view_dir,
            linestyle="-.",
            linewidth=1.6,
            color=color,
        )
    for beta in [-np.pi / 4, np.pi / 4]:
        plot_on_face(
            ax,
            s,
            np.full_like(s, beta),
            face=face,
            view_dir=view_dir,
            linestyle="-.",
            linewidth=1.6,
            color=color,
        )


def draw_single_panel(
    ax, face="+X", N=8, nghost=3, color="C0", view_dir=np.array([1, 0, 0])
):
    # all grid lines including ghosts
    draw_panel_grid(ax, face=face, N=N, nghost=nghost, color=color, view_dir=view_dir)

    # interior grid lines only
    draw_panel_grid(
        ax,
        face=face,
        N=N,
        nghost=0,
        view_dir=view_dir,
        linestyle="-",
        linewidth=1.2,
        facecolor=color,
        color=color,
    )


def draw_panel_seam(ax, N=8, nghost=3, view_dir=np.array([1.0, 1.0, 0.0])):
    draw_single_panel(ax, "+X", N=N, nghost=nghost, view_dir=view_dir, color="C0")
    draw_single_panel(ax, "+Y", N=N, nghost=nghost, view_dir=view_dir, color="C1")

    # ghost zone patches
    (a0, b0), (a1, b1) = panel_ab_limits((1, 0), N=N, nghost=nghost, exterior=True)
    verts_box = [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]
    poly = make_poly_patch(
        verts_box,
        face="+X",
        view_dir=view_dir,
        edgecolor="k",
        # grey with some transparency
        facecolor=(0.5, 0.5, 0.5, 0.3),
        linewidth=1.2,
    )
    ax.add_patch(poly)


def draw_panel_corner(ax, N=8, nghost=3, view_dir=np.array([1.0, 1.0, 1.0])):
    draw_single_panel(ax, "+X", N=N, nghost=nghost, view_dir=view_dir, color="C0")
    draw_single_panel(ax, "+Y", N=N, nghost=nghost, view_dir=view_dir, color="C1")
    draw_single_panel(ax, "+Z", N=N, nghost=nghost, view_dir=view_dir, color="C2")

    # ghost zone patches
    (a0, b0), (a1, b1) = panel_ab_limits((1, 1), N=N, nghost=nghost, exterior=True)
    verts_box = [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]
    poly = make_poly_patch(
        verts_box,
        face="+X",
        view_dir=view_dir,
        edgecolor="k",
        # grey with some transparency
        facecolor=(0.5, 0.5, 0.5, 0.3),
        linewidth=1.2,
    )
    ax.add_patch(poly)
