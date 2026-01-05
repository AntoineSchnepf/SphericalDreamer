import my_utils
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sample_circle(center, radius, normal, n_points, start_vector=None):
    """
    Sample points on a circle defined by (center, radius, normal).
    Points are ordered by increasing angle theta.
    
    Parameters
    ----------
    center : (3,) array
        Circle center in 3D.
    radius : float
        Circle radius.
    normal : (3,) array
        Normal vector of the circle plane.
    n_points : int
        Number of samples.
    start_vector : (3,) array, optional
        Reference vector in the plane to define theta=0 direction.
        If None, an arbitrary orthogonal vector is chosen.
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Find a vector orthogonal to the normal
    if start_vector is None:

        ref = np.array([0,0,1], dtype=float)
        u = np.cross(normal, ref)
    else:
        u = np.asarray(start_vector, dtype=float)
        # remove any component along normal
        u -= (u @ normal) * normal
    u /= np.linalg.norm(u)

    # v = n × u
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # Angles
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    # Points
    circle = center + radius*np.cos(theta)[:,None]*u + radius*np.sin(theta)[:,None]*v
    return circle

def sample_square(center, side_length, normal, n_points, start_vector=None):
    """
    Sample points on a square perimeter embedded in 3D.

    Convention (matches your circle convention):
      - theta=0 at the midpoint of the "right" edge: (a, 0) in (u,v) coords
      - theta increases CCW (sens trigonométrique) when looking along normal n
      - theta is arc-length along perimeter: theta in [0, 4*side_length)

    Returns:
      square_pts : (n_points, 3)
      thetas     : (n_points,)
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Build in-plane orthonormal basis (u,v).
    # u defines the "right" direction; v = n x u ensures CCW increase of theta.
    if start_vector is None:
        # choose an arbitrary vector not parallel to normal
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        u = np.cross(normal, ref)
        if np.linalg.norm(u) < 1e-12:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
            u = np.cross(normal, ref)
    else:
        u = np.asarray(start_vector, dtype=float)
        u -= (u @ normal) * normal  # project into plane

    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    a = 0.5 * side_length
    perimeter = 4.0 * side_length
    thetas = np.linspace(0.0, perimeter, n_points, endpoint=False)

    # We start at midpoint of right edge (a, 0) and go CCW:
    # Segment order (each length = side_length):
    # 0) right edge midpoint -> top-right corner      (y: 0 ->  a)
    # 1) top edge: top-right -> top-left             (x: a -> -a)
    # 2) left edge: top-left -> bottom-left          (y: a -> -a)
    # 3) bottom edge: bottom-left -> bottom-right    (x: -a -> a)
    # 4) bottom-right -> right edge midpoint         (y: -a -> 0) [completes loop]
    #
    # Note: this is a *closed* perimeter; with endpoint=False we don't duplicate the start.

    s = thetas
    pts_local = np.zeros((n_points, 2), dtype=float)

    # Segment 0: from (a, 0) to (a, a)
    beging = 0
    end = side_length/2
    m0 = (s >= beging) & (s < end)
    t0 = s[m0]
    pts_local[m0, 0] = a
    pts_local[m0, 1] = 0.0 + t0

    # Segment 1: from (a, a) to (-a, a)
    beging = end
    end = beging + side_length
    m1 = (s >= beging) & (s < end)
    t1 = s[m1] - beging
    pts_local[m1, 0] = a - t1
    pts_local[m1, 1] = a

    # Segment 2: from (-a, a) to (-a, -a)
    beging = end
    end = beging + side_length
    m2 = (s >= beging) & (s < end)
    t2 = s[m2] - beging

    pts_local[m2, 0] = -a
    pts_local[m2, 1] = a - t2

    # Segment 3: from (-a, -a) to (a, -a)
    beging = end
    end = beging + side_length
    m3 = (s >= beging) & (s < end)
    t3 = s[m3] - beging
    pts_local[m3, 0] = -a + t3
    pts_local[m3, 1] = -a

    # Segment 4: from (a, -a) to (a, 0)
    beging = end
    end = beging + side_length/2
    m4 = (s >= beging) & (s < end)
    t4 = s[m4] - beging
    pts_local[m4, 0] = a
    pts_local[m4, 1] = -a + t4

    # Lift to 3D
    square_pts = center + pts_local[:, 0, None] * u + pts_local[:, 1, None] * v
    return square_pts


def sample_star(center, r_outer, r_inner, n_branches, normal, n_points, start_vector=None):
    """
    Sample points along a regular star polygon perimeter embedded in 3D.

    Convention (same as circle/square):
      - theta=0 at the outer tip on the "right": center + r_outer * u
      - theta increases CCW when looking along normal n
      - theta is arc-length along the perimeter: theta in [0, perimeter)

    Parameters
    ----------
    center : (3,) array
    r_outer : float
        Outer radius (tip distance).
    r_inner : float
        Inner radius (indent distance). Must be < r_outer.
    n_branches : int
        Number of star arms (e.g., 5 for a classic star).
    normal : (3,) array
        Plane normal.
    n_points : int
        Number of samples along perimeter.
    start_vector : (3,) array, optional
        In-plane direction for "right". If None, chosen automatically.

    Returns
    -------
    star_pts : (n_points, 3)
    thetas   : (n_points,)
        Arc-length parameter along perimeter, starting at the rightmost outer tip.
    """
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    if r_inner <= 0 or r_outer <= 0 or r_inner >= r_outer:
        raise ValueError("Require 0 < r_inner < r_outer.")
    if n_branches < 2:
        raise ValueError("n_branches must be >= 2.")

    # Build in-plane basis (u,v), consistent with CCW theta increase.
    if start_vector is None:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        u = np.cross(normal, ref)
        if np.linalg.norm(u) < 1e-12:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
            u = np.cross(normal, ref)
    else:
        u = np.asarray(start_vector, dtype=float)
        u -= (u @ normal) * normal  # project into plane

    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # --- Build 2D star vertices in (u,v) coordinates ---
    # Outer tip at angle 0 => (r_outer, 0) => "middle-right".
    # Then alternate inner/outer vertices CCW.
    n_vertices = 2 * n_branches
    angles = np.arange(n_vertices) * (np.pi / n_branches)  # 0, pi/n, 2pi/n, ...
    radii = np.where(np.arange(n_vertices) % 2 == 0, r_outer, r_inner)

    verts_2d = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)  # (M,2)
    # verts_2d[0] == (r_outer, 0) by construction.

    # Close polygon
    verts_2d_closed = np.vstack([verts_2d, verts_2d[0:1]])

    # Edge lengths and cumulative arc-length
    edges = verts_2d_closed[1:] - verts_2d_closed[:-1]  # (M,2)
    edge_lens = np.linalg.norm(edges, axis=1)           # (M,)
    perimeter = edge_lens.sum()
    if perimeter <= 0:
        raise RuntimeError("Degenerate star perimeter.")

    cum = np.concatenate([[0.0], np.cumsum(edge_lens)])  # (M+1,)
    thetas = np.linspace(0.0, perimeter, n_points, endpoint=False)

    # For each theta, find which edge it falls on
    # idx in [0, M-1]
    idx = np.searchsorted(cum[1:], thetas, side="right")
    # Local interpolation factor along that edge
    t0 = cum[idx]
    seg_len = edge_lens[idx]
    alpha = (thetas - t0) / np.maximum(seg_len, 1e-12)

    p0 = verts_2d_closed[idx]
    p1 = verts_2d_closed[idx + 1]
    pts_2d = (1.0 - alpha)[:, None] * p0 + alpha[:, None] * p1  # (n_points,2)

    # Lift to 3D
    star_pts = center + pts_2d[:, 0, None] * u + pts_2d[:, 1, None] * v
    return star_pts

def get_points_colors(points):

    coords_min = points.min(axis=0)
    coords_max = points.max(axis=0)
    colors = (points - coords_min) / (coords_max - coords_min + 1e-12)
    return colors

def sample_sphere(n_points, radius):
    phi = np.random.rand(n_points) * 2*np.pi
    cos_theta = np.random.rand(n_points)*2 - 1
    theta = np.arccos(cos_theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.vstack([x,y,z]).T


    
    return points

def build_half_sphere_with_circle(radius, n_points_sphere, n_points_boundary, plane_x):
    # full sphere
    P = sample_sphere(n_points_sphere, radius)
    # keep only points below the plane (x <= plane_x)
    P_half = P[P[:,0] <= plane_x]

    circle_radius = np.sqrt(radius**2 - plane_x**2)
    circle_center = np.array([plane_x, 0, 0])
    circle_normal = np.array([1,0,0])  # normal of plane x=const
    circle = sample_circle(circle_center, circle_radius, circle_normal, n_points=n_points_boundary)


    return P_half, circle

def get_harmonic_toy_dset(
        sphere_radius=1.0, n_points_sphere=10000, n_points_boundary= 100, plane_x=0.80,
        target_circle_center=np.array([1.5, 0.0, 0.0]),
        target_circle_radius=0.4,
        target_circle_normal=np.array([1,0,0]),
        boundary_type="circle"
        ):
    
    # generate P data
    P, boundary = build_half_sphere_with_circle(sphere_radius, n_points_sphere, n_points_boundary, plane_x)

    # Generate target circle
    if boundary_type == "square":
        target_boundary = sample_square(target_circle_center, target_circle_radius, target_circle_normal, n_points=n_points_boundary)
    elif boundary_type == "circle":
        target_boundary = sample_circle(target_circle_center, target_circle_radius, target_circle_normal, n_points=n_points_boundary)
    elif boundary_type == "star":
        target_boundary = sample_star(
            center=target_circle_center,
            r_outer=target_circle_radius,
            r_inner=0.5*target_circle_radius,
            n_branches=5,
            normal=target_circle_normal,
            n_points=n_points_boundary
        )
    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")
    n1 = P.shape[0]
    n2 = boundary.shape[0]
    P_concat = np.concatenate([P, boundary], axis=0)
    mask_boundary = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
    colors = get_points_colors(P_concat)
    return P_concat, mask_boundary, colors, target_boundary, boundary

def plot_on_ax_before_def(ax, boundary, target_boundary, P, P_def, mask_fixed, colors, skip=10):
    # ax.plot(boundary[:,0], boundary[:,1], boundary[:,2], 
    #         color='purple', 
    #         label="boundary 1 (sphere cut)", linestyle=None, 
    #         marker='x', markersize=5)

    # viusalize fixed points
    ax.scatter(P[~mask_fixed][..., 0][::skip], P[~mask_fixed][..., 1][::skip], P[~mask_fixed][..., 2][::skip], s=3, alpha=0.3, c=colors[~mask_fixed][::skip], label="Deformable")
    ax.scatter(P[mask_fixed][..., 0][::skip], P[mask_fixed][..., 1][::skip], P[mask_fixed][..., 2][::skip], color='r', s=0.1, label="Fixed")
    ax.plot(target_boundary[:,0], target_boundary[:,1], target_boundary[:,2], color='g', label="Target")


    # ax.legend()
    ax.set_aspect('equal')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True

    from matplotlib.lines import Line2D


    # (Optional) also remove tick marks entirely
    # ax.tick_params(axis='both', which='both', length=0)

def plot_on_ax_after_def(ax, boundary, target_boundary, P, P_def, mask_fixed, colors, skip=10):
    
    ax.plot(target_boundary[:,0], target_boundary[:,1], target_boundary[:,2], color='g', label="Target")
    ax.scatter(P_def[~mask_fixed][..., 0][::skip], P_def[~mask_fixed][..., 1][::skip], P_def[~mask_fixed][..., 2][::skip], s=3, alpha=0.3, c=colors[~mask_fixed][::skip], label="Points")
    ax.scatter(P_def[mask_fixed][..., 0][::skip], P_def[mask_fixed][..., 1][::skip], P_def[mask_fixed][..., 2][::skip], color='r', s=0.1, label="Fixed points")
    ax.plot(P_def[mask_boundary][..., 0][::skip], P_def[mask_boundary][..., 1][::skip], P_def[mask_boundary][..., 2][::skip], color='orange', label="Deformed loop")


    # ax.legend()
    ax.set_aspect('equal')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True

    # (Optional) also remove tick marks entirely
    # ax.tick_params(axis='both', which='both', length=0)

if __name__ == "__main__":
    normal_types = {
        'flat': np.array([1,0,0]),
        'tilted': np.array([1,-0.7,-0.8]) 
    }
    normal_types['tilted'] /= np.linalg.norm(normal_types['tilted'])
    center_types = {
        'centered': np.array([1.5, 0.0, 0]),
        'offset': np.array([1.5, -0.7, -0.5])
    }
    shape_types = ['circle', 'square', 'star']

    where_save = "./Figures"
    target_normal_list = [
        normal_types['tilted'],
        normal_types['flat'],
        normal_types['flat'],
        normal_types['tilted'],
        normal_types['flat'],
        normal_types['tilted']
    ]
    target_centers_list = [
        center_types['offset'],
        center_types['centered'],
        center_types['centered'],
        center_types['offset'],
        center_types['centered'],
        center_types['offset']
    ]
    target_radius_list = [
        0.7,
        0.4,
        0.4*1.5,
        0.7*1.5,
        0.4,
        0.7
    ]
    target_shape_list = [
        "circle",
        "circle",
        "square",
        "square",
        "star",
        "star"
    ]
    fignames = [
        "circle_tilted_offset",
        "circle_flat_centered",
        "square_flat_centered",
        "square_tilted_offset",
        "star_flat_centered",
        "star_tilted_offset"
    ]
    for target_circle_normal, target_circle_center, target_circle_radius, boundary_type, figname in zip(
        target_normal_list,
        target_centers_list,
        target_radius_list,
        target_shape_list,
        fignames
        ):

        P, mask_boundary, colors, target_boundary, boundary = get_harmonic_toy_dset(
            n_points_sphere=1123200, 
            n_points_boundary=10000,
            target_circle_center=target_circle_center,
            target_circle_radius=target_circle_radius,
            target_circle_normal=target_circle_normal,
            boundary_type=boundary_type
            )

        x_fixed = 0.0
        mask_fixed = P[:,0] < x_fixed  # keep bottom hemisphere fixed
        pts_deform = P[ P[:,0] >= x_fixed]
        pts_fixed = P[ P[:,0]  < x_fixed]
        t0 = time.time()
        P_def, _ = my_utils.harmonic_deform_pipeline(
            P=P,
            mask_fixed=mask_fixed,
            mask_boundary=mask_boundary,
            target_boundary=target_boundary,
            n_coarse=10000,
            every=5,
            max_fixed=2000,
            k=10, m=3
        )
        t1 = time.time()
        print(f"Harmonic deformation took {t1 - t0:.1f}s")



        # viz
        skip = 1
        fig = plt.figure(figsize=(8, 4))

        # ---- First subplot
        ax1 = fig.add_subplot(121, projection='3d')
        plot_on_ax_before_def(
            ax1, boundary, target_boundary, P, P_def, mask_fixed, colors, skip=skip
        )

        # ---- Second subplot
        ax2 = fig.add_subplot(122, projection='3d')
        plot_on_ax_after_def(
            ax2, boundary, target_boundary, P, P_def, mask_fixed, colors, skip=skip
        )

        # ------------------------------------------------------------------
        # Shared legend (figure-level)
        # ------------------------------------------------------------------
        legend_handles = [
            Line2D([0], [0], marker='s', linestyle='None',
                markerfacecolor='red', markeredgecolor='red',
                markersize=8, label='Fixed'),

            Line2D([0], [0], marker='s', linestyle='None',
                markerfacecolor=(0.9, 0.6, 0.7), markeredgecolor='none',
                markersize=8, label='Deformable'),

            Line2D([0], [0], marker='s', linestyle='None',
                markerfacecolor='green', markeredgecolor='green',
                markersize=8, label='Target'),
        ]

        fig.legend(
            handles=legend_handles,
            loc='lower center',          # choose position
            bbox_to_anchor=(0.5, 0.15),  # fine control
            ncol=3,
            frameon=True
        )

        plt.savefig(f"{where_save}/x_HB_before_after_{figname}.png", dpi=300)
        plt.show()

        