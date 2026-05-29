import time
import my_utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata as interp_grid
from skimage.restoration import denoise_tv_chambolle

# -------------------------------------------------- #
# ------ Code for post-splatting interpolation ------ #
# -------------------------------------------------- #

def interpolate_missing_pixels(
    warped_img,        # (H, W, 3) float32 RGB in [0,1]
    warped_depth,      # (H, W)     float32 depth
    visited_pixels,    # (H, W)     bool mask: True = valid sample, False = missing
    method="linear",   # interpolation method for interp_grid
    fill_value=0.0    # passed to interp_grid; e.g., 0 or np.nan
):
    """
    Fill missing pixels (visited_pixels == False) by interpolating from the
    available samples on a regular grid.

    How it works
    ------------
    1) Collect known sample locations (u = x/column, v = y/row) from visited_pixels.
    2) Build a dense grid of all pixel centers of size (W*H, 2).
    3) Interpolate colors and depth at every grid location using `interp_grid`.

    Returns
    -------
    image_interp : (H, W, 3) float32
        Interpolated RGB image (defined everywhere on the grid).
    depth_interp : (H, W) float32
        Interpolated depth map (defined everywhere on the grid).

    Notes
    -----
    - Coordinates follow (u, v) = (x, y) convention (u: column, v: row).
    - Requires an `interp_grid(points, values, grid, method=..., fill_value=...)`
      function to be available in scope.
    - If there are no valid samples (no True in visited_pixels), this will raise.
    """
    # Basic checks & shapes
    assert warped_img.ndim == 3 and warped_img.shape[-1] == 3, "warped_img must be (H,W,3)"
    assert warped_depth.ndim == 2, "warped_depth must be (H,W)"
    assert visited_pixels.shape == warped_depth.shape == warped_img.shape[:2], \
        "All inputs must share (H,W)"
    height, width = warped_depth.shape

    # 1) Collect known sample locations and their values
    v_idx, u_idx = np.where(visited_pixels)  # rows (y), cols (x)
    if v_idx.size == 0:
        raise ValueError("No valid samples: visited_pixels has no True entries.")

    # points: (N,2) as (u, v) = (x, y)
    points = np.stack([u_idx.astype(np.float32), v_idx.astype(np.float32)], axis=1)

    # values at those points
    colors_samples = warped_img[visited_pixels].astype(np.float32)   # (N,3)
    depths_samples = warped_depth[visited_pixels].astype(np.float32) # (N,)

    # 2) Build the dense output grid of pixel centers (u, v)
    # meshgrid returns X with shape (H,W) varying along columns, Y varying along rows.
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    grid = np.stack([uu, vv], axis=-1).reshape(-1, 2)  # (H*W, 2)

    # 3) Interpolate colors and depth on the grid
    image_interp = interp_grid(
        points,
        colors_samples,
        grid,
        method=method,
        fill_value=fill_value
    ).reshape(height, width, 3).astype(np.float32)

    depth_interp = interp_grid(
        points,
        depths_samples,
        grid,
        method=method,
        fill_value=fill_value
    ).reshape(height, width).astype(np.float32)

    return image_interp, depth_interp

# ---------------------- #
# ----V0 of rendering--- # 
# ---------------------- #

# Use a z-buffer splatting approach.
def prepare_coords(coord_cam2, height, width, **additional_data):
    """
    Prepare coordinates for splatting:
    - Round to nearest integer pixel
    - Keep only those that fall inside the target frame
    All additional data (colors, depths, etc.) are filtered accordingly.
    coord_cam2 and additional_data must be numpy arrays of shape [N, ...].
    """
    assert coord_cam2 is not None
    colors = additional_data.get('colors')
    coord_cam1 = additional_data.get('coord_cam1')
    depth_cam2 = additional_data.get('depth_cam2')
    ldi_mask = additional_data.get('ldi_mask')

    # Round target coordinates to nearest integer pixel (u -> x/col, v -> y/row)
    u = coord_cam2[:, 0]
    v = coord_cam2[:, 1]
    u_r = np.rint(u).astype(np.int32)
    v_r = np.rint(v).astype(np.int32)

    # Keep only those that fall inside the target frame
    in_bounds = (u_r >= 0) & (u_r < width) & (v_r >= 0) & (v_r < height)
    assert np.any(in_bounds), "No points project inside the target frame!"

    # Restrict to valid points
    out = {
        'in_bounds': in_bounds,
        'u_r': u_r[in_bounds],
        'v_r': v_r[in_bounds],
        'u_float': u[in_bounds],
        'v_float': v[in_bounds],
    }

    # Prepare additional optional data
    if colors is not None:
        out['colors'] = colors[in_bounds]
    if depth_cam2 is not None:
        out['depth_cam2'] = depth_cam2[in_bounds].astype(np.float32)
    if coord_cam1 is not None:
        out['coord_cam1'] = coord_cam1[in_bounds]
    if ldi_mask is not None:
        out['ldi_mask'] = ldi_mask[in_bounds]
    
    return out

def get_winners_z_buffer_splatting(depth_cam2, coord_cam2, height, width):
    """
    z-buffer splatting to find winning points per pixel.
    args:
        - depth_cam2: np.array of shape [N,] with float values
        - coord_cam2: np.array of shape [N, 2] with float values
    """

    # Prepare coordinates (rounding, in-bounds filtering)
    out = prepare_coords(coord_cam2, height, width, depth_cam2=depth_cam2)

    u_r = out['u_r']
    v_r = out['v_r']
    depth_cam2 = out['depth_cam2']

    # Linearized target indices (row-major)
    tgt_lin = (v_r.astype(np.int64) * width + u_r.astype(np.int64))

    # Resolve collisions per target pixel with z-buffer: keep the *nearest* depth
    order = np.lexsort((depth_cam2, tgt_lin))        # primary: tgt_lin, secondary: depth (ascending)
    tgt_sorted = tgt_lin[order]
    _, first_idx = np.unique(tgt_sorted, return_index=True)
    winners = order[first_idx]

    # Winners' data
    return winners

def splat_with_z_buffer(colors, ldi_mask, depth_cam2, coord_cam2, height, width):
    """
    This function takes as input: 
        - `colors` an array of colors (N, 3) in [0-1]
        - `ldi_mask` an array of booleans (N,) indicating whether the point is from LDI or foreground
        - `depth_cam2` an array of depth values (N,) in [0-1]
        - `coord_cam2` an array of 2D coordinates in the image frame (N, 2) in pixel coordinates
    From this information, is returned 
        - the warped_image (no interpolation)
        - the warped_depth (no interpolation)
        - the binary mask of visited pixels
        - the interpolated_image (with interpolation)
        - the interpolated_depth (with interpolation)
    """

    # Basic checks
    assert colors.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert colors.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape

    # Flatten to [N, ...]
    colors   = colors.reshape((-1, 3))
    ldi_mask = ldi_mask.reshape((-1,))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    # Prepare coordinates (rounding, in-bounds filtering)
    out = prepare_coords(coord_cam2, height, width, colors=colors, ldi_mask=ldi_mask, depth_cam2=depth_cam2)

    # --- I. Z-buffer Splatting to find winners ----
    winners = get_winners_z_buffer_splatting(depth_cam2, coord_cam2, height, width)

    # Winners' data
    u_win_r = out['u_r'][winners]
    v_win_r = out['v_r'][winners]
    depths_win = out['depth_cam2'][winners]
    colors_win = out['colors'][winners]   
    ldi_mask_win = out['ldi_mask'][winners]
    u_win_f = out['u_float'][winners]       
    v_win_f = out['v_float'][winners]  

    # ---- II. Allocate outputs -----
    # 1. Visited mask
    visited = np.zeros((height, width), dtype=bool)
    visited[v_win_r, u_win_r] = True
    is_visited_ldi = np.zeros((height, width), dtype=bool)
    is_visited_ldi[v_win_r, u_win_r] = ldi_mask_win

    # 2. Naive without Interpolation
    warped_img   = np.zeros((height, width, 3), dtype=np.float32)
    warped_depth = np.full((height, width), 0.0, dtype=np.float32)
    warped_img[v_win_r, u_win_r]   = colors_win
    warped_depth[v_win_r, u_win_r] = depths_win

    return warped_img, warped_depth, visited, is_visited_ldi

def render_v0(all_pts_world, all_colors_world, all_ldi_mask, pose, height, width):

    # convert to ERP + Depth representation
    points_3D_cam2_sph = my_utils.world2cam_sph_3D(all_pts_world, pose)  
    depth_cam2 = points_3D_cam2_sph[..., 2] # [N,] 
    points_2D_cam2_sph = points_3D_cam2_sph[..., :2]
    points_2D_cam2_erp = my_utils.sph2erp_2D(points_2D_cam2_sph, height, width)  # [N, 2]

    # Splatting 
    warped_img, warped_depth, visited_pixels, is_visited_ldi = splat_with_z_buffer(
        colors=all_colors_world,
        ldi_mask=all_ldi_mask,
        depth_cam2=depth_cam2,
        coord_cam2=points_2D_cam2_erp,
        height=height,
        width=width,
    )
    
    # Post-splatting interpolation
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels,
    )

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels, is_visited_ldi

def render_perspective_v0(all_pts_world, all_colors_world, 
                          pose, height, width, fx, fy, cx, cy,):

    # World -> camera (Cartesian)
    points_3D_cam_carte = my_utils.world2cam_carte_3D(all_pts_world, pose)  # (N,3)

    # 3D -> perspective pixel coords + depth along the camera optical axis (+X)
    coords_persp, depths_persp = my_utils.carte2persp_3D(
        points_3D_cam_carte,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )  # (N,2)
    coords_persp = coords_persp.astype(np.float32)
    depths_persp = depths_persp.astype(np.float32)

    # Splatting 
    warped_img, warped_depth, visited_pixels = splat_with_z_buffer(
        colors=all_colors_world,
        depth_cam2=depths_persp,
        coord_cam2=coords_persp,
        height=height,
        width=width,
    )
    
    # Post-splatting interpolation
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels,
    )

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels

# ---------------------- #
# ----V1 of rendering--- # 
# ---------------------- #

# This implementation performs interpolation *during* splatting by blending
# all points that fall within a circular neighborhood around each pixel.
# Each point contributes according to a depth-dependent weight, giving higher
# influence to closer (foreground) samples — effectively modeling occlusion.
# A final interpolation pass fills any small remaining gaps in the image.

def _disk_offsets(R):
    """Integer (dx,dy) offsets inside a radius-R disk, including center."""
    r2 = R * R
    dxy = []
    for dy in range(-R, R + 1):
        # circle cap for this row
        max_dx = int(np.floor(np.sqrt(max(0, r2 - dy*dy))))
        for dx in range(-max_dx, max_dx + 1):
            dxy.append((dx, dy))
    offs = np.array(dxy, dtype=np.int32)
    return offs  # (M,2)

def _spatial_weights(u_frac, v_frac, dx, dy, R, mode="tent", sigma=None):
    """
    Compute spatial kernel at target pixels using subpixel distance.
    u_frac, v_frac: float arrays of fractional parts in [0,1)
    dx, dy: integer offsets arrays broadcastable to (#pts, M)
    """
    # Subpixel distance from projected center to pixel center
    # Pixel center is at integer coords; projected point at (floor(u)+u_frac, floor(v)+v_frac)
    du = (dx.astype(np.float32) - (1.0 - u_frac))  # distance from pixel center to point
    dv = (dy.astype(np.float32) - (1.0 - v_frac))
    r = np.sqrt(du*du + dv*dv)

    if mode == "tent":
        w = np.maximum(0.0, 1.0 - r / max(1e-6, R))
    elif mode == "gauss":
        if sigma is None:
            sigma = max(1e-6, R / 2.0)
        w = np.exp(-0.5 * (r / sigma) ** 2)
    else:
        raise ValueError(f"Unknown spatial kernel: {mode}")
    return w

def splat_with_weighted_disk(
    colors,          # (N,3), float32 in [0,1]
    depths,          # (N,),  float32, smaller = closer
    coords,          # (N,2), float32 (u,v) pixel coords
    height, width,
    radius=2,                           # integer pixels
    spatial_mode="tent",                # 'tent' or 'gauss'
    spatial_sigma=None,                 # used if 'gauss'
    depth_mode="exp",                   # 'exp' | 'linear' | 'softmax'
    tau=0.05,                           # depth scale (same units as depths)
    chunk_size=200000,                  # points per chunk to limit RAM
    return_depth=True,                  # also output blended depth
    eps=1e-8
):
    """
    Disk-gather renderer with depth-aware weights.

    Two-pass algorithm:
      Pass 1: per-pixel Dmin within radius
      Pass 2: accumulate colors using spatial * depth kernels (or softmax)

    depth_mode:
      - 'exp'   : w_d = exp(-(d - Dmin)/tau)          (relative exponential)
      - 'linear': w_d = max(0, 1 - (d - Dmin)/tau)    (relative linear)
      - 'softmax':
           Accumulate softmax over depth directly:
           num_logits += exp(-d/tau) * w_spatial
           img = numC / (num_logits + eps)
           (ignores Dmin)

    Returns:
      img: (H,W,3) float32
      dep: (H,W)   float32 (weighted by the same final weights) if return_depth else None
      visited: (H,W) bool
    """
    assert colors.shape[-1] == 3 and coords.shape[-1] == 2
    N = colors.shape[0]
    assert depths.shape[0] == N and coords.shape[0] == N

    colors = colors.astype(np.float32, copy=False)
    depths = depths.astype(np.float32, copy=False)
    coords = coords.astype(np.float32, copy=False)

    # Prepare outputs
    Dmin = np.full((height, width), np.inf, dtype=np.float32)

    # Precompute disk offsets
    offs = _disk_offsets(int(radius))      # (M,2)
    dx_all = offs[:, 0]
    dy_all = offs[:, 1]
    M = offs.shape[0]

    # Precompute integer anchor (floor) and fractional parts
    u = coords[:, 0]
    v = coords[:, 1]
    uf = np.floor(u).astype(np.int32)
    vf = np.floor(v).astype(np.int32)
    u_frac = u - uf    # fractional part in [0,1)
    v_frac = v - vf

    # -------- Pass 1: nearest depth per pixel --------
    # Chunked for memory friendliness
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        uf_c = uf[start:end][:, None] + dx_all[None, :]    # (#pts_chunk, M)
        vf_c = vf[start:end][:, None] + dy_all[None, :]

        # In-bounds mask
        inb = (uf_c >= 0) & (uf_c < width) & (vf_c >= 0) & (vf_c < height)

        # Flatten valid indices
        if not np.any(inb):
            continue
        ys = vf_c[inb].astype(np.intp)
        xs = uf_c[inb].astype(np.intp)
        dvals = np.repeat(depths[start:end], M)[inb.ravel()]

        # Per-pixel min
        # We can’t do true segmented min without sorting; scatter loop is OK because
        # #valid per chunk is ~N * disk_area, and this is just a few million ops.
        Dmin[ys, xs] = np.minimum(Dmin[ys, xs], dvals)

    # Pixels with no neighbor keep +inf (we'll mark visited later)

    # -------- Pass 2: accumulate colors with depth bias --------
    num = np.zeros((height, width, 3), dtype=np.float32)
    den = np.zeros((height, width), dtype=np.float32)
    dep_num = np.zeros((height, width), dtype=np.float32) if return_depth else None

    use_softmax = (depth_mode == "softmax")
    if use_softmax:
        logits_sum = np.zeros((height, width), dtype=np.float32)  # softmax denominator

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        n_chunk = end - start

        uf_chunk = uf[start:end]
        vf_chunk = vf[start:end]
        ufrac_c  = u_frac[start:end]
        vfrac_c  = v_frac[start:end]
        d_chunk  = depths[start:end]
        c_chunk  = colors[start:end]

        # (n_chunk, M) integer pixel coords for the disk
        U = uf_chunk[:, None] + dx_all[None, :]
        V = vf_chunk[:, None] + dy_all[None, :]

        inb = (U >= 0) & (U < width) & (V >= 0) & (V < height)
        if not np.any(inb):
            continue

        # Spatial weights using subpixel distance to pixel centers
        # Broadcast to (n_chunk, M)
        w_spatial = _spatial_weights(
            ufrac_c[:, None], vfrac_c[:, None],
            dx_all[None, :], dy_all[None, :],
            R=radius, mode=spatial_mode, sigma=spatial_sigma
        )

        # Keep valid entries only
        Uv = U[inb].astype(np.intp)
        Vv = V[inb].astype(np.intp)
        ws = w_spatial[inb].astype(np.float32)
        dv = np.repeat(d_chunk, M)[inb.ravel()]
        cv = np.repeat(c_chunk, M, axis=0)[inb.ravel()]

        if use_softmax:
            # Soft z: weight = exp(-d/tau) * w_spatial ; normalize per pixel implicitly by dividing afterwards
            wd = np.exp(-dv / max(1e-6, tau)).astype(np.float32)
            w  = ws * wd
            # Accumulate
            np.add.at(logits_sum, (Vv, Uv), w)
            np.add.at(num, (Vv, Uv, slice(None)), (w[:, None] * cv))
            if return_depth:
                np.add.at(dep_num, (Vv, Uv), (w * dv))
        else:
            # Relative-to-closest depth bias via Dmin
            Dmin_loc = Dmin[Vv, Uv]
            dd = (dv - Dmin_loc)  # >= 0

            if depth_mode == "exp":
                wd = np.exp(-dd / max(1e-6, tau)).astype(np.float32)
            elif depth_mode == "linear":
                wd = np.maximum(0.0, 1.0 - dd / max(1e-6, tau)).astype(np.float32)
            else:
                raise ValueError(f"Unknown depth_mode: {depth_mode}")

            w = ws * wd
            np.add.at(den, (Vv, Uv), w)
            np.add.at(num, (Vv, Uv, slice(None)), (w[:, None] * cv))
            if return_depth:
                np.add.at(dep_num, (Vv, Uv), (w * dv))

    # Finalize outputs
    if use_softmax:
        den_safe = logits_sum + eps
        img = num / den_safe[..., None]
        dep = (dep_num / den_safe) if return_depth else None
        visited = logits_sum > 0
    else:
        den_safe = den + eps
        img = num / den_safe[..., None]
        dep = (dep_num / den_safe) if return_depth else None
        visited = den > 0

    return img.astype(np.float32), (dep.astype(np.float32) if dep is not None else None), visited

def render_v1(
        all_pts_world, 
        all_colors_world, 
        pose, 
        height, 
        width,
        radius=1, 
        depth_mode='exp', 
        tau=0.05, 
        spatial_mode='tent'
    ):

    # convert to ERP + Depth representation
    pts_sph = my_utils.world2cam_sph_3D(all_pts_world, pose)
    depths  = pts_sph[..., 2].astype(np.float32)
    coords  = my_utils.sph2erp_2D(pts_sph[..., :2], height, width).astype(np.float32)

    # Depth-weighted disk splatting
    warped_img, warped_depth, visited_pixels = splat_with_weighted_disk(
        colors=all_colors_world.astype(np.float32),
        depths=depths,
        coords=coords,
        height=height,
        width=width,
        radius=radius,
        spatial_mode=spatial_mode,
        depth_mode=depth_mode,  # 'exp' | 'linear' | 'softmax'
        tau=tau
    )

    # Post-splatting interpolation
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels,
    )

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels

def render_perspective_v1(
        all_pts_world, 
        all_colors_world, 
        pose, 
        height, 
        width,
        fx, fy, cx, cy,
        radius=1, 
        depth_mode='exp', 
        tau=0.05, 
        spatial_mode='tent'
    ):

    # World -> camera (Cartesian)
    points_3D_cam_carte = my_utils.world2cam_carte_3D(all_pts_world, pose)  # (N,3)

    # 3D -> perspective pixel coords + depth along the camera optical axis (+X)
    coords_persp, depths_persp = my_utils.carte2persp_3D(
        points_3D_cam_carte,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )  # (N,2)
    coords_persp = coords_persp.astype(np.float32)
    depths_persp = depths_persp.astype(np.float32)

    # Depth-weighted disk splatting
    warped_img, warped_depth, visited_pixels = splat_with_weighted_disk(
        colors=all_colors_world.astype(np.float32),
        depths=depths_persp,
        coords=coords_persp,
        height=height,
        width=width,
        radius=radius,
        spatial_mode=spatial_mode,
        depth_mode=depth_mode,  # 'exp' | 'linear' | 'softmax'
        tau=tau
    )

    # Post-splatting interpolation
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels,
    )

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels


# ---------------------- #
# ----V2 of rendering--- # 
# ---------------------- #

# This rendering method gathers 3D point contributions per pixel within an adaptive radius that expands in regions of high depth variation.
# Each pixel blends nearby points using spatial proximity and depth-based occlusion weights, giving more influence to closer surfaces.
# Unlike z-buffer splatting, occlusion is computed locally and self-consistently, without relying on any external depth map.

def splat_with_modular_weighted_disk(
    colors,          # (N,3) float32 in [0,1]
    depths,          # (N,)  float32, smaller = closer
    coords,          # (N,2) float32 (u,v)
    height, width,
    # TV / radius controls
    R_tv=5,                      # fixed circle for TV computation
    R_min=0.5,                   # smallest per-pixel gather radius
    R_max=5,                     # largest per-pixel gather radius
    tv_lo=0.01,                  # TV where radius starts to grow
    tv_hi=0.25,                  # TV where radius saturates at R_max
    # weighting controls
    depth_mode="exp",            # 'exp' | 'linear' | 'softmax'
    tau=0.05,                    # depth scale for occlusion weighting
    spatial_mode="tent",         # 'tent' | 'gauss'
    spatial_sigma=None,          # used if spatial_mode == 'gauss'
    # perf / misc
    chunk_size=200000,
    return_depth=True,
    eps=1e-8,
):
    """
    Depth-aware, adaptive-radius point-cloud renderer.

    Principle (3 passes):
      0) Z-buffer splat (no circle): project points, take nearest depth per pixel.
         Produces a naive color/depth image (warped_img0/warped_depth0) and a visited mask.
      1) Local depth total variation (TV): within a *fixed* radius R_tv, compute
         mean absolute depth difference around each pixel, ignoring unvisited neighbors.
         Map TV ∈ [tv_lo, tv_hi] to a per-pixel gather radius R(u,v) ∈ [R_min, R_max].
      2) Adaptive gather: for each pixel, gather contributions from points whose
         projected centers fall inside its radius R(u,v). Combine spatial kernel
         (tent/gauss) with a depth-occlusion kernel ('exp'/'linear' or 'softmax')
         to produce the final color/depth.

    Returns a dictionary with:
      - "img":         (H,W,3) final rendered image
      - "depth":       (H,W)   final blended depth (or None if return_depth=False)
      - "visited":     (H,W)   boolean mask of pixels that received contributions
      - "Rmap":        (H,W)   per-pixel adaptive radius used in pass 2
      - "tv":          (H,W)   normalized local total variation (pass 1)
      - "zbuf_img":    (H,W,3) naive z-buffer image (pass 0 result)
      - "zbuf_depth":  (H,W)   naive z-buffer depth (pass 0 result)
      - "zbuf_visited":(H,W)   visited mask from pass 0

    Notes:
      * Occlusions: 'exp' with small tau gives the strongest foreground dominance.
      * The fixed TV circle R_tv is a hyperparameter: it controls how *sensitive*
        the adaptivity is to local depth changes, independently from R_min/R_max.
    """

    # -------------------------- Input checks & dtypes --------------------------
    assert colors.shape[-1] == 3 and coords.shape[-1] == 2, "Invalid shapes."
    num_points = colors.shape[0]
    assert depths.shape[0] == num_points and coords.shape[0] == num_points

    colors = colors.astype(np.float32, copy=False)
    depths = depths.astype(np.float32, copy=False)
    coords = coords.astype(np.float32, copy=False)

    # Integer pixel anchors and subpixel fractions for each point
    u_float = coords[:, 0]
    v_float = coords[:, 1]
    u_floor = np.floor(u_float).astype(np.int32)
    v_floor = np.floor(v_float).astype(np.int32)
    u_frac  = (u_float - u_floor).astype(np.float32)
    v_frac  = (v_float - v_floor).astype(np.float32)

    # ------------------------ Pass 0: Naive Z-buffer splat --------------------
    zbuf_img, zbuf_depth, zbuf_visited = _pass0_zbuffer_splat(
        colors=colors,
        depths=depths,
        coords=coords,
        height=height,
        width=width
    )

    # --------------------- Pass 1: TV within fixed circle ---------------------
    tv_map = _pass1_depth_tv_within_disk(
        depth_map=zbuf_depth,
        valid_mask=zbuf_visited,
        radius_tv=int(R_tv),
        eps=eps
    )

    # Map TV -> per-pixel radius in [R_min, R_max]
    radius_map_ = _map_tv_to_radius(
        tv=tv_map,
        tv_lo=tv_lo,
        tv_hi=tv_hi,
        R_min=R_min,
        R_max=R_max,
        eps=eps
    )
    radius_map = denoise_tv_chambolle(radius_map_, weight=1.0)

    # ---------------- Pass 2: Adaptive gather with depth occlusion ------------
    img, dep, visited = _pass2_adaptive_gather_self_occluding(
        colors=colors,
        depths=depths,
        u_floor=u_floor,
        v_floor=v_floor,
        u_frac=u_frac,
        v_frac=v_frac,
        height=height,
        width=width,
        radius_map=radius_map,
        spatial_mode=spatial_mode,
        spatial_sigma=spatial_sigma,
        depth_mode=depth_mode,
        tau=tau,
        R_max=R_max,
        chunk_size=chunk_size,
        return_depth=return_depth,
        eps=eps
    )

    return {
        "img": img,
        "depth": dep,
        "visited": visited,
        "Rmap": radius_map,
        "tv": tv_map,
        "zbuf_img": zbuf_img,
        "zbuf_depth": zbuf_depth,
        "zbuf_visited": zbuf_visited,
    }

def _disk_offsets(radius: int) -> np.ndarray:
    """All integer (dx,dy) offsets that lie inside a disk of given integer radius."""
    r2 = radius * radius
    offsets = []
    for dy in range(-radius, radius + 1):
        max_dx = int(np.floor(np.sqrt(max(0, r2 - dy * dy))))
        for dx in range(-max_dx, max_dx + 1):
            offsets.append((dx, dy))
    return np.array(offsets, dtype=np.int32)  # (M,2)

def _subpixel_distance(u_frac, v_frac, dx, dy):
    """
    Distance from a pixel center (integer grid) to the projected point location.
    u_frac, v_frac: fractional parts in [0,1)
    dx, dy: integer offset arrays (broadcastable with u_frac/v_frac)
    """
    du = (dx.astype(np.float32) - (1.0 - u_frac))
    dv = (dy.astype(np.float32) - (1.0 - v_frac))
    return np.sqrt(du * du + dv * dv)

def _pass0_zbuffer_splat(colors, depths, coords, height, width):
    """
    Z-buffer splat: pick the nearest (smallest depth) point per integer pixel.
    Returns: (img0, depth0, visited0)
    """
    # winners per pixel (indices into the *in-bounds* arrays prepared below)
    winners = get_winners_z_buffer_splatting(depths, coords, height, width)

    # Filter to in-bounds once and gather winners
    prep = prepare_coords(coords, height, width, colors=colors, depth_cam2=depths)
    u_r = prep['u_r']; v_r = prep['v_r']
    colors_valid = prep['colors'][winners]
    depths_valid = prep['depth_cam2'][winners]
    u_win = u_r[winners]; v_win = v_r[winners]

    # Allocate outputs
    visited0 = np.zeros((height, width), dtype=bool)
    img0   = np.zeros((height, width, 3), dtype=np.float32)
    depth0 = np.full((height, width), 0.0, dtype=np.float32)

    # Write winners
    visited0[v_win, u_win] = True
    img0[v_win, u_win]     = colors_valid
    depth0[v_win, u_win]   = depths_valid

    return img0, depth0, visited0

def _pass1_depth_tv_within_disk(depth_map, valid_mask, radius_tv: int, eps: float):
    """
    Compute mean absolute depth deviation (Total Variation) within a fixed disk, per pixel.
    Only compares pairs where BOTH center and neighbor are valid in `valid_mask`.
    """
    height, width = depth_map.shape
    offsets = _disk_offsets(radius_tv)
    dx_all = offsets[:, 0]; dy_all = offsets[:, 1]

    tv_sum   = np.zeros((height, width), dtype=np.float32)
    tv_count = np.zeros((height, width), dtype=np.float32)

    for k in range(len(dx_all)):
        dx = int(dx_all[k]); dy = int(dy_all[k])
        if dx == 0 and dy == 0:
            continue

        # Overlapping slices (shift by dx,dy)
        y0s = max(0,  dy); y0e = min(height, height + dy)
        x0s = max(0,  dx); x0e = min(width,  width  + dx)
        y1s = max(0, -dy); y1e = min(height, height - dy)
        x1s = max(0, -dx); x1e = min(width,  width  - dx)

        dep_c = depth_map[y0s:y0e, x0s:x0e]
        dep_n = depth_map[y1s:y1e, x1s:x1e]
        val_c = valid_mask[y0s:y0e, x0s:x0e]
        val_n = valid_mask[y1s:y1e, x1s:x1e]

        both_valid = val_c & val_n  # ignore black/unvisited pixels
        if not np.any(both_valid):
            continue

        diff = np.abs(dep_n - dep_c)
        tv_sum[y0s:y0e, x0s:x0e][both_valid]   += diff[both_valid]
        tv_count[y0s:y0e, x0s:x0e][both_valid] += 1.0

    tv = np.zeros_like(tv_sum)
    valid = tv_count > 0
    tv[valid] = tv_sum[valid] / (tv_count[valid] + eps)
    return tv

def _map_tv_to_radius(tv, tv_lo, tv_hi, R_min, R_max, eps):
    """
    Linearly map TV ∈ [tv_lo, tv_hi] to radius ∈ [R_min, R_max], clamped at the ends.
    """
    denom = max(eps, (tv_hi - tv_lo))
    s = np.clip((tv - tv_lo) / denom, 0.0, 1.0).astype(np.float32)
    return (R_min + s * (R_max - R_min)).astype(np.float32)

def _pass2_adaptive_gather_self_occluding(
    colors, depths,
    u_floor, v_floor, u_frac, v_frac,
    height, width,
    radius_map,
    spatial_mode, spatial_sigma,
    depth_mode, tau,
    R_max,
    chunk_size,
    return_depth,
    eps,
):
    """
    Adaptive gather without any external reference depth.

    For each pixel, only the points within its radius contribute.
    Occlusion is modeled using *only those contributing points*:

      - depth_mode='softmax':
          w ~ exp(-d/tau) * w_spatial, normalized per pixel (single pass).
      - depth_mode in {'exp','linear'} (two sub-passes):
          A) Build Dmin_local(u,v) = min depth over contributions to that pixel.
          B) For each contribution, Δd = d - Dmin_local(u,v) (>=0), then:
               'exp'   : w_d = exp(-Δd / tau)
               'linear': w_d = max(0, 1 - Δd / tau)
          Accumulate w = w_spatial * w_d.

    This uses only (u,v,depth,color) of the actual gather — no z-buffer.
    """
    # Candidate stencil up to R_max
    offsets = _disk_offsets(int(R_max))
    dx_all = offsets[:, 0]; dy_all = offsets[:, 1]; M = offsets.shape[0]

    # Accumulators for the final image/depth
    accum_rgb = np.zeros((height, width, 3), dtype=np.float32)
    accum_w   = np.zeros((height, width), dtype=np.float32)
    accum_z   = np.zeros((height, width), dtype=np.float32) if return_depth else None

    use_softmax = (depth_mode == "softmax")

    # ---------------------------- SOFTMAX MODE -----------------------------
    if use_softmax:
        for start in range(0, colors.shape[0], chunk_size):
            end = min(colors.shape[0], start + chunk_size)

            # Candidate integer pixels for each point
            U = u_floor[start:end, None] + dx_all[None, :]
            V = v_floor[start:end, None] + dy_all[None, :]

            in_bounds = (U >= 0) & (U < width) & (V >= 0) & (V < height)
            if not np.any(in_bounds):
                continue

            # Subpixel distances to pixel centers
            dist = _subpixel_distance(
                u_frac[start:end, None], v_frac[start:end, None],
                dx_all[None, :], dy_all[None, :]
            )

            # Flatten valid candidates
            Uv = U[in_bounds].astype(np.intp)
            Vv = V[in_bounds].astype(np.intp)
            rv = dist[in_bounds].astype(np.float32)
            dv = np.repeat(depths[start:end], M)[in_bounds.ravel()]
            cv = np.repeat(colors[start:end], M, axis=0)[in_bounds.ravel()]

            # Keep only those within each pixel's radius
            Rloc = radius_map[Vv, Uv]
            keep = (rv <= Rloc + 1e-6)
            if not np.any(keep):
                continue

            Uv = Uv[keep]; Vv = Vv[keep]; rv = rv[keep]; dv = dv[keep]; cv = cv[keep]
            Rloc_keep = Rloc[keep]

            # Spatial kernel
            if spatial_mode == "tent":
                ws = np.maximum(0.0, 1.0 - rv / np.maximum(1e-6, Rloc_keep)).astype(np.float32)
            elif spatial_mode == "gauss":
                sigma = spatial_sigma if (spatial_sigma is not None) else np.maximum(1e-6, Rloc_keep / 2.0)
                ws = np.exp(-0.5 * (rv / sigma) ** 2).astype(np.float32)
            else:
                raise ValueError("Unknown spatial_mode")

            # Absolute-depth softmax weights (normalized per pixel by division later)
            wd = np.exp(-dv / max(1e-6, tau)).astype(np.float32)
            w  = ws * wd

            np.add.at(accum_w,   (Vv, Uv), w)
            np.add.at(accum_rgb, (Vv, Uv, slice(None)), (w[:, None] * cv))
            if return_depth:
                np.add.at(accum_z, (Vv, Uv), (w * dv))

        denom = accum_w + eps
        out_img   = (accum_rgb / denom[..., None]).astype(np.float32)
        out_depth = (accum_z / denom).astype(np.float32) if return_depth else None
        visited   = (accum_w > 0)
        return out_img, out_depth, visited

    # --------------- EXP / LINEAR MODES (two sub-passes) -----------------

    # Sub-pass A: per-pixel local min depth among *kept* contributions
    Dmin_local = np.full((height, width), np.inf, dtype=np.float32)

    for start in range(0, colors.shape[0], chunk_size):
        end = min(colors.shape[0], start + chunk_size)

        U = u_floor[start:end, None] + dx_all[None, :]
        V = v_floor[start:end, None] + dy_all[None, :]

        in_bounds = (U >= 0) & (U < width) & (V >= 0) & (V < height)
        if not np.any(in_bounds):
            continue

        dist = _subpixel_distance(
            u_frac[start:end, None], v_frac[start:end, None],
            dx_all[None, :], dy_all[None, :]
        )

        Uv = U[in_bounds].astype(np.intp)
        Vv = V[in_bounds].astype(np.intp)
        rv = dist[in_bounds].astype(np.float32)
        dv = np.repeat(depths[start:end], M)[in_bounds.ravel()]

        Rloc = radius_map[Vv, Uv]
        keep = (rv <= Rloc + 1e-6)
        if not np.any(keep):
            continue

        Uv = Uv[keep]; Vv = Vv[keep]; dv = dv[keep]

        # robust per-pixel minimum using ufunc.at (handles repeated indices)
        np.minimum.at(Dmin_local, (Vv, Uv), dv)

    # Sub-pass B: accumulate with relative depth to that local min
    for start in range(0, colors.shape[0], chunk_size):
        end = min(colors.shape[0], start + chunk_size)

        U = u_floor[start:end, None] + dx_all[None, :]
        V = v_floor[start:end, None] + dy_all[None, :]

        in_bounds = (U >= 0) & (U < width) & (V >= 0) & (V < height)
        if not np.any(in_bounds):
            continue

        dist = _subpixel_distance(
            u_frac[start:end, None], v_frac[start:end, None],
            dx_all[None, :], dy_all[None, :]
        )

        Uv = U[in_bounds].astype(np.intp)
        Vv = V[in_bounds].astype(np.intp)
        rv = dist[in_bounds].astype(np.float32)
        dv = np.repeat(depths[start:end], M)[in_bounds.ravel()]
        cv = np.repeat(colors[start:end], M, axis=0)[in_bounds.ravel()]

        Rloc = radius_map[Vv, Uv]
        keep = (rv <= Rloc + 1e-6)
        if not np.any(keep):
            continue

        Uv = Uv[keep]; Vv = Vv[keep]; rv = rv[keep]; dv = dv[keep]; cv = cv[keep]
        Rloc_keep = Rloc[keep]

        # Spatial kernel
        if spatial_mode == "tent":
            ws = np.maximum(0.0, 1.0 - rv / np.maximum(1e-6, Rloc_keep)).astype(np.float32)
        elif spatial_mode == "gauss":
            sigma = spatial_sigma if (spatial_sigma is not None) else np.maximum(1e-6, Rloc_keep / 2.0)
            ws = np.exp(-0.5 * (rv / sigma) ** 2).astype(np.float32)
        else:
            raise ValueError("Unknown spatial_mode")

        # Relative depth to local minimum among current contributors
        dmin = Dmin_local[Vv, Uv]
        # Pixels with no contributors would not be here; dmin should be finite
        dd = dv - dmin
        dd = np.maximum(0.0, dd)

        if depth_mode == "exp":
            wd = np.exp(-dd / max(1e-6, tau)).astype(np.float32)
        elif depth_mode == "linear":
            wd = np.maximum(0.0, 1.0 - dd / max(1e-6, tau)).astype(np.float32)
        else:
            raise ValueError("depth_mode must be 'exp' or 'linear' (softmax handled above)")

        w = ws * wd
        np.add.at(accum_w,   (Vv, Uv), w)
        np.add.at(accum_rgb, (Vv, Uv, slice(None)), (w[:, None] * cv))
        if return_depth:
            np.add.at(accum_z, (Vv, Uv), (w * dv))

    denom = (accum_w + eps)
    out_img   = (accum_rgb / denom[..., None]).astype(np.float32)
    out_depth = (accum_z / denom).astype(np.float32) if return_depth else None
    visited   = (accum_w > 0)
    return out_img, out_depth, visited

def render_v2(
    all_pts_world,         # (N,3) float32 world coordinates
    all_colors_world,      # (N,3) float32 [0,1]
    pose,                  # camera pose (as expected by my_utils)
    height, width,
    # TV / radius controls
    R_tv=2.0,                    # fixed circle for TV computation (hyperparameter)
    R_min=1.0,                   # max radius used during final gather
    R_max=2.0,
    tv_lo=0.00,                  # TV where radius starts to grow
    tv_hi=0.10,                  # TV where radius saturates at R_max
    # weighting controls
    depth_mode="exp",            # 'exp' | 'linear' | 'softmax'
    tau=0.01,                    # depth scale for final occlusion
    spatial_mode="tent",         # 'tent' | 'gauss'
    spatial_sigma=None,          # used if 'gauss'
    # perf / misc
    chunk_size=200000,
    return_depth=True,
    eps=1e-8,
    return_extra=False,
):
    """
    Wrapper: world-space point cloud -> ERP coords + camera depth,
    then run Pass0 (z-buffer), Pass1 (TV over fixed circle R_tv), Pass2 (adaptive gather).

    Returns the same dict as splat_with_weighted_disk:
        {
          "img": (H,W,3) float32,            # final adaptive image
          "depth": (H,W) float32 or None,    # final blended depth
          "visited": (H,W) bool,             # pixels that received contributions
          "Rmap": (H,W) float32,             # per-pixel adaptive radius
          "tv": (H,W) float32,               # normalized local total variation
          "zbuf_img": (H,W,3) float32,       # naive z-buffer image (pass 0)
          "zbuf_depth": (H,W) float32,       # naive z-buffer depth (pass 0)
          "zbuf_visited": (H,W) bool,        # visited mask for z-buffer pass
        }
    """
    # 3D -> camera spherical (your util). Depth is z in that space.
    points_3D_cam2_sph = my_utils.world2cam_sph_3D(all_pts_world, pose)   # (N,3)
    depths  = points_3D_cam2_sph[..., 2].astype(np.float32)               # (N,)
    coords_sph = points_3D_cam2_sph[..., :2]
    coords_erp  = my_utils.sph2erp_2D(coords_sph, height, width).astype(np.float32)  # (N,2)

    # Call the zbuf->TV->adaptive renderer
    out = splat_with_modular_weighted_disk(
        colors=all_colors_world.astype(np.float32),
        depths=depths,
        coords=coords_erp,
        height=height,
        width=width,
        R_tv=R_tv,
        R_min=R_min,
        R_max=R_max,
        tv_lo=tv_lo,
        tv_hi=tv_hi,
        depth_mode=depth_mode,
        tau=tau,
        spatial_mode=spatial_mode,
        spatial_sigma=spatial_sigma,
        chunk_size=chunk_size,
        return_depth=return_depth,
        eps=eps,
    )
    warped_img = out["img"]
    warped_depth = out["depth"]
    visited_pixels = out["visited"]
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels
    )

    if return_extra:
        out["warped_img_interp"] = warped_img_interp
        out["warped_depth_interp"] = warped_depth_interp
        return out

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels

def render_perspective_v2(
    all_pts_world,         # (N,3) float32 world coordinates
    all_colors_world,      # (N,3) float32 [0,1]
    pose,                  # camera pose (as expected by my_utils)
    height, width,
    fx, fy, cx, cy,
    # TV / radius controls
    R_tv=2.0,                    # fixed circle for TV computation (hyperparameter)
    R_min=1.0,                   # max radius used during final gather
    R_max=2.0,
    tv_lo=0.00,                  # TV where radius starts to grow
    tv_hi=0.10,                  # TV where radius saturates at R_max
    # weighting controls
    depth_mode="exp",            # 'exp' | 'linear' | 'softmax'
    tau=0.01,                    # depth scale for final occlusion
    spatial_mode="tent",         # 'tent' | 'gauss'
    spatial_sigma=None,          # used if 'gauss'
    # perf / misc
    chunk_size=200000,
    return_depth=True,
    eps=1e-8,
    return_extra=False,
):
    """
    Wrapper: world-space point cloud -> ERP coords + camera depth,
    then run Pass0 (z-buffer), Pass1 (TV over fixed circle R_tv), Pass2 (adaptive gather).

    Returns the same dict as splat_with_weighted_disk:
        {
          "img": (H,W,3) float32,            # final adaptive image
          "depth": (H,W) float32 or None,    # final blended depth
          "visited": (H,W) bool,             # pixels that received contributions
          "Rmap": (H,W) float32,             # per-pixel adaptive radius
          "tv": (H,W) float32,               # normalized local total variation
          "zbuf_img": (H,W,3) float32,       # naive z-buffer image (pass 0)
          "zbuf_depth": (H,W) float32,       # naive z-buffer depth (pass 0)
          "zbuf_visited": (H,W) bool,        # visited mask for z-buffer pass
        }
    """

    # World -> camera (Cartesian)
    points_3D_cam_carte = my_utils.world2cam_carte_3D(all_pts_world, pose)  # (N,3)

    # 3D -> perspective pixel coords + depth along the camera optical axis (+X)
    coords_persp, depths_persp = my_utils.carte2persp_3D(
        points_3D_cam_carte,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )  # (N,2)
    coords_persp = coords_persp.astype(np.float32)
    depths_persp = depths_persp.astype(np.float32)

    # Call the zbuf->TV->adaptive renderer
    out = splat_with_modular_weighted_disk(
        colors=all_colors_world.astype(np.float32),
        depths=depths_persp,
        coords=coords_persp,
        height=height,
        width=width,
        R_tv=R_tv,
        R_min=R_min,
        R_max=R_max,
        tv_lo=tv_lo,
        tv_hi=tv_hi,
        depth_mode=depth_mode,
        tau=tau,
        spatial_mode=spatial_mode,
        spatial_sigma=spatial_sigma,
        chunk_size=chunk_size,
        return_depth=return_depth,
        eps=eps,
    )
    warped_img = out["img"]
    warped_depth = out["depth"]
    visited_pixels = out["visited"]
    warped_img_interp, warped_depth_interp = interpolate_missing_pixels(
        warped_img, warped_depth, visited_pixels
    )

    if return_extra:
        out["warped_img_interp"] = warped_img_interp
        out["warped_depth_interp"] = warped_depth_interp
        return out

    return warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels




if __name__ == "__main__":
    expname = '31_forest'
    test_rendering_eqr = False
    test_rendering_perspective = True
    width = 1440
    height = 720
    sphere_radius = 1.0
    save_dir = "OUTPUTS/SphericalDreamerRecurse"
    save_dir_ = f"{save_dir}/{expname}"
    translation_direction = np.array([1,0,0])
    delta_walk =  sphere_radius * np.pi / 2

    pose1 = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 1.0]])
    pose2 = my_utils.camera_translation(pose1, delta_walk * translation_direction)
    pose_intermediate = my_utils.camera_translation(pose2, -delta_walk/2 * translation_direction)

    pcd = my_utils.load_pcd(f"{save_dir_}/raw_dream_pcd.pkl")
    
    render_kwargs = {
        "all_pts_world": np.concatenate((
            pcd.pts,
        ), axis=0),

        "all_colors_world": np.concatenate((
            pcd.colors,
        ), axis=0),

        "pose": pose2,
        "height": height,
        "width": width,
    }
    
    if test_rendering_eqr:
        
        t0 = time.time()
        warped_img_v0, warped_depth_v0, warped_img_interp_v0, warped_depth_interp_v0, visited_pixels_v0 = render_v0(**render_kwargs)
        t1 = time.time()
        print(f"Render V0 Time: {t1 - t0:.4f} seconds")

        t0 = time.time()
        warped_img_v1, warped_depth_v1, warped_img_interp_v1, warped_depth_interp_v1, visited_pixels_v1 = render_v1(
            **render_kwargs, 
            radius=1.0,
            depth_mode='exp', 
            tau=0.05, 
            spatial_mode='tent'
        )
        t1 = time.time()
        print(f"Render V1 Time: {t1 - t0:.4f} seconds")
        
        t0 = time.time()
        res_v2 = render_v2(
            **render_kwargs,                 
            # TV / radius controls
            R_tv=2.0,                         # fixed circle for TV computation (hyperparameter)
            R_min=1.0,                   # max radius used during final gather
            R_max=2.0,
            tv_lo=0.00,                  # TV where radius starts to grow
            tv_hi=0.30,                  # TV where radius saturates at R_max
            # weighting controls
            depth_mode="exp",            # 'exp' | 'linear' | 'softmax'
            tau=0.01,                    # depth scale for final occlusion
            spatial_mode="tent",         # 'tent' | 'gauss'
            spatial_sigma=None,          # used if 'gauss'
            # perf / misc
            chunk_size=200000,
            return_depth=True,
            eps=1e-8,
            return_extra=True
        )
        t1 = time.time()
        print(f"Render V2 Time: {t1 - t0:.4f} seconds")
        
        warped_img_interp_v2, warped_depth_interp_v2 = res_v2["warped_img_interp"], res_v2["warped_depth_interp"]
        warped_img_interp_v0 = my_utils.numpy_to_PIL(warped_img_interp_v0)
        warped_img_interp_v1 = my_utils.numpy_to_PIL(warped_img_interp_v1)
        warped_img_interp_v2 = my_utils.numpy_to_PIL(warped_img_interp_v2)
        warped_depth_interp_v0=my_utils.numpy_to_PIL(warped_depth_interp_v0)
        warped_depth_interp_v1=my_utils.numpy_to_PIL(warped_depth_interp_v1)
        warped_depth_interp_v2=my_utils.numpy_to_PIL(warped_depth_interp_v2)
        img_res = my_utils.tile_image([warped_img_interp_v0, warped_img_interp_v1, warped_img_interp_v2])
        depth_res = my_utils.tile_image([warped_depth_interp_v0, warped_depth_interp_v1, warped_depth_interp_v2])

    if test_rendering_perspective:
        width = width // 2
        height = height // 2
        render_kwargs.update({
            "fx": 800.0,
            "fy": 800.0,
            "cx": width / 2,
            "cy": height / 2,
            "width": width,
            "height": height,
        })
        
        
        t0 = time.time()
        warped_img_v0, warped_depth_v0, warped_img_interp_v0, warped_depth_interp_v0, visited_pixels_v0 = render_perspective_v0(**render_kwargs)
        t1 = time.time()
        print(f"Render V0 Time: {t1 - t0:.4f} seconds")

        t0 = time.time()
        warped_img_v1, warped_depth_v1, warped_img_interp_v1, warped_depth_interp_v1, visited_pixels_v1 = render_perspective_v1(
            **render_kwargs, 
            radius=1.0,
            depth_mode='exp', 
            tau=0.05, 
            spatial_mode='tent'
        )
        t1 = time.time()
        print(f"Render V1 Time: {t1 - t0:.4f} seconds")
        
        t0 = time.time()
        res_v2 = render_perspective_v2(
            **render_kwargs,                 
            # TV / radius controls
            R_tv=2.0,                         # fixed circle for TV computation (hyperparameter)
            R_min=1.0,                   # max radius used during final gather
            R_max=2.0,
            tv_lo=0.00,                  # TV where radius starts to grow
            tv_hi=0.30,                  # TV where radius saturates at R_max
            # weighting controls
            depth_mode="exp",            # 'exp' | 'linear' | 'softmax'
            tau=0.01,                    # depth scale for final occlusion
            spatial_mode="tent",         # 'tent' | 'gauss'
            spatial_sigma=None,          # used if 'gauss'
            # perf / misc
            chunk_size=200000,
            return_depth=True,
            eps=1e-8,
            return_extra=True
        )
        t1 = time.time()
        print(f"Render V2 Time: {t1 - t0:.4f} seconds")
        
        warped_img_interp_v2, warped_depth_interp_v2 = res_v2["warped_img_interp"], res_v2["warped_depth_interp"]
        warped_img_interp_v0 = my_utils.numpy_to_PIL(warped_img_interp_v0)
        warped_img_interp_v1 = my_utils.numpy_to_PIL(warped_img_interp_v1)
        warped_img_interp_v2 = my_utils.numpy_to_PIL(warped_img_interp_v2)
        warped_depth_interp_v0=my_utils.numpy_to_PIL(warped_depth_interp_v0)
        warped_depth_interp_v1=my_utils.numpy_to_PIL(warped_depth_interp_v1)
        warped_depth_interp_v2=my_utils.numpy_to_PIL(warped_depth_interp_v2)
        img_res = my_utils.tile_image([warped_img_interp_v0, warped_img_interp_v1, warped_img_interp_v2])
        depth_res = my_utils.tile_image([warped_depth_interp_v0, warped_depth_interp_v1, warped_depth_interp_v2])

        img_res
