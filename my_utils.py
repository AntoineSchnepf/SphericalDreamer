import os
import sys 
import numpy as np
import logging
from PIL import Image
import copy
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
import cv2 
from scipy.interpolate import griddata as interp_grid
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import cg, splu



# ---------------------------- #
# ----- Computer Vision ------ #
# ---------------------------- #

def fill_mask(mask, flip=False):
    # mask: boolean NumPy array
    # Fill holes in False regions completely surrounded by True, by swapping True to False in such areas
    if flip:
        return ~ndimage.binary_fill_holes(~mask)
    return ndimage.binary_fill_holes(mask)

def close_mask(mask, size=5, flip=False):
     # Close False regions mostly surrounded by True, by swapping True to False in such areas
    structure = np.ones((size, size), dtype=bool)
    if flip:
        return ~ndimage.binary_closing(~mask, structure=structure)
    return ndimage.binary_closing(mask, structure=structure)

def erode_mask(mask, pixels=10):
    """
    Erode a boolean mask inward by `pixels`.
    """
    # Disk-shaped structuring element
    y, x = np.ogrid[-pixels:pixels+1, -pixels:pixels+1]
    selem = (x**2 + y**2) <= pixels**2   # circular footprint
    # Perform erosion
    eroded = ndimage.binary_erosion(mask, structure=selem)
    return eroded

def dilate_mask(mask, pixels=10):
    """
    Dilate (expand) a boolean mask outward by `pixels`.
    """
    # Disk-shaped structuring element
    y, x = np.ogrid[-pixels:pixels+1, -pixels:pixels+1]
    selem = (x**2 + y**2) <= pixels**2
    # Perform dilatation
    dilated = ndimage.binary_dilation(mask, structure=selem)
    return dilated

def seamless_blend(src, dst, mask):
    """
    Blend src into dst guided by mask (all PIL.Image objects).
    src and dst must be the same size.
    Returns a PIL.Image with seamless blending.
    """
    # Convert to OpenCV format
    src_cv  = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    dst_cv  = cv2.cvtColor(np.array(dst), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask.convert("L"))

    # Compute center of panoramic image
    height, width = mask_cv.shape

    # Blend
    # -- v2 --
    br = cv2.boundingRect(mask_cv) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    blended_cv = cv2.seamlessClone(src_cv, dst_cv, mask_cv, centerOfBR, cv2.NORMAL_CLONE)

    # -- v1 --
    # center = (width//2, height//2)
    # blended_cv = cv2.seamlessClone(src_cv, dst_cv, mask_cv, center, cv2.NORMAL_CLONE)


    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB))



# ---------------------------- #
# ------ Visualization ------- #
# ---------------------------- #

def show_masks(masks, alpha=0.5, background=None):
    """
    Visualize several boolean masks on the same image with color overlaps.

    Parameters
    ----------
    masks : list of np.ndarray
        List of boolean arrays (all same shape).
    alpha : float
        Transparency for overlays.
    background : np.ndarray or None
        Optional grayscale/RGB image to show under masks.
    """
    H, W = masks[0].shape
    # Assign distinct colors (cycle through tab colormap)
    cmap = plt.cm.get_cmap("tab10", len(masks))
    colors = [np.array(cmap(i)[:3]) for i in range(len(masks))]

    # Background (default = white)
    if background is None:
        img = np.ones((H, W, 3), dtype=float)
    else:
        # Normalize background to 0-1 RGB
        bg = np.array(background, dtype=float)
        if bg.ndim == 2:
            bg = np.stack([bg]*3, axis=-1)
        bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        img = bg

    # Blend each mask color
    for m, col in zip(masks, colors):
        m3 = np.stack([m]*3, axis=-1)
        img = np.where(m3, (1-alpha)*img + alpha*col, img)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

def depth_numpy_to_figure(depth, cmap='plasma', vmin=0.0, vmax=1.2, figsize=(12,12)):
    """
    Convert a depth map (numpy array) to a matplotlib figure for visualization.
    
    Parameters:
    - depth: np.ndarray
        2D array representing the depth map.
    - cmap: str
        Colormap to use for visualization.
    - vmin: float
        Minimum depth value for normalization.
    - vmax: float
        Maximum depth value for normalization.
    
    Returns:
    - fig: matplotlib.figure.Figure
        Figure object containing the visualized depth map.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize depth values to [vmin, vmax]
    plt.imshow(depth, cmap=cmap)
    plt.colorbar(ax=ax, label='Depth')
    plt.axis('off')
    plt.tight_layout()

    return fig

def xyz_to_rgb(pts, r=None, coord_type='cartesian'):
    """
    Map 3D points to RGB colors based on position.
    
    Parameters
    ----------
    pts : array-like (..., 3)
        Either Cartesian coordinates [x, y, z] or spherical [r, theta, phi].
        If spherical, `coord_type` must be 'spherical'.
    r : float, optional
        Reference radius for normalization. If None, it's estimated from |x|, |y|, |z|.
    coord_type : str, optional
        'cartesian' (default) or 'spherical'
        - 'cartesian': pts[...,0]=x, pts[...,1]=y, pts[...,2]=z
        - 'spherical': pts[...,0]=radius, pts[...,1]=theta, pts[...,2]=phi
          (theta: colatitude [0,π], phi: azimuth [-π,π])
    
    Returns
    -------
    colors : np.ndarray (..., 3)
        RGB values in [0, 1]
    """
    pts = np.asarray(pts, dtype=float)

    if coord_type == 'spherical':
        pts_carte = sph2carte_3D(pts)
        x, y, z = pts_carte[..., 0], pts_carte[..., 1], pts_carte[..., 2]
    elif coord_type == 'cartesian':
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    else:
        raise ValueError("coord_type must be 'cartesian' or 'spherical'.")

    if r is None:
        # estimate bounding radius (max absolute coord)
        r = np.max(np.abs([x, y, z]))

    R = (x + r) / (2 * r)
    G = (y + r) / (2 * r)
    B = (z + r) / (2 * r)

    return np.stack([R, G, B], axis=-1)

# ------------------------------------------ #
# ----- Numpy - PIL conversions / utils -----#
# ------------------------------------------ #

def cat_ones(array):
    return np.concatenate((array, np.ones((*array.shape[:-1], 1))), axis=-1)

def depth_numpy_to_PIL(depth):
    depth = copy.deepcopy(depth)
    depth[np.isnan(depth)] = 0.0
    depth_pil = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
    max_val = 65535
    depth_pil = (depth_pil * max_val).astype(np.uint16)              # Scale to [0, 65535]
    depth_pil = Image.fromarray(depth_pil)
    """Concatenate a column of ones to the input array."""
    return depth_pil

def numpy_to_PIL(image):
    """
    Convert a numpy array to a PIL Image. Automatically converts to RGB or Gray-Scale PIL Images.
    """
    if image.ndim == 2 :
        return Image.fromarray(np.uint8(image * 255.0)).convert('L')
    elif image.shape[2] == 1:
        assert image.ndim == 3
        return Image.fromarray(np.uint8(image * 255.0)).convert('L')
    else:
        return Image.fromarray(np.uint8(image * 255.0)).convert('RGB')

def PIL_to_numpy(pil_img):
    """
    Convert a PIL Image to a numpy array, normalized to [0, 1].
    Handles both grayscale ('L') and RGB images.
    """
    arr = np.array(pil_img).astype(np.float32) / 255.0
    
    return arr

def pil_mask_to_numpy_bool(pil_mask):
    return np.array(pil_mask.convert("L")) > 0

def numpy_bool_to_pil_mask(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay a binary mask on an RGB image ([0, 1] float) as a blue transparent filter.
    
    Parameters:
    - image: RGB image as a (H, W, 3) NumPy array with float values in [0, 1]
    - mask: Binary mask as a (H, W) NumPy array (values 0 or 1)
    - alpha: Transparency level of the overlay (0 = transparent, 1 = opaque)
    """
    # Ensure inputs are float and mask is boolean
    image = image.astype(np.float32)
    mask = mask.astype(bool)

    # Create a blue color layer [R, G, B] = [0, 0, 1]
    blue_layer = np.zeros_like(image)
    blue_layer[..., 2] = 1.0

    # Copy original image for overlay
    overlay = image.copy()

    # Apply alpha blending where mask is True
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * blue_layer[mask]

    # Clip to [0, 1] range
    overlay = np.clip(overlay, 0, 1)
    
    return overlay

def get_1px_red_line(image):

    width = image.shape[1]
    red_line = np.full((1, width, 3), (255, 0, 0), dtype=np.uint8)  # 1px red line
    return red_line

def tile_image(images, insert_red_lines=True):
    """
    Concatenate images vertically.
    image:PIL.Image or np.array
    """
    # convert to np
    images = [np.array(img) for img in images]

    # build stack
    stack = []
    for i, img in enumerate(images):
        if img.ndim == 2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        stack.append(img)
        if insert_red_lines and i < len(images) - 1:
            stack.append(get_1px_red_line(img))
    
    return Image.fromarray(np.vstack(stack))



# --------------------------------------------------------------------------- #
# ----- 3D Geometry: equirectangular, spherical & cartesian coordinates ----- #
# --------------------------------------------------------------------------- #

# ERP coordinate system:
#
# ┌─────────────────────────► u in [0, w-1]
# │
# │   [0,0]         [0, w-1]
# │     ●────────────●
# │     │            │
# │     │            │
# │     ●────────────●
# │   [h-1, 0]      [h-1,w-1]
# ▼
# v in [0, h-1]

#[u, v] reprensent a point on the unit sphere. 

# Spherical coordinate system:
#   
# ^ θ (elevation) in [-π/2, π/2]
# │
# │
# │
# │
# │
# ────────────────► φ (azimuth) in [-π, π[

# basics
def erp2sph_2D(erp_points:np.array, erp_image_height:int, erp_image_width:int):
    """
    Convert the point from erp image pixel location to spherical coordinate.
    The returned coordinates are such that the center of the ERP image correspond to (theta=0, phi=0) in spherical coordinates.

    args:
        :erp_points: np.array w. shape [..., 2]: array of 2D points coordinates in ERP coordinates, expressed in pixel unit. Convention X, Y.
        :erp_image_height: int: height (in pixels) of the ERP image
        :erp_image_width: int: width (in pixels) of the ERP image    

    The function will not fail if 3D points are given, but will ignore the z-coordinate.

    returns:
        :sph_points: np.array w. shape [..., 2] array of 2D points coordinates in spherical coordinates, expressed in radians. Convention theta, phi.
            - theta: Elevation in [-pi/2, pi/2]
            - phi: Azimuth in [-pi, pi[
    """

    H = erp_image_height
    W = erp_image_width

    erp_points_u = erp_points[..., 0]
    erp_points_v = erp_points[..., 1]

    offset_u = (W - 1) / 2
    offset_v = (H - 1) / 2

    points_phi = 2 * np.pi * (erp_points_u - offset_u) / W # azimuth in [-pi, pi]
    points_theta = - np.pi  * (erp_points_v - offset_v) / H # elevation in [-pi/2, pi/2]

    points_phi = np.where(points_phi == np.pi,  -np.pi, points_phi)
    points_theta = np.where(points_theta == -0.5 * np.pi, 0.5 * np.pi, points_theta)

    sph_point = np.stack((points_theta, points_phi), axis=-1)

    return sph_point

def sph2erp_2D(sph_point:np.array, erp_image_height:int, erp_image_width:int):
    """ 
    Transform the spherical coordinate location to ERP image pixel location.
    It is the inverse of the erp2sph function.
    args:   
        :sph_points: np.array w. shape [..., 2] array of 2D points coordinates in spherical coordinates, expressed in radians. Convention theta, phi.
            - theta: Elevation in [-pi/2, pi/2]
            - phi: Azimuth in [-pi, pi[

    The function will not fail if 3D points are given, but will ignore the r-coordinate.

    return:
        :erp_points: np.array w. shape [..., 2]: array of 2D points coordinates in ERP coordinates, expressed in pixel unit. Convention X, Y.
    """
    H = erp_image_height
    W = erp_image_width

    theta = sph_point[..., 0]  # elevation
    phi = sph_point[..., 1]  # azimuth

    erp_u = W * (phi / (2 * np.pi) + 0.5 ) - 0.5
    erp_v = H * (0.5 - theta / np.pi) - 0.5
    erp_point = np.stack((erp_u, erp_v), axis=-1)
    return erp_point

def sph2carte_3D(sph_point) :
    """
    Transform spherical coordinates to Cartesian coordinates.
    args:
        :sph_point: np.array w. shape [..., 3]: array of 3D points coordinates in spherical coordinates. Convention: theta, phi, r
    returns:
        :carte_points: np.array w. shape [..., 3]: array of 3D points coordinates in Cartesian coordinates. Convention X,Y,Z
    """

    
    theta, phi, r = sph_point[..., 0], sph_point[..., 1], sph_point[..., 2]
    X = r * np.cos(theta) * np.cos(phi)
    Y = r * np.cos(theta) * np.sin(phi)
    Z = r * np.sin (theta)
    carte_points = np.stack((X, Y, Z), axis=-1)

    return carte_points

def carte2sph_3D(carte_points):
    """
    Transform Cartesian coordinates to spherical coordinates.
    args: 
        :carte_points: np.array w. shape [..., 3]: array of 3D points coordinates in Cartesian coordinates. Convention X,Y,Z
    return:
        :sph_points: np.array w. shape [..., 3]: array of 3D points coordinates in spherical coordinates. Convention: theta, phi, r
    """

    X, Y, Z = carte_points[..., 0], carte_points[..., 1], carte_points[..., 2]
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arcsin(Z / r)  # elevation
    phi = np.arctan2(Y, X)  # azimuth

    sph_points = np.stack((theta, phi, r), axis=-1)
    return sph_points

# cam2world, world2cam and cam2cam functions
def cam_sph2world_3D(points_3D_cam_sph, pose):
    """
    Convert camera spherical coordinates to world coordinates.
    
    Args:
        points_3D_cam_sph (np.array): Camera spherical coordinates of shape [..., 3].
        pose (np.array): Camera pose matrix of shape [4, 4].
    
    Returns:
       points_3D_world_carte: np.array w. shape [..., 3]. World coordinates. Convention X, Y, Z.
    """
    points_3D_cam_carte = sph2carte_3D(points_3D_cam_sph)
    points_3D_world_carte = np.einsum('ij,...j->...i', pose, cat_ones(points_3D_cam_carte))[..., :3]
    return points_3D_world_carte

def world2cam_sph_3D(points_3D_world_carte, pose):
    """
    Convert world coordinates to camera spherical coordinates.
    
    Args:
        points_3D_world_carte (np.array): World coordinates of shape [..., 3].
        pose (np.array): Camera pose matrix of shape [4, 4].
    
    Returns:
       points_3D_cam_sph: np.array w. shape [..., 3]. Camera spherical coordinates. Convention theta, phi, r.
    """
    points_3D_cam_carte = np.einsum('ij,...j->...i', np.linalg.inv(pose), cat_ones(points_3D_world_carte))[..., :3]
    points_3D_cam_sph = carte2sph_3D(points_3D_cam_carte)
    return points_3D_cam_sph

def world2cam_carte_3D(points_3D_world_carte, pose): 
    """
    Convert world coordinates to camera Cartesian coordinates.
    
    Args:
        points_3D_world_carte (np.array): World coordinates of shape [..., 3].
        pose (np.array): Camera pose matrix of shape [4, 4].
    
    Returns:
       points_3D_cam_carte: np.array w. shape [..., 3]. Camera Cartesian coordinates. Convention X, Y, Z.
    """
    points_3D_cam_carte = np.einsum('ij,...j->...i', np.linalg.inv(pose), cat_ones(points_3D_world_carte))[..., :3]
    return points_3D_cam_carte


# depth2 functions (assumes depth is in [0,1] with shape [H,W])
def get_canonical_sph_pixels(height, width):
    points_2D_erp = np.stack((np.meshgrid(range(width), range(height))), axis=-1) 
    points_2D_sph = erp2sph_2D(points_2D_erp, erp_image_height=height, erp_image_width=width)
    return points_2D_sph

def depth2cam_sph(depth, sphere_radius, height, width):
    points_2D_sph = get_canonical_sph_pixels(height, width)
    assert depth.shape[0] == height and depth.shape[1] == width, f"Depth shape {depth.shape} does not match height {height} and width {width}"
    points_3D_cam_sph = np.concatenate((points_2D_sph, np.expand_dims(depth * sphere_radius, axis=-1)), axis=-1)
    return points_3D_cam_sph

def depth2cam_carte(depth, sphere_radius, height, width):
    points_3D_cam_sph = depth2cam_sph(depth, sphere_radius, height, width)
    points_3D_cam_carte = sph2carte_3D(points_3D_cam_sph)
    return points_3D_cam_carte

def depth2world(depth, pose, sphere_radius, height, width):
    points_3D_cam_carte = depth2cam_carte(depth, sphere_radius, height, width)
    points_3D_world_carte = np.einsum('ij,...j->...i', pose, cat_ones(points_3D_cam_carte))[..., :3]
    return points_3D_world_carte

# def cam_erp2cam_sph_3D(points_2D_cam_erp, height, width, depth, sphere_radius=1.0):
#     """
#     Convert Equirectangular coordinates to camera spherical coordinates.
    
#     Args:
#         points_2D_cam_erp (np.array): Equirectangular coordinates of shape [..., 2].
#         depth (np.array): Depth map of shape [...].
#         sphere_radius (float): Radius of the sphere.
    
#     Returns:
#        points_3D_cam_sph: np.array w. shape [..., 3]. Camera spherical coordinates. Convention theta, phi, r.
#     """
#     assert np.all(points_2D_cam_erp.shape[:-1] == depth.shape)
#     points_2D_cam_sph = erp2sph_2D(points_2D_cam_erp, erp_image_height=height, erp_image_width=width)
#     r = depth * sphere_radius
#     points_3D_cam_sph = np.concatenate((points_2D_cam_sph, np.expand_dims(r, axis=-1)), axis=-1)
#     return points_3D_cam_sph

# def cam_erp2world_3D(points_2D_cam_erp, height, width, depth, pose, sphere_radius=1.0):
#     """
#     Convert Equirectangular coordinates to world coordinates.
    
#     Args:
#         points_2D_cam_erp (np.array): Equirectangular coordinates of shape [..., 2].
#         depth (np.array): Depth map of shape [...].
#         pose (np.array): Camera pose matrix of shape [4, 4].
#         sphere_radius (float): Radius of the sphere.
    
#     Returns:
#        points_3D_world_carte: np.array w. shape [..., 3]. World coordinates. Convention X, Y, Z.
#     """
#     points_3D_cam_sph = cam_erp_to_cam_sph_3D(points_2D_cam_erp, height, width, depth, sphere_radius)
#     points_3D_world_carte = cam_sph_to_world_3D(points_3D_cam_sph, pose)
#     return points_3D_world_carte



# ---------------------------------------- #
# ---- Warping / Splatting functions ----- #
# ---------------------------------------- #

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

def splatting_and_interpolation(colors, depth_cam2, coord_cam2, height, width, interpolation_mode='original'):
    """
    This function takes as input: 
        - `colors` an array of colors (N, 3) in [0-1]
        - `depth_cam2` an array of depth values (N,) in [0-1]
        - `coord_cam2` an array of 2D coordinates in the image frame (N, 2) in pixel coordinates
    From this information, is returned 
        - the warped_image (no interpolation)
        - the warped_depth (no interpolation)
        - the binary mask of visited pixels
        - the interpolated_image (with interpolation)
        - the interpolated_depth (with interpolation)
    """
    #TODO: better docstring

    # Basic checks
    assert colors.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert colors.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape

    # Flatten to [N, ...]
    colors   = colors.reshape((-1, 3))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    # Prepare coordinates (rounding, in-bounds filtering)
    out = prepare_coords(coord_cam2, height, width, colors=colors, depth_cam2=depth_cam2)

    # --- I. Z-buffer Splatting to find winners ----
    winners = get_winners_z_buffer_splatting(depth_cam2, coord_cam2, height, width)

    # Winners' data
    u_win_r = out['u_r'][winners]
    v_win_r = out['v_r'][winners]
    depths_win = out['depth_cam2'][winners]
    colors_win = out['colors'][winners]   
    u_win_f = out['u_float'][winners]       
    v_win_f = out['v_float'][winners]        
    # ---- II. Allocate outputs -----
    # 1. Visited mask
    visited = np.zeros((height, width), dtype=bool)
    visited[v_win_r, u_win_r] = True

    # 2. Naive without Interpolation
    warped_img   = np.zeros((height, width, 3), dtype=np.float32)
    warped_depth = np.full((height, width), 0.0, dtype=np.float32)
    warped_img[v_win_r, u_win_r]   = colors_win
    warped_depth[v_win_r, u_win_r] = depths_win

    # 3. With Interpolation
    grid = np.stack((np.meshgrid(range(width), range(height))), axis=-1).reshape(-1, 2).astype(np.float32)
    if interpolation_mode == 'original':
        points_ = np.stack((u_win_f, v_win_f), axis=-1).astype(np.float32)
    elif interpolation_mode == 'rounded':
        points_ = np.stack((u_win_r, v_win_r), axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown interpolation mode: {interpolation_mode}")
    
    image_interp = interp_grid(
        points_,
        colors_win,
        grid, 
        method='linear', 
        # fill_value=0
    ).reshape(height,width,3)

    depth_interp = interp_grid(
        points_,
        depths_win,
        grid,
        method='linear',
        # fill_value=0
    ).reshape(height,width)



    return warped_img, warped_depth, image_interp, depth_interp, visited

def depth_aware_naive_splatting_vectorized(colors, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    This functions computes a new image, at a new camera location, based on:
        (i) a set coordinates of colored points in the new image: `coord_cam2` (float values)
        (ii) corresponding colors: `colors`
        (iii) corresponding depths: `depth_cam2`
    This functions rounds the float values, to obtain proper pixel location. If there are multiple pixels for a location, 
    the point that is the nearest to the camera is chosen. This models occlusions. 

    It is a simple form of splatting, using a z-buffer. 

    args:
        img: np.array with values in [0,1]. Shape [H, W, 3] or [HW, 3].Colors of the pixels at canonical locations(
            [[[0,0], ..., [0,W]],
            ...
            [H,0], ..., [H,W]]]
        ) in the source image, i.e. the source image itself
        coord_cam2: pixel coordinates in the new image (float values). Shape [H, W, 2] or [HW, 2].
        depth_cam2: np.array. Depth at new camera location (float values). Shape [H, W] or [HW].
    """
    # Basic checks
    assert colors.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert coord_cam1.shape[-1] == 2
    assert colors.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    # Flatten to [N, ...]
    colors   = colors.reshape((-1, 3))
    coord_cam1 = coord_cam1.reshape((-1, 2))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    # Prepare coordinates (rounding, in-bounds filtering)
    out = prepare_coords(coord_cam2, height, width, colors=colors, coord_cam1=coord_cam1, depth_cam2=depth_cam2)
    # z-buffer splatting to find winners
    winners = get_winners_z_buffer_splatting(depth_cam2, coord_cam2, height, width)

    # Winners' data
    u_win_r = out['u_r'][winners]
    v_win_r = out['v_r'][winners]
    depths_win = out['depth_cam2'][winners]
    colors_win = out['colors'][winners]
    coord1_win = out['coord_cam1'][winners]      # (a, b) source coordinates
    u_win_f = out['u_float'][winners]        # unrounded u for flow key
    v_win_f = out['v_float'][winners]        # unrounded v for flow key

    # Allocate outputs
    warped_img   = np.zeros((height, width, 3), dtype=np.float32)
    warped_depth = np.full((height, width), np.inf, dtype=np.float32)
    visited      = np.zeros((height, width), dtype=bool)

    # Scatter winners into outputs
    warped_img[v_win_r, u_win_r]   = colors_win
    warped_depth[v_win_r, u_win_r] = depths_win
    visited[v_win_r, u_win_r]      = True

    # Flow mapping: map exact (float u, float v) -> (a, b) from coord_cam1
    # Convert to native Python floats for dict keys/values
    flow = {(float(uf), float(vf)): (ab[0], ab[1])
            for uf, vf, ab in zip(u_win_f, v_win_f, coord1_win)}

    return warped_img, warped_depth, flow, visited

def depth_aware_naive_splatting(colors, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    This functions computes a new image, at a new camera location, based on:
        (i) a set coordinates of colored points in the new image: `coord_cam2` (float values)
        (ii) corresponding colors: `colors`
        (iii) corresponding depths: `depth_cam2`
    This functions rounds the float values, to obtain proper pixel location. If there are multiple pixels for a location, 
    the point that is the nearest to the camera is chosen. This models occlusions. 

    It is a simple form of splatting, using a z-buffer. 

    args:
        img: np.array with values in [0,1]. Shape [H, W, 3] or [HW, 3].Colors of the pixels at canonical locations(
            [[[0,0], ..., [0,W]],
            ...
            [H,0], ..., [H,W]]]
        ) in the source image, i.e. the source image itself
        coord_cam2: pixel coordinates in the new image (float values). Shape [H, W, 2] or [HW, 2].
        depth_cam2: np.array. Depth at new camera location (float values). Shape [H, W] or [HW].
    """
    assert colors.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert colors.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    colors = colors.reshape((-1, 3))
    coord_cam1 = coord_cam1.reshape((-1, 2))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    warped_img = np.zeros(shape=(height, width, 3), dtype=np.float32)
    warped_depth = np.full(shape=(height, width) , fill_value=np.inf, dtype=np.float32)
    visited_pixels = np.zeros(shape=(height, width), dtype=bool)  # keep track of visited pixels
    # more_than_once_visited = np.zero(shape=(height, width), dtype=bool)  # keep track of pixels visited more than once
    # visited_count = 0


    flow_mapping = {}
    # Iterate over all the 3D points 
    for k in range(len(coord_cam2)):
        (a,b) = coord_cam1[k] # (a, b) represent the coordinates of the current point in the source image
        (u,v) = coord_cam2[k] # (u, v) represent the coordinates of the current point in the target image

        u_ = int(round(u))  
        v_ = int(round(v))

        color = colors[k]
        depth2 = depth_cam2[k]
        
        if 0 <= u_ < width and 0 <= v_ < height:

            # /!\ Reference frame for an image inverts horizontal and vertical axis
            if warped_depth[v_, u_] > depth2 : # If this points is closer than previous
                warped_depth[v_, u_] = depth2 
                warped_img[v_, u_] = color 
                visited_pixels[v_, u_] = True
                flow_mapping[(u_, v_)] = ((a, b), (u, v))

    flow = {}
    for ((a, b), (u, v)) in flow_mapping.values():
        flow[(float(u), float(v))] = (a, b)

    return warped_img, warped_depth, flow, visited_pixels

def interpolate_with_flow(colors, depths, flow, mode='original'):
    """
    args:
        colors: np.array with values in [0,1]
        depths: np.array with float values 
        flow: dict with keys as pixel coordinates in source image and values as pixels coordinates in target (warped) image.
        It looks like {(u_, v_) : (i, j)}

    """
    width = colors.shape[1]
    height = colors.shape[0]

    grid = np.stack((np.meshgrid(range(width), range(height))), axis=-1).reshape(-1, 2).astype(np.float32)
    rgb_values = []
    depth_values = []
    points = []
    points_rounded = []
    for (u_, v_), (a,b) in flow.items():
        rgb_values.append(
            colors[b,a]
        )
        depth_values.append(
            depths[b,a]
        )
        points.append((u_, v_))
        points_rounded.append((round(u_), round(v_)))

    points = np.array(points)
    rgb_values = np.array(rgb_values)
    points_rounded = np.array(points_rounded)

    if mode == 'original':
        points_ = points
    elif mode == 'rounded':
        points_ = points_rounded
    else:
        raise ValueError("Mode must be 'rounded' or 'original'.")
    
    image_interp = interp_grid(
        points_,
        rgb_values,
        grid, 
        method='linear', 
        # fill_value=0
    ).reshape(height,width,3)

    depth_interp = interp_grid(
        points_,
        depth_values,
        grid,
        method='linear',
        # fill_value=0
    ).reshape(height,width)

    return image_interp, depth_interp


# ----------------------------- #
# ----- Utility functions ----- #
# ----------------------------- #

def get_norm_vector(v):
    """Normalize a vector or an array of vectors."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-10)

def normalize_angle(a):
    """Map any angle a (scalar or array) to [0, 2π)."""
    return np.mod(a, 2*np.pi)

def angle_diff(phi1, phi2):
    """
    Signed difference phi1 - phi2, wrapped to (-π, π].
    Works with scalars or numpy arrays.
    """
    return ( (phi1 - phi2 + np.pi) % (2*np.pi) ) - np.pi

def in_interval_mod(phi, start, end, closed='both', atol=1e-12):
    """
    Test if angle(s) phi lie in the modular interval [start, end] modulo 2π.

    - All angles are treated modulo 2π.
    - Interval semantics:
        closed='both'   -> inclusive on both ends
        closed='left'   -> inclusive start, open end
        closed='right'  -> open start, inclusive end
        closed='neither'-> open on both ends

    Works for wrap-around intervals (when start > end after normalization).
    """
    phi  = normalize_angle(phi)
    a    = normalize_angle(start)
    b    = normalize_angle(end)

    # Comparators with endpoint control
    if closed == 'both':
        le = lambda x, y: x <= y + atol
        ge = lambda x, y: x + atol >= y
    elif closed == 'left':
        le = lambda x, y: x < y - atol
        ge = lambda x, y: x + atol >= y
    elif closed == 'right':
        le = lambda x, y: x <= y + atol
        ge = lambda x, y: x > y + atol
    elif closed == 'neither':
        le = lambda x, y: x < y - atol
        ge = lambda x, y: x > y + atol
    else:
        raise ValueError("closed must be 'both', 'left', 'right', or 'neither'")

    if a <= b:
        # Normal (non-wrapping) interval
        return ge(phi, a) & le(phi, b)
    else:
        # Wrapping interval (crosses 0): [a, 2π) ∪ [0, b]
        return ge(phi, a) | le(phi, b)




# --------------------------------------------#
# ---- World Opening transformations ----- #
# --------------------------------------------#

def get_sphere_tangent(phi0, sph_points):
        """
        args:
        :phi0: float, radians: point on the sphere where the tangent is computed
        :sph_points: np.array w. shape [..., 3]: array of 3D points coordinates in spherical coordinates. Convention: theta, phi, r
        
        returns: np.array w. shape [..., 3]: array of 3D points coordinates in spherical coordinates, projected onto the tangent at phi0
        """

        theta, phi, r = sph_points[..., 0], sph_points[..., 1], sph_points[..., 2]

        # normal expression at phi0:
        X0 = r * np.cos(theta) * np.cos(phi0)
        Y0 = r * np.cos(theta) * np.sin(phi0)
        Z0 = r * np.sin (theta)
        P0 = np.stack([X0, Y0, Z0], axis=-1)  # shape [3]

        # derivative w.r.t. phi at phi0:
        dX_dphi0 = -r * np.cos(theta) * np.sin(phi0)
        dY_dphi0 =  r * np.cos(theta) * np.cos(phi0)
        dZ_dphi0 =  0.0 * r * np.sin (theta)
        dP0 = np.stack([dX_dphi0, dY_dphi0, dZ_dphi0], axis=-1)  # shape [3]

        # get projection of (X,Y,Z) onto the tangent at phi0:
        delta = angle_diff(phi, phi0) # shape [...]
        projection = P0[None, :] + dP0[None, :] * delta[..., None]  # shape [..., 3]
        #shape: [..., 3] = shape: [1, 3] + shape: [1, 3] * shape: [..., 1]
        return carte2sph_3D(projection)

def unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi):
    """
    Unfold points in the spherical coordinates, by projecting them onto the sphere tangents
    at the boundary of a cone of angle delta around forward_sph.
    All arguments expected in spherical coordinates (theta, phi, r).
    The returned points are also in spherical coordinates.
    """
    # a. Get the two boundary angles
    phi_forward = forward_sph[1]
    phi1 = normalize_angle(phi_forward + delta / 2)
    phi2 = normalize_angle(phi_forward - delta / 2)

    # b. gets arcs
    arc1_bounds = (phi_forward, phi1)
    arc2_bounds = (phi2, phi_forward)
    arc1_mask = in_interval_mod(pts_sph[..., 1], arc1_bounds[0], arc1_bounds[1], closed='both')
    arc2_mask = in_interval_mod(pts_sph[..., 1], arc2_bounds[0], arc2_bounds[1], closed='both')
    other_mask = ~(arc1_mask | arc2_mask)

    arc1_pts_sph = pts_sph[arc1_mask]
    arc2_pts_sph = pts_sph[arc2_mask]

    arc1_proj_pts_sph = get_sphere_tangent(phi1, arc1_pts_sph)
    arc2_proj_pts_sph = get_sphere_tangent(phi2, arc2_pts_sph)
    other_pts_sph = pts_sph[other_mask]


    res_sph = np.zeros_like(pts_sph)
    res_sph[arc1_mask] = arc1_proj_pts_sph
    res_sph[arc2_mask] = arc2_proj_pts_sph
    res_sph[other_mask] = other_pts_sph

    return res_sph

def remove_points_within_cone(pts_sph, forward_sph, delta=np.pi/2, eps=1e-12):
    """
    Remove points inside a cone of aperture `delta` (radians) centered on `direction`,
    optionally restricted to points within a sphere of radius `radius` around `center`.

    Args:
        pts_sph (..., 3): point cloud in camera frame (Spherical).
        forward_sph (3,): cone axis in camera frame (Spherical, doesn't need to be normalized).
        delta (float): cone angle in radians. Points with angle <= delta / 2 are removed.
    Returns:
        pts_cut_sph (M, 3): (flattened) points outside the cone (and within radius if provided).
        mask_keep (...,): boolean mask of kept points.
    """

    points = sph2carte_3D(pts_sph)          # (N, 3)
    direction = sph2carte_3D(forward_sph) # (3,)

    # Normalize the cone axis
    d_norm = np.linalg.norm(direction)
    if d_norm < eps:
        raise ValueError("`direction` must be non-zero.")
    d = direction / d_norm

    # Translate to local coordinates
    r = np.linalg.norm(points, axis=-1)                     # (N,)
    assert np.allclose(r, pts_sph[..., 2])

    # Cosine of angle to direction: cos(theta) = (v · d) / ||v||
    # Safe divide to handle points exactly at the center.
    cos_theta = np.einsum('...j,j->...', points, d) / np.maximum(r, eps)
    cos_hdelta = np.cos(delta / 2)

    # Points inside the cone have theta <= delta  <=>  cos(theta) >= cos(delta)
    in_cone = cos_theta >= cos_hdelta

    # Keep: NOT in_cone
    mask_keep = ~in_cone
    pts_cut_sph = pts_sph[mask_keep]
    return pts_cut_sph, mask_keep

def unfold_points_on_cylinder(pts_sph, forward_sph, base_radius=1.0, eps=1e-12):
    """
    Unfolds the northern hemisphere of a (possibly perturbed) sphere into a cylinder.

    Inputs
    ------
    forward_sph : (3,) array-like
        Direction vector (in spherical coordinates [theta, phi, r])
        indicating the 'up' or 'north' direction of the deformation.
        Only the orientation (theta, phi) is used; r is ignored.
    pts_sph : (..., 3) array-like
        Input points in spherical coordinates [theta, phi, R],
        where theta ∈ [-pi/2, pi/2] (elevation),
              phi   ∈ [-pi, pi]     (azimuth),
              R     > 0             (radius).
    base_radius : float or None
        Reference radius R0 for the target cylinder. If None, uses median(R).
    eps : float
        Small value to avoid numerical issues.

    Returns
    -------
    pts_prime_sph : (..., 3)
        Deformed points in spherical coordinates [theta, phi, R],
        in the same world frame and convention as the input.
    """

    # --- Helpers: rotation & coordinate transforms ---
    def _rotation_align_a_to_b(a, b, eps=1e-12):
        """Compute rotation matrix that aligns vector a to vector b."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        a /= max(np.linalg.norm(a), eps)
        b /= max(np.linalg.norm(b), eps)
        v = np.cross(a, b)
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        s = np.linalg.norm(v)
        if s < eps:
            if c > 0:
                return np.eye(3)
            # 180° rotation: choose any orthogonal axis
            axis = np.array([1., 0., 0.]) if abs(a[0]) <= 0.9 else np.array([0., 1., 0.])
            v = np.cross(a, axis)
            v /= max(np.linalg.norm(v), eps)
            vx, vy, vz = v
            K = np.array([[0, -vz,  vy],
                          [vz,  0, -vx],
                          [-vy, vx,  0]])
            return np.eye(3) + 2 * K @ K
        v /= s
        vx, vy, vz = v
        K = np.array([[0, -vz,  vy],
                      [vz,  0, -vx],
                      [-vy, vx,  0]])
        return np.eye(3) + K * s + K @ K * (1 - c)

    # --- Convert direction (spherical -> cartesian) ---
    from math import cos, sin
    theta_f, phi_f, r_f = forward_sph
    forward_dir = np.array([
        cos(theta_f) * cos(phi_f),
        cos(theta_f) * sin(phi_f),
        sin(theta_f),
    ])

    # --- Align the forward direction to +Z ---
    R_world_to_z = _rotation_align_a_to_b(forward_dir, np.array([0., 0., 1.]))
    R_z_to_world = R_world_to_z.T

    # --- Convert input spherical to cartesian in world frame ---
    P_cart_world = sph2carte_3D(pts_sph)
    # Align to +Z frame
    P_cart_aligned = np.tensordot(P_cart_world, R_world_to_z.T, axes=([P_cart_world.ndim - 1], [0]))
    # Convert to spherical (aligned frame)
    S_aligned = carte2sph_3D(P_cart_aligned)

    theta = S_aligned[..., 0]  # elevation [-pi/2, pi/2]
    phi   = S_aligned[..., 1]
    R     = S_aligned[..., 2]

    # --- Compute base radius and perturbations ---
    R0 = np.median(R) if base_radius is None else float(base_radius)
    delta = R - R0

    # Split north/south hemispheres (θ≥0 → north)
    north = theta >= 0
    south = ~north

    Pp_aligned = np.empty_like(P_cart_aligned)

    # --- SOUTH: unchanged (keep spherical radius R0 + δ) ---
    if np.any(south):
        sph_south = np.stack([theta[south], phi[south], R0 + delta[south]], axis=-1)
        Pp_aligned[south] = sph2carte_3D(sph_south)

    # --- NORTH: unfold onto a cylinder ---
    if np.any(north):
        thetan = theta[north]
        phin   = phi[north]
        dn     = delta[north]

        # Cylinder base (aligned frame): radius = R0, height = R0 * theta
        x0 = R0 * np.cos(phin)
        y0 = R0 * np.sin(phin)
        z0 = R0 * thetan

        # Apply radius variation (δ) along cylinder normal (cos φ, sin φ, 0)
        xn = x0 + dn * np.cos(phin)
        yn = y0 + dn * np.sin(phin)
        zn = z0
        Pp_aligned[north] = np.stack([xn, yn, zn], axis=-1)

    # --- Rotate back to world frame and convert to spherical ---
    Pp_world = np.tensordot(Pp_aligned, R_z_to_world.T, axes=([Pp_aligned.ndim - 1], [0]))
    pts_prime_sph = carte2sph_3D(Pp_world)
    return pts_prime_sph

def filter_points_by_plane(pts_sph, forward_sph, cut_distance):
    """
    Remove points that lie *behind* the plane orthogonal to `forward`
    and located at a distance `cut_distance` along that direction.

    Parameters
    ----------
    pts_sph : np.ndarray, shape (..., 3)
        Point cloud in spherical coordinates (theta, phi, r).
        Convention: theta=elevation, phi=azimuth, r=radius
    forward : np.ndarray, shape (3,)
        Direction vector of the plane's normal.
    cut_distance : float
        Distance of the plane along `forward` (in same units as points).

    Returns
    -------
    kept_pts_sph : np.ndarray, shape (M, 3)
        Points that are *in front of or on* the plane, i.e. (p · f̂) >= cut_distance.
    mask_keep : np.ndarray of bool, shape (N,)
        Boolean mask of kept points.
    """

    # Convert spherical → Cartesian
    pts_xyz = sph2carte_3D(pts_sph)

    # Normalize direction vector
    forward = sph2carte_3D(forward_sph)
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("`forward` vector must be non-zero.")
    forward_hat = forward / norm

    # Project each point onto the forward direction
    proj = pts_xyz @ forward_hat  # dot product

    # Keep points in behing of (or on) the plane
    mask_keep = (proj <= cut_distance)

    # Select those points and return them in spherical coords
    kept_xyz = pts_xyz[mask_keep]
    kept_pts_sph = carte2sph_3D(kept_xyz)

    return kept_pts_sph, mask_keep

def compute_cut_distance_based_on_percentile(pts_sph, forward_sph, percentile=90):
    """
    Compute the cut distance along `forward_sph` such that a given percentile
    of points lie behind the cutting plane.

    Parameters
    ----------
    pts_sph : np.ndarray, shape (..., 3)
        Point cloud in spherical coordinates (theta, phi, r).
        Convention: theta=elevation, phi=azimuth, r=radius
    forward_sph : np.ndarray, shape (3,)
        Direction vector of the plane's normal.
    percentile : float
        Percentile (0-100) of points to be behind the cutting plane.

    Returns
    -------
    cut_distance : float
        Distance along `forward_sph` such that the specified percentile of points
        lie behind the cutting plane.
    """

    # Convert spherical → Cartesian
    pts_xyz = sph2carte_3D(pts_sph)

    # Normalize direction vector
    forward = sph2carte_3D(forward_sph)
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("`forward` vector must be non-zero.")
    forward_hat = forward / norm

    # Project each point onto the forward direction
    proj = pts_xyz @ forward_hat  # dot product

    # Compute and return the desired percentile
    cut_distance = np.percentile(proj, percentile)
    return cut_distance


def open_world(forward_sph, pts_sph, mode='cut+cylinder', delta_cut=np.pi/2, cut_distance=None, cut_distance_percentile=90):
    assert mode in ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut']
    if mode == 'wall':
        pts_opened = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        mask_keep = np.ones_like(pts_sph[..., 0], dtype=bool)
    elif mode == 'cut+wall':
        pts_opened = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif mode == 'cut+cylinder':
        pts_opened = unfold_points_on_cylinder(pts_sph, forward_sph, base_radius=1.0)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif mode == 'remove_within_cone':
        pts_opened = pts_sph
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif mode == 'straight_cut':
        if cut_distance is None:
            cut_distance = compute_cut_distance_based_on_percentile(
                pts_sph, forward_sph, percentile=cut_distance_percentile
            )
        pts_opened = pts_sph
        _, mask_keep = filter_points_by_plane(pts_sph, forward_sph, cut_distance=cut_distance)

    return pts_opened[mask_keep], pts_opened, mask_keep

# ------------------------------------------- #
# ---- Harmonic Deformation of 3D Points ---- #
# ------------------------------------------- #

# 1. Build Laplacian
def build_graph_laplacian(P, k=10, symmetrize=True):
    G = kneighbors_graph(P, n_neighbors=k, opening_mode='distance',
                         include_self=False, n_jobs=-1)
    W = G.tocoo(copy=True)
    W.data = 1.0 / (W.data + 1e-12)
    if symmetrize:
        W = 0.5*(W + W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = diags(deg) - W.tocsr()
    return L

# 2. Constraint subsampling (mask-based)
def subsample_constraints(mask_boundary, target_boundary, mask_fixed,
                          every=5, max_fixed=5000, seed=0):
    rng = np.random.default_rng(seed)

    idx_boundary = np.where(mask_boundary)[0]
    idx_fixed = np.where(mask_fixed)[0]

    # Boundary: keep every Nth point
    idx_boundary_sub = idx_boundary[::every]
    target_boundary_sub = target_boundary[::every]

    # Fixed: random subsample if too many
    if len(idx_fixed) > max_fixed:
        sel = rng.choice(len(idx_fixed), max_fixed, replace=False)
        idx_fixed_sub = idx_fixed[sel]
    else:
        idx_fixed_sub = idx_fixed

    return idx_boundary_sub, target_boundary_sub, idx_fixed_sub

# 3. Coarse set selection
def kmeans_downsample(P, n_samples, seed=0):
    mbk = MiniBatchKMeans(n_clusters=n_samples,
                          batch_size=4096, max_iter=100,
                          n_init="auto", random_state=seed)
    mbk.fit(P)
    centers = mbk.cluster_centers_
    nn = NearestNeighbors(n_neighbors=1).fit(P)
    _, idx = nn.kneighbors(centers)
    return np.unique(idx[:,0])

# 4. Hard Dirichlet solver
def harmonic_deform_dirichlet(P, idx_fixed, idx_boundary, target_boundary, k=10, solver='cg'):
    N = P.shape[0]
    B = np.unique(np.concatenate([idx_fixed, idx_boundary]))
    M = np.setdiff1d(np.arange(N), B, assume_unique=False)

    L = build_graph_laplacian(P, k=k)

    # Displacements on boundary
    uB = np.zeros((len(B), 3))
    if len(idx_boundary) > 0:
        pos_in_B = {b:i for i,b in enumerate(B)}
        jj = np.array([pos_in_B[i] for i in idx_boundary], dtype=int)
        uB[jj,:] = (target_boundary - P[idx_boundary])

    L_MM = L[M][:, M]
    L_MB = L[M][:, B]
    rhs  = - L_MB @ uB

    U = np.zeros_like(P)
    if solver == 'cg':
        for d in range(3):
            uM, info = cg(L_MM, rhs[:,d], atol=0, rtol=1e-6, maxiter=2000)
            if info != 0:
                lu = splu(L_MM.tocsc())
                uM = lu.solve(rhs[:,d])
            U[M,d] = uM
        U[B,:] = uB
    else:
        lu = splu(L_MM.tocsc())
        for d in range(3):
            U[M,d] = lu.solve(rhs[:,d])
        U[B,:] = uB
    return P + U

# 5. Prolongation (IDW)
def prolongate_displacements(P_full, P_coarse, U_coarse, m=3, power=2):
    nbr = NearestNeighbors(n_neighbors=m).fit(P_coarse)
    d, j = nbr.kneighbors(P_full, return_distance=True)
    w = 1.0 / (d**power + 1e-12)
    w /= w.sum(axis=1, keepdims=True)
    U_full = (w[...,None] * U_coarse[j]).sum(axis=1)
    return U_full

# 6. Pipeline (mask-based)
def harmonic_deform_pipeline(P, mask_fixed, mask_boundary, target_boundary,
                             n_coarse=10000, every=5, max_fixed=5000,
                             k=10, m=3, power=2, seed=0):
    """
    - Subsample constraints
    - Build coarse set of mobile points with k-means
    - Solve Dirichlet system on coarse subset
    - Prolongate displacements back to full set
    """
    idx_boundary_sub, target_boundary_sub, idx_fixed_sub = subsample_constraints(
        mask_boundary, target_boundary, mask_fixed, every, max_fixed, seed)

    must_keep = np.unique(np.concatenate([idx_fixed_sub, idx_boundary_sub]))
    free_pool = np.setdiff1d(np.arange(len(P)), must_keep)
    if len(free_pool) > 0 and n_coarse > len(must_keep):
        idx_free = kmeans_downsample(P[free_pool],
                                     n_samples=n_coarse - len(must_keep),
                                     seed=seed)
        coarse_idx = np.concatenate([must_keep, free_pool[idx_free]])
    else:
        coarse_idx = must_keep

    # Deformation on coarse subset
    P_coarse = P[coarse_idx]
    map_to_coarse = -np.ones(len(P), dtype=int)
    map_to_coarse[coarse_idx] = np.arange(len(coarse_idx))
    idx_fixed_c = map_to_coarse[idx_fixed_sub]
    idx_boundary_c = map_to_coarse[idx_boundary_sub]
    target_boundary_c = target_boundary_sub
    P_coarse_def = harmonic_deform_dirichlet(
        P_coarse, idx_fixed_c, idx_boundary_c, target_boundary_c, k=k, solver='cg')
    U_coarse = P_coarse_def - P_coarse

    # Prolongation to full set
    U_full = prolongate_displacements(P, P_coarse, U_coarse, m=m, power=power)
    P_def = P + U_full
    return P_def, coarse_idx



# ------------------------------------------------ #
# ----- Generate Images with color gradients ----- #
# ------------------------------------------------ #
def _part1by1(n: np.ndarray) -> np.ndarray:
    """
    Vectorized bit interleaving helper: expand 16-bit values so that the bits
    occupy the even positions (..b15 0 b14 0 ... b1 0 b0 0).
    Works for n in [0, 65535], we’ll only use up to 12 bits (0..4095).
    """
    n = n.astype(np.uint32) & 0x0000FFFF
    n = (n | (n << 8))  & 0x00FF00FF
    n = (n | (n << 4))  & 0x0F0F0F0F
    n = (n | (n << 2))  & 0x33333333
    n = (n | (n << 1))  & 0x55555555
    return n

def morton_ids(width: int, height: int) -> np.ndarray:
    """
    Return a (H, W) array of unique 24-bit IDs in Morton (Z-order).
    Constraint for uniqueness within 24-bit RGB:
      width * height <= 16,777,216 (2^24)
    For the Morton mapping below, best keep width, height <= 4096 (12 bits each).
    """
    if width * height > (1 << 24):
        raise ValueError("width*height must be ≤ 16,777,216 (24-bit RGB limit).")
    if width  > 4096 or height > 4096:
        raise ValueError("Use width/height ≤ 4096 to stay within 12 bits for Morton interleave.")

    # Coordinate grids
    y = np.arange(height, dtype=np.uint32)[:, None]  # (H,1)
    x = np.arange(width,  dtype=np.uint32)[None, :]  # (1,W)

    # Interleave bits: morton = interleave(y,x) -> y bits in odd positions, x in even (or vice-versa)
    my = _part1by1(y)
    mx = _part1by1(x)
    morton = (my << 1) | mx  # (H,W), up to 24 bits when x,y <= 12 bits
    return morton

def unique_gradient_image(width=1024, height=1024) -> Image.Image:
    """
    Create an image where each pixel has a unique RGB value, arranged so
    colors vary smoothly in both directions (Z-order gradient).
    """
    ids = morton_ids(width, height)  # (H,W) uint32

    # Map 24-bit id -> RGB
    R = (ids >> 16).astype(np.uint8)
    G = (ids >> 8 ).astype(np.uint8)
    B = (ids      ).astype(np.uint8)

    img = np.dstack([R, G, B])
    return Image.fromarray(img, mode="RGB")






# ----- TESTS -----
if __name__ == "__main__":
    # test erp2sph_2D and sph2erp_2D

    H, W = 100, 200
    pixel_u, pixel_v = np.meshgrid(np.arange(H), np.arange(W), indexing='ij') #2D points on canonical grid
    erp_points = np.stack((pixel_u, pixel_v), axis=-1)  

    sph_points = erp2sph_2D(erp_points, H, W)
    recovered_erp_points = sph2erp_2D(sph_points, H, W)
    assert np.allclose(erp_points, recovered_erp_points), "Error: recovered ERP points do not match original ERP points"

    sph_points = np.random.rand(100, 2) * np.array([np.pi/2, np.pi])  # random 2D spherical points
    erp_points = sph2erp_2D(sph_points, H, W)
    recovered_sph_point = erp2sph_2D(erp_points, H, W)
    assert np.allclose(sph_points, recovered_sph_point), "Error: recovered spherical points do not match original spherical points"

    # test sph2_carte_3D and carte2sph_3D

    carte_points = np.random.rand(100, 3)   # random 3D points
    sph_points = carte2sph_3D(carte_points)
    recovered_carte_points = sph2carte_3D(sph_points)
    assert np.allclose(carte_points, recovered_carte_points), "Error: recovered Cartesian points do not match original Cartesian points"

    sph_points_3D = np.random.rand(100, 3) * np.array([1, np.pi/2, np.pi])  # random 3D spherical points
    carte_points_3D = sph2carte_3D(sph_points_3D)
    recovered_sph_points_3D = carte2sph_3D(carte_points_3D)
    assert np.allclose(sph_points_3D, recovered_sph_points_3D), "Error: recovered spherical points do not match original spherical points"

    # test cam_sph2world_3D and world2cam_sph_3D
    points_3D_cam_sph = np.random.rand(100, 3) * np.array([1, np.pi/2, 5])  # random 3D spherical points
    pose = np.eye(4)
    translation = np.array([1, 2, 3])
    pose[:3, 3] = translation
    points_3D_world_carte = cam_sph2world_3D(points_3D_cam_sph, pose)
    recovered_points_3D_cam_sph = world2cam_sph_3D(points_3D_world_carte, pose)
    assert np.allclose(points_3D_cam_sph, recovered_points_3D_cam_sph), "Error: recovered camera spherical points do not match original camera spherical points"


    # test that numpy_to_PIL and PIL_to_numpy as loss-free

    # start with uint8
    image = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)  # random RGB image
    pil_image = Image.fromarray(image)

    # convert to float
    image_float = np.array(pil_image) / 255.0

    # back to unint8
    pil_image2 = numpy_to_PIL(image_float)

    # assert everything still okay
    recovered_image = np.array(pil_image2) 
    assert np.all(image == recovered_image)

    # --- existing setup (your code above this) ---
    perturn_scale = 0.0
    N_th, N_ph = 400, 400
    theta = np.linspace(-np.pi/2, np.pi/2, N_th)
    phi = np.linspace(-np.pi, np.pi, N_ph, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    r = np.ones_like(TH)
    r += perturn_scale * (2 * np.random.rand(*r.shape) - 1.0)
    pts_sph = np.stack([TH, PH, r], axis=-1).reshape(-1, 3)

    forward = np.array([1.0, -1.0, 0.0])
    forward_sph = carte2sph_3D(forward)

    # Optional: make 3D axes have equal aspect (so spheres look like spheres)
    def set_equal_aspect_3d(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        xmid = np.mean(xlim); ymid = np.mean(ylim); zmid = np.mean(zlim)
        radius = max((xlim[1]-xlim[0]), (ylim[1]-ylim[0]), (zlim[1]-zlim[0])) / 2
        ax.set_xlim3d([xmid - radius, xmid + radius])
        ax.set_ylim3d([ymid - radius, ymid + radius])
        ax.set_zlim3d([zmid - radius, zmid + radius])

    # --- plotting section fix ---
    methods = ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut']

    # +1 row for the original point cloud
    n_rows = 1 + 1  # first row = original, second row = all methods
    n_cols = len(methods)

    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': '3d'}, figsize=(4*n_cols, 8))

    # Ensure axes is always 2D array
    axes = np.atleast_2d(axes)

    # --- Row 0: Original point cloud ---
    for jj in range(n_cols):
        ax = axes[0, jj]
        pts_xyz = sph2carte_3D(pts_sph)
        ax.scatter(*pts_xyz.T, s=1, c=xyz_to_rgb(pts_sph, coord_type='spherical'))
        if jj == 0:
            ax.set_title("Original point cloud")
        else:
            ax.set_title("")  # only show title on first plot
        set_equal_aspect_3d(ax)

    # --- Row 1: Transformed by each method ---
    for jj, mode in enumerate(methods):
        pts_opened_sph, _, _ = open_world(forward_sph, pts_sph, mode=mode, delta_cut=2*np.pi/3)
        ax = axes[1, jj]
        pts_opened_xyz = sph2carte_3D(pts_opened_sph)
        ax.scatter(*pts_opened_xyz.T, s=1, c=xyz_to_rgb(pts_opened_sph, coord_type='spherical'))
        ax.set_title(f"Open world ({mode})")
        set_equal_aspect_3d(ax)

    plt.tight_layout()
    plt.show()