import os
import sys 
import numpy as np
from PIL import Image
import copy
import matplotlib.pyplot as plt
from scipy import ndimage
import copy
import cv2 
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import diags
from scipy.sparse.linalg import cg, splu
import time
from scipy.interpolate import RegularGridInterpolator
import pickle
import yaml
import sys
import collections.abc
from prodict import Prodict
import argparse
import pyfiglet
import shutil
from pathlib import Path

# -------------------------------------------- #
# --------------- Config utils ---------------- #
# -------------------------------------------- #

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        assert key in source.keys(), f"key {key} not in source"
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]

    return source

def yaml_load(cfg_name, load_dir):
    config_path = os.path.join(load_dir, cfg_name)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_config(cfg_name, load_dir, from_default=False, default_cfg_name='_default.yaml') :
    """Load a configuration file. If from_default is True, load 
    the default config and update it with the config file"""
    
    config = yaml_load(cfg_name, load_dir)

    if from_default :
        default_config = yaml_load(default_cfg_name, load_dir)
        config = deep_update(default_config, config)

    return config

def save_config(config, cfg_name, save_dir) :
    config_path = os.path.join(save_dir, cfg_name)
    if isinstance(config, Prodict) :
        config = Prodict.to_dict(config, is_recursive=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

# Custom print function that lets you print with colors
def printc(str, color=None):
    """Print string with color"""
    if color is None:
        print(str)
    else:
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "end": "\033[0m"
        }
        print(f"{colors[color]}{str}{colors['end']}")


def copy_phase_folders(folder_start_with: str, item_start_with: str,
                       source_dir: Path, dest_dir: Path):

    if source_dir == dest_dir:
        return
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Copy root-level files that match item_start_with ---
    for item in source_dir.iterdir():
        if item.is_file() and item.name.startswith(item_start_with):
            shutil.copy2(item, dest_dir / item.name)

    # --- 2. Copy folders that match folder_start_with ---
    for folder in source_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(folder_start_with):

            dst_folder = dest_dir / folder.name
            dst_folder.mkdir(parents=True, exist_ok=True)

            # --- 3. Inside the folder, copy only items with item_start_with ---
            for sub in folder.iterdir():
                if sub.name.startswith(item_start_with):

                    dst_item = dst_folder / sub.name
                    if sub.is_dir():
                        shutil.copytree(sub, dst_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(sub, dst_item)

def fetch_config_via_parser(debug, debug_parser_override=[], return_img_name=False):
    repo_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="_default.yaml")
    parser.add_argument('--config_dir', type=str, default=os.path.join(repo_path, "configs"))
    # TODO: remove lines below
    parser.add_argument('--img_name', type=str, default='FD0')
    print("WARNING(Antoine): added a stuppid line in utils.py to run some quick exp. To remove later.")

    # ---- script args ----
    if debug:
        debug_message = pyfiglet.figlet_format("!Debug mode!", font="slant")
        printc(debug_message, color="red")
        args = parser.parse_args(debug_parser_override)
    else:
        args = parser.parse_args()

    config = Prodict.from_dict(load_config(args.config, args.config_dir, from_default=True, default_cfg_name="_default.yaml"))
    
    if return_img_name:
        return config, args.img_name
    
    return config

def setup(config):
    seeds = [config.seed + offset for offset in config.seed_offsets]
    if config.depth_model == 'egformer':
        width = 1024
        height = 512
        print("WARNING: EGFormer depth model selected: Forcing panorama resolution to 1024x512")
    else:
        width = config.width
        height = config.height
        
    save_dir_ = f"{config.save_dir}/{config.expname}"
    pose_init = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    translation_direction = np.array(config.translation_direction, dtype=np.float32)
    pose_end = camera_translation(pose_init, config.delta_walk * translation_direction * (config.num_dreams-1))
    return seeds, width, height, save_dir_, pose_init, pose_end, translation_direction

# -------------------------------------------- #
# ------ Classic Computer Vision utils ------- #
# -------------------------------------------- #


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

def opencv_resize(img, new_h, new_w, mode='bilinear'):
    """Resize using OpenCV for potentially better performance."""
    if mode == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif mode == 'nearest':
        interp = cv2.INTER_NEAREST
    else:
        raise ValueError("Unsupported mode: choose 'nearest' or 'bilinear'")
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized_img

def mask_resize(mask, new_h, new_w):
    """Resize a binary mask using nearest neighbor interpolation."""
    return opencv_resize(mask.astype(np.uint8), new_h, new_w, mode="nearest").astype(bool)

def resize_depth():
   #TODO
   pass
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
    Convert a numpy array to a PIL Image.
    Handles NaNs safely by replacing them with 0 before uint8 conversion.

    Supports:
        - Grayscale 2D arrays
        - (H, W, 1)
        - (H, W, 3) RGB arrays
    """
    image = np.asarray(image)

    # Replace NaN with 0 (black). You can choose another fill value if you prefer.
    safe_img = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    # Normalize to uint8 range
    safe_img = np.clip(safe_img * 255.0, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return Image.fromarray(safe_img, mode='L')

    elif image.ndim == 3 and image.shape[2] == 1:
        return Image.fromarray(safe_img[..., 0], mode='L')

    elif image.ndim == 3 and image.shape[2] == 3:
        return Image.fromarray(safe_img, mode='RGB')

    else:
        raise ValueError("Unsupported image shape for PIL conversion:", image.shape)

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
    image=np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
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

# perspective projection functions

def carte2persp_3D(carte_points: np.ndarray,
                    fx: float,
                    fy: float,
                    cx: float,
                    cy: float):
    """
    Project 3D points in Cartesian coordinates to a perspective image plane.

    Camera convention:
        - Camera center at origin.
        - Forward / optical axis along +X (consistent with sph2carte_3D).
        - Image plane parameterized by (u, v):
              u horizontal, aligned with -Y,
              v vertical,   aligned with -Z.
        - Pinhole projection:
              u = fx * (-Y / X) + cx
              v = fy * (-Z / X) + cy

    Args:
        carte_points: np.array w. shape [..., 3]
            3D points in Cartesian coordinates, convention (X, Y, Z).
        fx: float
            Focal length along u-axis (in pixels).
        fy: float
            Focal length along v-axis (in pixels).
        cx: float
            Principal point offset along u-axis (in pixels).
        cy: float
            Principal point offset along v-axis (in pixels).

    Returns:
        persp_points: np.array w. shape [..., 2]
            2D points in the perspective image plane, in pixel units (u, v).
            Points with X <= 0 (behind the camera or on the camera plane)
            are assigned NaN in both coordinates.
        persp_depth: np.array w. shape [...]
            Depth along +X for valid points, NaN otherwise.
    """
    carte_points = np.asarray(carte_points, dtype=float)
    if carte_points.shape[-1] != 3:
        raise ValueError("carte_points must have shape [..., 3].")

    X = carte_points[..., 0]
    Y = carte_points[..., 1]
    Z = carte_points[..., 2]

    # Avoid division by zero / points behind camera
    valid = X > 0

    u = np.full_like(X, np.nan, dtype=float)
    v = np.full_like(X, np.nan, dtype=float)

    # u aligned with -Y, v aligned with -Z
    u[valid] = fx * (-Y[valid] / X[valid]) + cx
    v[valid] = fy * (-Z[valid] / X[valid]) + cy

    persp_points = np.stack((u, v), axis=-1)
    persp_depth = np.where(valid, X, np.nan)  # Depth along +X for valid points, NaN otherwise
    return persp_points, persp_depth

def persp2carte_3D(persp_points: np.ndarray,
                    persp_depth: np.ndarray,
                    fx: float,
                    fy: float,
                    cx: float,
                    cy: float):
    """
    Back-project 2D perspective image points to 3D Cartesian coordinates,
    assuming a given depth along the camera optical axis (+X).

    Camera convention:
        - Same as carte2persp_3D above.
        - depth corresponds to X (distance along +X).

        Forward = +X
        u aligned with -Y
        v aligned with -Z

        From projection:
            u = fx * (-Y / X) + cx  =>  Y = -(u - cx) * X / fx
            v = fy * (-Z / X) + cy  =>  Z = -(v - cy) * X / fy

    Args:
        persp_points: np.array w. shape [..., 2]
            2D points in pixel units (u, v).
        persp_depth: np.array broadcastable to persp_points[..., 0]
            Depth along +X (i.e., X coordinate in camera frame).
        fx, fy, cx, cy: floats
            Intrinsic parameters as in carte2persp_3D.

    Returns:
        carte_points: np.array w. shape [..., 3]
            3D points in Cartesian coordinates (X, Y, Z).
    """
    persp_points = np.asarray(persp_points, dtype=float)
    if persp_points.shape[-1] != 2:
        raise ValueError("persp_points must have shape [..., 2].")

    persp_depth = np.asarray(persp_depth, dtype=float)

    u = persp_points[..., 0]
    v = persp_points[..., 1]

    X = persp_depth
    # Inverse of the new projection with u/v aligned to -Y/-Z
    Y = -(u - cx) * X / fx
    Z = -(v - cy) * X / fy

    carte_points = np.stack((X, Y, Z), axis=-1)
    return carte_points

# cylindrical coordinates functions
def cart2cyl_xaxis(pts):
    """
    Convert Cartesian (X, Y, Z) -> cylindrical coordinates aligned with the X-axis.
    
    Works with any input shape [..., 3].

    Convention:
      - x = X               (axis along +X)
      - p = sqrt(Y^2 + Z^2) (radial distance)
      - theta = atan2(Z, Y) in [-pi, pi], with theta=0 pointing toward +Y.

    Parameters
    ----------
    pts : np.ndarray, shape (..., 3)
        Cartesian points [X, Y, Z].

    Returns
    -------
    cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates [x, p, theta].
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Input must have shape [..., 3].")
        
    X, Y, Z = np.moveaxis(pts, -1, 0)
    x = X
    p = np.sqrt(Y**2 + Z**2)
    theta = np.arctan2(Z, Y)  # θ=0 along +Y, increases toward +Z

    return np.stack((x, p, theta), axis=-1)

def cyl2cart_xaxis(cyl):
    """
    Convert cylindrical coordinates (aligned with X-axis) -> Cartesian (X, Y, Z).
    
    Works with any input shape [..., 3].

    Convention (inverse of cart2cyl_xaxis):
      - X = x
      - Y = p * cos(theta)
      - Z = p * sin(theta)

    Parameters
    ----------
    cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates [x, p, theta].

    Returns
    -------
    pts : np.ndarray, shape (..., 3)
        Cartesian points [X, Y, Z].
    """
    cyl = np.asarray(cyl, dtype=float)
    if cyl.shape[-1] != 3:
        raise ValueError("Input must have shape [..., 3].")
        
    x, p, theta = np.moveaxis(cyl, -1, 0)
    X = x
    Y = p * np.cos(theta)
    Z = p * np.sin(theta)

    return np.stack((X, Y, Z), axis=-1)

def cart2cyl_zaxis(pts):
    """
    Convert Cartesian (X, Y, Z) -> cylindrical coordinates aligned with the Z-axis.

    Works with any input shape [..., 3].

    Convention:
      - z = Z               (axis along +Z)
      - p = sqrt(X^2 + Y^2) (radial distance in the XY plane)
      - theta = atan2(Y, X) in [-pi, pi], with theta=0 pointing toward +X.

    Parameters
    ----------
    pts : np.ndarray, shape (..., 3)
        Cartesian points [X, Y, Z].

    Returns
    -------
    cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates [z, p, theta].
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError("Input must have shape [..., 3].")

    X, Y, Z = np.moveaxis(pts, -1, 0)
    z = Z
    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)  # θ=0 along +X, increases toward +Y

    return np.stack((z, p, theta), axis=-1)

def cyl2cart_zaxis(cyl):
    """
    Convert cylindrical coordinates (aligned with Z-axis) -> Cartesian (X, Y, Z).

    Works with any input shape [..., 3].

    Convention (inverse of cart2cyl_zaxis):
      - Z = z
      - X = p * cos(theta)
      - Y = p * sin(theta)

    Parameters
    ----------
    cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates [z, p, theta].

    Returns
    -------
    pts : np.ndarray, shape (..., 3)
        Cartesian points [X, Y, Z].
    """
    cyl = np.asarray(cyl, dtype=float)
    if cyl.shape[-1] != 3:
        raise ValueError("Input must have shape [..., 3].")

    z, p, theta = np.moveaxis(cyl, -1, 0)
    Z = z
    X = p * np.cos(theta)
    Y = p * np.sin(theta)

    return np.stack((X, Y, Z), axis=-1)


# spherical / erp coordinates functions
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

def cam_carte2world_3D(points_3D_cam_carte, pose):
    """
    Convert camera Cartesian coordinates to world coordinates.
    
    Args:
        points_3D_cam_carte (np.array): Camera Cartesian coordinates of shape [..., 3].
        pose (np.array): Camera pose matrix of shape [4, 4].
    
    Returns:
       points_3D_world_carte: np.array w. shape [..., 3]. World coordinates. Convention X, Y, Z.
    """
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


# depth2 functions (assumes has shape [H,W])
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


# ---------------------------------------- #
# ----- Panorama / Pointcloud utils  ----- #
# ---------------------------------------- #
def load_rgbd_pano(dream, save_dir_, override_depth_with_ones=False):
    load_dir__ = os.path.join(save_dir_, f"dream_{dream:02d}")
    pano_rgb = Image.open(os.path.join(load_dir__, "XX_pano_rgb.png"))
    colors = PIL_to_numpy(pano_rgb)
    depth = np.load(os.path.join(load_dir__, "XX_depth.npy"))
    if override_depth_with_ones:
        depth = np.ones_like(depth)  
        print("WARNING: depth override to ones for debugging purposes")
    return colors, depth

def save_rgbd_pano(pano_rgb, depth, dream, save_dir_):
    save_dir__ = os.path.join(save_dir_, f"dream_{dream:02d}")
    os.makedirs(save_dir__, exist_ok=True)
    pano_rgb.save(os.path.join(save_dir__, "XX_pano_rgb.png"))
    np.save(os.path.join(save_dir__, "XX_depth.npy"), depth)
    depth_numpy_to_PIL(depth).save(os.path.join(save_dir__, "XX_depth.png"))
    depth_numpy_to_figure(depth).savefig(os.path.join(save_dir__, "XX_depth_figure.png"))

def save_rgbd_ldi_pano(pano_rgb_bg, depth_bg, mask_bg, dream, save_dir_, phase):
    if phase == 1:
        save_dir__ = os.path.join(save_dir_, f"dream_{dream:02d}")
    elif phase == 2:
        save_dir__ = os.path.join(save_dir_, f"align_{dream:02d}")
    else:
        raise ValueError("phase must be 1 or 2, received:", phase)
    
    os.makedirs(save_dir__, exist_ok=True)
    pano_rgb_bg.save(os.path.join(save_dir__, "ZZ_ldi_pano_rgb.png"))
    np.save(os.path.join(save_dir__, "ZZ_ldi_depth.npy"), depth_bg)
    depth_numpy_to_PIL(depth_bg).save(os.path.join(save_dir__, "ZZ_ldi_depth.png"))
    depth_numpy_to_figure(depth_bg).savefig(os.path.join(save_dir__, "ZZ_ldi_depth_figure.png"))
    numpy_bool_to_pil_mask(mask_bg).save(os.path.join(save_dir__, "ZZ_ldi_mask.png"))

def load_rgbd_ldi_pano(dream, save_dir_, phase):
    if phase == 1:
        load_dir__ = os.path.join(save_dir_, f"dream_{dream:02d}")
    elif phase == 2:
        load_dir__ = os.path.join(save_dir_, f"align_{dream:02d}")
    else:
        raise ValueError("phase must be 1 or 2, received:", phase)
    
    pano_rgb_bg = Image.open(os.path.join(load_dir__, "ZZ_ldi_pano_rgb.png"))
    colors_bg = PIL_to_numpy(pano_rgb_bg)
    depth_bg = np.load(os.path.join(load_dir__, "ZZ_ldi_depth.npy"))
    mask_bg = pil_mask_to_numpy_bool(Image.open(os.path.join(load_dir__, "ZZ_ldi_mask.png")))

    return colors_bg, depth_bg, mask_bg


class PointCloud:
    def __init__(self, pts, colors):
        """
        pts: np.array of shape [..., 3]
        colors: np.array of shape [..., 3] with values in [0-1]
        """
        self.pts = pts.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)
        assert self.pts.shape[0] == self.colors.shape[0], "Error: pts and colors must have the same number of points"

    def get_o3d_pointcloud(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd

    # def get_o3d_pointcloud(self):
    #     import open3d as o3d
    #     t = o3d.core.Tensor(self.pts)
    #     pcd = o3d.t.geometry.PointCloud(t)
    #     pcd.point['colors'] = o3d.core.Tensor(self.colors)
    #     self.pcd = pcd
    #     return self

    # @property
    # def points(self):
    #     return self.pcd.point['points']

    # @property
    # def colors(self):
    #     return self.pcd.point['colors']

    # @points.setter
    # def points(self, new_points):
    #     self.pcd.point['points'] = o3d.core.Tensor(new_points)

    # @colors.setter
    # def colors(self, new_colors):
    #     self.pcd.point['colors'] = o3d.core.Tensor(new_colors)

class SphereState:
    def __init__(self, pts_carte, colors, pose):
        """everything in spherical coordinates"""
        self.pts = pts_carte
        self.colors = colors
        self.pose = pose
        self.is_world_pcd_init=False
    
    def init_world_pcd(self):
        assert not self.is_world_pcd_init, "World pointcloud is already initialized"
        assert self.pose is not None, "Pose must be defined to compute world pointcloud"

        self.world_pcd = PointCloud(
            pts=cam_carte2world_3D(self.pts, self.pose),
            colors=self.colors
        )
        self.is_world_pcd_init=True

    def get_world_pcd(self):
        "returns pointcloud in world coordinates"

        if not self.is_world_pcd_init:
            self.init_world_pcd()

        return self.world_pcd

    def update_pose(self, new_pose):
        self.pose = new_pose
        self.is_world_pcd_init=False

_default_opening_kwargs = {
    'opening_mode': 'cut+cylinder',
    'delta_cut': 2*np.pi/3,
    'cut_distance_percentile': 90,
}

class Sphere:

    def __init__(self, pose, pts_carte, colors, forward_sph=None, forward_carte=None, opening_kwargs=_default_opening_kwargs):
        """
        Can be derived in four different ways: open left, open right, open both, open none
        Input points are expected not to be opened ye.
        A sphere has a forward direction, which can be expressed in spherical coordinates or cartesian coordinates
        """
        assert (forward_sph is not None) or (forward_carte is not None), "Error: forward direction must be provided in either spherical or cartesian coordinates"
        assert pts_carte.shape[-1] == 3, "Error: pts_carte must have shape [N, 3]"
        assert colors.shape[-1] == 3, "Error: colors must have shape [N, 3]"
        assert pts_carte.reshape(-1, 3).shape[0] == colors.reshape(-1, 3).shape[0], "Error: pts_carte and colors must have the same number of points"
        self.forward_sph = forward_sph if forward_sph is not None else carte2sph_3D(forward_carte)
        self.forward_carte = forward_carte if forward_carte is not None else sph2carte_3D(forward_sph)
        self.opening_kwargs = opening_kwargs
        self.pose = pose

        # open the Sphere in all four ways to get all states
        t = time.time()
        self.init_states(pts_carte, colors)
        print(f"Sphere init took {time.time() - t:.1f}s")

    @staticmethod
    def filter_nan(pts_carte, colors):
        mask_finite = np.isfinite(pts_carte).all(axis=-1) & np.isfinite(colors).all(axis=-1)
        pts_carte = pts_carte[mask_finite]
        colors = colors[mask_finite]
        return pts_carte, colors
    
    def init_states(self, pts_carte, colors):
        # filter nan
        self.pts_carte, self.colors = self.filter_nan(pts_carte, colors)
        # compute all openings
        self.closed = self._close(self.pts_carte, self.colors)
        self.both_opened = self._open_both(self.pts_carte, self.colors)
        self.left_opened = self._open_left(self.pts_carte, self.colors)
        self.right_opened = self._open_right(self.pts_carte, self.colors)

    def get_state(self, open_left, open_right):
        if open_left and open_right:
            return self.both_opened_sphere
        elif open_left and not open_right:
            return self.left_opened_sphere
        elif not open_left and open_right:
            return self.right_opened_sphere
        else:
            return self.closed_sphere
    
    def _close(self, pts_carte, colors):
        sphere_closed = SphereState(
            pts_carte=pts_carte, 
            colors=colors,
            pose=self.pose
        )
        return sphere_closed
    
    def _open_left(self, pts_carte, colors):
        _, open_pts_carte, mask_opening = open_world_carte(
            forward_carte=-self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        sphere_opened_left = SphereState(
            pts_carte=open_pts_carte[mask_opening], 
            colors=colors[mask_opening],
            pose=self.pose
        )
        return sphere_opened_left

    def _open_right(self, pts_carte, colors):
        _, open_pts_carte, mask_opening = open_world_carte(
            forward_carte=self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        sphere_opened_right = SphereState(
            pts_carte=open_pts_carte[mask_opening], 
            colors=colors[mask_opening],
            pose=self.pose
        )
        return sphere_opened_right

    def _open_both(self, pts_carte, colors):
        _, open_pts_carte, mask_opening1 = open_world_carte(
            forward_carte=self.forward_carte,
            pts_carte=pts_carte,
            **self.opening_kwargs
        )
        open_pts_carte = open_pts_carte[mask_opening1]
        colors = colors[mask_opening1]

        _, open_pts_carte, mask_opening2 = open_world_carte(
            forward_carte=-self.forward_carte,
            pts_carte=open_pts_carte,
            **self.opening_kwargs
        )
        open_pts_carte = open_pts_carte[mask_opening2]
        colors = colors[mask_opening2]
        sphere_opened_both = SphereState(
            pts_carte=open_pts_carte, 
            colors=colors,
            pose=self.pose
        )

        return sphere_opened_both

    def add_new_points(self, new_pts_carte, new_colors):
        pts_carte = np.concatenate((self.pts_carte, new_pts_carte.reshape(-1, 3)), axis=0)
        colors = np.concatenate((self.colors, new_colors.reshape(-1, 3)), axis=0)
        self.init_states(pts_carte, colors)

    def update_pose(self, new_pose):
        self.pose = new_pose
        for state in [self.closed, self.both_opened, self.left_opened, self.right_opened]:
            state.update_pose(new_pose)

    def save_dict(self, path):
        """
        Save the current Sphere to `path`, including metadata and base points/colors.
        The different opened/closed states will be recomputed when loading.
        """
        data = {
            "pose": self.pose,
            "forward_sph": self.forward_sph,
            "forward_carte": self.forward_carte,
            "opening_kwargs": self.opening_kwargs,
            "pts_carte": self.pts_carte,
            "colors": self.colors,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def instanciate_from_saved_dict(path):
        """
        Load a Sphere previously saved with `saved_dict` and return an equivalent instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle possible older saves that might not have all keys
        pose = data["pose"]
        pts_carte = data["pts_carte"]
        colors = data["colors"]
        forward_sph = data.get("forward_sph", None)
        forward_carte = data.get("forward_carte", None)
        opening_kwargs = data.get("opening_kwargs", _default_opening_kwargs)

        sphere = Sphere(
            pose=pose,
            pts_carte=pts_carte,
            colors=colors,
            forward_sph=forward_sph,
            forward_carte=forward_carte,
            opening_kwargs=opening_kwargs,
        )
        return sphere

def camera_translation(pose, translation):
    """
    pose: np.array of shape [4,4]
    translation: np.array of shape [3,] in world coordinates
    """
    pose2 = pose.copy()
    pose2[:3, 3] += translation
    return pose2


# ---------------------------------------- #
# ------- Geometry Correction utils  ----- #
# ---------------------------------------- #
class Regression1D:

    @staticmethod
    def fit_nw_grid_interpolator_1d(X, Y, bandwidth, grid_size=1024, margin=3.0):
        """
        Fit-time:
        - builds a 1D grid that extends beyond data by `margin * bandwidth`,
        - evaluates Nadaraya–Watson (Gaussian) on that grid,
        - returns an inference-only f(x) that linearly interpolates on the grid.

        Parameters
        ----------
        X : array-like, shape (n,)
            Training inputs.
        Y : array-like, shape (n,)
            Training targets.
        bandwidth : float
            Gaussian kernel width (σ). Larger => smoother.
        grid_size : int, default 1024
            Number of grid points to precompute.
        margin : float, default 3.0
            Extra range (in units of σ) added on both sides of [min(X), max(X)]
            to stabilize edge behavior and improve clamped extrapolation.

        Returns
        -------
        f : callable
            Inference-only function. Accepts batched x with shape [...] and returns shape [...].
            Interpolates within the grid; clamps to edge values outside the grid.
        """
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        assert X.ndim == 1 and Y.ndim == 1 and X.size == Y.size, "X and Y must be 1D and same length."
        assert bandwidth > 0.0, "bandwidth must be positive."

        # Build grid with padding to mitigate boundary artifacts
        x_min = X.min() - margin * bandwidth
        x_max = X.max() + margin * bandwidth
        x_grid = np.linspace(x_min, x_max, int(grid_size))

        # Evaluate NW smoother on the grid (vectorized, O(grid_size * n))
        D = (x_grid[:, None] - X[None, :]) / bandwidth                  # shape (G, n)
        W = np.exp(-0.5 * D**2)                                         # Gaussian kernels
        W_sum = W.sum(axis=1) + 1e-12                                   # avoid divide-by-zero
        y_grid = (W @ Y) / W_sum                                        # shape (G,)

        # Inference-only interpolator: piecewise-linear + clamped extrapolation
        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            # np.interp clamps to left/right values if outside the grid
            y_flat = np.interp(x_flat, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])
            return y_flat.reshape(x.shape)

        return f

    @staticmethod
    def fit_local_min_knots_interpolator_1d(
        X, Y, bandwidth, *, handle_empty="skip", tie_break="center"
    ):
        
        """
        One-shot: build (X_min, Y_min) knots per bin, and return an
        inference-only monotone interpolator.
        """
                
        X = np.asarray(X, dtype=float).ravel()
        Y = np.asarray(Y, dtype=float).ravel()
        if X.size != Y.size:
            raise ValueError("X and Y must have the same length.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("X and Y must be finite.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0.")

        x_min, x_max = X.min(), X.max()
        n_bins = max(1, int(np.ceil((x_max - x_min) / bandwidth)))
        bin_edges = x_min + np.arange(n_bins + 1) * bandwidth
        if bin_edges[-1] < x_max:
            bin_edges[-1] = x_max

        bin_idx = np.digitize(X, bin_edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        X_k = np.full(n_bins, np.nan)
        Y_k = np.full(n_bins, np.nan)

        for b in range(n_bins):
            mask = (bin_idx == b)
            if not np.any(mask):
                continue
            Xb = X[mask]
            Yb = Y[mask]
            y_min = np.min(Yb)
            tie_mask = (Yb == y_min)
            if tie_break == "first":
                i = np.argmax(tie_mask)
            elif tie_break == "last":
                i = len(Yb) - 1 - np.argmax(tie_mask[::-1])
            else:  # "center"
                center = 0.5 * (bin_edges[b] + bin_edges[b + 1])
                idxs = np.flatnonzero(tie_mask)
                i = idxs[np.argmin(np.abs(Xb[idxs] - center))]
            X_k[b] = Xb[i]
            Y_k[b] = y_min

        if handle_empty == "nearest":
            # forward fill
            for i in range(1, n_bins):
                if not np.isfinite(Y_k[i]):
                    X_k[i] = X_k[i - 1]
                    Y_k[i] = Y_k[i - 1]
            # backward fill
            for i in range(n_bins - 2, -1, -1):
                if not np.isfinite(Y_k[i]):
                    X_k[i] = X_k[i + 1]
                    Y_k[i] = Y_k[i + 1]

        finite = np.isfinite(X_k) & np.isfinite(Y_k)
        X_k = X_k[finite]
        Y_k = Y_k[finite]
        if X_k.size == 0:
            def f_nan(x):
                x = np.asarray(x, dtype=float)
                return np.full_like(x, np.nan)
            return f_nan, X_k, Y_k

        order = np.argsort(X_k)
        X_k = X_k[order]
        Y_k = Y_k[order]

        uniq_x, inv = np.unique(X_k, return_inverse=True)
        if uniq_x.size != X_k.size:
            y_min_by_x = np.full_like(uniq_x, np.inf, dtype=float)
            np.minimum.at(y_min_by_x, inv, Y_k)
            X_k, Y_k = uniq_x, y_min_by_x

        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            y_flat = np.interp(x_flat, X_k, Y_k, left=Y_k[0], right=Y_k[-1])
            return y_flat.reshape(x.shape)

        return f, X_k, Y_k

    @staticmethod
    def _isotonic_l2_pav(y, w=None):
        """
        Pool-Adjacent-Violators for nondecreasing isotonic regression (L2).
        Returns the closest (in weighted L2) nondecreasing vector to y.
        """
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return y
        if w is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
        # Initialize blocks
        v = y.copy()
        wv = w.copy()
        # Stack of block end indices
        end = [0]
        for i in range(n):
            v[i] = y[i]
            wv[i] = w[i]
            end.append(i + 1)
            # Merge while violating monotonicity
            while len(end) >= 3:
                i2 = end[-1]      # end of last block
                i1 = end[-2]      # start of last block
                i0 = end[-3]      # start of penultimate block
                if v[i1 - 1] <= v[i2 - 1]:
                    break
                # pool blocks [i0:i1] and [i1:i2]
                tot_w = wv[i1 - 1] + wv[i2 - 1]
                avg = (wv[i1 - 1] * v[i1 - 1] + wv[i2 - 1] * v[i2 - 1]) / tot_w
                v[i1 - 1] = avg
                wv[i1 - 1] = tot_w
                # pop last block
                end.pop()
                end[-1] = i2  # extend previous block to new end
        # Expand block means
        y_iso = np.empty_like(y)
        start = 0
        for e in end[1:]:
            y_iso[start:e] = v[e - 1]
            start = e
        return y_iso

    @staticmethod
    def _make_monotone_increasing_from_knots(
        X_knots, Y_knots, *, weights=None, lower=None, upper=None,
        strict=False, eps=1e-12
    ):
        """
        Enforce nondecreasing Y over X_knots via isotonic regression, then
        return an inference-only linear interpolator over (X_knots, Y_iso).

        - 'sticks' to Y_knots wherever they already satisfy monotonicity.
        - optional bounds 'lower'/'upper' clamp the final curve.
        - if strict=True, nudges flat segments by tiny eps to be strictly increasing.

        Returns f_mono, (X_knots_sorted, Y_iso)
        """
        Xk = np.asarray(X_knots, dtype=float)
        Yk = np.asarray(Y_knots, dtype=float)
        mask = np.isfinite(Xk) & np.isfinite(Yk)
        Xk, Yk = Xk[mask], Yk[mask]
        if Xk.size == 0:
            def f_nan(x):
                x = np.asarray(x, dtype=float)
                return np.full_like(x, np.nan)
            return f_nan, (Xk, Yk)

        order = np.argsort(Xk)
        Xk = Xk[order]
        Yk = Yk[order]
        if weights is not None:
            w = np.asarray(weights, dtype=float)[mask][order]
        else:
            w = None

        Y_iso = Regression1D._isotonic_l2_pav(Yk, w=w)

        if lower is not None:
            Y_iso = np.maximum(Y_iso, lower)
        if upper is not None:
            Y_iso = np.minimum(Y_iso, upper)

        if strict:
            # Make strictly increasing by adding tiny offsets to flat runs
            # while staying within bounds if provided.
            diffs = np.diff(Y_iso)
            flat_idx = np.where(diffs <= 0)[0]
            k = 0
            for i in flat_idx:
                k += 1
                Y_iso[i + 1] = max(Y_iso[i + 1], Y_iso[i] + eps)
            # optional: re-clip
            if lower is not None:
                Y_iso = np.maximum(Y_iso, lower)
            if upper is not None:
                Y_iso = np.minimum(Y_iso, upper)

        def f(x):
            x = np.asarray(x, dtype=float)
            x_flat = x.ravel()
            y_flat = np.interp(x_flat, Xk, Y_iso, left=Y_iso[0], right=Y_iso[-1])
            return y_flat.reshape(x.shape)

        return f, (Xk, Y_iso)

    @staticmethod
    def fit_local_min_knots_monotone_interpolator_1d(
        X, Y, bandwidth, *, handle_empty="skip", tie_break="center",
        lower=None, upper=None, strict=False, weights=None
    ):
        """
        One-shot: build (X_min, Y_min) knots per bin, then enforce
        nondecreasing Y via isotonic regression, and return an
        inference-only monotone interpolator.
        """
        _, Xk, Yk = Regression1D.fit_local_min_knots_interpolator_1d(
            X, Y, bandwidth, handle_empty=handle_empty, tie_break=tie_break
        )
        f_mono, (Xk_sorted, Y_iso) = Regression1D._make_monotone_increasing_from_knots(
            Xk, Yk, weights=weights, lower=lower, upper=upper, strict=strict
        )
        
        return f_mono

class GeometryTransforms:

    @staticmethod
    def depth_transform(
        D_raw: np.ndarray,
        n: float = 0.5,      # near (meters)
        f: float = 200.0,    # far  (meters)
        method: str = "inv",  # "inv", "exp", "gamma", 'threshold'
        gamma: float = 0.6,  # used if method=="gamma" (gamma<1 expands far)
        k: float = 3.0,       # used if method=="exp"   (larger k → more far expansion)
        plot: bool = False
    ) -> np.ndarray:
        """
        Map depth D in [0,1] (near≈0, far≈1) to metric range Z in [n,f], monotonically increasing.

        Methods:
        - "linear":   Z = n + D*(f-n)  (identity, no correction)
        - "inv":  assumes D is ~linear in 1/Z but with near→0, far→1.
                        Z = 1 / ( 1/n - D*(1/n - 1/f) )
                        (Excellent default to remove far stacking.)
        - "exp":       convex exponential easing toward f:
                        s = (exp(k*D)-1)/(exp(k)-1); Z = n + s*(f-n)
                        (k>0; increases separation at large D.)
        - "gamma":     gamma pre-warp then linear:
                        Dg = D**gamma (gamma<1 expands far); Z = n + Dg*(f-n)
        """
        D = np.asarray(D_raw, dtype=np.float32)
        D = np.clip(D, 0.0, 1.0)
        
        if plot: 
            plot_d = np.linspace(0, 1, 500)

        if method == "linear":
            Z = D * (f - n) + n
            if plot:
                Z_plot = plot_d * (f - n) + n

        elif method =='threshold':
            def corr(D):
                D[D>0.9] = f
                return D
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "inv":
            n = max(n, 1e-3)  # avoid div-by-zero
            def corr(D):
                denom = (1.0 / n) - D * (1.0 / n - 1.0 / f)
                Z = 1.0 / np.clip(denom, 1e-9, None)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "exp":
            def corr(D):
                s = (np.exp(k * D) - 1.0) / (np.exp(k) - 1.0 + 1e-9)
                Z = n + s * (f - n)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        elif method == "gamma":
            def corr(D):
                Dg = D ** gamma   # gamma<1 expands high-D region
                Z = n + Dg * (f - n)
                return Z
            Z = corr(D)
            if plot:
                Z_plot = corr(plot_d)

        else:
            raise ValueError(f"Unknown method: {method}")

        if plot:
            plt.figure()
            plt.plot(plot_d, Z_plot, label='corrected')
            plt.plot(plot_d, plot_d * (f - n) + n, '--', label='linear')
            plt.legend()
            plt.xlabel("Input D (0=near, 1=far)")
            plt.ylabel("Output Z (meters)")
            plt.title(f"Depth Linearization: method={method}, n={n}, f={f}")
            plt.grid()
            plt.show()
        # Optional: set invalid/zero inputs to NaN
        Z[~np.isfinite(D)] = np.nan
        return Z

    @staticmethod
    def _l2_errors(x, y):
        "norm over last axis"
        return np.sqrt((x - y)**2)

    @staticmethod
    def _l1_errors(x, y):
        "norm over last axis"
        return np.abs(x - y)

    @staticmethod
    def correct_floor_old(P, depth_map_eqr, error_type='l1', plot=False):
        """
        Correct points using trigonometry and an heuristic to make the floor flat
        The next version is better. Keeping this one just for reference.
        """
        thetas = get_canonical_sph_pixels(height, width)[..., 0]
        avg_depth_vertical = np.nanmean(depth_map_eqr, axis=1)  # [H, ]
        r_horizon_theta_range=(
            np.deg2rad(-10), np.deg2rad(-1) 
        )
        r_horizon_band_mask = (thetas[:,0] >= r_horizon_theta_range[0]) & (thetas[:,0] <= r_horizon_theta_range[1])
        r_horizon = np.nanmean(avg_depth_vertical[r_horizon_band_mask])  # scalar

        if error_type=='l2':
            strength = GeometryTransforms._l2_errors(depth_map_eqr, r_horizon)
        elif error_type=='l1':
            strength = GeometryTransforms._l1_errors(depth_map_eqr, r_horizon)
        else:
            raise ValueError(f"Unknown error type: {error_type}. Choose from 'l1' or 'l2'.")
        
        strength = (strength - np.min(strength)) / (np.max(strength) - np.min(strength) + 1e-8)
        strength[thetas >= 0] = 0.0

        correction_raw =  (avg_depth_vertical[:, None] * np.cos(thetas + np.pi/2))[..., None] * np.array([0, 0, 1])
        correction = strength[..., None] * correction_raw
        corrected_pts = P + correction

        if plot:
            fig, axes = plt.subplots(3,2, figsize=(8,16))

            axes[0,0].set_title("Depth")
            axes[0,0].imshow(depth_map_eqr)
            fig.colorbar(axes[0,0].imshow(depth_map_eqr), ax=axes[0,0])

            axes[0,1].set_title("Depth Correction")
            axes[0,1].imshow(correction[..., 2])
            fig.colorbar(axes[0,1].imshow(correction[..., 2]), ax=axes[0,1])
            
            axes[1,0].set_title("Correction Raw")
            axes[1,0].imshow(correction_raw)
            fig.colorbar(axes[1,0].imshow(correction_raw[..., 2]), ax=axes[1,0])


            axes[1,1].set_title("Correction Strength")
            axes[1,1].imshow(strength)
            fig.colorbar(axes[1,1].imshow(strength), ax=axes[1,1])

            axes[2,0].set_title("Depth and Correction Profile")
            correction_profile = np.nanmean(correction[..., 2], axis=1)
            axes[2,0].plot(thetas[:,0], correction_profile, label="Average depth correction")
            theta_band = np.nanmean(thetas, axis=1)
            avg_depth_vertical = np.nanmean(depth_map_eqr, axis=1)
            axes[2,0].plot(theta_band, avg_depth_vertical, label="average depth (before)")
            axes[2,0].legend()
            axes[2,0].set_xlabel("Elevation (radians)")
            axes[2,0].set_ylabel("Average Depth")

            axes[2,1].set_title("Z value Profiles before/after correction")
            z_before = P[..., 2].mean(axis=1)
            axes[2,1].plot(theta_band, z_before, label="Before correction")
            axes[2,1].set_xlabel("Elevation (radians)")
            axes[2,1].set_ylabel("Average Z value")

            z_after = corrected_pts[..., 2].mean(axis=1)
            axes[2,1].plot(theta_band, z_after, label="After correction")
            axes[2,1].legend()
            plt.tight_layout()
            plt.show()

            # show z axis 
            fig, axes = plt.subplots(2,1)
            fig.suptitle("Z values before/after correction")
            axes[0].set_title("Before Correction")
            im0 = axes[0].imshow(P[..., 2], vmin=-1, vmax=1)
            fig.colorbar(im0, ax=axes[0])
            axes[1].set_title("After Correction")
            im1 = axes[1].imshow(corrected_pts[..., 2], vmin=-1, vmax=1)
            fig.colorbar(im1, ax=axes[1])
            plt.tight_layout()
            plt.show()


        return corrected_pts, correction, correction_raw, strength

    @staticmethod
    def build_floor_correction(
        X, Y, Z, theta,
        *,
        correct_until_depth_metric,
        dx=0.05, dy=0.05,
        q=10.0,
        theta_min=-np.pi, theta_max=0.0,
        min_pts_per_bin=10,
        gaussian_sigma_xy=(1.0, 1.0),
        reference_level='horizon',  # 'horizon' | 'median' | 'mean' | float
        epsilon=0.0,                 # optional C^1 blend width at Y boundaries (meters)
        plot_horizon=False,
        plot_floor_profile=False,
        horizon_deg=(-10.0, -1.0),
        horizon_colors=None,         # optional colors aligned with input if you plot horizon
    ):
        """
        Tools to estimate a road-floor correction surface C(x,y) from a 3D point cloud,
        using only points below the horizon (theta in (-pi, 0)), then apply this correction
        to all points. The correction is computed on a (x,y) grid as a *low percentile*
        (robust floor) of Z per cell, filled and smoothed to form a continuous surface.

        Key features
        ------------
        - Robust floor estimate via low-percentile per (x,y) bin.
        - Holes filled by nearest-value, then Gaussian-smoothed.
        - Reference level can be the "horizon" band, grid median/mean, or a constant.
        - `C(x,y)` is *truncated by continuity* outside |Y| <= correct_until_depth_metric:
            C(x,y<Ymin) := C(x, Ymin)  and  C(x,y>Ymax) := C(x, Ymax)
        (Optionally with a C^1 soft blend width `epsilon`.)
        - All plotting is optional and OFF by default.

        Parameters
        ----------
        X, Y, Z, theta : array_like (same shape)
            Point cloud coordinates and polar angle per point (radians).
            Only points with theta ∈ (theta_min, theta_max) are used to estimate the floor.
        correct_until_depth_metric : float
            Y-band half-width; `C(x,y)` is prolonged by continuity outside
            [-correct_until_depth_metric, +correct_until_depth_metric]:
                C(x,y<Ymin) := C(x, Ymin),  C(x,y>Ymax) := C(x, Ymax).
        dx, dy : float
            Grid resolutions along X and Y for the floor estimation (meters).
        q : float
            Low percentile (e.g., 5–10) used as robust floor per (x,y) cell.
        theta_min, theta_max : float
            Horizon mask range (radians) used for selecting floor points. Default (-π, 0).
        min_pts_per_bin : int
            Minimum number of points to accept a bin; others are filled by nearest.
        gaussian_sigma_xy : (float, float)
            Gaussian smoothing sigmas (in *cells*, not meters) along (X,Y).
        reference_level : {'horizon','median','mean'} or float
            Base level z0 for the correction: horizon-band mean, grid median/mean, or fixed float.
        epsilon : float
            Optional blend half-width (meters) to make the boundary transitions C^1. Set 0 for hard.
        plot_horizon : bool
            If True, show the 3D scatter of the horizon band used for 'horizon' reference.
        plot_floor_profile : bool
            If True, show the estimated floor Z(x,y) as an image.
        horizon_deg : (float, float)
            Angular band in degrees for the horizon (used if reference_level='horizon').
        horizon_colors : array_like or None
            Optional per-point colors for the horizon 3D scatter (if plotted).

        Returns
        -------
        C_func : callable
            Function C(x, y) returning the correction. Continuous prolongation is enforced
            along Y outside [-correct_until_depth_metric, +correct_until_depth_metric].
        grid : tuple of (x_centers, y_centers)
            Regular grid used to estimate Zfloor.
        Zfloor : ndarray, shape (len(x_centers), len(y_centers))
            Smoothed floor surface on the grid (before subtracting z0).
        """
        # --- 0) sanitize input ---
        X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z); theta = np.asarray(theta)
        if not (X.shape == Y.shape == Z.shape == theta.shape):
            raise ValueError("X, Y, Z, theta must have the same shape.")

        # --- 1) mask by theta in (theta_min, theta_max) ---
        mask = (theta > theta_min) & (theta < theta_max) & np.isfinite(Z)
        Xg, Yg, Zg = X[mask], Y[mask], Z[mask]
        if Xg.size == 0:
            raise ValueError("No points remain after theta masking; check theta range.")

        # --- 2) build grid along X,Y using masked points extent ---
        x_min, x_max = np.min(Xg), np.max(Xg)
        y_min, y_max = np.min(Yg), np.max(Yg)
        x_edges = np.arange(x_min, x_max + dx, dx)
        y_edges = np.arange(y_min, y_max + dy, dy)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        H, W = x_centers.size, y_centers.size

        # bin indices for masked points
        ix = np.clip(np.digitize(Xg, x_edges) - 1, 0, H - 1)
        iy = np.clip(np.digitize(Yg, y_edges) - 1, 0, W - 1)

        # --- 3) per-bin low-percentile floor ---
        flat_idx = ix * W + iy
        order = np.argsort(flat_idx)
        flat_idx_sorted = flat_idx[order]
        Z_sorted = Zg[order]

        Zfloor = np.full((H, W), np.nan, dtype=float)
        counts = np.zeros((H, W), dtype=int)

        start = 0
        n = flat_idx_sorted.size
        while start < n:
            stop = start + 1
            key = flat_idx_sorted[start]
            while stop < n and flat_idx_sorted[stop] == key:
                stop += 1
            i, j = divmod(key, W)
            Zij = Z_sorted[start:stop]
            counts[i, j] = Zij.size
            if Zij.size >= min_pts_per_bin:
                Zfloor[i, j] = np.percentile(Zij, q)
            start = stop

        # --- 4) nearest fill for holes ---
        valid = np.isfinite(Zfloor)
        if not np.any(valid):
            raise ValueError("All bins empty; increase min_pts_per_bin or adjust dx/dy.")
        _, (ii, jj) = ndimage.distance_transform_edt(~valid, return_indices=True)
        Zfilled = Zfloor.copy()
        Zfilled[~valid] = Zfloor[ii[~valid], jj[~valid]]

        # --- 5) smooth the surface ---
        sx, sy = gaussian_sigma_xy
        if sx > 0 or sy > 0:
            Zsmooth = ndimage.gaussian_filter(Zfilled, sigma=(sx, sy))
        else:
            Zsmooth = Zfilled

        # --- 6) choose reference level z0 ---
        if reference_level == "horizon":
            lo, hi = np.deg2rad(horizon_deg[0]), np.deg2rad(horizon_deg[1])
            # two symmetric bands around pi for wrap
            mask1 = in_interval_mod(theta, lo, hi)
            # symmetric around pi: (π - lo, π - hi) but order might flip:
            lo2, hi2 = (np.pi - lo), (np.pi - hi)
            mask2 = in_interval_mod(theta, min(lo2, hi2), max(lo2, hi2))
            horizon_mask = (mask1 | mask2) & np.isfinite(Z)
            if not np.any(horizon_mask):
                raise ValueError("No points in horizon band; adjust horizon_deg or data.")
            z0 = np.nanmean(Z[horizon_mask])

            if plot_horizon:
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111, projection='3d')
                if horizon_colors is None:
                    ax.scatter(X[horizon_mask], Y[horizon_mask], Z[horizon_mask], s=1, c='orange')
                else:
                    ax.scatter(X[horizon_mask], Y[horizon_mask], Z[horizon_mask], s=1,
                               c=np.asarray(horizon_colors)[horizon_mask])
                ax.set_title('Horizon band points used for reference level')
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
                plt.tight_layout(); plt.show()
        elif reference_level == "median":
            z0 = np.nanmedian(Zsmooth)
        elif reference_level == "mean":
            z0 = np.nanmean(Zsmooth)
        else:
            z0 = float(reference_level)

        # --- 7) build correction grid and continuous prolongation C(x,y) ---
        C_grid = -(Zsmooth - z0)  # so Z_corrected = Z + C(x,y)

        C_interp = RegularGridInterpolator(
            (x_centers, y_centers), C_grid, bounds_error=False, fill_value=None
        )

        # boundaries for continuity prolongation along Y
        Ymin_band = -float(correct_until_depth_metric)
        Ymax_band = +float(correct_until_depth_metric)
        # ensure we evaluate *on the grid* when we sample boundary traces:
        Ymin_eval = float(np.clip(Ymin_band, y_centers[0], y_centers[-1]))
        Ymax_eval = float(np.clip(Ymax_band, y_centers[0], y_centers[-1]))

        def _smoothstep(t):
            return t*t*(3 - 2*t)

        def C_func(x, y):
            """
            Evaluate the correction with continuous prolongation outside the Y band.

            For y < Ymin: returns C(x, Ymin).
            For y > Ymax: returns C(x, Ymax).
            For y within [Ymin, Ymax]: returns C(x, y) from the interpolator.
            If epsilon > 0, blends across [Ymin-eps, Ymin] and [Ymax, Ymax+eps] for C^1.
            """
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            shp = np.shape(x)

            xf = x.ravel()
            yf = y.ravel()

            # Evaluate boundary traces (functions of x)
            c_low  = C_interp(np.column_stack([xf, np.full_like(xf, Ymin_eval)]))
            c_high = C_interp(np.column_stack([xf, np.full_like(xf, Ymax_eval)]))

            c = np.empty_like(xf, dtype=float)

            if epsilon <= 0:
                # Hard (C^0) continuation
                mask_low  = (yf <= Ymin_band)
                mask_high = (yf >= Ymax_band)
                mask_mid  = ~(mask_low | mask_high)

                # interior (clamped to grid Y-range)
                y_mid = np.clip(yf[mask_mid], y_centers[0], y_centers[-1])
                c[mask_mid] = C_interp(np.column_stack([xf[mask_mid], y_mid]))
                c[mask_low]  = c_low[mask_low]
                c[mask_high] = c_high[mask_high]
            else:
                # Smooth (C^1) blend of width epsilon
                mask_low_outer   = (yf <= Ymin_band - epsilon)
                mask_low_blend   = (yf >  Ymin_band - epsilon) & (yf < Ymin_band)
                mask_mid         = (yf >= Ymin_band) & (yf <= Ymax_band)
                mask_high_blend  = (yf >  Ymax_band) & (yf < Ymax_band + epsilon)
                mask_high_outer  = (yf >= Ymax_band + epsilon)

                # outer regions
                c[mask_low_outer]  = c_low[mask_low_outer]
                c[mask_high_outer] = c_high[mask_high_outer]

                # mid region
                if np.any(mask_mid):
                    y_mid = np.clip(yf[mask_mid], y_centers[0], y_centers[-1])
                    c[mask_mid] = C_interp(np.column_stack([xf[mask_mid], y_mid]))

                # blends
                if np.any(mask_low_blend):
                    yb = yf[mask_low_blend]
                    t = (yb - (Ymin_band - epsilon)) / epsilon  # 0..1
                    w = _smoothstep(t)
                    y_eval = np.clip(np.minimum(yb, Ymin_band), y_centers[0], y_centers[-1])
                    c_int = C_interp(np.column_stack([xf[mask_low_blend], y_eval]))
                    c[mask_low_blend] = (1 - w) * c_low[mask_low_blend] + w * c_int

                if np.any(mask_high_blend):
                    yb = yf[mask_high_blend]
                    t = (yb - Ymax_band) / epsilon  # 0..1
                    w = _smoothstep(t)
                    y_eval = np.clip(np.maximum(yb, Ymax_band), y_centers[0], y_centers[-1])
                    c_int = C_interp(np.column_stack([xf[mask_high_blend], y_eval]))
                    c[mask_high_blend] = (1 - w) * c_int + w * c_high[mask_high_blend]

            return c.reshape(shp)

        # --- optional floor plot ---
        if plot_floor_profile:
            # Evaluate C_func on a dense grid to visualize the prolongation
            Xg, Yg = np.meshgrid(x_centers, y_centers, indexing='ij')
            C_vis = C_func(Xg, Yg)

            fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

            # --- Left: floor map ---
            im0 = axes[0].imshow(
                Zsmooth.T, origin='lower', aspect='auto',
                extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                cmap='viridis'
            )
            axes[0].set_title('Estimated floor Z(x, y)')
            axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
            fig.colorbar(im0, ax=axes[0], label='Z_floor')

            # --- Right: correction map ---
            im1 = axes[1].imshow(
                C_vis.T, origin='lower', aspect='auto',
                extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
                cmap='plasma'
            )
            axes[1].axhline(-correct_until_depth_metric, color='w', ls='--', lw=1)
            axes[1].axhline(+correct_until_depth_metric, color='w', ls='--', lw=1)
            axes[1].set_title('Correction surface C(x, y)\n(with prolongation beyond Y-band)')
            axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
            fig.colorbar(im1, ax=axes[1], label='C(x, y)')

            plt.show()

        return C_func, (x_centers, y_centers), Zsmooth

    @staticmethod
    def correct_floor_v3(
        pts_carte, theta, colors, correct_until_depth_metric,
        *,
        dx=0.1, dy=0.1, q=10.0,
        theta_min=-np.pi, theta_max=0.0,
        min_pts_per_bin=10,
        gaussian_sigma_xy=(1.0, 1.0),
        reference_level='horizon',
        epsilon=0.0,
        plot=False,
        horizon_deg=(-10.0, -1.0),
    ):
        """
        Build the correction C(x,y), apply it to all points, and (optionally) plot.

        Parameters
        ----------
        pts_carte : array_like (..., 3)
        theta : array_like (...)
        colors : array_like (...)
            Only used if plot_horizon=True (for 3D scatter coloring).
        correct_until_depth_metric : float
            Y half-width of correction band for continuous prolongation.
        dx, dy, q, theta_min, theta_max, min_pts_per_bin, gaussian_sigma_xy,
        reference_level, epsilon, plot_horizon, plot_floor_profile, horizon_deg : see
            `build_floor_correction`.

        Returns
        -------
        X, Y, Z_corrected : ndarrays
        """

        X, Y, Z = pts_carte[..., 0], pts_carte[..., 1], pts_carte[..., 2]

        C_func, (xc, yc), Zfloor = GeometryTransforms.build_floor_correction(
            X, Y, Z, theta,
            correct_until_depth_metric=correct_until_depth_metric,
            dx=dx, dy=dy, q=q,
            theta_min=theta_min, theta_max=theta_max,
            min_pts_per_bin=min_pts_per_bin,
            gaussian_sigma_xy=gaussian_sigma_xy,
            reference_level=reference_level,
            epsilon=epsilon,
            plot_horizon=plot,
            plot_floor_profile=plot,
            horizon_deg=horizon_deg,
            horizon_colors=colors,
        )

        C_all = C_func(X, Y)
        Z_corrected = Z + C_all

        if plot:
            mask = (theta > -np.pi) & (theta < 0)
            plt.figure(figsize=(10,4))
            plt.subplot(2,2,1)
            plt.scatter(Y[mask], Z[mask], s=1, c='steelblue'); plt.grid(True)
            plt.xlabel('Y'); plt.ylabel('Z'); plt.title('Before (masked θ∈(-π,0))')
            plt.subplot(2,2,2)
            plt.scatter(Y[mask], Z_corrected[mask], s=1, c='crimson'); plt.grid(True)
            plt.xlabel('Y'); plt.ylabel('Z corrected'); plt.title('After correction (masked)')

            plt.subplot(2,2,3)
            plt.scatter(Y, Z, s=1, c='steelblue'); plt.grid(True)
            plt.xlabel('Y'); plt.ylabel('Z'); plt.title('Before ')
            plt.subplot(2,2,4)
            plt.scatter(Y, Z_corrected, s=1, c='crimson'); plt.grid(True)
            plt.xlabel('Y'); plt.ylabel('Z corrected'); plt.title('After correction')

            plt.tight_layout(); plt.show()
        
        pts_corrected = np.stack((X, Y, Z_corrected), axis=-1)
        return pts_corrected

    @staticmethod
    def get_sky_mask(
            depth_map,         
            height,
            width,
            thetas_range_for_sky_detection = (np.deg2rad(80), np.deg2rad(90)),
            eps = 0.05
        ):

        thetas = get_canonical_sph_pixels(height, width)[..., 0]
        sky_theta_mask = (thetas >= thetas_range_for_sky_detection[0]) & (thetas <= thetas_range_for_sky_detection[1])
        depth_sky_values = depth_map[sky_theta_mask]
        threshold = np.nanmean(depth_sky_values) - eps # * np.nanstd(depth_sky_values)
        sky_mask = depth_map >= threshold
        return sky_mask

    @staticmethod
    def _smoothstep(a, b, x):
        """Cubic smoothstep from 0→1 on [a,b]."""
        t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def correct_walls_v1(
            pts_sph,
            theta_range,
            edge=np.deg2rad(8.0),
            sky_mask= None    
        ):
        """
        Smoothly map points from sphere to cylinder only within an elevation band,
        with soft transitions near the edges.

        Conventions:
        - phi: azimuth in [0, 2π)
        - theta  : elevation in [-π/2, π/2] (0 at equator)
        - r    : radius of sphere/cylinder
        - phi_range: (phi_min, phi_max) where mapping is 'active'
        - edge : half-width (radians) of the soft transition at each band edge
        - sky_mask: optional boolean mask where sky points are untouched
        Returns:
        x_out, y_out, z_out : arrays of mapped Cartesian points
        w                   : blend weight in [0,1] (0=sphere, 1=cylinder)
        """
        theta_min, theta_max = theta_range

        theta, phi, r = pts_sph[..., 0], pts_sph[..., 1], pts_sph[..., 2]

        # Sphere coords (elevation convention)
        x_s = r * np.cos(theta) * np.cos(phi)
        y_s = r * np.cos(theta) * np.sin(phi)
        z_s = r * np.sin(theta)

        # Cylinder coords (same z)
        x_c = r * np.cos(phi)
        y_c = r * np.sin(phi)
        z_c = z_s  # unchanged

        # Build a smooth "band" weight:
        #  - ramp up across lower edge:  smoothstep(theta_min - edge, theta_min + edge, theta)
        #  - ramp down across upper edge: 1 - smoothstep(theta_max - edge, theta_max + edge, theta)
        w_up   = GeometryTransforms._smoothstep(theta_min - edge, theta_min + edge, theta)
        w_down = 1.0 - GeometryTransforms._smoothstep(theta_max - edge, theta_max + edge, theta)
        w = np.clip(w_up * w_down, 0.0, 1.0)

        # Blend between sphere (0) and cylinder (1)
        x_out = (1.0 - w) * x_s + w * x_c
        y_out = (1.0 - w) * y_s + w * y_c
        z_out = z_s  # identical in both, so blending unnecessary; kept for clarity

        # Optionally preserve sky points from modification
        if sky_mask is not None:
            x_out[sky_mask] = x_s[sky_mask]
            y_out[sky_mask] = y_s[sky_mask]
            z_out[sky_mask] = z_s[sky_mask]

        pts_corrected_carte = np.stack((x_out, y_out, z_out), axis=-1)
        return pts_corrected_carte
    
    @staticmethod
    def correct_walls_lp(pts_carte, p=6.0):
        """
        Correct wall geometry using Lp transform for a sphere. Assumes Z-axis is up and apply the transform in all radial directions.
        """
        pt1_cyl = cart2cyl_zaxis(pts_carte)
        r_c, z = pt1_cyl[..., 0], pt1_cyl[..., 1]

        mask = z > 0
        r_c_corr = r_c.copy()
        z_corr = z.copy()

        theta = np.arctan2(z, r_c)
        r = np.sqrt(r_c**2 + z**2)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        rho = 1.0 / (np.abs(cos_t)**p + np.abs(sin_t)**p)**(1.0/p)
        r_c = r * rho * cos_t
        z = r * rho * sin_t

        r_c_corr[mask] = r_c[mask]
        z_corr[mask] = z[mask]

        pt1_cyl[..., 0] = r_c_corr
        pt1_cyl[..., 1] = z_corr
        pts1_carte_corrected = cyl2cart_zaxis(pt1_cyl)

        return pts1_carte_corrected

    @staticmethod
    def correct_walls_sphere_unfold(pts_carte, sphere_center=np.array([0,0,0])):
        """
        Unfolds sphere into a cylinder of prinpal axis Z.
        pts_carte: np.array[..., 3] in Cartesian coordinates experessed in the local sphere frame.
        """
        pts_sph = carte2sph_3D(pts_carte-sphere_center)
        up = np.array([0, 0, 1])
        up_sph = carte2sph_3D(up)
        pts_prime_sph = unfold_sphere_in_cylinder_uniform(pts_sph, up_sph)
        pts_prime = sph2carte_3D(pts_prime_sph) + sphere_center
        return pts_prime

    @staticmethod
    def correct_walls_cylinder_unfold(pts_carte):
        """
        Unfolds the curved walls of a cylinder of principal axis +X into
        a straight wall (tangent plane) facing +Z.

        Only points whose cylindrical angle around the +X axis
        satisfies theta ∈ [0, π] are affected.

        pts_carte: np.array[..., 3] in Cartesian coordinates.
        returns:   np.array[..., 3] in Cartesian coordinates.
        """
        pts_carte = np.asarray(pts_carte)
        if pts_carte.shape[-1] != 3:
            raise ValueError("pts_carte must have shape [..., 3].")

        # Cartesian -> cylindrical [x, p, theta] around +X
        pts_cyl = cart2cyl_xaxis(pts_carte)

        # Unfold appropriate walls

        up = np.array([0, 0, 1])
        up_cyl = cart2cyl_xaxis(up)

        pts_cyl_unfold = unfold_cylinder_on_tangents(pts_cyl, up_cyl, delta=np.pi)

        # Back to Cartesian
        pts_prime = cyl2cart_xaxis(pts_cyl_unfold)

        return pts_prime

    @staticmethod
    def remove_statistical_outliers(pts, colors, nb_neighbors=20, std_ratio=1.8):
        """
        Remove statistical outliers from a point cloud using Open3D's 
        statistical outlier removal (SOR) filter.

        This method analyzes the local neighborhood of each point and removes
        those whose average neighbor distance is significantly larger than the
        global average, based on a configurable standard deviation threshold.
        It is useful for denoising raw point clouds or eliminating isolated
        artifacts.

        Parameters
        ----------
        pts : (N, 3) array-like
            Input 3D point positions in Cartesian coordinates.

        colors : (N, 3) array-like
            RGB colors associated with each point. Values are expected in [0, 1].

        nb_neighbors : int, optional (default: 20)
            Number of nearest neighbors to consider when computing the mean 
            distance for each point. Larger values produce smoother filtering 
            but may remove more detail.

        std_ratio : float, optional (default: 1.8)
            Threshold that defines how many standard deviations above the mean 
            distance a point must be to be considered an outlier. Lower values 
            remove more points; higher values keep more points.

        Returns
        -------
        inlier_pts : (M, 3) ndarray
            The 3D coordinates of the inlier points after filtering.

        inlier_colors : (M, 3) ndarray
            The RGB colors corresponding to the inlier points.

        Notes
        -----
        - This function internally constructs an Open3D `PointCloud` object 
        from `(pts, colors)`.
        - Only inliers are returned; outliers are discarded.
        """
        import open3d as o3d
        pcd = PointCloud(pts, colors).get_o3d_pointcloud()
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_pts = np.asarray(cl.points)
        inlier_colors = np.asarray(cl.colors)
        return inlier_pts, inlier_colors

    @staticmethod
    def run_corrective_pipeline_depreciated(colors, depth, sphere_radius, height, width, correct_depth, near, far, correct_floor, correct_walls, remove_sky, indoor_or_outdoor, remove_outliers, verbose=False):
        #TODO:delete this after testing the next one

        assert not np.any(np.isnan(depth)), "Depth contains NaNs!"
        assert indoor_or_outdoor in ['indoor', 'outdoor', None], "indoor_or_outdoor must be either 'indoor' or 'outdoor'"

        # 1. Get Metric Depth
        if correct_depth:
            depth_corrected = GeometryTransforms.depth_transform(
                depth, 
                method="inv", 
                n=near, 
                f=far, 
                gamma=5,
                plot=True
            )
            if verbose:
                print("a. Metric Depth Obtained.")
        else:
            depth_corrected = depth
            
        # 2. Project to Camera Space in Cartesian Coordinates
        pts_cam_cartesian = depth2cam_carte(
            depth=depth_corrected,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
        ) # [H, W, 3]

        # 3. Correct Floor
        if correct_floor:
            pts_cam_cartesian = GeometryTransforms.correct_floor(
                pts_cam_cartesian,
                plot=True
            )
            if verbose:
                print("b. Floor Corrected.")

        # 4. Correct Walls
        if correct_walls:
            if indoor_or_outdoor == 'outdoor':
                sky_mask = GeometryTransforms.get_sky_mask(depth_corrected, height=height, width=width)
            else:
                sky_mask = None
                
            pts_cam_cartesian = GeometryTransforms.correct_walls(
                pts_sph=carte2sph_3D(pts_cam_cartesian),
                theta_range=(np.deg2rad(0), np.deg2rad(70)),
                edge=np.deg2rad(15),
                sky_mask=sky_mask
            )
            if verbose:
                print("c. Walls Corrected.")

        #5. remove sky points
        if remove_sky:
            assert indoor_or_outdoor == 'outdoor', "Sky removal can only be done for outdoor scenes."
            sky_mask = GeometryTransforms.get_sky_mask(depth_corrected, height=height, width=width)
            pts_cam_cartesian[sky_mask] = np.array([np.nan, np.nan, np.nan])
            #TODO: plot a figure showing the sky mask as an overlay over the image.

            if verbose:
                print("d. (Optional) Sky Removed.")

        # 6. Remove statistical outliers
        if remove_outliers: 
            n_before = len(pts_cam_cartesian.reshape(-1,3))
            pts_cam_cartesian, colors = GeometryTransforms.remove_statistical_outliers(
                pts=pts_cam_cartesian,
                colors=colors,
                nb_neighbors=20,
                std_ratio=2.0
            )
            n_after = len(pts_cam_cartesian.reshape(-1,3))
            if verbose:
                print(f"e. (Optional) Outliers Removed ({(n_before - n_after) / n_before * 100:.2f}%)")

        return pts_cam_cartesian, colors


def run_corrective_pipeline_on_sphere(
        pts, # in cartesian coordinates (local camera frame)
        colors, 
        height, width, 
        correct_depth, 
        near, 
        far, 
        correct_walls, 
        correct_floor, 
        depth_threshold_for_floor_correction, 
        remove_sky, 
        remove_outliers, 
        verbose=False,
        plot=False,
    ):
    "assunmes points in cartesian coordinates"


    # 1.  Convert to spherical coordinates
    final_pts = pts.copy()
    pts_sph = carte2sph_3D(pts)

    # 2. Get Metric Depth
    if correct_depth:
        depth = pts_sph[..., 2]  # radial distances
        depth_corrected = GeometryTransforms.depth_transform(
            depth, 
            method="inv", 
            n=near, 
            f=far, 
            gamma=5,
            plot=plot
        )
        pts_sph[..., 2] = depth_corrected
        if verbose:
            print("a. Metric Depth Obtained.")
    else:
        depth_corrected = pts_sph[..., 2]
    # 3. Merge back 
    final_pts = sph2carte_3D(pts_sph)

    # 4. Correct Walls
    if correct_walls:
        # final_pts = GeometryTransforms.correct_walls_lp(
        #     pts_carte=final_pts,
        #     p=6.0 #correction strenght
        # )
        final_pts = GeometryTransforms.correct_walls_sphere_unfold(
            pts_carte=final_pts
        )
    
    # 5. Correct Floor
    if correct_floor:
        if correct_depth:
            correct_until_depth_metric = GeometryTransforms.depth_transform(
                np.array([depth_threshold_for_floor_correction]), 
                method="inv", 
                n=near, 
                f=far, 
                gamma=5,
                plot=plot
            )[0]
            print(f"Using depth threshold for floor correction (metric): {correct_until_depth_metric:.2f}m")
        else:
            correct_until_depth_metric = depth_threshold_for_floor_correction
            
        theta = carte2sph_3D(final_pts)[..., 0]
        final_pts = GeometryTransforms.correct_floor_v3(
            pts_carte=final_pts,
            theta=theta,
            colors=colors,
            correct_until_depth_metric=correct_until_depth_metric,
            dx=0.05,
            dy=0.05,
            plot=plot,
        )

        if verbose:
            print("b. Cylindrical Floor Corrected.")

    # 6. remove sky points
    if remove_sky:
        sky_mask = GeometryTransforms.get_sky_mask(depth_corrected, height=height, width=width)
        final_pts = final_pts[~sky_mask] 
        colors = colors[~sky_mask]
        #TODO: plot a figure showing the sky mask as an overlay over the image.

        if verbose:
            print("d. (Optional) Sky Removed.")

    # 7. Remove statistical outliers
    if remove_outliers: 
        n_before = len(final_pts.reshape(-1,3))
        final_pts, colors = GeometryTransforms.remove_statistical_outliers(
            pts=final_pts,
            colors=colors,
            nb_neighbors=20,
            std_ratio=1.8
        )
        n_after = len(final_pts.reshape(-1,3))
        if verbose:
            print(f"e. (Optional) Outliers Removed ({(n_before - n_after) / n_before * 100:.2f}%)")
    
    return final_pts, colors

def run_corrective_pipeline_on_world(
    pts, 
    colors,
    pose_left,
    pose_right,
    translation_direction,
    correct_depth, 
    near, 
    far, 
    correct_walls, 
    correct_floor, 
    depth_threshold_for_floor_correction, 
    remove_outliers,
    verbose=False,
    plot=False,
):
    
    """expects points in cartesian coordinates. Cylinder is assumed to be along the X-axis"""

    # 1. Seprate cylindrical right sphere and left sphere.
    final_pts = pts.copy()
    cam_left=pose_left[:3,3]
    cam_right=pose_right[:3,3]
    _, mask_keep_left = filter_points_by_plane_cartesian(pts, forward_carte=-translation_direction, cut_distance=np.linalg.norm(cam_left))
    _, mask_keep_right = filter_points_by_plane_cartesian(pts, forward_carte=translation_direction, cut_distance=np.linalg.norm(cam_right))
    
    # in world coordinates
    pts_left = pts[~mask_keep_left]
    pts_right = pts[~mask_keep_right]
    pts_cyl = pts[mask_keep_right & mask_keep_left]

    # in adapated coordinates
    pts_left = world2cam_sph_3D(pts_left, pose_left)
    pts_right = world2cam_sph_3D(pts_right, pose_right)
    pts_cyl = cart2cyl_xaxis(pts_cyl)
    

    # 2. Metric Depth Correction 
    if correct_depth:
        # 2.a for cylindrical points
        p_r = pts_cyl[..., 1]     # radial distance
        p_r_corrected = GeometryTransforms.depth_transform(
            p_r, 
            method="inv", 
            n=near, 
            f=far, 
            gamma=5,
            plot=plot
        ) 
        pts_cyl[..., 1] = p_r_corrected

        # 2.b. for left sphere points
        depth_left = pts_left[..., 2]
        depth_left_corrected = GeometryTransforms.depth_transform(
            depth_left, 
            method="inv", 
            n=near, 
            f=far, 
            gamma=5,
            plot=plot
        )
        pts_left[..., 2] = depth_left_corrected

        #2.c. for right sphere points
        depth_right = pts_right[..., 2]
        depth_right_corrected = GeometryTransforms.depth_transform(
            depth_right, 
            method="inv", 
            n=near, 
            f=far, 
            gamma=5,
            plot=plot
        )
        pts_right[..., 2] = depth_right_corrected

        if verbose:
            print("a. Depth Corrected.")

    # 3. Merge back left and right sphere points
    pts_left = cam_sph2world_3D(pts_left, pose_left)
    pts_right = cam_sph2world_3D(pts_right, pose_right)
    pts_cyl = cyl2cart_xaxis(pts_cyl)

    final_pts[~mask_keep_left] = pts_left
    final_pts[~mask_keep_right] = pts_right
    final_pts[mask_keep_left & mask_keep_right] = pts_cyl
    
    # 4. Correct Walls
    if correct_walls:
        # final_pts = GeometryTransforms.correct_walls_lp(
        #     pts_carte=final_pts,
        #     p=6.0 #correction strenght
        # )
        pts_cyl = GeometryTransforms.correct_walls_cylinder_unfold(
            pts_carte=pts_cyl
        )

        pts_left = GeometryTransforms.correct_walls_sphere_unfold(
            pts_carte=pts_left,
            sphere_center=pose_left[:3,3]
        )
        pts_right = GeometryTransforms.correct_walls_sphere_unfold(
            pts_carte=pts_right,
            sphere_center=pose_right[:3,3]
        )

        final_pts[mask_keep_left & mask_keep_right] = pts_cyl
        final_pts[~mask_keep_left] = pts_left
        final_pts[~mask_keep_right] = pts_right


    

    # 5. Correct Floor
    if correct_floor:
        if correct_depth:
            correct_until_depth_metric = GeometryTransforms.depth_transform(
                np.array([depth_threshold_for_floor_correction]), 
                method="inv", 
                n=near, 
                f=far, 
                gamma=5,
                plot=False
            )[0]
        else:
            correct_until_depth_metric = depth_threshold_for_floor_correction

        theta = carte2sph_3D(final_pts)[..., 0]
        final_pts = GeometryTransforms.correct_floor_v3(
            pts_carte=final_pts,
            theta=theta,
            colors=colors,
            correct_until_depth_metric=correct_until_depth_metric,
            plot=plot,
        )

        if verbose:
            print("b. Cylindrical Floor Corrected.")

    # 7. Remove statistical outliers
    if remove_outliers: 
        n_before = len(final_pts.reshape(-1,3))
        final_pts, colors = GeometryTransforms.remove_statistical_outliers(
            pts=final_pts,
            colors=colors,
            nb_neighbors=20,
            std_ratio=1.8
        )
        n_after = len(final_pts.reshape(-1,3))
        if verbose:
            print(f"e. (Optional) Outliers Removed ({(n_before - n_after) / n_before * 100:.2f}%)")

    return final_pts, colors
    

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

def _rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3)

    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # Find the major diagonal element
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)

def _quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)
    return R

def _slerp(q0, q1, t):
    """
    Spherical linear interpolation between two quaternions q0, q1 at parameter t in [0, 1].
    q0, q1: (..., 4)
    t: scalar or array broadcastable to q0.shape[:-1]
    """
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)

    # Normalize just in case
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # If dot < 0, the interpolation will take the long way around the sphere.
    # Fix by reversing one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close, fall back to linear interpolation to avoid numerical issues.
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)

    # theta is angle between input quaternions
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0

    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)

def rotation_matrix_z(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=float)
    return R

def get_intermediate_camera_poses(
        start_pose,
        end_pose,
        num_steps,
        perturb_z=0.0,
        perturb_y=0.0,
        perturb_x=0.0,
    ):
    """
    Generate a list of intermediate camera poses between start_pose and end_pose.

    Each pose is a 4x4 transformation matrix. Rotations are interpolated
    using quaternion SLERP; translations are linearly interpolated and can
    be optionally perturbed by Gaussian noise along each axis.

    Parameters
    ----------
    start_pose : array-like, shape (4, 4)
        Starting camera pose (world-to-camera or camera-to-world; must be
        consistent with end_pose).
    end_pose : array-like, shape (4, 4)
        Ending camera pose.
    num_steps : int
        Number of poses to generate, including both start and end.
        Must be >= 2.
    perturb_z : float, default 0.0
        Standard deviation of Gaussian noise added to the Z translation
        component at each step.
    perturb_y : float, default 0.0
        Standard deviation of Gaussian noise added to the Y translation
        component at each step.
    perturb_x : float, default 0.0
        Standard deviation of Gaussian noise added to the X translation
        component at each step.

    Returns
    -------
    poses : np.ndarray, shape (num_steps, 4, 4)
        Interpolated camera poses.
    """
    start_pose = np.asarray(start_pose, dtype=float)
    end_pose = np.asarray(end_pose, dtype=float)

    if start_pose.shape != (4, 4) or end_pose.shape != (4, 4):
        raise ValueError("start_pose and end_pose must be 4x4 matrices.")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2.")

    # Extract rotations (3x3) and translations (3,)
    R0 = start_pose[:3, :3]
    t0 = start_pose[:3, 3]

    R1 = end_pose[:3, :3]
    t1 = end_pose[:3, 3]

    # Convert rotations to quaternions
    q0 = _rotation_matrix_to_quaternion(R0)
    q1 = _rotation_matrix_to_quaternion(R1)

    # Interpolation parameters
    ts = np.linspace(0.0, 1.0, num_steps)

    poses = np.zeros((num_steps, 4, 4), dtype=float)

    for i, alpha in enumerate(ts):
        # SLERP between rotations
        qi = _slerp(q0, q1, alpha)
        Ri = _quaternion_to_rotation_matrix(qi)

        # Linear interpolation between translations
        ti = (1.0 - alpha) * t0 + alpha * t1

        # Add optional Gaussian perturbations on translation
        if (perturb_x != 0.0) or (perturb_y != 0.0) or (perturb_z != 0.0):
            noise = np.array([
                np.random.normal(0.0, perturb_x),
                np.random.normal(0.0, perturb_y),
                np.random.normal(0.0, perturb_z),
            ])
            ti = ti + noise

        # Build 4x4 pose
        pose_i = np.eye(4, dtype=float)
        pose_i[:3, :3] = Ri
        pose_i[:3, 3] = ti

        poses[i] = pose_i

    return poses

def load_pcd(filename):
    """
    Load a PointCloud object (class defined in this file) from a pickle file.
    Parameters
    ----------
    filename : str
        Path to the pickle file.
    Returns
    -------
    pcd : PointCloud
        Loaded PointCloud object.
    """
    with open(filename, 'rb') as f:
        pcd = pickle.load(f)
    return pcd

# --------------------------------------------#
# ----   World Opening transformations -----  #
# --------------------------------------------#
#TODO: remove the functions: 
# get_sphere_tangent, unfold_sphere_on_tangents


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

def unfold_sphere_on_tangents(pts_sph, forward_sph, delta=np.pi):
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

def get_cylinder_tangent(theta0, cyl_points):
    """
    Project cylindrical points (aligned with +X) onto the tangent plane
    of the cylinder at angular position theta0.

    Args
    ----
    theta0 : float
        Angle in radians where the tangent is computed (around +X).
    cyl_points : np.ndarray, shape (..., 3)
        Cylindrical coordinates [x, p, theta] with:
          - x     : coordinate along +X
          - p     : radial distance sqrt(Y^2 + Z^2)
          - theta : atan2(Z, Y), 0 along +Y, π/2 along +Z

    Returns
    -------
    proj_cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates [x, p, theta] of projected points.
    """
    cyl_points = np.asarray(cyl_points)
    if cyl_points.shape[-1] != 3:
        raise ValueError("cyl_points must have shape [..., 3].")

    x   = cyl_points[..., 0]
    p   = cyl_points[..., 1]
    th  = cyl_points[..., 2]

    # Original Cartesian coordinates
    X = x
    Y = p * np.cos(th)
    Z = p * np.sin(th)

    # Base point on the cylinder at theta0, same x and p
    X0 = X
    Y0 = p * np.cos(theta0)
    Z0 = p * np.sin(theta0)
    P0 = np.stack([X0, Y0, Z0], axis=-1)       # (..., 3)

    # Tangent direction wrt theta at theta0
    dX_dtheta0 = 0.0 * X
    dY_dtheta0 = -p * np.sin(theta0)
    dZ_dtheta0 =  p * np.cos(theta0)
    dP0 = np.stack([dX_dtheta0, dY_dtheta0, dZ_dtheta0], axis=-1)  # (..., 3)

    # Angular difference to theta0
    delta = angle_diff(th, theta0)             # (...,)

    # Projection on tangent plane at theta0
    projection = P0 + dP0 * delta[..., None]   # (..., 3)

    # Back to cylindrical [x, p, theta]
    proj_cyl = cart2cyl_xaxis(projection)
    return proj_cyl

def unfold_cylinder_on_tangents(pts_cyl, up_cyl, delta=np.pi):
    """
    Unfold points on a cylinder of principal axis +X by projecting them
    onto the cylinder tangents at the boundary of an angular sector of
    size `delta` around the 'up' direction.

    All arguments and return values are in cylindrical coordinates
    (x, r, theta) around the +X axis.
      - x:     coordinate along +X
      - r:     radial distance in YZ-plane
      - theta: angle around +X (0 along +Y, π/2 along +Z, etc.)

    Args
    ----
    pts_cyl : np.ndarray, shape (..., 3)
        Cylindrical coordinates (theta, r, x) to be unfolded.
    up_cyl : np.ndarray, shape (3,)
        Cylindrical coordinates (theta, r, x) of the 'up' direction
        around which the angular sector of width `delta` is defined.
        Only up_cyl[0] (theta_up) is used.
    delta : float, optional
        Total angular width of the sector to unfold (in radians).
        The tangents are taken at angles theta1 and theta2 defined as
        theta_up ± delta/2.

    Returns
    -------
    res_cyl : np.ndarray, shape (..., 3)
        Unfolded points, in cylindrical coordinates (theta, r, x).
    """
    theta_all = pts_cyl[..., 2]

    # a. Get the two boundary angles, analogous to phi1 / phi2 in the sphere case
    theta_up = up_cyl[2]
    theta1 = normalize_angle(theta_up + delta / 2.0)
    theta2 = normalize_angle(theta_up - delta / 2.0)

    # b. Get arcs (two angular intervals around theta_up)
    arc1_bounds = (theta_up, theta1)
    arc2_bounds = (theta2, theta_up)

    arc1_mask = in_interval_mod(theta_all, arc1_bounds[0], arc1_bounds[1], closed='both')
    arc2_mask = in_interval_mod(theta_all, arc2_bounds[0], arc2_bounds[1], closed='both')
    other_mask = ~(arc1_mask | arc2_mask)

    arc1_pts_cyl = pts_cyl[arc1_mask]
    arc2_pts_cyl = pts_cyl[arc2_mask]

    # c. Project onto tangents at theta1 and theta2
    arc1_proj_pts_cyl = get_cylinder_tangent(theta1, arc1_pts_cyl)
    arc2_proj_pts_cyl = get_cylinder_tangent(theta2, arc2_pts_cyl)
    other_pts_cyl = pts_cyl[other_mask]

    # d. Re-assemble
    res_cyl = np.zeros_like(pts_cyl)
    res_cyl[arc1_mask]  = arc1_proj_pts_cyl
    res_cyl[arc2_mask]  = arc2_proj_pts_cyl
    res_cyl[other_mask] = other_pts_cyl

    return res_cyl

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

def unfold_sphere_in_cylinder_uniform(pts_sph, forward_sph, base_radius=1.0, eps=1e-12):
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

def filter_points_by_plane_sph(pts_sph, forward_sph, cut_distance):
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

    # Keep points behind (or on) the plane
    mask_keep = (proj <= cut_distance)

    # Select those points and return them in spherical coords
    kept_xyz = pts_xyz[mask_keep]
    kept_pts_sph = carte2sph_3D(kept_xyz)

    return kept_pts_sph, mask_keep

def filter_points_by_plane_cartesian(pts_carte, forward_carte, cut_distance):
    """
    Remove points that lie *beyond* the plane orthogonal to `forward_carte`
    and located at a distance `cut_distance` along that direction.

    Parameters
    ----------
    pts_carte : np.ndarray, shape (..., 3)
        Point cloud in Cartesian coordinates (X, Y, Z).
    forward_carte : np.ndarray, shape (3,)
        Direction vector of the plane's normal.
    cut_distance : float
        Distance of the plane along `forward_carte` (in same units as points).

    Returns
    -------
    kept_pts_carte : np.ndarray, shape (M, 3)
        Points that are *in front of or on* the plane, i.e. (p · f̂) >= cut_distance.
    mask_keep : np.ndarray of bool, shape (N,)
        Boolean mask of kept points.
    """

    pts_carte = np.asarray(pts_carte, dtype=float)
    forward_carte = np.asarray(forward_carte, dtype=float)

    # Normalize direction vector
    norm = np.linalg.norm(forward_carte)
    if norm == 0:
        raise ValueError("`forward_carte` vector must be non-zero.")
    forward_carte_hat = forward_carte / norm

    # Project each point onto the forward_carte direction
    proj = pts_carte @ forward_carte_hat  # dot product gives distance along forward_carte_hat

    # Keep points behind (or on) the plane
    mask_keep = proj <= cut_distance

    # Apply mask
    kept_pts_carte = pts_carte[mask_keep]

    return kept_pts_carte, mask_keep

def compute_cut_distance_based_on_percentile(pts_sph=None, pts_carte=None, forward_carte=None, forward_sph=None, percentile=90):
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
    
    if pts_carte is not None:
        pts_xyz = pts_carte
    elif pts_sph is not None:
        pts_xyz = sph2carte_3D(pts_sph)
    else:
        raise ValueError("Either `pts_sph` or `pts_carte` must be provided.")
    
    if forward_carte is not None:
        forward = forward_carte
    elif forward_sph is not None:
        forward = sph2carte_3D(forward_sph)
    else:
        raise ValueError("Either `forward_sph` or `forward_carte` must be provided.")

    # Normalize direction vector
    norm = np.linalg.norm(forward)
    if norm == 0:
        raise ValueError("`forward` vector must be non-zero.")
    forward_hat = forward / norm

    # Project each point onto the forward direction
    proj = pts_xyz @ forward_hat  # dot product

    # Compute and return the desired percentile
    cut_distance = np.percentile(proj, percentile)
    return cut_distance

def straight_cut(forward_carte, pts_carte, cut_distance=None, cut_distance_percentile=90):
    """
    Open the world by cutting points behind a plane orthogonal to `forward_carte`.

    Parameters
    ----------
    forward_carte : np.ndarray, shape (3,)
        Direction vector of the plane's normal in Cartesian coordinates.
    pts_carte : np.ndarray, shape (..., 3)
        Point cloud in Cartesian coordinates.
    cut_distance : float or None
        Distance of the plane along `forward_carte`. If None, computed based on percentile.
    cut_distance_percentile : float
        Percentile (0-100) of points to be behind the cutting plane if `cut_distance` is None.

    Returns
    -------
    kept_pts_carte : np.ndarray, shape (M, 3)
        Points that are in front of or on the plane.
    pts_carte : np.ndarray, shape (..., 3)
        Original point cloud.
    mask_keep : np.ndarray of bool, shape (N,)
        Boolean mask of kept points.
    """
    if cut_distance is None:
        cut_distance = compute_cut_distance_based_on_percentile(
            pts_carte=pts_carte,
            forward_carte=forward_carte,
            percentile=cut_distance_percentile
        )
    kept_pts_carte, mask_keep = filter_points_by_plane_cartesian(
        pts_carte=pts_carte,
        forward_carte=forward_carte,
        cut_distance=cut_distance
    )
    return kept_pts_carte, pts_carte, mask_keep, cut_distance

def build_disk_to_square_displacement_fn(
    center=(0.0, 0.0),
    radius=1.0,
    num_points=700000,
    threshold=0.8,
    forward=np.array([1.0, 0.0]),
    n_arc_points=200,
    n_line_points=50,
    delta=np.pi,
    plot=False,
    max_arrows=1000,
):
    """
    Build a harmonic displacement function for a cut disk setup and
    generate diagnostic plots.

    Parameters
    ----------
    center : (2,) tuple or array-like
        Center of the disk.
    radius : float
        Radius of the disk.
    num_points : int
        Approximate number of sampling points for the disk.
    threshold : float
        Plane threshold distance (relative to center) along `forward`.
    forward : (2,) array-like
        Normal to the cutting plane.
    n_arc_points : int
        Number of points to sample on the circular arc of the boundary.
    n_line_points : int
        Number of points to sample on the straight chord of the boundary.
    delta : float
        Angular opening used in the unfolding step.
    closed : str
        Interval closure for arc splitting ('both', 'left', 'right', 'neither').
    max_arrows : int
        Max number of arrows for quiver debug plots.

    Returns
    -------
    displacement_fn : callable
        A function f(P) -> P_def that applies the learned deformation
        to any (N, 2) array of points.
    """
    closed='neither'


    def get_points_on_disk(center, radius, num_points):
        """
        Generate approximately `num_points` grid-spaced points inside a disk.

        Parameters
        ----------
        center : (2,) array-like
            Center of the disk (x, y).
        radius : float
            Disk radius.
        num_points : int
            Approximate number of points desired.

        Returns
        -------
        points : (M, 2) ndarray
            Grid points inside the disk, M ≈ num_points.
        """
        cx, cy = center
        area = np.pi * radius * radius
        spacing = np.sqrt(area / num_points)

        gx = np.arange(cx - radius, cx + radius + spacing, spacing)
        gy = np.arange(cy - radius, cy + radius + spacing, spacing)
        xx, yy = np.meshgrid(gx, gy)
        pts = np.column_stack([xx.ravel(), yy.ravel()])

        d2 = (pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2
        inside = d2 <= radius**2
        return pts[inside]

    def get_dummy_points(center, radius, forward, num_points, threshold):
        """
        Convenience function:
        - Generate points on a disk
        - Remove points beyond a plane with normal `forward`
        at distance `threshold` from `center` along that normal.

        Parameters
        ----------
        center : (2,) array-like
            Center of the disk.
        radius : float
            Radius of the disk.
        forward : (2,) array-like
            Normal to the cutting plane.
        num_points : int
            Approximate number of grid points inside the disk.
        threshold : float
            Distance along `forward` from `center` where the cut occurs.

        Returns
        -------
        points : (M, 2) ndarray
            Filtered points inside the disk and on the kept side of the plane.
        """
        pts = get_points_on_disk(center, radius, num_points)
        pts_filtered, mask = filter_points_by_plane_cartesian(pts - center, forward, threshold)
        return pts_filtered

    def get_line(p_start, p_end, n_points):
        """
        Return n_points evenly spaced along the straight line segment
        from p_start to p_end (inclusive).
        """
        p_start = np.asarray(p_start, dtype=float)
        p_end   = np.asarray(p_end, dtype=float)

        t = np.linspace(0.0, 1.0, n_points)
        return (1 - t)[:, None] * p_start[None, :] + t[:, None] * p_end[None, :]

    def get_cut_disk_boundary(center, radius, forward, threshold,
                            n_arc_points=200, n_line_points=50):
        """
        Get the boundary of a disk cut by the half-space <p, n> <= threshold,
        where n is the normalized `forward` vector.

        The boundary is:
        - A circular arc of the disk
        - Plus a straight chord segment on the cutting line
            <p, n> = threshold, between the two intersection points
            with the circle.

        Parameters
        ----------
        center : (2,) array-like
            Center of the disk (cx, cy).
        radius : float
            Radius of the disk.
        forward : (2,) array-like
            Normal to the cutting line (direction of "beyond").
            Does not need to be normalized.
        threshold : float
            Threshold value along the normalized `forward` direction.
            The cutting line is { p | <p, n> = threshold }.
        n_arc_points : int
            Number of sample points along the circular arc part.
        n_line_points : int
            Number of sample points along the straight chord segment.

        Returns
        -------
        arc_points : (Na, 2) ndarray
            Points sampled along the circular arc of the kept region.
        line_points : (Nl, 2) ndarray
            Points sampled along the chord segment on the cutting line.
            If there is no cut or no intersection, may be empty.

        Notes
        -----
        - If the plane is entirely outside the disk on the "beyond" side,
        you get the full circle and an empty line (no chord).
        - If the plane is entirely inside the disk on the opposite side
        (disk completely beyond), both outputs are empty.
        """
        center = np.asarray(center, dtype=float)
        cx, cy = center
        r = float(radius)

        forward = np.asarray(forward, dtype=float)
        n = forward / np.linalg.norm(forward)  # normalized normal

        # Signed distance of center along n
        d_center = np.dot(center, n)

        # Equation on the unit circle direction u(θ):
        #   <c + r u(θ), n> = threshold  =>  <u(θ), n> = (threshold - <c,n>) / r
        s = (threshold - d_center) / r  # must be in [-1, 1] to intersect

        # Case 1: plane is outside on the "beyond" side => full circle kept, no chord
        if s >= 1.0:
            theta = np.linspace(0.0, 2.0 * np.pi, n_arc_points, endpoint=True)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            arc_points = np.column_stack((x, y))
            line_points = np.zeros((0, 2))
            return arc_points, line_points

        # Case 2: plane is so far inside that disk is entirely beyond => nothing kept
        if s <= -1.0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        # Genuine cut
        # Let n = (cos ψ, sin ψ). Then <u(θ), n> = cos(θ - ψ).
        # We need cos(θ - ψ) = s => θ - ψ = ±α, α = arccos(s).
        # Region kept is where <p, n> <= threshold => cos(θ - ψ) <= s,
        # which corresponds to (θ - ψ) ∈ [α, 2π - α].
        psi = np.arctan2(n[1], n[0])
        alpha = np.arccos(s)

        theta_arc = psi + np.linspace(alpha, 2.0 * np.pi - alpha, n_arc_points, endpoint=True)
        x_arc = cx + r * np.cos(theta_arc)
        y_arc = cy + r * np.sin(theta_arc)
        arc_points = np.column_stack((x_arc, y_arc))

        # Intersection points at θ1 = ψ + α, θ2 = ψ - α (≡ ψ + 2π - α)
        theta1 = psi + alpha
        theta2 = psi - alpha  # equivalent to psi + 2π - alpha modulo 2π

        p1 = center + r * np.array([np.cos(theta1), np.sin(theta1)])
        p2 = center + r * np.array([np.cos(theta2), np.sin(theta2)])

        line_points = get_line(p1, p2, n_line_points)
        return arc_points, line_points

    def carte2polar_xy(x, y):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return phi % (2*np.pi), r

    def carte2polar_2D(points):
        x = points[:, 0]
        y = points[:, 1]
        return carte2polar_xy(x, y)

    def circle_point(phi, r=1.0):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.stack([x, y], axis=-1)

    def circle_tangent_dir(phi0, r=1.0):
        dx = -r * np.sin(phi0)
        dy =  r * np.cos(phi0)
        return np.array([dx, dy])

    def project_on_tangent(phi0, phi, r=1.0):
        phi  = np.asarray(phi)
        base = circle_point(np.array([phi0]), r=r)[0]
        direction = circle_tangent_dir(phi0, r=r)
        delta = angle_diff(phi, phi0)
        return base[None, :] + delta[:, None] * direction[None, :]

    def compute_forward_angles(forward, delta=np.pi):
        """
        Given a forward direction and an opening angle delta,
        compute phi1, phi2 and phi_forward.
        """
        forward = np.asarray(forward, dtype=float)
        forward /= np.linalg.norm(forward)
        phi_forward, _ = carte2polar_xy(forward[0], forward[1])
        phi_forward = normalize_angle(phi_forward)

        phi1 = normalize_angle(phi_forward + delta/2.0)
        phi2 = normalize_angle(phi_forward - delta/2.0)
        return phi1, phi2, phi_forward

    def unfold_circle_from_cartesian(points, phi1, phi2, forward, closed='neither'):
        """
        Given cartesian points on a circle, split them into two arcs and 'other'
        using forward, phi1, phi2, and project the two arcs onto the tangents.
        """
        points = np.asarray(points)
        assert points.ndim == 2 and points.shape[1] == 2, "points must be (N, 2)"

        forward = np.asarray(forward, dtype=float)
        forward /= np.linalg.norm(forward)
        phi_forward, _ = carte2polar_xy(forward[0], forward[1])
        phi_forward = normalize_angle(phi_forward)

        phi1 = normalize_angle(phi1)
        phi2 = normalize_angle(phi2)

        phis, radii = carte2polar_2D(points)
        r = radii.mean()

        arc1_bounds = (phi_forward, phi1)
        arc2_bounds = (phi2, phi_forward)

        mask_arc1 = in_interval_mod(phis, arc1_bounds[0], arc1_bounds[1], closed=closed)
        mask_arc2 = in_interval_mod(phis, arc2_bounds[0], arc2_bounds[1], closed=closed)
        mask_other = ~(mask_arc1 | mask_arc2)

        arc1_points = points[mask_arc1]
        arc2_points = points[mask_arc2]
        other_points = points[mask_other]

        phis_arc1 = phis[mask_arc1]
        phis_arc2 = phis[mask_arc2]

        arc1_proj = project_on_tangent(phi1, phis_arc1, r=r)
        arc2_proj = project_on_tangent(phi2, phis_arc2, r=r)

        return {
            "arc1": arc1_points,
            "arc2": arc2_points,
            "other": other_points,
            "arc1_proj": arc1_proj,
            "arc2_proj": arc2_proj,
            "phi_forward": phi_forward,
            "radius": r,
            "mask_arc1": mask_arc1,
            "mask_arc2": mask_arc2,
            "mask_other": mask_other,
        }

    def build_cut_disk_unfolded(center,
                                radius,
                                threshold,
                                forward=np.array([1.0, 0.0]),
                                delta=np.pi,
                                n_arc_points=200,
                                n_line_points=50,
                                closed='neither'):
        """
        Full pipeline wrapper.

        Steps:
        - Build cut disk boundary (arc_points, line_points)
        - Split arc_points into arc1, arc2, other in angle-space
        - Project arc1, arc2 onto tangents at phi1, phi2
        - Find extremes of the straight boundary and project them
        - Build a straight line between those extremes in both spaces

        Returns
        -------
        circle_concat : (K, 2) ndarray
            Concatenation of (arc1, arc2, line_points)

        proj_concat : (K, 2) ndarray
            Concatenation of (arc1_proj, arc2_proj, line_points_proj)
        """
        # 1) cut disk boundary
        arc_points, line_points = get_cut_disk_boundary(
            center, radius, forward, threshold,
            n_arc_points=n_arc_points,
            n_line_points=n_line_points
        )

        if arc_points.shape[0] == 0 or line_points.shape[0] == 0:
            # Degenerate cases: nothing interesting to unfold
            return np.zeros((0, 2)), np.zeros((0, 2))

        # 2) angles for unfolding
        phi1, phi2, _ = compute_forward_angles(forward, delta=delta)

        # 3) unfold only the arc part
        data = unfold_circle_from_cartesian(arc_points, phi1, phi2, forward, closed=closed)

        # 4) extremes of the boundary line (two intersection points)
        extreme_points = np.vstack([line_points[0], line_points[-1]])
        data_extremes = unfold_circle_from_cartesian(extreme_points, phi1, phi2, forward, closed=closed)

        # 5) line in projected (tangent) space between projected extremes
        p_start_proj = data_extremes["arc1_proj"][0]
        p_end_proj   = data_extremes["arc2_proj"][0]
        line_points_proj = get_line(p_start_proj, p_end_proj, n_line_points)

        # 6) concatenations requested
        circle_concat = np.vstack([data["arc1"], data["arc2"], line_points])
        proj_concat   = np.vstack([data["arc1_proj"], data["arc2_proj"], line_points_proj])

        return circle_concat, proj_concat

    def plot_quiver_mapping(circle_concat, proj_concat, max_arrows=100):
        """
        Plot a quiver field mapping each point in circle_concat to its
        corresponding point in proj_concat, using a subset to avoid clutter.

        Parameters
        ----------
        circle_concat : (N, 2) ndarray
            Source points (e.g. on the circle / cut boundary).
        proj_concat : (N, 2) ndarray
            Target points (e.g. projected onto tangents / line).
            Must have one-to-one correspondence with circle_concat.
        max_arrows : int
            Maximum number of arrows to draw (subsampled uniformly).
        """
        circle_concat = np.asarray(circle_concat, dtype=float)
        proj_concat   = np.asarray(proj_concat, dtype=float)

        assert circle_concat.shape == proj_concat.shape, \
            "circle_concat and proj_concat must have the same shape"
        assert circle_concat.ndim == 2 and circle_concat.shape[1] == 2, \
            "Inputs must be of shape (N, 2)"

        N = circle_concat.shape[0]
        if N == 0:
            print("No points to plot.")
            return

        # Uniform subsampling
        if N > max_arrows:
            step = int(np.ceil(N / max_arrows))
            idx = np.arange(0, N, step)
        else:
            idx = np.arange(N)

        P = circle_concat[idx]       # tails
        Q = proj_concat[idx]         # heads
        U = Q[:, 0] - P[:, 0]        # dx
        V = Q[:, 1] - P[:, 1]        # dy


        # Quiver for the subsampled pairs
        plt.quiver(P[:, 0], P[:, 1], U, V,
                angles='xy', scale_units='xy', scale=1,
                width=0.003, alpha=0.9, color='tab:red',
                label="mapping")


    center = np.asarray(center, dtype=float)
    forward = np.asarray(forward, dtype=float)

    # ---------------------------------------------------------------
    # 1. Sample points inside the cut disk
    # ---------------------------------------------------------------
    pts = get_dummy_points(center, radius, forward, num_points, threshold)

    # ---------------------------------------------------------------
    # 2. Build boundary & target boundary (unfolded)
    # ---------------------------------------------------------------
    boundary, target_boundary = build_cut_disk_unfolded(
        center=center,
        radius=radius,
        threshold=threshold,
        forward=forward,
        delta=delta,
        n_arc_points=n_arc_points,
        n_line_points=n_line_points,
        closed=closed,
    )

    # ---------------------------------------------------------------
    # 3. Prepare constraints for harmonic_deform_pipeline
    # ---------------------------------------------------------------
    pts_with_boundary = np.vstack([pts, boundary])

    boundary_mask = np.concatenate([
        np.zeros(len(pts), dtype=bool),      # pts → not boundary
        np.ones(len(boundary), dtype=bool)   # boundary → boundary
    ])

    # Fixed region: everything "behind" plane through center (< p-center, n > <= 0)
    _, fixed_mask = filter_points_by_plane_cartesian(
        pts_with_boundary - center,
        forward_carte=forward,
        cut_distance=0.0
    )

    # ---------------------------------------------------------------
    # 4. Run harmonic deformation to get displacement function
    # ---------------------------------------------------------------
    _, _, displacement_fn = harmonic_deform_pipeline(
        P=pts_with_boundary,
        mask_fixed=fixed_mask,
        mask_boundary=boundary_mask,
        target_boundary=target_boundary,
        return_displacement_fn=True,
    )

    if plot:

        # ---------------------------------------------------------------
        # 6. Debug plot : deformation of a global grid
        # ---------------------------------------------------------------
        grid_x, grid_y = np.meshgrid(
            np.linspace(-1.5, 1.5, 100),
            np.linspace(-1.5, 1.5, 100)
        )
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid_points)  # randomize for better quiver subsampling

        grid_points_def = displacement_fn(grid_points)

        plt.figure(figsize=(8, 8))
        plt.scatter(
            grid_points[:, 0], grid_points[:, 1],
            s=1, alpha=0.3, label="grid points"
        )
        plt.scatter(
            grid_points_def[:, 0], grid_points_def[:, 1],
            s=1, alpha=0.3, label="deformed grid points"
        )
        plt.scatter(
            boundary[:, 0], boundary[:, 1],
            s=10, c='red', label="boundary"
        )
        plt.scatter(
            target_boundary[:, 0], target_boundary[:, 1],
            s=10, c='blue', label="target boundary"
        )

        plot_quiver_mapping(
            grid_points,
            grid_points_def,
            max_arrows=max_arrows
        )

        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.legend()
        plt.title("Harmonic deformation on global grid")
        plt.show()

    # ---------------------------------------------------------------
    # 7. Return the displacement function
    # ---------------------------------------------------------------
    return displacement_fn

def open_world_carte(forward_carte, pts_carte, opening_mode='cut+cylinder', delta_cut=np.pi/2, cut_distance=None, cut_distance_percentile=90):
    assert opening_mode in ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut', 'straight_cut+disk_to_square_displacement']

    if opening_mode == 'wall':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_sphere_on_tangents(pts_sph, forward_sph, delta=np.pi)
        pts_opened = sph2carte_3D(pts_opened_sph)
        mask_keep = np.ones_like(pts_sph[..., 0], dtype=bool)
    elif opening_mode == 'cut+wall':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_sphere_on_tangents(pts_sph, forward_sph, delta=np.pi)
        pts_opened = sph2carte_3D(pts_opened_sph)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'cut+cylinder':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_sphere_in_cylinder_uniform(pts_sph, forward_sph, base_radius=1.0)
        pts_opened = sph2carte_3D(pts_opened_sph)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'remove_within_cone':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened = pts_sph
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'straight_cut':
        pts_opened = pts_carte
        kept_pts_carte, pts_carte, mask_keep, cut_distance = straight_cut(
            forward_carte=forward_carte,
            pts_carte=pts_carte,
            cut_distance=cut_distance,
            cut_distance_percentile=cut_distance_percentile
        )
    elif opening_mode == 'straight_cut+disk_to_square_displacement':
        if cut_distance is None:
            cut_distance = compute_cut_distance_based_on_percentile(
                pts_carte=pts_carte,
                forward_carte=forward_carte,
                percentile=cut_distance_percentile
            )
        kept_pts_carte, pts_carte, mask_keep, cut_distance = straight_cut(
            forward_carte=forward_carte,
            pts_carte=pts_carte,
            cut_distance=cut_distance,
        )
        # TODO: optimize performance here doing this init only once
        displacement_fn = build_disk_to_square_displacement_fn(
            center=(0.0, 0.0),
            radius=1.0,
            threshold=cut_distance,
            forward=forward_carte[:2],
            plot=False
        )
        pts_xy = pts_carte[:, :2]
        pts_z  = pts_carte[:, 2:3]
        pts_xy_def = displacement_fn(pts_xy)
        pts_opened = np.hstack([pts_xy_def, pts_z])

    return pts_opened[mask_keep], pts_opened, mask_keep


# ------------------------------------------- #
# ---- Harmonic Deformation of 3D Points ---- #
# ------------------------------------------- #

# 1. Build Laplacian
def build_graph_laplacian(P, k=10, symmetrize=True):
    G = kneighbors_graph(P, n_neighbors=k, mode='distance',
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
    N, D = P.shape
    B = np.unique(np.concatenate([idx_fixed, idx_boundary]))
    M = np.setdiff1d(np.arange(N), B, assume_unique=False)

    L = build_graph_laplacian(P, k=k)

    # Displacements on boundary
    uB = np.zeros((len(B), D))
    if len(idx_boundary) > 0:
        pos_in_B = {b:i for i,b in enumerate(B)}
        jj = np.array([pos_in_B[i] for i in idx_boundary], dtype=int)
        uB[jj,:] = (target_boundary - P[idx_boundary])

    L_MM = L[M][:, M]
    L_MB = L[M][:, B]
    rhs  = - L_MB @ uB

    U = np.zeros_like(P)
    if solver == 'cg':
        for d in range(D):
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
                             k=10, m=3, power=2, seed=0, return_displacement_fn=False):
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
    def displacement_fn(P_input):
        U_full = prolongate_displacements(P_input, P_coarse, U_coarse, m=m, power=power)
        P_def = P_input + U_full
        return P_def
    
    P_def = displacement_fn(P)
    if return_displacement_fn:
        return P_def, coarse_idx, displacement_fn
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


# ------------------------------------------------ #
# ------------------ Video Utils ----------------- #
# ------------------------------------------------ #
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

def save_video_from_o3d_images(
    images,
    out_path,
    fps=20,
    codec="libx264"
):
    """
    Save a list of Open3D images (renderer.render_to_image()) as a video.

    Args:
        images: list of open3d.geometry.Image (or numpy arrays).
        out_path: output video path, e.g. "video.mp4".
        fps: frames per second.
        codec: video codec (for mp4, "libx264" is common; may require ffmpeg).

    Note:
        Requires: `pip install "imageio[ffmpeg]"`.
    """
    if len(images) == 0:
        raise ValueError("No images provided to save_video_from_o3d_images")

    writer = imageio.get_writer(out_path, fps=fps, codec=codec)

    try:
        for img in images:
            # Convert Open3D image to numpy array if needed
            if isinstance(img, o3d.geometry.Image):
                frame = np.asarray(img)
            else:
                frame = np.asarray(img)

            # Ensure 3 channels (RGB)
            if frame.ndim == 2:  # grayscale -> RGB
                frame = np.repeat(frame[..., None], 3, axis=-1)
            elif frame.shape[2] == 4:  # RGBA -> RGB
                frame = frame[..., :3]

            # Ensure uint8
            if frame.dtype != np.uint8:
                # Assume in [0, 1] if float
                if np.issubdtype(frame.dtype, np.floating):
                    frame = np.clip(frame, 0.0, 1.0)
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Saved video to {out_path}")

def set_camera_from_elev_azim(scene_camera,
                              cam_pos,
                              elev_deg,
                              azim_deg,
                              fov_deg,
                              width,
                              height,
                              near,
                              far):
    """
    Set Open3D rendering camera from:
      - camera position (world coords),
      - elevation angle (deg) above XY plane,
      - azimuth angle (deg) around Z,
      - perspective intrinsics (fov, near, far).

    Convention:
      - World Z is "up".
      - Azimuth = 0° looks along +X, increases toward +Y.
      - Elevation = 0° in XY plane, +90° straight up (+Z), -90° straight down.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # Forward direction from spherical angles
    fx = np.cos(elev) * np.cos(azim)
    fy = np.cos(elev) * np.sin(azim)
    fz = np.sin(elev)
    forward = np.array([fx, fy, fz], dtype=float)

    # Default world up
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    # Avoid collinearity between forward and up
    if np.abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

    # Orthonormal basis: right, up, forward
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Look-at point (along forward)
    lookat = cam_pos + forward

    # Use the correct FovType enum (Vertical or Horizontal)
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical

    scene_camera.set_projection(fov_deg,
                                width / height,
                                near,
                                far,
                                fov_type)
    scene_camera.look_at(lookat, cam_pos, up)

def interpolate_lists(a, b, steps=4):
    """
    Interpolate element-wise between lists a and b using np.linspace.

    a, b: lists (or arrays) of equal length
    steps: number of interpolated points including start and end
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Lists must have the same shape.")

    # Stack linspace results for each dimension
    return np.array([np.linspace(a[i], b[i], steps) for i in range(len(a))]).T

def stretch_append(*lists):
    # Length of the longest list
    max_len = max(len(lst) for lst in lists)

    result = []
    for j in range(max_len):
        row = []
        for lst in lists:
            n = len(lst)
            if n == 0:
                raise ValueError("Lists must be non-empty")

            # Map position j in [0, max_len) to an index in [0, n)
            # This creates contiguous chunks per element.
            idx = min(int(j * n / max_len), n - 1)
            row.append(lst[idx])
        result.append(row)

    return result

def get_template_tranjectories(trajectory):
    # x, y, z, elev, azims
    if trajectory == 'test_walk':
        test_walk = [
        [0, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 60],
        [0.5, 0, 0, 0, -60],
        [0.5, 0, 0, 0, 60],
        [0.75, 0, 0, -10, 0],
        [1, 0, 0, 0, 20],
    ]
        return test_walk
    elif trajectory == 'walk':
        walk = [
        [0, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0],
    ]
        return walk
    elif trajectory == 'walk_look':
        walk_look = [
        [0, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0],
        [0.25, 0, 0, 0, -45],
        [0.25, 0, 0, 0, 45],
        [0.25, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0],
        [0.5, 0, 0, 0, -45],
        [0.5, 0, 0, 0, 45],
        [0.5, 0, 0, 0, 0],
        [0.75, 0, 0, 0, 0],
        [0.75, 0, 0, 0, -45],
        [0.75, 0, 0, 0, 45],
        [0.75, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, -45],
        [1, 0, 0, 0, 45],
        [1, 0, 0, 0, 0],
    ]
        return walk_look
    elif trajectory == 'walk_lookaround':
        walk_lookaround = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 360],
        [0.25, 0, 0, 0, 360],
        [0.25, 0, 0, 0, 360*2],
        [0.5, 0, 0, 0, 360*2],
        [0.5, 0, 0, 0, 360*3],
        [0.75, 0, 0, 0, 360*3],
        [0.75, 0, 0, 0, 360*4],
        [1, 0, 0, 0, 360*4],
        [1, 0, 0, 0, 360*5],
    ]
        return walk_lookaround
    elif trajectory == 'zigzag':
        zigzag_y = 0.25
        zigzag_az = 45
        zigzag = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, zigzag_az],
        [0.25, zigzag_y, 0, 0, zigzag_az],
        [0.25, zigzag_y, 0, 0, -zigzag_az],
        [0.5, -zigzag_y, 0, 0, -zigzag_az],
        [0.5, -zigzag_y, 0, 0, zigzag_az],
        [0.75, zigzag_y, 0, 0, zigzag_az],
        [0.75, zigzag_y, 0, 0, -zigzag_az],
        [1.0, -zigzag_y, 0, 0, -zigzag_az],
        [1.0, -zigzag_y, 0, 0, zigzag_az],
    ]
        return zigzag
    elif trajectory == 'walk_lookupdown':
        lookup_angle = 45
        lookdown_angle = -45
        walk_lookupdown = [
        [0, 0, 0, 0, 0],
        [0.25, 0, 0, 0, 0],
        [0.25, 0, 0, lookup_angle, 0],
        [0.25, 0, 0, lookdown_angle, 0],
        [0.25, 0, 0, 0, 0],
        [0.5, 0, 0, 0, 0],
        [0.5, 0, 0, lookup_angle, 0],
        [0.5, 0, 0, lookdown_angle, 0],
        [0.5, 0, 0, 0, 0],
        [0.75, 0, 0, 0, 0],
        [0.75, 0, 0, lookup_angle, 0],
        [0.75, 0, 0, lookdown_angle, 0],
        [0.75, 0, 0, 0, 0],
        [1.0, 0, 0, 0, 0],
        [1.0, 0, 0, lookup_angle, 0],
        [1.0, 0, 0, lookdown_angle, 0],
        [1.0, 0, 0, 0, 0],
    ]
        return walk_lookupdown

def interpolate_camera_keypoints(camera_keypoints, fpm, fpd_e, fpd_a, max_x):
    all_cameras = []
    for i in range(len(camera_keypoints)-1):
        start = camera_keypoints[i]
        end = camera_keypoints[i+1]

        num_cameras_x = np.abs(end[0] - start[0]) * max_x * fpm 
        num_cameras_y = np.abs(end[1] - start[1]) * fpm
        num_cameras_z = np.abs(end[2] - start[2]) * fpm
        num_camers_elevs = np.abs(end[3] - start[3]) * fpd_e
        num_camers_azims = np.abs(end[4] - start[4]) * fpd_a

        num_cameras = max(num_cameras_x, num_cameras_y, num_cameras_z, num_camers_elevs, num_camers_azims)
        num_cameras = int(num_cameras) + 1

        # print(f"num_cameras_x: {num_cameras_x}, num_cameras_y: {num_cameras_y}, num_cameras_z: {num_cameras_z}")
        # print(f"num_camers_elevs: {num_camers_elevs}, num_camers_azims: {num_camers_azims}")

        all_x = np.linspace(start[0], end[0], num_cameras)
        all_y = np.linspace(start[1], end[1], num_cameras)
        all_z = np.linspace(start[2], end[2], num_cameras)
        all_elevs = np.linspace(start[3], end[3], num_cameras)
        all_azims = np.linspace(start[4], end[4], num_cameras)

        all_cameras.extend(stretch_append(all_x, all_y, all_z, all_elevs, all_azims))
    return all_cameras

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

    # Test open3d statistical outlier removal
    colors = np.random.rand(1000, 3)  # random colors
    pts = (np.random.rand(1000, 3)-0.5) * 10.0  # random 3D points
    pts[::50] += np.random.rand(20, 3) * 50
    pts_clean, colors_clean = GeometryTransforms.remove_statistical_outliers(pts, colors, nb_neighbors=20, std_ratio=1.8)
    print(f"Original points: {pts.shape[0]}, Cleaned points: {pts_clean.shape[0]}")
    # --- TEST OPEN WORLD ---- 

    # Optional: make 3D axes have equal aspect (so spheres look like spheres)
    def set_equal_aspect_3d(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        xmid = np.mean(xlim); ymid = np.mean(ylim); zmid = np.mean(zlim)
        radius = max((xlim[1]-xlim[0]), (ylim[1]-ylim[0]), (zlim[1]-zlim[0])) / 2
        ax.set_xlim3d([xmid - radius, xmid + radius])
        ax.set_ylim3d([ymid - radius, ymid + radius])
        ax.set_zlim3d([zmid - radius, zmid + radius])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


    pts_xyz = (np.random.rand(5000, 3)-0.5) * 10.0  # random 3D points
    forward = np.array([1.0, 0.0, 0.0])
    kept_pts_carte, pts_carte, mask_keep = open_world_carte(
        forward_carte=forward,
        pts_carte=pts_xyz, 
        opening_mode='cut+cylinder',
        delta_cut=np.pi/3
    )
    pts_new = kept_pts_carte
    assert np.allclose(pts_new, pts_carte[mask_keep])
    #visualize
    # --- Visualization ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(
        pts_xyz[:, 0], pts_xyz[:, 1], pts_xyz[:, 2],
        s=5, alpha=0.5, color='steelblue', label='Original'
    )

    # Plot transformed points
    ax.scatter(
        pts_new[:, 0], pts_new[:, 1], pts_new[:, 2],
        s=2, alpha=0.8, color='red', label='After open_world'
    )

    # Labels & style
    ax.set_title("Point Cloud Before and After open_world()")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_equal_aspect_3d(ax)

    plt.tight_layout()
    plt.show()

    # test build_disk_to_square_displacement_fn
    test_cylinder_wall_unfolding=False
    if test_cylinder_wall_unfolding:

        def cylinder_point_cloud(N=10000, radius=1.0, length=2.0):
            # Uniform samples along the axis
            u = np.random.uniform(-length/2, length/2, N)
            
            # Uniform angle around the axis
            theta = np.random.uniform(0, 2*np.pi, N)
            
            # Points on cylinder surface
            x = u
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            return np.vstack((x, y, z)).T

        pts_carte = cylinder_point_cloud(N=10000, radius=1.0, length=4.0)
        pts_carte_prime = GeometryTransforms.correct_walls_cylinder_unfold(pts_carte)

        # Visualization
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(pts_carte[:, 0], pts_carte[:, 1],
                    pts_carte[:, 2], s=1, c='blue')
        ax1.set_title("Original Cylinder Point Cloud")
        set_equal_aspect_3d(ax1)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(pts_carte_prime[:, 0], pts_carte_prime[:, 1],
                    pts_carte_prime[:, 2], s=1, c='green')
        ax2.set_title("After Cylinder Wall Unfolding")
        set_equal_aspect_3d(ax2)
        plt.tight_layout()
        plt.show()

    test_displancement_fn=True
    if test_displancement_fn:
        displacement_fn = build_disk_to_square_displacement_fn(
            center=(0.0, 0.0),
            radius=1.0,
            num_points=700000,
            threshold=0.8,
            forward=np.array([1.0, -1.0]),
            n_arc_points=200,
            n_line_points=50,
            plot=True,
        )

    test_old=True
    if test_old:
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



        # --- plotting section fix ---
        methods = ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut', 'straight_cut+disk_to_square_displacement']

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
            forward_carte = sph2carte_3D(forward_sph)
            pts_carte = sph2carte_3D(pts_sph)
            pts_opened_xyz, _, _ = open_world_carte(forward_carte, pts_carte, opening_mode=mode, delta_cut=2*np.pi/3)
            ax = axes[1, jj]
       
            ax.scatter(*pts_opened_xyz.T, s=1, c=xyz_to_rgb(pts_opened_xyz, coord_type='cartesian'))
            ax.set_title(f"Open world ({mode})")
            set_equal_aspect_3d(ax)

        plt.tight_layout()
        plt.show()


        # open3d visualization of pts_opened_xyz
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_opened_xyz)
            colors = xyz_to_rgb(pts_opened_xyz, coord_type='cartesian')
            pcd.colors = o3d.utility.Vector3dVector(colors)

            o3d.visualization.draw_geometries([pcd], window_name="Opened Point Cloud")
        except ImportError:
            print("Open3D is not installed. Skipping Open3D visualization.")    