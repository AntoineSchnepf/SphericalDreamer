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
import time



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
# ----- Panorama / Pointcloud utils  ----- #
# ---------------------------------------- #
def load_rgbd_pano(dream, save_dir_, override_depth_with_ones=False):

    pano_rgb = Image.open(f"{save_dir_}/dream_{dream:02d}/XX_pano_rgb.png")
    depth = np.load(f"{save_dir_}/dream_{dream:02d}/XX_depth.npy")
    if override_depth_with_ones:
        depth = np.ones_like(depth)  
        print("WARNING: depth override to ones for debugging purposes")
    colors = np.array(pano_rgb)/255.0
    return colors, depth

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

def camera_translation(pose, translation):
    """
    pose: np.array of shape [4,4]
    translation: np.array of shape [3,] in world coordinates
    """
    pose2 = pose.copy()
    pose2[:3, 3] += translation
    return pose2


# ---------------------------------------- #
# ------- Depth Correction utils  -------- #
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
    def correct_floor_v1(P, depth_map_eqr, error_type='l1', plot=False):
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
    def correct_floor(P, height, width, plot=True):
        """
        Corrects a 3D point cloud so that the detected floor becomes flat.

        This function identifies and models the lowest (floor) surface of a 
        point cloud, assuming the floor corresponds to the lower envelope 
        of the (Z, sqrt(X² + Y²)) profile below the horizon. It fits a 
        non-decreasing local-minimum interpolator to estimate the floor height 
        as a function of radial distance from the origin, and then vertically 
        adjusts all points so that this estimated floor becomes flat.

        Parameters
        ----------
        P : ndarray of shape (..., 3)
            Input 3D point cloud. Each row or voxel corresponds to (X, Y, Z)
            coordinates in world or sensor space.
        plot : bool, optional, default=True
            If True, plots the original (C, Z) data below the horizon and the 
            fitted floor profile for visual inspection.

        Returns
        -------
        P_corrected : ndarray of shape (..., 3)
            The point cloud after applying the vertical correction that flattens 
            the floor. The Z-axis is adjusted such that the estimated floor 
            becomes level at the mean horizon height.
        correction_raw : ndarray of shape (..., 3)
            The vertical displacement applied to each point (ΔX, ΔY, ΔZ). 
            Only the Z component is non-zero.

        Notes
        -----
        The correction is derived as follows:

        1. Convert the 3D points (X, Y, Z) to polar coordinates 
        `C = sqrt(X² + Y²)` and select only the region below the horizon.
        2. Fit a smooth, non-decreasing local-minimum interpolator 
        `Z_floor(C)` to the lowest observed points in each radial bin.
        This ensures that the estimated floor height is monotonic with 
        respect to distance and not affected by local noise or clutter.
        3. Estimate the mean horizon height `Z_horizon` using points near 
        the horizon band.
        4. Compute the per-point correction along Z as 
        `ΔZ = Z_horizon - Z_floor(C)`, and apply it to flatten the floor.

        The result is a geometrically corrected point cloud where the 
        ground plane appears level, useful for visualization, mapping, 
        or downstream processing.

        See Also
        --------
        Regression1D.fit_local_min_knots_monotone_interpolator_1d : 
            Used internally to fit the monotone local-minimum interpolator.
        get_canonical_sph_pixels : 
            Provides the spherical pixel mapping used to select points 
            below the horizon.
        """
        sph_canon = get_canonical_sph_pixels(height, width)
        phi_range = (
            np.deg2rad(0), np.deg2rad(360)
        )
        below_horizon_mask_and_phi_in_range = (sph_canon[..., 0] < 0) & (sph_canon[..., 1] >= phi_range[0]) & (sph_canon[..., 1] <= phi_range[1])
        X_bh, Y_bh, Z_bh = P[below_horizon_mask_and_phi_in_range].T
        c_bh = np.sqrt(X_bh**2 + Y_bh**2)

        bandwidth = 0.05 * (np.max(c_bh) - np.min(c_bh)) 
        t0 = time.time()
        f_approx = Regression1D.fit_local_min_knots_monotone_interpolator_1d(c_bh, Z_bh, bandwidth=bandwidth)
        print("Regression for depth correction took:", time.time() - t0)
        
        # compute correction
        thetas = sph_canon[..., 0]
        horizon_theta_range=(
            np.deg2rad(-10), np.deg2rad(-1) 
        )
        horizon_band_mask = (thetas[:,0] >= horizon_theta_range[0]) & (thetas[:,0] <= horizon_theta_range[1])
        les_z_in_band = P[horizon_band_mask, ..., 2]
        z_horizon = np.nanmean(les_z_in_band)

        correction_raw = np.zeros_like(P)
        c_input = np.sqrt(P[...,0]**2 + P[...,1]**2)
        correction_raw = z_horizon - f_approx(c_input)
        correction_raw = correction_raw[..., None] * np.array([0, 0, 1])
        P_corrected = P + correction_raw

        if plot:
            plt.figure()
            plt.title("Z values below horizon")
            plt.scatter(c_bh, Z_bh, s=1, label='Data Points')
            les_c = np.linspace(np.min(c_bh), np.max(c_bh), 100)

            les_z_approx = f_approx(les_c)
            plt.plot(les_c, les_z_approx, color='r', label='Kernel Regression Fit')
            plt.xlabel("C = sqrt(X^2 + Y^2)")
            plt.ylabel("Z values")
            plt.legend()
            plt.show()

            plt.figure()
            plt.imshow(correction_raw[..., 2], cmap='jet')
            plt.title("Raw Depth Correction (new method)")
            plt.colorbar()
            plt.show()
        
        return P_corrected

    @staticmethod
    def get_sky_mask(
            depth_map,         
            height,
            width,
            thetas_range_for_sky_detection = (np.deg2rad(80), np.deg2rad(90)),
            eps = 0.5
        ):

        thetas = get_canonical_sph_pixels(height, width)[..., 0]
        sky_theta_mask = (thetas >= thetas_range_for_sky_detection[0]) & (thetas <= thetas_range_for_sky_detection[1])
        depth_sky_values = depth_map[sky_theta_mask]
        threshold = np.nanmean(depth_sky_values) - eps * np.nanstd(depth_sky_values)
        sky_mask = depth_map >= threshold
        return sky_mask

    @staticmethod
    def _smoothstep(a, b, x):
        """Cubic smoothstep from 0→1 on [a,b]."""
        t = np.clip((x - a) / (b - a + 1e-12), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def correct_walls(
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
    def remove_statistical_outliers(pts, colors, nb_neighbors=20, std_ratio=1.8):
        import open3d as o3d
        pcd = PointCloud(pts, colors).get_o3d_pointcloud()
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_pts = np.asarray(cl.points)
        inlier_colors = np.asarray(cl.colors)
        return inlier_pts, inlier_colors

    @staticmethod
    def run_corrective_pipeline(colors, depth, sphere_radius, height, width, correct_depth, near, far, correct_floor, correct_walls, remove_sky, indoor_or_outdoor, remove_outliers, verbose=False):
        #TODO:
        # - docstring

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
                std_ratio=1.8
            )
            n_after = len(pts_cam_cartesian.reshape(-1,3))
            if verbose:
                print(f"e. (Optional) Outliers Removed ({(n_before - n_after) / n_before * 100:.2f}%)")

        return pts_cam_cartesian, colors


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
# ----   World Opening transformations -----  #
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
    Remove points that lie *behind* the plane orthogonal to `forward_carte`
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

def open_world_sph(forward_sph, pts_sph, opening_mode='cut+cylinder', delta_cut=np.pi/2, cut_distance=None, cut_distance_percentile=90):
    assert opening_mode in ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut']
    if opening_mode == 'wall':
        pts_opened = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        mask_keep = np.ones_like(pts_sph[..., 0], dtype=bool)
    elif opening_mode == 'cut+wall':
        pts_opened = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'cut+cylinder':
        pts_opened = unfold_points_on_cylinder(pts_sph, forward_sph, base_radius=1.0)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'remove_within_cone':
        pts_opened = pts_sph
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'straight_cut':
        if cut_distance is None:
            cut_distance = compute_cut_distance_based_on_percentile(
                pts_sph, forward_sph, percentile=cut_distance_percentile
            )
        pts_opened = pts_sph
        _, mask_keep = filter_points_by_plane_sph(pts_sph, forward_sph, cut_distance=cut_distance)

    return pts_opened[mask_keep], pts_opened, mask_keep

def open_world_carte(forward_carte, pts_carte, opening_mode='cut+cylinder', delta_cut=np.pi/2, cut_distance=None, cut_distance_percentile=90):
    assert opening_mode in ['wall', 'cut+wall', 'cut+cylinder', 'remove_within_cone', 'straight_cut']

    if opening_mode == 'wall':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        pts_opened = sph2carte_3D(pts_opened_sph)
        mask_keep = np.ones_like(pts_sph[..., 0], dtype=bool)
    elif opening_mode == 'cut+wall':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_points_on_walls(pts_sph, forward_sph, delta=np.pi)
        pts_opened = sph2carte_3D(pts_opened_sph)
        _, mask_keep = remove_points_within_cone(pts_sph, forward_sph, delta=delta_cut)
    elif opening_mode == 'cut+cylinder':
        pts_sph = carte2sph_3D(pts_carte)
        forward_sph = carte2sph_3D(forward_carte)
        pts_opened_sph = unfold_points_on_cylinder(pts_sph, forward_sph, base_radius=1.0)
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

    return pts_opened[mask_keep], pts_opened, mask_keep

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


    test_old=False
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
            pts_opened_sph, _, _ = open_world_sph(forward_sph, pts_sph, mode=mode, delta_cut=2*np.pi/3)
            ax = axes[1, jj]
            pts_opened_xyz = sph2carte_3D(pts_opened_sph)
            ax.scatter(*pts_opened_xyz.T, s=1, c=xyz_to_rgb(pts_opened_sph, coord_type='spherical'))
            ax.set_title(f"Open world ({mode})")
            set_equal_aspect_3d(ax)

        plt.tight_layout()
        plt.show()