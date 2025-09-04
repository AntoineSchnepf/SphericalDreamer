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

# -----Computer Vision ------#
def fill_mask(mask):
    # mask: boolean NumPy array
    # Fill holes (False regions completely surrounded by True)
    return ndimage.binary_fill_holes(mask)

def close_mask(mask, size=5):
    # size controls how aggressively to close gaps
    structure = np.ones((size, size), dtype=bool)
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


# ----- Numpy - PIL conversions / utils -----#
def cat_ones(array):
def depth_numpy_to_PIL(depth):
    depth = copy.deepcopy(depth)
    depth[np.isnan(depth)] = 0.0
    depth_pil = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
    max_val = 65535
    depth_pil = (depth_pil * max_val).astype(np.uint16)              # Scale to [0, 65535]
    depth_pil = Image.fromarray(depth_pil)
    """Concatenate a column of ones to the input array."""
    return np.concatenate((array, np.ones((*array.shape[:-1], 1))), axis=-1)
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

#----- 3D Geometry: equirectangular, spherical & cartesian coordinates ----- #

# ERP image coordinate system:
# ┌─────────────────────────► u 
# │
# │   [0,0]         [0, w-1]
# │     ●────────────●
# │     │            │
# │     │            │
# │     ●────────────●
# │   [h-1, 0]      [h-1,w-1]
# ▼
# v 

#[u, v] reprensent a point on the unit sphere. 

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

def erp_to_world(points_2D_cam_erp, height, width, depth, pose, sphere_radius=1.0):
    """
    Convert Equirectangular coordinates to world coordinates.
    
    Args:
        points_2D_cam_erp (np.array): Equirectangular coordinates of shape [..., 2].
        depth (np.array): Depth map of shape [...].
        pose (np.array): Camera pose matrix of shape [4, 4].
        sphere_radius (float): Radius of the sphere.
    
    Returns:
       points_3D_world_carte: np.array w. shape [..., 3]. World coordinates. Convention X, Y, Z.
    """
    assert np.all(points_2D_cam_erp.shape[:-1] == depth.shape)
    points_2D_cam_sph = erp2sph_2D(points_2D_cam_erp, erp_image_height=height, erp_image_width=width)
    r = depth * sphere_radius
    points_3D_cam_sph = np.concatenate((points_2D_cam_sph, np.expand_dims(r, axis=-1)), axis=-1)
    points_3D_cam_carte = sph2carte_3D(points_3D_cam_sph)
    points_3D_world_carte = np.einsum('ij,...j->...i', pose, cat_ones(points_3D_cam_carte))[..., :3]
    return points_3D_world_carte


# ---- Warping / Splatting functions -----
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

def depth_aware_naive_splatting_vectorized(colors1, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    Vectorized depth-aware splatting with a z-buffer.
    Supports inputs shaped [H, W, *] or flattened [HW, *].
    """
    # Basic checks
    assert colors1.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert coord_cam1.shape[-1] == 2
    assert colors1.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    # Flatten to [N, ...]
    colors1   = colors1.reshape((-1, 3))
    coord_cam1 = coord_cam1.reshape((-1, 2))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    # Round target coordinates to nearest integer pixel (u -> x/col, v -> y/row)
    u = coord_cam2[:, 0]
    v = coord_cam2[:, 1]
    u_r = np.rint(u).astype(np.int32)
    v_r = np.rint(v).astype(np.int32)

    # Keep only those that fall inside the target frame
    in_bounds = (u_r >= 0) & (u_r < width) & (v_r >= 0) & (v_r < height)
    if not np.any(in_bounds):
        warped_img   = np.zeros((height, width, 3), dtype=np.float32)
        warped_depth = np.full((height, width), np.inf, dtype=np.float32)
        visited      = np.zeros((height, width), dtype=bool)
        return warped_img, warped_depth, {}, visited

    # Restrict to valid points
    u_r = u_r[in_bounds]
    v_r = v_r[in_bounds]
    depths = depth_cam2[in_bounds].astype(np.float32)
    colors = colors1[in_bounds]
    coord1 = coord_cam1[in_bounds]
    u_float = u[in_bounds]
    v_float = v[in_bounds]

    # Linearized target indices (row-major)
    tgt_lin = (v_r.astype(np.int64) * width + u_r.astype(np.int64))

    # Resolve collisions per target pixel with z-buffer: keep the *nearest* depth
    order = np.lexsort((depths, tgt_lin))        # primary: tgt_lin, secondary: depth (ascending)
    tgt_sorted = tgt_lin[order]
    _, first_idx = np.unique(tgt_sorted, return_index=True)
    winners = order[first_idx]

    # Winners' data
    u_win_r = u_r[winners]
    v_win_r = v_r[winners]
    depths_win = depths[winners]
    colors_win = colors[winners]
    coord1_win = coord1[winners]      # (a, b) source coordinates
    u_win_f = u_float[winners]        # unrounded u for flow key
    v_win_f = v_float[winners]        # unrounded v for flow key

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

def depth_aware_naive_splatting(colors1, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    This functions computes a new image, at a new camera location, based on:
        (i) a set coordinates of colored points in the new image: `coord_cam2` (float values)
        (ii) corresponding colors: `colors1`
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

        Modif 20 Aug: img, coord_cam2 and depth_cam2 can be flattened
    """
    assert colors1.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert colors1.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    colors1 = colors1.reshape((-1, 3))
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

        color = colors1[k]
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

    print("all tests run successfully")


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


