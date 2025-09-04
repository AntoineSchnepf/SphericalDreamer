import os
import sys 
import numpy as np
import logging
from PIL import Image
import copy
import matplotlib.pyplot as plt
from scipy import ndimage
import copy

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



def cat_ones(array):
    """Concatenate a column of ones to the input array."""
    return np.concatenate((array, np.ones((*array.shape[:-1], 1))), axis=-1)


def depth_numpy_to_PIL(depth):
    depth = copy.deepcopy(depth)
    depth[np.isnan(depth)] = 0.0
    depth_pil = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to [0, 1]
    max_val = 65535
    depth_pil = (depth_pil * max_val).astype(np.uint16)              # Scale to [0, 65535]
    depth_pil = Image.fromarray(depth_pil)
    return depth_pil


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

#[u, v] actually reprensent a point on the unit sphere. 

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


