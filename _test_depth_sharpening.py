
import pickle
from cv2 import threshold
import numpy as np 
import open3d as o3d
from PIL import Image
import os
import sys
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
import my_utils 
import sys
from functools import reduce
import cv2
# bilateral filtering (copyright: 3DP)
def sparse_bilateral_filtering(
    depth, image, config, HR=False, mask=None, gsHR=True, edge_id=None, num_iter=None, num_gs_iter=None, spdb=False
):
    """
    config:
    - filter_size
    """
    import time

    save_images = []
    save_depths = []
    save_discontinuities = []
    vis_depth = depth.copy()
    backup_vis_depth = vis_depth.copy()

    depth_max = vis_depth.max()
    depth_min = vis_depth.min()
    vis_image = image.copy()
    for i in range(num_iter):
        if isinstance(config["filter_size"], list):
            window_size = config["filter_size"][i]
        else:
            window_size = config["filter_size"]
        vis_image = image.copy()
        save_images.append(vis_image)
        save_depths.append(vis_depth)
        u_over, b_over, l_over, r_over = vis_depth_discontinuity(vis_depth, config, mask=mask)
        vis_image[u_over > 0] = np.array([0, 0, 0])
        vis_image[b_over > 0] = np.array([0, 0, 0])
        vis_image[l_over > 0] = np.array([0, 0, 0])
        vis_image[r_over > 0] = np.array([0, 0, 0])

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1
        save_discontinuities.append(discontinuity_map)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter(
            vis_depth, config, discontinuity_map=discontinuity_map, HR=HR, mask=mask, window_size=window_size
        )

    return vis_depth

def vis_depth_discontinuity(depth, config, vis_diff=False, label=False, mask=None):
    """
    config:
    - 
    """
    if label == False:
        disp = 1./depth
        u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > config['depth_threshold']).astype(np.float32)
        b_over = (np.abs(b_diff) > config['depth_threshold']).astype(np.float32)
        l_over = (np.abs(l_diff) > config['depth_threshold']).astype(np.float32)
        r_over = (np.abs(r_diff) > config['depth_threshold']).astype(np.float32)
    else:
        disp = depth
        u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > 0).astype(np.float32)
        b_over = (np.abs(b_diff) > 0).astype(np.float32)
        l_over = (np.abs(l_diff) > 0).astype(np.float32)
        r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode='constant')
    b_over = np.pad(b_over, 1, mode='constant')
    l_over = np.pad(l_over, 1, mode='constant')
    r_over = np.pad(r_over, 1, mode='constant')
    u_diff = np.pad(u_diff, 1, mode='constant')
    b_diff = np.pad(b_diff, 1, mode='constant')
    l_diff = np.pad(l_diff, 1, mode='constant')
    r_diff = np.pad(r_diff, 1, mode='constant')

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]

def bilateral_filter(depth, config, discontinuity_map=None, HR=False, mask=None, window_size=False):
    sort_time = 0
    replace_time = 0
    filter_time = 0
    init_time = 0
    filtering_time = 0
    sigma_s = config['sigma_s']
    sigma_r = config['sigma_r']
    if window_size == False:
        window_size = config['filter_size']
    midpt = window_size//2
    ax = np.arange(-midpt, midpt+1.)
    xx, yy = np.meshgrid(ax, ax)
    if discontinuity_map is not None:
        spatial_term = np.exp(-(xx**2 + yy**2) / (2. * sigma_s**2))

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1,1), (1,1)), 'edge')
    pad_depth = np.pad(depth, (midpt,midpt), 'edge')
    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1,1), (1,1)), 'edge')
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt,midpt), 'edge')
        pad_discontinuity_hole = 1 - pad_discontinuity_map
    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1,1])
    if discontinuity_map is not None:
        pad_discontinuity_patches = rolling_window(pad_discontinuity_map, [window_size, window_size], [1,1])
        pad_discontinuity_hole_patches = rolling_window(pad_discontinuity_hole, [window_size, window_size], [1,1])

    if mask is not None:
        pad_mask = np.pad(mask, (midpt,midpt), 'constant')
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1,1])
    from itertools import product
    if discontinuity_map is not None:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue
                if discontinuity_map is not None:
                    if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]
                if discontinuity_map is not None:
                    coef = discontinuity_holes.astype(np.float32)
                    if mask is not None:
                        coef = coef * pad_mask_patches[pi, pj]
                else:
                    range_term = np.exp(-(depth_patch-patch_midpt)**2 / (2. * sigma_r**2))
                    coef = spatial_term * range_term
                if coef.max() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.max() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef/(coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if discontinuity_map is not None:
                    if pad_discontinuity_patches[pi, pj][window_size//2, window_size//2] == 1:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = (1. - discontinuity_patch)
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]
                range_term = np.exp(-(depth_patch-patch_midpt)**2 / (2. * sigma_r**2))
                if discontinuity_map is not None:
                    coef = spatial_term * range_term * discontinuity_holes
                else:
                    coef = spatial_term * range_term
                if coef.sum() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.sum() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef/(coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output

def rolling_window(a, window, strides):
    assert len(a.shape)==len(window)==len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"
    shape_fn = lambda i,w,s: (a.shape[i]-w)//s + 1
    shape = [shape_fn(i,w,s) for i,(w,s) in enumerate(zip(window, strides))] + list(window)
    def acc_shape(i):
        if i+1>=len(a.shape):
            return 1
        else:
            return reduce(lambda x,y:x*y, a.shape[i+1:])
    _strides = [acc_shape(i)*s*a.itemsize for i,s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)

# detect edges from depth (my code)
def sharpen_depth_sparse_bilateral(depth, image, config, mask=None, num_iter=None):
    """
    Apply your existing sparse_bilateral_filtering to a depth map and return
    the final sharpened depth.

    Parameters
    ----------
    depth : (H, W) float32
        Depth map (in meters or any consistent unit).
    image : (H, W, 3) uint8
        RGB image aligned with depth (used by the filtering).
    config : dict
        Must contain at least:
            - 'filter_size' (int or list of ints)
            - 'depth_threshold'
            - 'sigma_s'
            - 'sigma_r'
    mask : (H, W) bool or 0/1, optional
        Validity mask; invalid pixels can be ignored.
    num_iter : int, optional
        Number of iterations. If None, tries to infer from config['filter_size']
        when it's a list, or defaults to 1.

    Returns
    -------
    depth_filtered : (H, W) float32
        Final sharpened depth.
    all_depths : list of np.ndarray
        List of intermediate depth maps (including the original).
    """
    # Infer num_iter if needed
    if num_iter is None:
        if isinstance(config.get("filter_size", 3), list):
            num_iter = len(config["filter_size"])
        else:
            num_iter = 1

    # Call your original function
    depth_filtered = sparse_bilateral_filtering(
        depth=depth,
        image=image,
        config=config,
        HR=False,
        mask=mask,
        gsHR=True,
        edge_id=None,
        num_iter=num_iter,
        num_gs_iter=None,
        spdb=False,
    )

    return depth_filtered

def sobel_edges_from_depth(depth, mask=None, ksize=3):
    """
    Compute Sobel edge magnitude from a depth map.

    Parameters
    ----------
    depth : (H, W) float32
        Depth map (can contain NaNs).
    mask : (H, W) bool or 0/1, optional
        Valid mask; outside mask can be treated as NaN.
    ksize : int
        Sobel kernel size (1, 3, 5, ...); 3 is standard.

    Returns
    -------
    edges_uint8 : (H, W) uint8
        Edge magnitude image scaled to [0, 255].
    """
    depth_proc = depth.copy().astype(np.float32)

    # Apply mask if provided
    if mask is not None:
        depth_proc[~mask] = np.nan

    # Replace NaNs with median of valid depth
    valid = np.isfinite(depth_proc)
    if not np.any(valid):
        raise ValueError("No valid depth values for edge detection.")
    median_val = np.median(depth_proc[valid])
    depth_proc[~valid] = median_val

    # Normalize depth to [0, 255] for Sobel (optional but helps)
    dmin, dmax = depth_proc.min(), depth_proc.max()
    if dmax > dmin:
        depth_norm = (depth_proc - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth_proc)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)

    # Sobel gradients
    sobelx = cv2.Sobel(depth_8u, cv2.CV_32F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(depth_8u, cv2.CV_32F, 0, 1, ksize=ksize)

    # Gradient magnitude
    mag = cv2.magnitude(sobelx, sobely)

    # Normalize to [0, 255] and convert to uint8
    edges_uint8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return edges_uint8

def canny_edges_from_depth(depth, mask=None, low=50, high=150):
    edges_mag = sobel_edges_from_depth(depth, mask=mask)
    edges_canny = cv2.Canny(edges_mag, low, high)
    return edges_canny

depth_sharpen_default_config = {
        "filter_size": 5,          # or [5, 5, 5] for multiple iterations
        "depth_threshold": 0.01,   # threshold on disparity diff in vis_depth_discontinuity
        "sigma_s": 3.0,            # spatial sigma
        "sigma_r": 0.1,            # range sigma (in depth / disparity units)
}
def get_canny_sobel_edges(depth, image, edged_sobel_ksize=3, canny_low_t=15, canny_high_t=50, depth_sharpen_config=depth_sharpen_default_config):
    # Optional depth mask (e.g., depth > 0)
    depth_mask = np.isfinite(depth) & (depth > 0)

    # 1) Sparse bilateral filtering (sharpen depth)
    if depth_sharpen_config.apply:
        depth_sharpened = sharpen_depth_sparse_bilateral(
            depth=depth,
            image=image,
            config=depth_sharpen_config,
            mask=depth_mask,
            num_iter=None,  # will infer from config
        )
    else:
        depth_sharpened = depth.copy()

    # 2) Edge detection on filtered depth
    edges_sobel = sobel_edges_from_depth(depth_sharpened, mask=depth_mask, ksize=edged_sobel_ksize)
    edges_canny = canny_edges_from_depth(depth_sharpened, mask=depth_mask, low=canny_low_t, high=canny_high_t)

    return edges_canny, edges_sobel, depth_sharpened
    
def visualize_canny_sobel_edges(image, depth_origin, depth, depth_sharpened, edges_sobel, edges_canny):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax = axes.flatten()

    # Original image
    ax[0].imshow(image)
    ax[0].set_title("Original image")
    ax[0].axis("off")

    # Original depth
    im1 = ax[1].imshow(depth_origin, cmap="plasma")
    ax[1].set_title("Original depth")
    ax[1].axis("off")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    # Low frequency depth
    im2 = ax[2].imshow(depth, cmap="plasma")
    ax[2].set_title("Low-frequency depth removed")
    ax[2].axis("off")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    # Sharpened depth
    im3 = ax[3].imshow(depth_sharpened, cmap="plasma")
    ax[3].set_title("Sharpened depth (sparse bilateral)")
    ax[3].axis("off")
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    # Sobel edges
    ax[4].imshow(edges_sobel, cmap="gray")
    ax[4].set_title("Sobel edges from depth")
    ax[4].axis("off")

    # Canny edges (optional)
    ax[5].imshow(edges_canny, cmap="gray")
    ax[5].set_title("Canny edges")
    ax[5].axis("off")

    # Empty / reserved

    plt.tight_layout()
    plt.show()


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter



if __name__ == "__main__":

    # --- parse args: which sphere to load --- #
    config = my_utils.fetch_config_via_parser(
        debug=True, 
        debug_parser_override=["--config", "Antoine/F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    # overriding args for mac testing   
    # ---------------------------------------- #
    expname = "24_forest"
    dream_iter = 1
    open_right = False
    open_left = False
    # ---------------------------------------- #


    save_dir = "/Users/a.schnepf/Documents/code/phd/scene_gen/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse"
    save_dir_ = f"{save_dir}/{expname}"

    # --- script init --- 
    width = 1440
    height = 720
    sphere_radius = 1.0
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    FAR=2.0
    NEAR=0.2
    pcd_upsampling_factor = 4
    apply_sharpening = False
    apply_canny_edge_removal = False
    remove_3D_outliers = True
    config.pcd_upsampling_factor = int(pcd_upsampling_factor)

          
    config.ldi.masking.edges_detection.depth_sharpening.apply = True
    config.ldi.masking.edges_detection.depth_sharpening.filter_size = 5          
    config.ldi.masking.edges_detection.depth_sharpening.depth_threshold = 0.01   
    config.ldi.masking.edges_detection.depth_sharpening.sigma_s = 3.0         
    config.ldi.masking.edges_detection.depth_sharpening.sigma_r = 0.1  
    

    opening_kwargs = {
        'opening_mode': 'cut+cylinder',
        'delta_cut': 2*np.pi/3,
    }
    pose1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # --- script init end --- 
    
    sphere_correction_kwargs = {
        "correct_depth": False,
        "near": NEAR,
        "far": FAR,
        "correct_walls": False,
        "correct_floor": True,
        "depth_threshold_for_floor_correction": 0.6,
        "remove_sky": False,
        "remove_outliers": remove_3D_outliers,
        "outlier_removal_options": {
            "nb_neighbors": 20,
            "std_ratio": 1.8,
        },
        "verbose": False,
        "plot": True,
    }

    colors1, depth1 = my_utils.load_rgbd_pano(
        dream=0,
        save_dir_=save_dir_
    )

    # depth sharpening
    if apply_sharpening:
        t0 = time.time()
        depth_sharpened = sharpen_depth_sparse_bilateral(
            depth=depth1,
            image=(colors1*255).astype(np.uint8),
            config=config.ldi.masking.edges_detection.depth_sharpening,
            mask=None,
            num_iter=None,  # will infer from config
        )
        t1 = time.time()
        print(f"Depth sharpening took {t1-t0:.2f} seconds.")

        # vizualize
        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        min_ = min(depth1.min(), depth_sharpened.min())
        max_ = max(depth1.max(), depth_sharpened.max())
        axes[0].imshow(depth1[200:250, 150:250], cmap='plasma', vmin=min_, vmax=max_)
        axes[1].imshow(depth_sharpened[200:250, 150:250], cmap='plasma', vmin=min_, vmax=max_)
        plt.show()
    else : 
        depth_sharpened = depth1

    if apply_canny_edge_removal:
        edges_sobel = sobel_edges_from_depth(depth1, mask=None, ksize=config.ldi.masking.edges_detection.sobel.ksize)
        # edges_canny = canny_edges_from_depth(depth1, mask=None, low=config.ldi.masking.edges_detection.canny.low_t, high=config.ldi.masking.edges_detection.canny.high_t)
        edges_low_t = 35 # Antoine: Maybe take it from       config.ldi.masking.edges_detection.canny.low_t: 35
        edges = (edges_sobel > edges_low_t).astype(bool)

        colors1[edges] = np.nan  # black out edges in color
        depth_sharpened[edges] = np.nan  # black out edges in depth

    if pcd_upsampling_factor>1:
        colors1 = my_utils.opencv_resize(colors1, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        depth_sharpened = my_utils.opencv_resize(depth_sharpened, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
    

    pts1_carte = my_utils.depth2cam_carte(
        depth=depth_sharpened,
        sphere_radius=config.sphere_radius,
        height=height*config.pcd_upsampling_factor,
        width=width*config.pcd_upsampling_factor,
    )

    nan_mask = np.isnan(pts1_carte).any(axis=-1)
    pts1_carte = pts1_carte[~nan_mask]
    colors1 = colors1.reshape(-1, 3)[~nan_mask.reshape(-1)]


    pts1_carte_corrected, colors1_corrected = my_utils.run_corrective_pipeline_on_sphere(
        pts1_carte, # in cartesian coordinates
        colors1, 
        height, width, 
        **sphere_correction_kwargs
    )


    sphere1 = my_utils.Sphere(
        pose1, pts1_carte_corrected, colors1_corrected, 
        forward_carte=translation_direction,
        opening_kwargs=opening_kwargs,
    )

    # open 3d viewer
    if open_right and not open_left:
        pcd = sphere1.right_opened.get_world_pcd().get_o3d_pointcloud()
    elif open_left and not open_right:
        pcd = sphere1.left_opened.get_world_pcd().get_o3d_pointcloud()
    elif open_right and open_left:
        pcd = sphere1.both_opened.get_world_pcd().get_o3d_pointcloud()
    else:
        pcd = sphere1.closed.get_world_pcd().get_o3d_pointcloud()

    o3d.visualization.draw_geometries([pcd])




