import copy
import os
import time
import torch
import pickle
import random
from functools import reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter1d,
)
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPTextModel, CLIPTokenizer
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)
from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from tqdm import tqdm
from prodict import Prodict
from PIL import Image
import my_utils
from sphericaldreamer import SphericalDreamer
from src.Infusion.depth_inpainting.utils.seed_all import seed_all
from src.Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half import (
    DepthEstimationInpaintPipeline,
)
from matplotlib import colors
from harmonic_blending import harmonic_blend_of_depths
from pathlib import Path

# STEP 1: FOREGROUD OBJECT MASK GENERATION
# remove low frequency from depth (my code)
def minmax_norm(x, out_min=0.0, out_max=1.0):
    """
    Normalize an array to a custom range [out_min, out_max].

    Parameters
    ----------
    x : np.ndarray
        Input array.
    out_min : float
        Lower bound of the normalized output range.
    out_max : float
        Upper bound of the normalized output range.

    Returns
    -------
    x_norm : np.ndarray
        Array normalized to [out_min, out_max], dtype float32.
    """
    x = np.asarray(x, dtype=np.float32)

    # Compute input min & max while ignoring NaNs
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    # Handle constant arrays → return midpoint value
    if xmax - xmin < 1e-12:
        return np.full_like(x, (out_min + out_max) / 2, dtype=np.float32)

    # Normalize to [0, 1]
    x01 = (x - xmin) / (xmax - xmin)

    # Scale to custom range
    return x01 * (out_max - out_min) + out_min

def get_low_freq_via_fft(img, cutoff_frac=0.1):
    """
    Extract a vertically low-frequency version of an image.
    Frequencies are computed ONLY along the vertical axis, and
    the final image is invariant along the horizontal axis
    (each row has a constant value across all columns).

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (H, W, C), dtype float or uint8.
    cutoff_frac : float
        Fraction of vertical frequencies to keep around DC (0..0.5).
        e.g. 0.1 => keep lowest 10% vertical frequencies.

    Returns
    -------
    low_img : np.ndarray
        Low-frequency, row-constant image (same shape as input).
    """
    img = np.asarray(img)
    is_uint8 = img.dtype == np.uint8
    img_f = img.astype(np.float32)

    # If grayscale / depth (H, W), add a channel dimension
    squeeze_channel = False
    if img_f.ndim == 2:
        img_f = img_f[..., None]  # (H, W, 1)
        squeeze_channel = True

    H, W, C = img_f.shape

    # 1) Collapse along horizontal axis: mean over width
    #    vertical_profile: shape (H, C)
    vertical_profile = img_f.mean(axis=1)  # (H, C)

    # 2) 1D FFT along vertical axis
    F = np.fft.fft(vertical_profile, axis=0)        # (H, C)
    F_shift = np.fft.fftshift(F, axes=0)           # center zero-freq vertically

    # 3) Build vertical low-pass mask along axis 0
    k_max = int(cutoff_frac * (H / 2.0))
    print("k_max:", k_max)
    k_max = max(1, k_max)

    mask = np.zeros(H, dtype=bool)
    center = H // 2
    start = max(0, center - k_max)
    end   = min(H, center + k_max + 1)
    mask[start:end] = True  # keep low vertical frequencies

    # 4) Zero out high vertical frequencies
    F_shift_filtered = np.zeros_like(F_shift)
    F_shift_filtered[mask, :] = F_shift[mask, :]

    # 5) Inverse FFT → low-frequency vertical profile
    F_filtered = np.fft.ifftshift(F_shift_filtered, axes=0)
    low_profile = np.fft.ifft(F_filtered, axis=0).real  # (H, C)

    # 6) Broadcast back across width → row-constant image
    low_img = np.repeat(low_profile[:, None, :], W, axis=1)  # (H, W, C)

    # 7) Restore original dimensionality / dtype
    if squeeze_channel:
        low_img = low_img[..., 0]

    if is_uint8:
        low_img = np.clip(low_img, 0, 255).astype(np.uint8)

    return low_img

def get_low_freq_via_gaussian_filter(img, sigma_h=50.0, sigma_v=50.0):
    """
    Extract a vertically low-frequency version of an image using 1D Gaussian
    filters along both axes, but return a row-constant image obtained from
    a 1D vertical profile.

    Pipeline:
      1) Horizontal 1D Gaussian blur (low-pass in x).
      2) Collapse to vertical profile (H, C) via mean over width.
      3) Vertical 1D Gaussian blur on that profile (low-pass in y).
      4) Broadcast back over width → each row is constant (row-constant image).

    Parameters
    ----------
    img : np.ndarray
        Image array of shape (H, W) or (H, W, C), dtype float or uint8.
    sigma_h : float
        Gaussian sigma for horizontal low-pass (along axis=1).
    sigma_v : float
        Gaussian sigma for vertical low-pass (along axis=0) on the profile.

    Returns
    -------
    low_img : np.ndarray
        Low-frequency, row-constant image (same shape as input).
    """
    img = np.asarray(img)
    is_uint8 = (img.dtype == np.uint8)
    img_f = img.astype(np.float32)

    # If grayscale / depth (H, W), add a channel dim
    squeeze_channel = False
    if img_f.ndim == 2:
        img_f = img_f[..., None]  # (H, W, 1)
        squeeze_channel = True

    H, W, C = img_f.shape

    # 1) Horizontal Gaussian filter (low-pass in x)
    img_hf = gaussian_filter1d(img_f, sigma=sigma_h, axis=1)

    # 2) Collapse over width → vertical profile (H, C)
    vertical_profile = img_hf.mean(axis=1)  # (H, C)

    # 3) Vertical Gaussian filter on the profile (low-pass in y)
    low_profile = gaussian_filter1d(vertical_profile, sigma=sigma_v, axis=0)  # (H, C)

    # 4) Broadcast back across width → row-constant image
    low_img = np.repeat(low_profile[:, None, :], W, axis=1)  # (H, W, C)

    # Restore original dimensionality / dtype
    if squeeze_channel:
        low_img = low_img[..., 0]

    if is_uint8:
        low_img = np.clip(low_img, 0, 255).astype(np.uint8)

    return low_img

def get_low_freq_via_hmax(img, smooth_v=None, ignore_invalid=True):
    """
    Row-constant "low-frequency" image built from the horizontal maximum.

    For each row y, we compute:
        v[y, c] = max_x img[y, x, c]
    and broadcast it back over width, so each row is constant across columns.

    Parameters
    ----------
    img : np.ndarray
        Array of shape (H, W) or (H, W, C), dtype float or uint8.
        (Works for depth too; typically (H, W).)
    smooth_v : int or None
        If not None, applies a vertical max-filter of size `smooth_v`
        on the per-row profile (helps reduce row-wise noise).
    ignore_invalid : bool
        If True, ignore NaNs and non-positive values when taking the max
        (useful for depth maps with zeros/NaNs). Falls back to 0 if a row has no valid values.

    Returns
    -------
    low_img : np.ndarray
        Row-constant image with same shape and dtype semantics as input.
    """
    img = np.asarray(img)
    is_uint8 = (img.dtype == np.uint8)
    img_f = img.astype(np.float32)

    # If grayscale / depth (H, W), add a channel dimension
    squeeze_channel = False
    if img_f.ndim == 2:
        img_f = img_f[..., None]  # (H, W, 1)
        squeeze_channel = True

    H, W, C = img_f.shape

    if ignore_invalid:
        # treat NaNs and <=0 as invalid (common for depth)
        valid = np.isfinite(img_f) & (img_f > 0)
        tmp = img_f.copy()
        tmp[~valid] = -np.inf
        row_profile = np.max(tmp, axis=1)  # (H, C)
        row_profile[~np.isfinite(row_profile)] = 0.0  # rows with no valid pixels
    else:
        row_profile = np.max(img_f, axis=1)  # (H, C)

    # Optional vertical smoothing on the profile
    if smooth_v is not None and smooth_v > 1:
        row_profile = maximum_filter1d(row_profile, size=int(smooth_v), axis=0)

    # Broadcast back across width → row-constant image
    low_img = np.repeat(row_profile[:, None, :], W, axis=1)  # (H, W, C)

    # Restore original dimensionality / dtype
    if squeeze_channel:
        low_img = low_img[..., 0]

    if is_uint8:
        low_img = np.clip(low_img, 0, 255).astype(np.uint8)

    return low_img

def remove_low_freq(depth, config):
    """
    Remove low vertical frequencies from a depth map.

    Parameters
    ----------
    depth : np.ndarray
        Depth map of shape (H, W), dtype float32.
    cutoff_frac : float
        Fraction of vertical frequencies to remove around DC (0..0.5).
        e.g. 0.1 => remove lowest 10% vertical frequencies.

    Returns
    -------
    depth_high : np.ndarray
        High-frequency depth map (same shape as input).
    low_freq : np.ndarray
        Low-frequency component that was removed (same shape as input).
    """

    """config:
        - method: 'fourier' or 'gaussian'
        - fourier:
            - cutoff_frac
        - gaussian:
            - sigma
    """
    if config.method == 'fourier':
        cutoff_frac = config.fourier.cutoff_frac
        low_freq = get_low_freq_via_fft(depth, cutoff_frac=cutoff_frac)
    elif config.method == 'gaussian':
        sigma_v = config.gaussian.sigma_v
        sigma_h = config.gaussian.sigma_h
        low_freq = get_low_freq_via_gaussian_filter(depth, sigma_v=sigma_v, sigma_h=sigma_h)
    elif config.method == 'hmax':
        smooth_v = config.hmax.smooth_v
        smooth_v = config.hmax.smooth_v if 'smooth_v' in config.hmax else None
        low_freq = get_low_freq_via_hmax(depth, smooth_v=smooth_v, ignore_invalid=True)
    else:
        raise ValueError(f"Unknown method: {config.method}. Should be either 'fourier' or 'gaussian'.")
    depth_high = depth - low_freq 
    return depth_high, low_freq

def visualize_low_freq_removal(depth_origin, low_freq, depth, title_prefix="", save_path=None):
    """
    Visualize low-frequency removal on a depth map.

    Parameters
    ----------
    depth_origin : (H, W) array-like
        Original depth map.
    low_freq : (H, W) array-like
        Low-frequency component (e.g. from get_low_freq_via_fft or Gaussian).
    depth : (H, W) array-like
        Depth after removal of low-frequencies (typically depth_origin - low_freq),
        i.e. the high-frequency component.
    title_prefix : str, optional
        Optional prefix for subplot titles (e.g. scene name).
    """
    depth_origin = np.asarray(depth_origin, dtype=np.float32)
    low_freq     = np.asarray(low_freq,     dtype=np.float32)
    depth        = np.asarray(depth,        dtype=np.float32)

    # Mask NaNs for nicer plotting
    orig_masked = np.ma.masked_invalid(depth_origin)
    low_masked  = np.ma.masked_invalid(low_freq)
    high_masked = np.ma.masked_invalid(depth)

    # Shared vmin/vmax for original & low-freq to compare scales
    vmin = np.nanmin(depth_origin)
    vmax = np.nanmax(depth_origin)

    # For high-frequency component, center around 0
    high_abs = np.nanmax(np.abs(depth))
    if not np.isfinite(high_abs) or high_abs == 0:
        high_abs = 1.0  # avoid degenerate range

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if title_prefix:
        fig.suptitle(f"{title_prefix} – Low-Frequency Removal", fontsize=16)

    # 1) Original depth
    im0 = axes[0].imshow(orig_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original depth")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2) Low-frequency component
    im1 = axes[1].imshow(low_masked, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Low-frequency vertical component")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3) High-frequency / residual depth (after removal)
    im2 = axes[2].imshow(
        high_masked,
        cmap="seismic",          # diverging colormap for +/- values
        vmin=-high_abs,
        vmax= high_abs
    )
    axes[2].set_title("High-frequency depth (origin - low_freq)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


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

def vis_depth_discontinuity(depth, config, vis_diff=False, label=False, mask=None, save_path=None):
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

    edges_canny = cv2.Canny(depth_8u, low, high)

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
    
def visualize_canny_sobel_edges(image, depth_origin, depth, depth_sharpened, edges_sobel, edges_canny, save_path=None):
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
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


# segmask scoring functions
def get_mask_score_via_hdistance_to_max_depth(
    seg: np.ndarray,
    depth_rect: np.ndarray,
    *,
    use_median: bool = False
) -> float:
    """
    Score a segmentation mask using the rectified depth map.

    Parameters
    ----------
    seg : (H, W) bool or {0,1} array
        Segmentation mask.
    depth_rect : (H, W) float
        Rectified depth map (horizontal maxima removed).
    use_median : bool
        If True, use median depth; otherwise use mean.

    Returns
    -------
    score : float
        Average depth value inside the mask.
        Lower values typically indicate foreground (closer).
    """
    seg = seg.astype(bool)

    dvals = depth_rect[seg]
    if dvals.size == 0:
        return np.inf

    segmask_depth = np.median(dvals) if use_median else np.mean(dvals) # betweem 0. and 1.
    return segmask_depth

def mask_boundary(seg, radius=1):
    """
    seg : (H, W) bool
    radius : thickness of boundary (1 is usually enough)
    """
    struct = np.ones((2*radius+1, 2*radius+1), dtype=bool)
    eroded = binary_erosion(seg, structure=struct, border_value=0)
    boundary = seg & ~eroded   # pixels that disappear under erosion
    return boundary

def depth_edge_gradient_score(
    seg,
    depth,
    edges_canny,
    max_edge_dist=0,
    step_along_normal=1.0,
    min_pairs=30
):
    """
    Measure how strongly depth increases when moving from INSIDE the mask
    to OUTSIDE the mask, but ONLY ALONG CANNY EDGES.

    Parameters
    ----------
    seg : (H, W) bool
        SAM mask.
    depth : (H, W) float32
        Depth map.
    edges_canny : (H, W) uint8 or bool
        Canny edges (nonzero = edge).
    max_edge_dist : int
        If > 0, allow boundary points that are within this distance of an edge
        (via dilation). If 0, only exact edge pixels are used.
    step_along_normal : float
        Step in pixels along the normal for sampling inside/outside depth.
    min_pairs : int
        Minimum number of valid inside/outside pairs to compute a meaningful score.

    Returns
    -------
    result : dict or None
        dict with:
            - score
            - mean_positive_jump
            - frac_positive
            - mean_jump
            - n_pairs
        or None if not enough valid pairs.
    """
    seg = seg.astype(bool)
    H, W = seg.shape
    depth = depth.astype(np.float32)

    # --- 1) Boundary of the mask ---
    boundary = mask_boundary(seg, radius=1)

    # --- 2) Restrict boundary to Canny edges (or near them) ---
    edges_bool = edges_canny.astype(bool)
    if max_edge_dist > 0:
        struct = np.ones((2*max_edge_dist+1, 2*max_edge_dist+1), dtype=bool)
        edges_band = binary_dilation(edges_bool, structure=struct)
    else:
        edges_band = edges_bool

    boundary_on_edges = boundary & edges_band
    if not np.any(boundary_on_edges):
        return None

    ys, xs = np.where(boundary_on_edges)

    # --- 3) Compute mask gradient to estimate inward normal ---
    # seg = 1 inside, 0 outside → gradient points from outside → inside
    seg_float = seg.astype(np.float32)
    gx = cv2.Sobel(seg_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg_float, cv2.CV_32F, 0, 1, ksize=3)

    # Inward normal (toward inside of mask)
    mag = np.sqrt(gx**2 + gy**2) + 1e-6
    nx_in = gx / mag
    ny_in = gy / mag

    # Outward normal (toward outside)
    nx_out = -nx_in
    ny_out = -ny_in

    depth_in_list = []
    depth_out_list = []

    for y, x in zip(ys, xs):
        # inward sample (inside mask)
        dy_in = ny_in[y, x] * step_along_normal
        dx_in = nx_in[y, x] * step_along_normal

        # outward sample (outside mask)
        dy_out = ny_out[y, x] * step_along_normal
        dx_out = nx_out[y, x] * step_along_normal

        y_in = int(round(y + dy_in))
        x_in = int(round(x + dx_in))
        y_out = int(round(y + dy_out))
        x_out = int(round(x + dx_out))

        # Clamp to image bounds
        if (0 <= y_in < H and 0 <= x_in < W and
            0 <= y_out < H and 0 <= x_out < W):
            d_in  = depth[y_in, x_in]
            d_out = depth[y_out, x_out]

            # require finite, positive depth
            if np.isfinite(d_in) and np.isfinite(d_out) and d_in > 0 and d_out > 0:
                depth_in_list.append(d_in)
                depth_out_list.append(d_out)

    depth_in_list  = np.array(depth_in_list, dtype=np.float32)
    depth_out_list = np.array(depth_out_list, dtype=np.float32)

    if len(depth_in_list) < min_pairs:
        return None

    # Signed jump when going inside → outside
    jumps = depth_out_list - depth_in_list  # we want this large and > 0

    mean_jump = float(np.mean(jumps))
    pos_jumps = jumps[jumps > 0]
    mean_pos_jump = float(np.mean(pos_jumps)) if len(pos_jumps) > 0 else 0.0
    frac_pos = float((jumps > 0).sum() / len(jumps))

    # Final score: encourage many positive jumps AND big positive jumps
    score = mean_pos_jump * frac_pos

    return dict(
        score=score,
        mean_positive_jump=mean_pos_jump,
        frac_positive=frac_pos,
        mean_jump=mean_jump,
        n_pairs=len(jumps)
    )

def visualize_sam_masks(
    img,
    sam_masks,
    alpha=0.2,
    draw_bbox=True,
    draw_points=True,
    max_masks=None,
    figsize=(6, 12),
    suptitle="SAM Masks Visualization",
    save_path=None,
):
    """
    Visualize Segment Anything masks on top of an image.

    Parameters
    ----------
    img : np.ndarray or PIL.Image
        Original image. Shape (H, W, 3), uint8 or float in [0,1].
    sam_masks : list of dict
        Output of mask_generator.generate(img). Each dict must contain:
        - 'segmentation' : bool array (H, W)
        - 'bbox'         : [x, y, w, h]
        - 'point_coords' : [[x, y], ...]
    alpha : float
        Transparency for mask overlay.
    draw_bbox : bool
        If True, draw bounding boxes for each mask.
    draw_points : bool
        If True, draw seed points used by SAM.
    max_masks : int or None
        If not None, only visualize the first `max_masks` masks.
    figsize : tuple
        Figure size for matplotlib.
    """
    # --- prepare image as numpy uint8 ---
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype != np.uint8:
        # assume image is in [0,1] float or similar
        img_vis = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    else:
        img_vis = img.copy()

    H, W = img_vis.shape[:2]

    # --- base and overlay canvas ---
    overlay = img_vis.copy().astype(np.float32)

    # we’ll also build an outline-only image (optional)
    outline = np.zeros_like(img_vis, dtype=np.uint8)

    # Limit number of masks if desired
    masks_to_use = sam_masks if max_masks is None else sam_masks[:max_masks]

    for i, m in enumerate(masks_to_use):
        seg = m["segmentation"]  # bool array (H, W)

        # random color for this mask
        color = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.float32)

        # --- fill region: alpha-blend directly on overlay ---
        mask_idx = seg.astype(bool)
        overlay[mask_idx] = (1 - alpha) * overlay[mask_idx] + alpha * color

        # --- optional: draw mask boundaries into outline image ---
        # boundary = seg ^ cv2.erode(seg.astype(np.uint8), None).astype(bool)
        seg_uint8 = seg.astype(np.uint8) * 255
        # find edges by Canny or morphological gradient
        edges = cv2.Canny(seg_uint8, 50, 150)
        outline[edges > 0] = (0, 255, 0)  # green edges

        # --- optional: draw bbox & seed points directly on overlay ---
        if draw_bbox and "bbox" in m:
            x, y, w, h = m["bbox"]
            x2, y2 = x + w, y + h
            cv2.rectangle(
                overlay,
                (int(x), int(y)),
                (int(x2), int(y2)),
                color=(255, 255, 255),
                thickness=1,
            )

        if draw_points and "point_coords" in m:
            for (px, py) in m["point_coords"]:
                cv2.circle(
                    overlay,
                    (int(px), int(py)),
                    radius=3,
                    color=(255, 255, 255),
                    thickness=-1,
                )

    overlay = overlay.astype(np.uint8)

    # --- show original and overlay side-by-side ---
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(suptitle)
    axes[0].imshow(img_vis)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Masks overlay")
    axes[1].axis("off")

    axes[2].imshow(img_vis)
    axes[2].imshow(outline, alpha=0.9)  # just boundaries
    axes[2].set_title("Mask boundaries")
    axes[2].axis("off")

    if save_path is not None:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()

def visualize_sam_masks_light(
    img,
    sam_masks,
    alpha=0.2,
    draw_bbox=True,
    draw_points=True,
    max_masks=None,
    figsize=(6, 12//3),
    suptitle="SAM Masks Visualization",
    save_path=None,
):
    """
    Visualize Segment Anything masks on top of an image.

    Parameters
    ----------
    img : np.ndarray or PIL.Image
        Original image. Shape (H, W, 3), uint8 or float in [0,1].
    sam_masks : list of dict
        Output of mask_generator.generate(img). Each dict must contain:
        - 'segmentation' : bool array (H, W)
        - 'bbox'         : [x, y, w, h]
        - 'point_coords' : [[x, y], ...]
    alpha : float
        Transparency for mask overlay.
    draw_bbox : bool
        If True, draw bounding boxes for each mask.
    draw_points : bool
        If True, draw seed points used by SAM.
    max_masks : int or None
        If not None, only visualize the first `max_masks` masks.
    figsize : tuple
        Figure size for matplotlib.
    """
    # --- prepare image as numpy uint8 ---
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype != np.uint8:
        # assume image is in [0,1] float or similar
        img_vis = (255 * np.clip(img, 0, 1)).astype(np.uint8)
    else:
        img_vis = img.copy()

    H, W = img_vis.shape[:2]

    # --- base and overlay canvas ---
    overlay = img_vis.copy().astype(np.float32)

    # we’ll also build an outline-only image (optional)
    outline = np.zeros_like(img_vis, dtype=np.uint8)

    # Limit number of masks if desired
    masks_to_use = sam_masks if max_masks is None else sam_masks[:max_masks]

    for i, m in enumerate(masks_to_use):
        seg = m["segmentation"]  # bool array (H, W)

        # random color for this mask
        color = np.array([
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ], dtype=np.float32)

        # --- fill region: alpha-blend directly on overlay ---
        mask_idx = seg.astype(bool)
        overlay[mask_idx] = (1 - alpha) * overlay[mask_idx] + alpha * color

        # --- optional: draw mask boundaries into outline image ---
        # boundary = seg ^ cv2.erode(seg.astype(np.uint8), None).astype(bool)
        seg_uint8 = seg.astype(np.uint8) * 255
        # find edges by Canny or morphological gradient
        edges = cv2.Canny(seg_uint8, 50, 150)
        outline[edges > 0] = (0, 255, 0)  # green edges

        # --- optional: draw bbox & seed points directly on overlay ---
        if draw_bbox and "bbox" in m:
            x, y, w, h = m["bbox"]
            x2, y2 = x + w, y + h
            cv2.rectangle(
                overlay,
                (int(x), int(y)),
                (int(x2), int(y2)),
                color=(255, 255, 255),
                thickness=1,
            )

        if draw_points and "point_coords" in m:
            for (px, py) in m["point_coords"]:
                cv2.circle(
                    overlay,
                    (int(px), int(py)),
                    radius=3,
                    color=(255, 255, 255),
                    thickness=-1,
                )

    overlay = overlay.astype(np.uint8)

    # --- show original and overlay side-by-side ---
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(suptitle)

    axes.imshow(overlay)
    axes.set_title("Masks overlay")
    axes.axis("off")

    if save_path is not None:
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()

def filter_sam_masks_by_depth(
    sam_masks,
    depth,
    *,
    score_thresh,
    use_median=False
):
    """
    Filter SAM masks using mean/median depth.

    Parameters
    ----------
    sam_masks : list of dict
        SAM outputs, each containing a 'segmentation' key.
    depth : (H, W) float
        Rectified depth map.
    score_thresh : float
        Threshold on depth score (lower = closer = foreground).
    use_median : bool
        Whether to use median instead of mean depth.

    Returns
    -------
    final_mask : (H, W) bool
        Union of selected foreground masks.
    candidates : list of (score, mask_dict)
        Sorted candidate masks with scores.
    """
    candidates = []

    for m in sam_masks:
        seg = m["segmentation"].astype(bool)
        score = get_mask_score_via_hdistance_to_max_depth(
            seg,
            depth,
            use_median=use_median
        )
        candidates.append((score, m))

    # Sort from closest (foreground) to farthest
    candidates.sort(key=lambda x: x[0])

    selected_masks = [
        m["segmentation"]
        for score, m in candidates
        if score < score_thresh
    ]

    if len(selected_masks) == 0:
        final_mask = np.zeros_like(depth, dtype=bool)
    else:
        final_mask = np.any(np.stack(selected_masks, axis=-1), axis=-1)

    return final_mask, candidates

def filter_sam_masks_by_depth_edges(
    sam_masks,
    depth,
    edges_canny,
    *,
    score_threshold,
    max_edge_dist=1,
    step_along_normal=1.0,
    min_pairs=30,
):
    """
    Filter SAM masks based on depth gradients across mask boundaries,
    evaluated only along image edges.

    Parameters
    ----------
    sam_masks : list of dict
        SAM outputs, each containing a 'segmentation' key.
    depth : (H, W) float
        Depth map used for scoring (e.g., sharpened or raw).
    edges_canny : (H, W) bool or uint8
        Canny edge map.
    score_threshold : float
        Minimum score for a mask to be kept.
    max_edge_dist : int
        Allow boundary points within this pixel distance from edges.
    step_along_normal : float
        Step size (in pixels) along the boundary normal.
    min_pairs : int
        Minimum valid depth pairs required to score a mask.
    plot_results : bool
        Whether to visualize top-scoring masks.
    img : optional image
        Image used for visualization.
    visualize_fn : callable, optional
        Function to visualize SAM masks (e.g., visualize_sam_masks).

    Returns
    -------
    final_mask : (H, W) bool
        Union of selected foreground masks.
    candidates : list of tuples
        (score, mask_dict, score_details), sorted by descending score.
    """
    candidates = []

    for m in sam_masks:
        seg = m["segmentation"].astype(bool)
        s = depth_edge_gradient_score(
            seg,
            depth,
            edges_canny,
            max_edge_dist=max_edge_dist,
            step_along_normal=step_along_normal,
            min_pairs=min_pairs
        )
        if s is not None:
            candidates.append((s["score"], m, s))

    # Sort from strongest foreground evidence to weakest
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Optional visualization

    selected_masks = [
        m["segmentation"]
        for score, m, _ in candidates
        if score > score_threshold
    ]

    if len(selected_masks) == 0:
        final_mask = np.zeros_like(depth, dtype=bool)
    else:
        final_mask = np.any(np.stack(selected_masks, axis=-1), axis=-1)

    return final_mask, candidates

def get_foreground_segmask(config, mask_generator, img, depth_origin, plot_results=False, save_path=None):
    # 0.generate SAM masks
    sam_masks = mask_generator.generate(img)
    if plot_results:
        print("0. Generating SAM masks...")
        visualize_sam_masks(img, sam_masks, alpha=0.5, suptitle="Detected SAM Masks", save_path=save_path/"01_detected_sam_masks.png")


    # I. Get Mask depth on mean depth
    depth, low_freq = remove_low_freq(depth_origin, config=config.ldi.masking.depth_mean_based.remove_depth_low_freq)
    if plot_results:
        print("I. MASK VIA MEAN DEPTH: Removing low-frequency from depth for mean-depth based filtering...")
        visualize_low_freq_removal(depth_origin, low_freq, depth, save_path=save_path/"02_depth_lowfreq_removal_hmax.png")  

    final_mask_mean_depth, candidates = filter_sam_masks_by_depth(
        sam_masks,
        depth,
        score_thresh=config.ldi.masking.depth_mean_based.segmask_scoring.score_threshold,
        use_median=config.ldi.masking.depth_mean_based.segmask_scoring.use_median_depth
    )
    if plot_results:
        print("I. MASK VIA MEAN DEPTH: Computing score and accumularing masks...")
        visualize_sam_masks_light(img, [{"segmentation": final_mask_mean_depth}], alpha=0.8, max_masks=1, suptitle="Selected Mask after Mean-Depth Filtering", save_path=save_path/"03_mean_depth_selected_mask.png")
    
    
    # II. Get  Mask depth on depth edges
    if config.ldi.masking.depth_edges_based.remove_depth_low_freq.apply:
        depth, low_freq = remove_low_freq(depth_origin, config=config.ldi.masking.depth_edges_based.remove_depth_low_freq)
        if plot_results:
            print("II. MASK VIA DEPTH EDGES: Removing low-frequency from depth for depth-edges based filtering...")
            visualize_low_freq_removal(depth_origin, low_freq, depth, save_path=save_path/"04_depth_lowfreq_removal_gaussian.png") 
        depth = minmax_norm(depth, out_min=0.0, out_max=1.0)
    else:
        depth = depth_origin.copy()
    
    edges_canny, edges_sobel, depth_sharpened = get_canny_sobel_edges(
        depth, img, 
        edged_sobel_ksize=config.ldi.masking.depth_edges_based.edges_detection.sobel.ksize,
        canny_low_t=config.ldi.masking.depth_edges_based.edges_detection.canny.low_t, 
        canny_high_t=config.ldi.masking.depth_edges_based.edges_detection.canny.high_t, 
        depth_sharpen_config=config.ldi.masking.depth_edges_based.edges_detection.depth_sharpening
    )
    if plot_results:
        print("II. MASK VIA DEPTH EDGES: Computing Canny and Sobel edges from depth...")
        visualize_canny_sobel_edges(img, depth_origin, depth, depth_sharpened, edges_sobel, edges_canny, save_path=save_path/"05_canny_sobel_edges.png")
    
    final_mask_depth_edges, candidates = filter_sam_masks_by_depth_edges(
        sam_masks,
        depth_sharpened,
        edges_canny,
        score_threshold=config.ldi.masking.depth_edges_based.segmask_scoring.score_threshold,
        max_edge_dist=config.ldi.masking.depth_edges_based.segmask_scoring.max_edge_dist,
        step_along_normal=config.ldi.masking.depth_edges_based.segmask_scoring.step_along_normal,
        min_pairs=config.ldi.masking.depth_edges_based.segmask_scoring.min_pairs,
    )
    if plot_results:
        print("II. MASK VIA DEPTH EDGES: Computing score and accumularing masks...")
        visualize_sam_masks_light(img, [{"segmentation": final_mask_depth_edges}], alpha=0.8, max_masks=1, suptitle="Selected Mask after Edge-Alignment Filtering", save_path=save_path/"06_edge_aligned_selected_mask.png")
    
    # III. Combine both masks
    final_mask = final_mask_mean_depth & final_mask_depth_edges
   
    if plot_results:
        print("III. FINAL MASK: Combining both masks...")
        visualize_sam_masks_light(img, [{"segmentation": final_mask}], alpha=0.8, max_masks=1, suptitle="Final Selected Mask", save_path=save_path/"07_final_selected_mask.png")
    
    return final_mask

# STEP II: Double inpainting with LAMA and FLUX (copyright: LayerPano3D)
def generate_caption(model, processor, raw_image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "you are a powerful image captioner. Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Keep it short and simple.Minimize aesthetic descriptions as much as possible. Beside, Start with The image captures a xxx"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    caption = processor.decode(output[0][2:], skip_special_tokens=True)
    caption = caption[355:]
    caption = caption.replace("The image captures ", "")
    
    return caption

def get_smooth_mask(general_mask, ksize = 50):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    mask_array = cv2.dilate(general_mask.astype(np.uint8), kernel)              #[1024, 2048] uint8 1
    mask_array = (mask_array>0).astype(np.uint8)
    return mask_array

def viz_lama_flux_double_inpainting(
    img,
    mask_smooth,
    pano_lama_pil,
    pano_flux_pil,
    prompt,
    config,
    save_path=None,
):
    aspect = config.ldi.inpainting.flux_inpainting_resolution.width / config.ldi.inpainting.flux_inpainting_resolution.height 
    n_rows, n_cols = 3, 2
    s = 4  # scale factor, adjust as needed
    alpha=0.2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * aspect * s, n_rows * s)
    )
    axes = axes.flatten()
        
    # MAIN TITLE
    fig.suptitle(f"Inpainting Results. \n strength={config.ldi.inpainting.strength}\n Prompt (truncated)='{prompt[:50]} ...'", fontsize=20, y=1.00)

    # Row 0: original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(my_utils.overlay_mask(img, mask_smooth, alpha=alpha))
    axes[1].set_title("Original Image (with mask overlay)")
    axes[1].axis("off")

    # Inpainted image [LAMA]
    axes[2].imshow(my_utils.PIL_to_numpy(pano_lama_pil))
    axes[2].set_title(f"Inpainted Image [LAMA]")
    axes[2].axis("off")
    axes[3].imshow(my_utils.overlay_mask(my_utils.PIL_to_numpy(pano_lama_pil), mask_smooth, alpha=alpha))
    axes[3].set_title(f"Inpainted Image [LAMA] (with mask overlay)")
    axes[3].axis("off")

    # Inpainted image [FLUX]
    axes[4].imshow(my_utils.PIL_to_numpy(pano_flux_pil))
    axes[4].set_title(f"Inpainted Image [FLUX]")
    axes[4].axis("off")
    axes[5].imshow(my_utils.overlay_mask(my_utils.PIL_to_numpy(pano_flux_pil), my_utils.mask_resize(mask_smooth, config.ldi.inpainting.flux_inpainting_resolution.height, config.ldi.inpainting.flux_inpainting_resolution.width), alpha=alpha))
    axes[5].set_title(f"Inpainted Image [FLUX] (with mask overlay)")
    axes[5].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()

def lama_flux_double_inpainting_p1(
        config, 
        spherical_dreamer, 
        llm_model,
        processor, 
        image:np.ndarray[float],
        mask:np.ndarray[bool],

):

    # step 1: lama inpainting on a reduced resolution
    mask_smooth = get_smooth_mask(np.asarray(mask), ksize = config.ldi.inpainting.mask_dilatation_px)
    lama_inpainting_resolution = config.ldi.inpainting.lama_inpainting_resolution
    inpaint_pano_lama = spherical_dreamer.lama_inpaint(
        image=my_utils.numpy_to_PIL(my_utils.opencv_resize(image, lama_inpainting_resolution.height, lama_inpainting_resolution.width, )),
        mask= my_utils.numpy_bool_to_pil_mask(my_utils.mask_resize(mask_smooth, lama_inpainting_resolution.height, lama_inpainting_resolution.width)),
    ).resize((config.width, config.height))
    inpaint_pano_lama = my_utils.PIL_to_numpy(inpaint_pano_lama)
    inpaint_pano_lama = inpaint_pano_lama * mask_smooth[..., None] + (1-mask_smooth)[..., None] * image
    
    inpaint_pano_lama_pil = my_utils.numpy_to_PIL(inpaint_pano_lama)
    mask_smooth_pil = my_utils.numpy_bool_to_pil_mask(mask_smooth)

    # step2 : caption generation
    prompt = generate_caption(llm_model, processor, inpaint_pano_lama_pil)

    viz_kwargs = {
        "img": image,
        "mask_smooth": mask_smooth,
        "pano_lama_pil": inpaint_pano_lama_pil,
        "pano_flux_pil": None,  # Will be filled in part 2
        "prompt": prompt,
        "config": config,
    }

    return prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs

def lama_flux_double_inpainting_p2(
        config, 
        spherical_dreamer, 
        prompt:str,
        mask_smooth_pil:Image.Image,
        inpaint_pano_lama_pil:Image.Image,
        viz_kwargs,
        plot_results:bool=False,
        save_path=None,
    ):
    # step 3: flux inpainting
    inpaint_pano_flux_pil = spherical_dreamer.inpaint_pano(
        prompt=prompt,
        pano_rgb=inpaint_pano_lama_pil,  
        mask=mask_smooth_pil, 
        strength= config.ldi.inpainting.strength,
        height=config.ldi.inpainting.flux_inpainting_resolution.height,
        width=config.ldi.inpainting.flux_inpainting_resolution.width,
    )

    if plot_results:
        viz_kwargs["pano_flux_pil"] = inpaint_pano_flux_pil
        viz_lama_flux_double_inpainting(
            **viz_kwargs,
            save_path=save_path/"08_lama_flux_double_inpainting.png",
        )
        # also save the inpainting mask as numpy and PILLOW image
        np.save(save_path/"08_inpainting_mask.npy", mask_smooth_pil)
        mask_smooth_pil.save(save_path/"08_inpainting_mask.png")

    return inpaint_pano_flux_pil, mask_smooth_pil


# STEP III: Depth Inpainting Pipeline (copyright: Infusion, LayerPano3D)
def pad_equirectangular(depth, pad_width, mask=None, rgb=None):
    """
    Pad an equirectangular depth map horizontally by wrapping columns:
    - Left pad = last `pad_width` columns
    - Right pad = first `pad_width` columns

    Optionally applies the same padding to mask and RGB image.

    Parameters
    ----------
    depth : (H, W) array
        Depth map (float, typically in [0,1]).
    pad_width : int
        How many columns to pad left and right.
    mask : (H, W) array, optional
        Boolean or uint8 mask.
    rgb : (H, W, 3) array, optional
        RGB image.

    Returns
    -------
    depth_padded : (H, W + 2*pad_width)
    mask_padded  : same or None
    rgb_padded   : same or None
    """
    H, W = depth.shape

    # Wrap padding
    left  = depth[:, -pad_width:]
    right = depth[:, :pad_width]
    depth_padded = np.concatenate([left, depth, right], axis=1)

    mask_padded = None
    if mask is not None:
        left_m  = mask[:, -pad_width:]
        right_m = mask[:, :pad_width]
        mask_padded = np.concatenate([left_m, mask, right_m], axis=1)

    rgb_padded = None
    if rgb is not None:
        left_rgb  = rgb[:, -pad_width:, :]
        right_rgb = rgb[:, :pad_width, :]
        rgb_padded = np.concatenate([left_rgb, rgb, right_rgb], axis=1)

    return depth_padded, mask_padded, rgb_padded

def unpad_equirectangular(depth_padded, pad_width, mask_padded=None, rgb_padded=None):
    """
    Remove equirectangular wrap padding added by `pad_equirectangular`.

    Parameters
    ----------
    depth_padded : (H, W + 2*pad_width)
    pad_width : int
        Number of padded columns to remove on each side.
    mask_padded : optional
    rgb_padded : optional

    Returns
    -------
    depth : (H, original_W)
    mask  : same or None
    rgb   : same or None
    """
    # Remove left pad_width and right pad_width
    depth = depth_padded[:, pad_width:-pad_width]

    mask = None
    if mask_padded is not None:
        mask = mask_padded[:, pad_width:-pad_width]

    rgb = None
    if rgb_padded is not None:
        rgb = rgb_padded[:, pad_width:-pad_width, :]

    return depth, mask, rgb

def load_depth_inpaint_pipeline(
    model_path="checkpoints/Infusion",
    device="cuda",
    dtype=torch.float16,
):
    """
    Load the Infusion depth-inpainting pipeline (same as in your Gen_traindata).

    Returns
    -------
    pipe_dp : DepthEstimationInpaintPipeline
    """
    seed = int(time.time())
    seed_all(seed)

    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        model_path, subfolder="scheduler", torch_dtype=dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", torch_dtype=dtype
    )

    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        in_channels=13,
        sample_size=96,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )

    pipe_dp = DepthEstimationInpaintPipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    try:
        pipe_dp.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    pipe_dp = pipe_dp.to(device)
    return pipe_dp

def inpaint_bg_depth_infusion(
    image,
    depth,
    image_bg,
    bg_mask,
    pipe_dp,
    rescale_to_min_depth=True,
    pad_width=None,
    plot_results=False,
    save_path=None,
):
    """
    Inpaint the depth map in regions where background from `image_bg`
    replaces foreground in `image`.

    Parameters
    ----------
    image : (H, W, 3) uint8
        Original RGB image with foreground objects.
        (Only used for sanity; not passed to the model here.)
    depth : (H, W) float32 or float64
        Depth map corresponding to `image`. Assumed in [0, 1] like your
        Pano_depth_estimation output.
    image_bg : (H, W, 3) uint8
        Composited image where background content has been filled in.
        In pixels where `bg_mask` is True, background objects from `image_bg`
        cover the original foreground from `image`.
    bg_mask : (H, W) bool or uint8
        True where the background was inserted and depth should be inpainted.
        Outside this mask, `image` and `image_bg` are assumed equal.
    pipe_dp : DepthEstimationInpaintPipeline
        The Infusion depth-inpainting pipeline returned by
        `load_depth_inpaint_pipeline`.
    rescale_to_min_depth : bool, default True
        If True, mimic your original code:
            depth_pred = min_depth + depth_pred * (1 - min_depth)
        where min_depth = depth.min().
        This keeps predicted depths >= min_depth so they stay compatible
        with existing layers.

    Returns
    -------
    depth_inpainted : (H, W) float32
        Depth map where background regions (bg_mask==True) have been
        inpainted for `image_bg`.
    """
    # --- 0) Basic checks / casting ---
    image = np.asarray(image)
    image_bg = np.asarray(image_bg)
    depth = np.asarray(depth, dtype=np.float32)
    bg_mask = np.asarray(bg_mask)

    assert image.shape[:2] == depth.shape[:2], "image and depth must match spatially."
    assert image_bg.shape[:2] == depth.shape[:2], "image_bg and depth must match spatially."

    H, W = depth.shape

    if pad_width:
        depth, bg_mask, image_bg = pad_equirectangular(
            depth,
            pad_width=pad_width,
            mask=bg_mask,
            rgb=image_bg,
        )

    # Ensure mask is 0/1 float in [0, 1]
    if bg_mask.dtype == bool:
        mask = bg_mask.astype(np.float32)
    else:
        mask = bg_mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = np.clip(mask, 0.0, 1.0)

    # If depth range isn't [0,1], you may want to normalize here
    # based on your actual pipeline. For now we assume [0,1] as in your code.
    # Optionally keep min depth to re-scale predictions later
    min_depth = float(np.min(depth)) if rescale_to_min_depth else 0.0

    # --- 1) Call the Infusion depth-inpainting model ---
    #   The original code uses:
    #       pipe_out = self.pipe_dp(input_image=pano_rgb,
    #                               depth_numpy=pano_depth_base_i,
    #                               mask = mask)
    #   and then:
    #       depth_pred = pipe_out.depth_np
    pipe_out = pipe_dp(
        input_image=image_bg,   # we want depth for the background-composited image
        depth_numpy=depth,      # original depth as guidance
        mask=mask,              # where to inpaint depth
    )

    depth_pred = np.asarray(pipe_out.depth_np, dtype=np.float32)  # [0,1] predicted

    # --- 2) Optional re-scaling as in your generate_traindata ---
    if rescale_to_min_depth:
        # depth_pred in [0,1] → [min_depth, 1]
        depth_pred = min_depth + depth_pred * (1.0 - min_depth)

    # --- 3) Merge prediction into original depth ---
    depth_inpainted = depth_pred.copy()
    # bg_mask_bool = mask > 0.5
    # depth_inpainted[~bg_mask_bool] = depth[~bg_mask_bool]

    if pad_width:
        depth_inpainted, bg_mask, image_bg = unpad_equirectangular(
            depth_inpainted,
            pad_width=pad_width,
            mask_padded=bg_mask,
            rgb_padded=image_bg,
        )
        depth, _, _ = unpad_equirectangular(
            depth,
            pad_width=pad_width
            )

    if plot_results:
        visualize_bg_depth_inpainting(
            image=image,
            depth=depth,
            image_bg=image_bg,
            bg_mask=bg_mask,
            depth_inpainted=depth_inpainted,
            suptitle=f"Background depth inpainting",
            save_path=save_path,
        )

    return depth_inpainted

def interpolate_depth_nearest(
        depth,
        bg_mask,
        pad_width=15,
    ):
    """
    Inpaint depth in background regions using nearest-neighbor interpolation.

    Parameters
    ----------
    depth : (H, W) float32/float64
        Depth map corresponding to `image`.
    bg_mask : (H, W) bool or 0/1
        True where background from `image_bg` replaces foreground in `image`,
        i.e. where depth should be inpainted.

    Returns
    -------
    depth_inpainted : (H, W) float32
        Depth map where pixels under bg_mask were filled by nearest-neighbor
        interpolation using surrounding valid depths.
    """
    depth = np.asarray(depth, dtype=np.float32)
    bg_mask = np.asarray(bg_mask).astype(bool)

    # --- 1) Pad depth for better interpolation near the horizontal seam ---
    depth_padded, mask_padded, _ = pad_equirectangular(depth, pad_width=pad_width, mask=bg_mask)
    depth_masked = depth_padded.copy()

    # Mark inpainting region
    depth_masked[mask_padded] = np.nan

    invalid = np.isnan(depth_masked)
    if np.all(invalid):
        raise ValueError("All padded depth is NaN; cannot interpolate (nearest).")

    # --- 2) Nearest-neighbor interpolation ---
    indices = ndimage.distance_transform_edt(
        invalid,
        return_distances=False,
        return_indices=True
    )
    depth_masked[invalid] = depth_masked[tuple(indices[:, invalid])]

    depth_filled_padded = depth_masked

    # --- 3) Unpad → back to original shape ---
    depth_filled, _, _ = unpad_equirectangular(depth_filled_padded, pad_width=pad_width)

    return depth_filled

def interpolate_depth_bilinear_plus_nn(
        depth,
        bg_mask,
        pad_width=15,
    ):
    """
    Inpaint depth in background regions using bilinear interpolation
    (griddata) with a nearest-neighbor fallback for remaining holes.

    Parameters
    ----------
    depth : (H, W) float32/float64
        Depth map corresponding to `image`.
    bg_mask : (H, W) bool or 0/1
        True where background from `image_bg` replaces foreground in `image`,
        i.e. where depth should be inpainted.

    Returns
    -------
    depth_inpainted : (H, W) float32
        Depth map where pixels under bg_mask were filled by bilinear interpolation
        plus nearest-neighbor fallback.
    """
    depth = np.asarray(depth, dtype=np.float32)
    bg_mask = np.asarray(bg_mask).astype(bool)

    # --- 1) Pad depth and mask ---
    depth_padded, mask_padded, _ = pad_equirectangular(depth, pad_width=pad_width, mask=bg_mask)

    # Mask out unknown depths
    depth_lin = depth_padded.copy()
    depth_lin[mask_padded] = np.nan
    H, W = depth_lin.shape

    # Grid
    yy, xx = np.indices((H, W))

    # Valid points
    valid = ~np.isnan(depth_lin)
    if not np.any(valid):
        raise ValueError("No valid depth to interpolate (bilinear+nn).")

    points = np.stack([xx[valid], yy[valid]], axis=-1)
    values = depth_lin[valid]

    # --- 2) Bilinear interpolation ---
    depth_interp = griddata(
        points,
        values,
        (xx, yy),
        method="linear"
    )

    # --- 3) Nearest-neighbor fallback ---
    depth_interp_filled = depth_interp.copy()
    nan_mask = np.isnan(depth_interp_filled)

    if np.any(nan_mask):
        tmp = depth_interp_filled.copy()
        tmp[nan_mask] = np.nan

        invalid2 = np.isnan(tmp)
        indices2 = ndimage.distance_transform_edt(
            invalid2,
            return_distances=False,
            return_indices=True,
        )
        depth_interp_filled[nan_mask] = tmp[tuple(indices2[:, nan_mask])]

    depth_filled_padded = depth_interp_filled.astype(np.float32)

    # --- 4) Unpad to original shape ---
    depth_filled, _, _ = unpad_equirectangular(depth_filled_padded, pad_width=pad_width)

    return depth_filled

def visualize_bg_depth_inpainting(
    image,
    depth,
    image_bg,
    bg_mask,
    depth_inpainted,
    suptitle="Background depth inpainting",
    save_path=None,
):
    """
    Visualize original vs background-composited image, mask, and depths.

    Parameters
    ----------
    image          : PIL.Image or np.ndarray (H, W, 3), uint8
    depth   : np.ndarray (H, W), float
    image_bg       : PIL.Image or np.ndarray (H, W, 3), uint8
    bg_mask        : np.ndarray (H, W), bool or 0/1
    depth_inpainted: np.ndarray (H, W), float
    save_path      : str or None
        If given, save the figure to this path.
    suptitle       : str
        Title of the whole figure.
    """
    # --- Normalize inputs ---
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = np.asarray(image)

    if isinstance(image_bg, Image.Image):
        img_bg_np = np.array(image_bg)
    else:
        img_bg_np = np.asarray(image_bg)

    mask_bool = bg_mask.astype(bool)

    depth = np.asarray(depth, dtype=np.float32)
    depth_inpainted = np.asarray(depth_inpainted, dtype=np.float32)

    # Depth diff (only care inside mask for visualization)
    depth_diff = depth_inpainted - depth
    depth_diff_masked = depth_diff.copy()
    depth_diff_masked[~mask_bool] = 0.0

    # Common vmin/vmax for original & inpainted depth
    valid_orig = np.isfinite(depth)
    valid_inp = np.isfinite(depth_inpainted)
    valid = valid_orig & valid_inp

    if np.any(valid):
        vmin = float(min(depth[valid].min(), depth_inpainted[valid].min()))
        vmax = float(max(depth[valid].max(), depth_inpainted[valid].max()))
    else:
        vmin, vmax = 0.0, 1.0

    # For difference, use symmetric limits
    diff_valid = np.isfinite(depth_diff_masked)
    if np.any(diff_valid):
        max_abs = float(np.abs(depth_diff_masked[diff_valid]).max())
        vmin_diff, vmax_diff = -max_abs, max_abs
    else:
        vmin_diff, vmax_diff = -1.0, 1.0

    # --- Build figure ---
    # Pano is 2:1, 3 columns x 2 rows → use wider figure
    fig, axes = plt.subplots(2, 3, figsize=(3 * 5, 2 * 5))
    axes = axes.reshape(2, 3)

    # Row 0: images
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_bg_np)
    axes[0, 1].set_title("Background composite (image_bg)")
    axes[0, 1].axis("off")

    overlaid = my_utils.overlay_mask(img_bg_np, mask_bool, alpha=0.5)
    axes[0, 2].imshow(overlaid)
    axes[0, 2].set_title("image_bg with background mask (blue)")
    axes[0, 2].axis("off")

    # Row 1: depths
    im0 = axes[1, 0].imshow(depth, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Original depth")
    axes[1, 0].axis("off")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(depth_inpainted, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Inpainted depth (background filled)")
    axes[1, 1].axis("off")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 2].imshow(depth_diff_masked, cmap="seismic", vmin=vmin_diff, vmax=vmax_diff)
    axes[1, 2].set_title("Depth difference (inpainted - original)\n(only shown in bg mask)")
    axes[1, 2].axis("off")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    plt.show()

def post_process_inpainted_depth(
    depth_bg,
    depth_fg,
    bg_mask,
    eps=1.1e-3,
    plot=False,
    save_path=None,
):
    """
    Post-process an inpainted background depth map so that:
      - background is always farther (larger depth) than foreground
      - corrections are applied smoothly (via bilinear + NN interpolation)
        instead of setting invalid points to NaN.

    Parameters
    ----------
    depth_bg : (H, W) float
        Inpainted background depth (to be corrected).
    depth_fg : (H, W) float
        Original foreground depth.
    bg_mask : (H, W) bool
        True where background is present / relevant.
    eps : float
        Small safety margin to ensure depth_bg > depth_fg + eps.

    Returns
    -------
    depth_bg_corrected : (H, W) float
        Corrected background depth.
    """
    depth_bg = np.asarray(depth_bg, dtype=np.float32)
    depth_fg = np.asarray(depth_fg, dtype=np.float32)
    bg_mask  = np.asarray(bg_mask,  dtype=bool)

    # 1) Pixels where background is NOT behind foreground and within bg_mask
    wrong = bg_mask & (depth_bg <= depth_fg)

    # If nothing is wrong, just return a copy
    if not np.any(wrong):
        return depth_bg.copy()

    # 2) Raw correction: how much we need to push depth_bg back
    #    so that depth_bg_new = depth_bg + correction >= depth_fg + eps
    correction_raw = np.zeros_like(depth_bg, dtype=np.float32)
    correction_raw[wrong] = (depth_fg[wrong] - depth_bg[wrong]) + eps

    # 3) Build a correction map with NaNs outside the "wrong" region
    #    -> we will extend / smooth these corrections by interpolation
    correction_masked = np.full_like(depth_bg, np.nan, dtype=np.float32)
    correction_masked[wrong] = correction_raw[wrong]

    # 4) Use your bilinear + NN interpolation to obtain a smooth correction field
    #    Note: interpolate_depth_bilinear_plus_nn fills NaNs in `depth`
    #    based on valid neighbors; we pass bg_mask to keep the interface consistent.
    correction_smooth = interpolate_depth_bilinear_plus_nn(
        depth=correction_masked,
        bg_mask=bg_mask & ~wrong,   # region of interest; function may or may not use it internally
    )

    # 5) Apply the smooth correction ONLY on the wrong pixels
    depth_bg_corrected = depth_bg.copy()
    depth_bg_corrected[wrong] += correction_smooth[wrong]

    if plot:
        # Shared vmin/vmax for raw & smooth corrections (excluding NaNs)
        valid_corr = ~np.isnan(correction_raw)
        if np.any(valid_corr):
            vmin = float(np.nanmin(correction_raw))
            vmax = float(np.nanmax(correction_raw))
        else:
            vmin, vmax = 0.0, 1.0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1) Raw correction (only defined on 'wrong' pixels)
        im0 = axes[0].imshow(correction_raw, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Raw correction (wrong pixels)")
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 2) Smooth correction field
        im1 = axes[1].imshow(correction_smooth, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title("Smoothed correction (bilinear+NN)")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 3) Final corrected depth
        im2 = axes[2].imshow(depth_bg_corrected, cmap="Spectral_r")
        axes[2].set_title("Corrected background depth")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path / "09_bring_bg_behind_fg.png", dpi=150)

        plt.show()

    return depth_bg_corrected

def prepare_inpainting(config, img, depth_origin, inpaint_mask_pil):
    he = config.ldi.inpainting.flux_inpainting_resolution.height
    wi = config.ldi.inpainting.flux_inpainting_resolution.width
    img_pil = my_utils.numpy_to_PIL(my_utils.opencv_resize(img, he, wi, mode="bilinear"))
    depth_origin = my_utils.opencv_resize(depth_origin, he, wi, mode="bilinear") # FLAG: depth resize
    inpaint_mask_pil_ = inpaint_mask_pil.resize((wi, he), resample=Image.NEAREST)
    inpaint_mask_bool_ = my_utils.pil_mask_to_numpy_bool(inpaint_mask_pil_)

    if config.ldi.depth_inpainting.additionnal_mask_dilation_px > 0:
        inpaint_mask_bool_ = my_utils.dilate_mask(
            inpaint_mask_bool_,
            pixels=config.ldi.depth_inpainting.additionnal_mask_dilation_px
        )

    if config.ldi.depth_inpainting.fill_holes:
        inpaint_mask_bool_ = my_utils.fill_mask(inpaint_mask_bool_)

    inpaint_mask_pil_ = my_utils.numpy_bool_to_pil_mask(inpaint_mask_bool_)

    return img_pil, depth_origin, inpaint_mask_pil_, inpaint_mask_bool_


# INSTANCIATIONS

def instanciate_sam(config):
    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth").to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **config.ldi.masking.segmask_detection
    )
    return sam, mask_generator

def instanciate_llm_and_processor():
    model_id = "llava-hf/llava-1.5-7b-hf"
    llm_model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    return llm_model, processor

def instanciate_pipe_dp():
    pipe_dp = load_depth_inpaint_pipeline(
        model_path="checkpoints/Infusion",
        device="cuda",
        dtype=torch.float16,
    )
    return pipe_dp

def visualize_depth_inpainting(
    # RGB / mask (optional)
    img_pil=None,
    inpaint_pano_pil=None,
    inpaint_mask_pil=None,

    # Depths (all optional)
    depth_origin=None,
    depth_origin_pp=None,

    depth_inpainted_hblending=None,
    depth_inpainted_hblending_pp=None,

    depth_inpainted_hminprior=None,          # NEW
    depth_inpainted_hminprior_pp=None,       # NEW

    depth_inpainted_infusion=None,
    depth_inpainted_infusion_pp=None,

    depth_inpainted_nn=None,
    depth_inpainted_nn_pp=None,

    depth_inpainted_bilinear_nn=None,
    depth_inpainted_bilinear_nn_pp=None,

    suptitle="Depth Inpainting Results",
    save_path=None,
):
    """
    Flexible visualization:
    - Only plots what is provided (all args optional).
    - Automatically adapts number of rows.
    - 3 columns per row:
        col 0: raw depth (or RGB)
        col 1: post-processed depth (or same RGB)
        col 2: blended view (depth_origin with depth_pp inside mask), if possible.
              Otherwise falls back to "pp + mask overlay" when possible.

    Notes
    -----
    - mask is assumed boolean (converted from inpaint_mask_pil if provided).
    - depths assumed float in [0,1], but normalization is global over provided depths.
    """

    # -----------------------------
    # 0) Helpers
    # -----------------------------
    def _as_rgb01(x):
        """PIL or np -> float RGB in [0,1], shape (H,W,3)."""
        if x is None:
            return None
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def _as_depth(x):
        """Depth -> float32 (H,W)."""
        if x is None:
            return None
        d = np.asarray(x, dtype=np.float32)
        if d.ndim != 2:
            raise ValueError(f"Depth must be (H,W), got {d.shape}")
        return d

    def _overlay_mask_rgb(rgb01, mask, alpha=0.35):
        """Blue overlay on rgb01 where mask==True."""
        if rgb01 is None or mask is None:
            return rgb01
        out = rgb01.copy()
        blue = np.zeros_like(out)
        blue[..., 2] = 1.0
        m = mask.astype(bool)
        out[m] = (1 - alpha) * out[m] + alpha * blue[m]
        return np.clip(out, 0.0, 1.0)

    def _blend_depth(depth_fg, depth_bg, mask):
        """Return depth_fg except on mask where it's depth_bg."""
        blended = depth_fg.copy()
        blended[mask] = depth_bg[mask]
        return blended

    def _off(ax):
        ax.axis("off")

    # -----------------------------
    # 1) Convert inputs
    # -----------------------------
    img_rgb     = _as_rgb01(img_pil)
    inpaint_rgb = _as_rgb01(inpaint_pano_pil)

    mask = None
    if inpaint_mask_pil is not None:
        if isinstance(inpaint_mask_pil, Image.Image):
            mask = my_utils.pil_mask_to_numpy_bool(inpaint_mask_pil)
        mask = np.asarray(inpaint_mask_pil, dtype=bool)

    # Depth conversion
    d0    = _as_depth(depth_origin)
    d0_pp = _as_depth(depth_origin_pp)

    d_hb    = _as_depth(depth_inpainted_hblending)
    d_hb_pp = _as_depth(depth_inpainted_hblending_pp)

    d_hmin    = _as_depth(depth_inpainted_hminprior)     # NEW
    d_hmin_pp = _as_depth(depth_inpainted_hminprior_pp)  # NEW

    d_inf    = _as_depth(depth_inpainted_infusion)
    d_inf_pp = _as_depth(depth_inpainted_infusion_pp)

    d_nn    = _as_depth(depth_inpainted_nn)
    d_nn_pp = _as_depth(depth_inpainted_nn_pp)

    d_bl    = _as_depth(depth_inpainted_bilinear_nn)
    d_bl_pp = _as_depth(depth_inpainted_bilinear_nn_pp)

    # Infer (H,W) from anything available (needed for consistency checks)
    hw = None
    for candidate in [
        d0, d0_pp,
        d_hb, d_hb_pp,
        d_hmin, d_hmin_pp,   # NEW
        d_inf, d_inf_pp,
        d_nn, d_nn_pp,
        d_bl, d_bl_pp
    ]:
        if candidate is not None:
            hw = candidate.shape
            break

    if hw is not None:
        # Ensure all provided depths share same shape
        for name, candidate in [
            ("depth_origin", d0), ("depth_origin_pp", d0_pp),
            ("depth_hblending", d_hb), ("depth_hblending_pp", d_hb_pp),
            ("depth_hminprior", d_hmin), ("depth_hminprior_pp", d_hmin_pp),  # NEW
            ("depth_infusion", d_inf), ("depth_infusion_pp", d_inf_pp),
            ("depth_nn", d_nn), ("depth_nn_pp", d_nn_pp),
            ("depth_bilinear_nn", d_bl), ("depth_bilinear_nn_pp", d_bl_pp),
        ]:
            if candidate is not None and candidate.shape != hw:
                raise ValueError(f"{name} has shape {candidate.shape} but expected {hw}")

        if mask is not None and mask.shape != hw:
            raise ValueError(f"mask has shape {mask.shape} but expected {hw}")

    # -----------------------------
    # 2) Build rows dynamically
    # -----------------------------
    rows = []

    # RGB rows if available
    if img_rgb is not None:
        rows.append(("Original RGB", img_rgb, img_rgb, _overlay_mask_rgb(img_rgb, mask), "rgb"))
    if inpaint_rgb is not None:
        rows.append(("Inpainted RGB", inpaint_rgb, inpaint_rgb, _overlay_mask_rgb(inpaint_rgb, mask), "rgb"))

    depth_pairs = [
        ("Depth origin", d0, d0_pp),
        ("Depth (H-blending)", d_hb, d_hb_pp),
        ("Depth (H-min prior)", d_hmin, d_hmin_pp),  # NEW
        ("Depth (Infusion)", d_inf, d_inf_pp),
        ("Depth (Nearest)", d_nn, d_nn_pp),
        ("Depth (Bilinear+NN)", d_bl, d_bl_pp),
    ]

    for title, raw, pp in depth_pairs:
        if raw is None and pp is None:
            continue

        col0 = raw if raw is not None else pp
        col1 = pp if pp is not None else raw

        if (d0 is not None) and (pp is not None) and (mask is not None):
            col2 = _blend_depth(d0, pp, mask)
        elif (mask is not None):
            col2 = col1  # overlay later
        else:
            col2 = col1

        rows.append((title, col0, col1, col2, "depth"))

    if len(rows) == 0:
        raise ValueError("Nothing to plot: provide at least one RGB or depth argument.")

    # -----------------------------
    # 3) Shared depth colormap + normalization
    # -----------------------------
    depth_for_norm = []
    for _, c0, c1, c2, kind in rows:
        if kind == "depth":
            depth_for_norm.extend([c0, c1, c2])

    if len(depth_for_norm) > 0:
        all_depths = np.stack([d for d in depth_for_norm if d is not None], axis=0)
        vmin = float(np.nanmin(all_depths))
        vmax = float(np.nanmax(all_depths))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("Spectral_r")

        def depth_to_rgb(depth):
            depth_norm = norm(depth)
            return cmap(depth_norm)[..., :3]
    else:
        norm, cmap = None, None
        depth_to_rgb = None

    # -----------------------------
    # 4) Plot
    # -----------------------------
    nrows = len(rows)
    ncols = 3
    fig_w = 18
    fig_h = max(2.5 * nrows, 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    col_titles = ["Raw", "Post-processed", "PP + mask (or blended)"]
    for j in range(ncols):
        axes[0, j].set_title(col_titles[j], fontsize=12)

    for i, (title, c0, c1, c2, kind) in enumerate(rows):
        axes[i, 0].text(
            0.01, 1.02, title, transform=axes[i, 0].transAxes,
            fontsize=12, fontweight="bold", va="bottom"
        )

        if kind == "rgb":
            axes[i, 0].imshow(c0)
            axes[i, 1].imshow(c1)
            axes[i, 2].imshow(c2 if c2 is not None else c1)
            _off(axes[i, 0]); _off(axes[i, 1]); _off(axes[i, 2])
        else:
            axes[i, 0].imshow(c0, cmap=cmap, norm=norm)
            axes[i, 1].imshow(c1, cmap=cmap, norm=norm)

            if (mask is not None) and (c2 is c1):
                rgb = depth_to_rgb(c2)
                axes[i, 2].imshow(_overlay_mask_rgb(rgb, mask))
            else:
                axes[i, 2].imshow(c2, cmap=cmap, norm=norm)

            _off(axes[i, 0]); _off(axes[i, 1]); _off(axes[i, 2])

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def visualize_depth_inpainting_(
    img_pil,
    inpaint_pano_pil,
    inpaint_mask_pil,
    depth_origin,
    depth_origin_pp,
    depth_inpainted_hblending,
    depth_inpainted_hblending_pp,
    depth_inpainted_infusion,
    depth_inpainted_infusion_pp,
    depth_inpainted_nn,
    depth_inpainted_nn_pp,
    depth_inpainted_bilinear_nn,
    depth_inpainted_bilinear_nn_pp,
    suptitle="Depth Inpainting Results",
    save_path=None,
):
    """
    Create a figure with 3 columns:

      col 0: raw depth (or RGB)
      col 1: post-processed depth (or same RGB)
      col 2: post-processed view with mask overlay

    Rows:
      0: original RGB
      1: inpainted RGB
      2: original depth
      3: Infusion inpainted depth
      4: NN inpainted depth
      5: Bilinear+NN inpainted depth

    Notes
    -----
    - mask is boolean
    - depth_* and depth_*_pp are assumed in [0, 1] (float)
    - All depth images share a global colormap / normalization.
    """

    # --- 1. Convert inputs to numpy ---
    img_rgb     = np.array(img_pil) / 255.0              # (H, W, 3) float in [0,1]
    inpaint_rgb = np.array(inpaint_pano_pil) / 255.0     # (H, W, 3) float in [0,1]
    mask        = my_utils.pil_mask_to_numpy_bool(inpaint_mask_pil)  # (H, W) bool

    # Raw depths
    d0 = np.asarray(depth_origin, dtype=np.float32)
    d1 = np.asarray(depth_inpainted_hblending, dtype=np.float32)
    d2 = np.asarray(depth_inpainted_infusion, dtype=np.float32)
    d3 = np.asarray(depth_inpainted_nn, dtype=np.float32)
    d4 = np.asarray(depth_inpainted_bilinear_nn, dtype=np.float32)

    # Post-processed depths
    d0_pp = np.asarray(depth_origin_pp, dtype=np.float32)
    d1_pp = np.asarray(depth_inpainted_hblending_pp, dtype=np.float32)
    d2_pp = np.asarray(depth_inpainted_infusion_pp, dtype=np.float32)
    d3_pp = np.asarray(depth_inpainted_nn_pp, dtype=np.float32)
    d4_pp = np.asarray(depth_inpainted_bilinear_nn_pp, dtype=np.float32)

    # Ensure shapes are compatible
    assert d0.shape == d1.shape == d2.shape == d3.shape == d0_pp.shape == d1_pp.shape == d2_pp.shape == d3_pp.shape, \
        "All depth maps (raw & post-processed) must share the same shape."
    H, W = d0.shape

    # --- 2. Shared colormap & normalization across ALL depth maps ---
    all_depths = np.stack([d0, d1, d2, d3, d0_pp, d1_pp, d2_pp, d3_pp], axis=0)
    vmin = float(np.nanmin(all_depths))
    vmax = float(np.nanmax(all_depths))

    # Safety: if depths are constant or NaN, fallback
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Spectral_r")

    def depth_to_rgb(depth):
        """Map depth (H, W) -> RGB (H, W, 3) using the shared colormap."""
        depth_norm = norm(depth)               # (H, W) in [0,1]
        depth_rgb  = cmap(depth_norm)[..., :3] # (H, W, 3) float in [0,1]
        return depth_rgb

    depth0_rgb    = depth_to_rgb(d0)
    depth1_rgb    = depth_to_rgb(d1)
    depth2_rgb    = depth_to_rgb(d2)
    depth3_rgb    = depth_to_rgb(d3)
    depth0_pp_rgb = depth_to_rgb(d0_pp)
    depth1_pp_rgb = depth_to_rgb(d1_pp)
    depth2_pp_rgb = depth_to_rgb(d2_pp)
    depth3_pp_rgb = depth_to_rgb(d3_pp)

    # --- 3. Figure layout ---
    rows = 7
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5))
    axes = np.atleast_2d(axes)

    # Helper for hiding axis
    def off(ax):
        ax.axis("off")

    # --- 4. RGB rows ---

    # Row 0: original RGB
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original RGB")
    off(axes[0, 0])

    axes[0, 1].imshow(img_rgb)
    axes[0, 1].set_title("Original RGB (copy)")
    off(axes[0, 1])

    axes[0, 2].imshow(my_utils.overlay_mask(img_rgb, mask, alpha=0.5))
    axes[0, 2].set_title("Original RGB + mask")
    off(axes[0, 2])

    # Row 1: inpainted RGB
    axes[1, 0].imshow(inpaint_rgb)
    axes[1, 0].set_title("Inpainted RGB")
    off(axes[1, 0])

    axes[1, 1].imshow(inpaint_rgb)
    axes[1, 1].set_title("Inpainted RGB (copy)")
    off(axes[1, 1])

    axes[1, 2].imshow(my_utils.overlay_mask(inpaint_rgb, mask, alpha=0.5))
    axes[1, 2].set_title("Inpainted RGB + mask")
    off(axes[1, 2])

    # --- 5. Depth rows (all share same vmin/vmax & colormap) ---

    def show_depth(ax, depth, title):
        im = ax.imshow(depth, cmap=cmap, norm=norm)
        ax.set_title(title)
        off(ax)
        return im

    def blend_depth(depth_fg, depth_bg, mask):
        """Blend two depth maps using a mask."""
        blended = depth_fg.copy()
        blended[mask] = depth_bg[mask]
        return blended

    # Row 2: original depth
    show_depth(axes[2, 0], d0, "Depth origin (raw)")
    show_depth(axes[2, 1], d0_pp, "Depth origin (post-processed)")
    # axes[2, 2].imshow(my_utils.overlay_mask(depth0_pp_rgb, mask, alpha=0.5))
    # axes[2, 2].set_title("Depth origin (pp) + mask")
    off(axes[2, 2])

    # Row 3 : H-blending depth
    show_depth(axes[3, 0], d1, "Depth (H-blending, raw)")
    show_depth(axes[3, 1], d1_pp, "Depth (H-blending, pp)")
    show_depth(axes[3, 2], blend_depth(d0, d1_pp, mask), "Depth (H-blending, pp) + blended")

    # Row 4: Infusion depth
    show_depth(axes[4, 0], d2, "Depth (Infusion, raw)")
    show_depth(axes[4, 1], d2_pp, "Depth (Infusion, pp)")
    show_depth(axes[4, 2], blend_depth(d0, d2_pp, mask), "Depth (Infusion, pp) + blended")
    # axes[4, 2].imshow(my_utils.overlay_mask(depth1_pp_rgb, mask, alpha=0.5))
    # axes[4, 2].set_title("Depth (Infusion, pp) + mask")
    off(axes[4, 2])

    # Row 5: Nearest-neighbor depth
    show_depth(axes[5, 0], d3, "Depth (Nearest, raw)")
    show_depth(axes[5, 1], d3_pp, "Depth (Nearest, pp)")
    show_depth(axes[5, 2], blend_depth(d0, d3_pp, mask), "Depth (Nearest, pp) + blended")
    # axes[5, 2].imshow(my_utils.overlay_mask(depth2_pp_rgb, mask, alpha=0.5))
    # axes[5, 2].set_title("Depth (Nearest, pp) + mask")
    off(axes[5, 2])

    # Row 6: Bilinear+NN depth
    show_depth(axes[6, 0], d4, "Depth (Bilinear+NN, raw)")
    show_depth(axes[6, 1], d4_pp, "Depth (Bilinear+NN, pp)")
    show_depth(axes[6, 2], blend_depth(d0, d4_pp, mask), "Depth (Bilinear+NN, pp) + blended")
    # axes[6, 2].imshow(my_utils.overlay_mask(depth3_pp_rgb, mask, alpha=0.5))
    # axes[6, 2].set_title("Depth (Bilinear+NN, pp) + mask")
    off(axes[6, 2])

    fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    config, img_name = my_utils.fetch_config_via_parser(
        debug=True, 
        debug_parser_override=["--config", "Antoine/ldi.yaml"],
        return_img_name=True
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    img_name = "FD0"
    data_dir= "/home/a.schnepf/phd/SphericalDreamer/OUTPUTS"
    depth_path = f"{data_dir}/gen_depths_bckp/{img_name}.npy"
    image_path = f"{data_dir}/gen_images_bckp/{img_name}.png"  # or .jpg
    depth_origin = my_utils.depth_resize(np.load(depth_path), height, width)
    img = my_utils.opencv_resize( my_utils.PIL_to_numpy(Image.open(image_path)), height, width, mode="bilinear")
    plot_results = False
    shortcut_I = False
    shortcut_II = False
    shortcut_III = True
    savedir = Path(f"OUTPUTS/ldi_tests/{img_name}")
    os.makedirs(savedir, exist_ok=True)

    if not shortcut_I:
        # -----------------------------------------
        # I. COMPUTE SEGMAP FOR FORGROUND OBJECTS
        # -----------------------------------------
        sam, mask_generator = instanciate_sam(config)
        final_mask = get_foreground_segmask(
            config,
            mask_generator, 
            img,
            depth_origin,
            plot_results=plot_results,
            save_path=savedir
        )
        del sam
        del mask_generator
        torch.cuda.empty_cache()
    
    if not shortcut_II:
        # --------------------------------
        # II. INPAINTING WITH LAMA + FLUX
        # --------------------------------
        llm_model, processor = instanciate_llm_and_processor()
        prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs = lama_flux_double_inpainting_p1(
            config,
            spherical_dreamer,
            llm_model,
            processor,
            image=img,
            mask=final_mask,
        )

        spherical_dreamer._release_lama_memory()
        del llm_model
        del processor
        torch.cuda.empty_cache()

        inpaint_pano_pil, inpaint_mask_pil = lama_flux_double_inpainting_p2(
            config,
            spherical_dreamer,
            prompt,
            mask_smooth_pil,
            inpaint_pano_lama_pil,
            viz_kwargs,
            plot_results=plot_results,
            save_path=savedir,
        )

        spherical_dreamer._release_flux_inpainting_memory()
        torch.cuda.empty_cache()

        # depth_360_mono = spherical_dreamer.estimate_pano_depth(inpaint_pano_pil)
        # -- debugging save time code ---
        current_variables = {
            'inpaint_pano_pil' : inpaint_pano_pil,
            'inpaint_mask_pil' : inpaint_mask_pil,
            'final_mask_loose' : final_mask_loose,
            # 'depth_360_mono' : depth_360_mono
        }
        with open(os.path.join(savedir, "tmp_inpaint_data.pkl"), "wb") as f:
            pickle.dump(current_variables, f)
    
    # New simpler depth inpainting
    with open(os.path.join(savedir, "tmp_inpaint_data.pkl"), "rb") as f:
        loaded_variables = pickle.load(f)


    inpaint_pano_pil = loaded_variables['inpaint_pano_pil']
    inpaint_mask_pil = loaded_variables['inpaint_mask_pil']
    
    img_pil, depth_origin, _, _ = prepare_inpainting(
        config,
        img,
        depth_origin,
        inpaint_mask_pil,
    )
    _, depth_prior = remove_low_freq(depth_origin, config=config.ldi.masking.depth_mean_based.remove_depth_low_freq)
    inpaint_mask_bool_ = np.ones_like(depth_prior, dtype=bool)

    np.savez(
        f"{savedir}/{img_name}_o3d_local_viz_data.npz",

        # Original image + depth
        my_original_image=img_pil,
        my_original_depth=depth_origin,  # already inverted

        # Background image & mask
        my_new_bg=inpaint_pano_pil,
        my_new_bg_mask=inpaint_mask_bool_,

        depth_prior=depth_prior,
    )

    if not shortcut_III:
        with open(os.path.join(savedir, "tmp_inpaint_data.pkl"), "rb") as f:
            loaded_variables = pickle.load(f)

        inpaint_pano_pil = loaded_variables['inpaint_pano_pil']
        inpaint_mask_pil = loaded_variables['inpaint_mask_pil']
        depth_360_mono = loaded_variables['depth_360_mono']
        final_mask_loose = loaded_variables['final_mask_loose']
        # -- end of debugging save time code ---

        # -------------------------------------------------
        # III. DEPTH INPAINTING (at resolution 1024 * 2048)
        # -------------------------------------------------
        pipe_dp = instanciate_pipe_dp()

        img_pil, depth_origin, inpaint_mask_pil_, inpaint_mask_bool_ = prepare_inpainting(
            config,
            img,
            depth_origin,
            inpaint_mask_pil,
        )
        # Inpainting begins.... 

        # hblending
        inpaint_pano = np.array(inpaint_pano_pil) / 255.0
        _, _, _, depth_inpainted_hblending = harmonic_blend_of_depths(
            colors=inpaint_pano, 
            warped_depth_interp=depth_origin, #gt depth
            depth_estimated=depth_360_mono, # new depth
            missing_info_mask=inpaint_mask_bool_,
            pose= np.eye(4).astype(np.float32),
            sphere_radius=1.0,
            height=inpaint_pano.shape[0],
            width=inpaint_pano.shape[1],
            phase="1",
            logging=plot_results, 
            where_save=savedir,
        )
        if config.ldi.depth_inpainting.apply_post_processing:
            depth_inpainted_hblending_pp = post_process_inpainted_depth(
                depth_bg=depth_inpainted_hblending,
                depth_fg=depth_origin,
                bg_mask=inpaint_mask_bool_,
                plot=plot_results,
            )
        else:
            depth_inpainted_hblending_pp = depth_inpainted_hblending

        # infusion
        depth_inpainted_infusion = inpaint_bg_depth_infusion(
            image=img_pil,
            depth=depth_origin,
            image_bg=inpaint_pano_pil,
            bg_mask=inpaint_mask_pil_,
            pipe_dp=pipe_dp,
            rescale_to_min_depth=False,
            plot_results=plot_results,
            pad_width=config.ldi.depth_inpainting.pad_width,
        )
        if config.ldi.depth_inpainting.apply_post_processing:
            depth_inpainted_infusion_pp = post_process_inpainted_depth(
                depth_bg=depth_inpainted_infusion,
                depth_fg=depth_origin,
                bg_mask=inpaint_mask_bool_,
                plot=plot_results,
            )
        else:
            depth_inpainted_infusion_pp = depth_inpainted_infusion

        # simple interpolations
        depth_inpainted_nn = interpolate_depth_nearest(
            depth=depth_origin,
            bg_mask=inpaint_mask_bool_,
            pad_width=config.ldi.depth_inpainting.pad_width,
        )
        if config.ldi.depth_inpainting.apply_post_processing:
            depth_inpainted_nn_pp = post_process_inpainted_depth(
                depth_bg=depth_inpainted_nn,
                depth_fg=depth_origin,
                bg_mask=inpaint_mask_bool_,
                plot=plot_results,
            )
        else:
            depth_inpainted_nn_pp = depth_inpainted_nn

        depth_inpainted_bilinear_nn = interpolate_depth_bilinear_plus_nn(
            depth=depth_origin,
            bg_mask=inpaint_mask_bool_,
            pad_width=config.ldi.depth_inpainting.pad_width,
        )
        if config.ldi.depth_inpainting.apply_post_processing:
            depth_inpainted_bilinear_nn_pp = post_process_inpainted_depth(
                depth_bg=depth_inpainted_bilinear_nn,
                depth_fg=depth_origin,
                bg_mask=inpaint_mask_bool_,
                plot=plot_results,
            )
        else:
            depth_inpainted_bilinear_nn_pp = depth_inpainted_bilinear_nn
            
        del pipe_dp
        torch.cuda.empty_cache()

        if config.ldi.depth_inpainting.apply_post_processing:
            depth_origin_pp = post_process_inpainted_depth(
                depth_bg=depth_origin,
                depth_fg=depth_origin,
                bg_mask=inpaint_mask_bool_,
            )
        else:
            depth_origin_pp = depth_origin

        suptitle=f""" == Depth Inpainting Results == 
        Additionnal mask dilation: {config.ldi.depth_inpainting.additionnal_mask_dilation_px} px
        Fill holes: {config.ldi.depth_inpainting.fill_holes}
        Depth padding width: {config.ldi.depth_inpainting.pad_width} px
        """ 
        print(suptitle)
        visualize_depth_inpainting(
            img_pil,
            inpaint_pano_pil,
            inpaint_mask_pil_,
            depth_origin,
            depth_origin_pp,
            depth_inpainted_hblending,
            depth_inpainted_hblending_pp,
            depth_inpainted_infusion,
            depth_inpainted_infusion_pp,
            depth_inpainted_nn,
            depth_inpainted_nn_pp,
            depth_inpainted_bilinear_nn,
            depth_inpainted_bilinear_nn_pp,
            suptitle=suptitle,
            save_path=f"{savedir}/main-viz.png"
        )


        # all numpy
        # images PIL uint
        # depth in [0,1] numpy
        np.savez(
            f"{savedir}/{img_name}_o3d_local_viz_data.npz",

            # Original image + depth
            my_original_image=img_pil,
            my_original_depth=depth_origin,  # already inverted

            # Background image & mask
            my_new_bg=inpaint_pano_pil,
            my_new_bg_mask=my_utils.pil_mask_to_numpy_bool(inpaint_mask_pil_),

            depth_infusion=depth_inpainted_infusion_pp,
            depth_nearest=depth_inpainted_nn_pp,
            depth_bilinear_nn=depth_inpainted_bilinear_nn_pp,
            depth_hblending=depth_inpainted_hblending_pp,
        )


# ---------------------------------------------------------------------------
# High-level LDI phase runner (shared by 1b_ldi.py and 2b_ldi.py)
# ---------------------------------------------------------------------------

def run_ldi_phase(
    config,
    save_dir_,
    spherical_dreamer,
    *,
    phase_tag,
    phase_cfg,
    load_phase_from,
    dream_indices,
    load_data_fn,
    save_path_fn,
    viz_filename,
):
    """
    Shared LDI inpainting body used by phases 1b and 2b.

    Parameters
    ----------
    config : Prodict
        Full pipeline config.
    save_dir_ : Path
        Per-experiment output directory.
    spherical_dreamer : SphericalDreamer
        Loaded model wrapper.
    phase_tag : str
        Short phase identifier (e.g. "1b" or "2b") for logging.
    phase_cfg : Prodict
        Phase-specific config slice (config.phase1 or config.phase2).
    load_phase_from : str or None
        Value of config.load_phase1b_from / config.load_phase2b_from.
    dream_indices : iterable of int
        Dream indices to process (range(0, N) for 1b, range(1, N) for 2b).
    load_data_fn : callable (int) -> (np.ndarray, np.ndarray)
        Loads (img, depth) for a given dream index.
    save_path_fn : callable (int) -> Path
        Returns the per-dream visualization directory for dream index i.
    viz_filename : str
        Filename for the depth-inpainting visualization PNG.
    """
    import os
    import torch
    import time

    plot_results = config.ldi.save_plots

    my_utils.printc(f"=== [PHASE {phase_tag}] EXPERIMENT: {config.expname} ===", color='cyan')

    if phase_cfg.apply_ldi:
        if not load_phase_from:
            my_utils.printc(f"=== PHASE {phase_tag}: LDI INPAINTING ===", color='green')

            # -------------------------------------------------
            # 0. LOAD INPUT IMAGES + DEPTH
            # -------------------------------------------------
            list_img = []
            list_depth_origin = []
            for i in dream_indices:
                my_utils.printc(f"--- {phase_tag}: load image  {i:02d} / {config.num_dreams} ---", color='yellow')
                img, depth_origin = load_data_fn(i)
                list_img.append(img)
                list_depth_origin.append(depth_origin)

            # -------------------------------------------------
            # 1. COMPUTE MASK FOR FOREGROUND OBJECTS
            # -------------------------------------------------
            t0 = time.time()
            list_mask = []
            sam, mask_generator = instanciate_sam(config)

            for list_idx, i in enumerate(dream_indices):
                my_utils.printc(f"--- {phase_tag}: Compute mask for foreground object  {i:02d} / {config.num_dreams} ---", color='yellow')
                save_viz_path = save_path_fn(i)
                os.makedirs(save_viz_path, exist_ok=True)
                mask = get_foreground_segmask(
                    config,
                    mask_generator,
                    list_img[list_idx],
                    list_depth_origin[list_idx],
                    plot_results=plot_results,
                    save_path=save_viz_path,
                )
                list_mask.append(mask)

            del sam
            del mask_generator
            torch.cuda.empty_cache()
            print(f"Foreground mask computed in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

            # -------------------------------------------------
            # 2. INPAINTING WITH LAMA
            # -------------------------------------------------
            t0 = time.time()
            list_prompt = []
            list_mask_smooth_pil = []
            list_inpaint_pano_lama_pil = []
            list_viz_kwargs = []
            llm_model, processor = instanciate_llm_and_processor()

            for list_idx, i in enumerate(dream_indices):
                my_utils.printc(f"--- {phase_tag}: Lama Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
                prompt, mask_smooth_pil, inpaint_pano_lama_pil, viz_kwargs = lama_flux_double_inpainting_p1(
                    config,
                    spherical_dreamer,
                    llm_model,
                    processor,
                    image=list_img[list_idx],
                    mask=list_mask[list_idx],
                )
                list_prompt.append(prompt)
                list_mask_smooth_pil.append(mask_smooth_pil)
                list_inpaint_pano_lama_pil.append(inpaint_pano_lama_pil)
                list_viz_kwargs.append(viz_kwargs)

            spherical_dreamer._release_lama_memory()
            del llm_model
            del processor
            torch.cuda.empty_cache()
            print(f"Lama inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

            # -------------------------------------------------
            # 3. INPAINTING WITH FLUX
            # -------------------------------------------------
            t0 = time.time()
            list_inpaint_mask_pil = []
            list_inpaint_pano_pil = []

            for list_idx, i in enumerate(dream_indices):
                my_utils.printc(f"--- {phase_tag}: Flux Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')
                inpaint_pano_pil, inpaint_mask_pil = lama_flux_double_inpainting_p2(
                    config,
                    spherical_dreamer,
                    list_prompt[list_idx],
                    list_mask_smooth_pil[list_idx],
                    list_inpaint_pano_lama_pil[list_idx],
                    list_viz_kwargs[list_idx],
                    plot_results=plot_results,
                    save_path=save_path_fn(i),
                )
                list_inpaint_mask_pil.append(inpaint_mask_pil)
                list_inpaint_pano_pil.append(inpaint_pano_pil)

            spherical_dreamer._release_flux_inpainting_memory()
            torch.cuda.empty_cache()
            print(f"FLUX inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images.")

            # -------------------------------------------------
            # 4. DEPTH INPAINTING (at resolution 1024 * 2048)
            # -------------------------------------------------
            t0 = time.time()
            list_depth_inpainted = []
            list_mask_inpaint_resized = []
            list_depth_origin_resized = []
            list_img_pil_resized = []

            if config.ldi.depth_inpainting.method == "infusion":
                pipe_dp = instanciate_pipe_dp()

            for list_idx, i in enumerate(dream_indices):
                my_utils.printc(f"--- {phase_tag}: Depth Inpainting  {i:02d} / {config.num_dreams} ---", color='yellow')

                img_pil, depth_origin, _, inpaint_mask_bool_ = prepare_inpainting(
                    config,
                    list_img[list_idx],
                    list_depth_origin[list_idx],
                    list_inpaint_mask_pil[list_idx],
                )

                if config.ldi.depth_inpainting.method == "horizontal_min_prior":
                    _, depth_prior = remove_low_freq(depth_origin, config=config.ldi.masking.depth_mean_based.remove_depth_low_freq)
                    inpaint_mask_bool_ = np.ones_like(inpaint_mask_bool_, dtype=bool)
                    _ = my_utils.numpy_bool_to_pil_mask(inpaint_mask_bool_)
                    depth_inpainted = depth_prior

                elif config.ldi.depth_inpainting.method == "harmonic_blending":
                    depth_360_mono = spherical_dreamer.estimate_pano_depth(inpaint_pano_pil)
                    inpaint_pano = np.array(inpaint_pano_pil) / 255.0
                    _, _, _, depth_inpainted_hblending = harmonic_blend_of_depths(
                        colors=inpaint_pano,
                        warped_depth_interp=depth_origin,
                        depth_estimated=depth_360_mono,
                        missing_info_mask=inpaint_mask_bool_,
                        pose=np.eye(4).astype(np.float32),
                        sphere_radius=1.0,
                        height=inpaint_pano.shape[0],
                        width=inpaint_pano.shape[1],
                        phase=phase_tag,
                        logging=plot_results,
                        where_save=save_viz_path,
                    )

                elif config.ldi.depth_inpainting.method == "infusion":
                    depth_inpainted = inpaint_bg_depth_infusion(
                        image=img_pil,
                        depth=depth_origin,
                        image_bg=list_inpaint_pano_pil[list_idx],
                        bg_mask=inpaint_mask_pil,
                        pipe_dp=pipe_dp,
                        rescale_to_min_depth=True,
                        pad_width=config.ldi.depth_inpainting.pad_width,
                        plot_results=plot_results,
                        save_path=save_viz_path,
                    )

                elif config.ldi.depth_inpainting.method == "nearest":
                    depth_inpainted = interpolate_depth_nearest(
                        depth=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                        pad_width=config.ldi.depth_inpainting.pad_width,
                    )

                elif config.ldi.depth_inpainting.method == "bilinear_plus_nn":
                    depth_inpainted = interpolate_depth_bilinear_plus_nn(
                        depth=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                        pad_width=config.ldi.depth_inpainting.pad_width,
                    )

                else:
                    raise ValueError(f"Unknown depth inpainting method: {config.ldi.depth_inpainting.method}")

                depth_inpainted[~inpaint_mask_bool_] = np.nan
                if config.ldi.depth_inpainting.apply_post_processing:
                    depth_inpainted = post_process_inpainted_depth(
                        depth_bg=depth_inpainted,
                        depth_fg=depth_origin,
                        bg_mask=inpaint_mask_bool_,
                        plot=plot_results,
                        save_path=save_path_fn(i),
                    )

                list_depth_inpainted.append(depth_inpainted)
                list_mask_inpaint_resized.append(inpaint_mask_bool_)
                list_depth_origin_resized.append(depth_origin)
                list_img_pil_resized.append(img_pil)

            if config.ldi.depth_inpainting.method == "infusion":
                del pipe_dp
            torch.cuda.empty_cache()
            print(f"Depth inpainting done in {time.time() - t0:.1f} seconds for {config.num_dreams} images with method {config.ldi.depth_inpainting.method}.")

            # -------------------------------------------------
            # SAVE RESULTS
            # -------------------------------------------------
            _method_to_suffix = {
                "horizontal_min_prior": "hminprior",
                "harmonic_blending":    "hblending",
                "infusion":             "infusion",
                "nearest":              "nn",
                "bilinear_plus_nn":     "bilinear_nn",
            }
            suffix = _method_to_suffix[config.ldi.depth_inpainting.method]
            for list_idx, i in enumerate(dream_indices):
                kwargs = {
                    "img_pil": list_img_pil_resized[list_idx],
                    "inpaint_pano_pil": list_inpaint_pano_pil[list_idx],
                    "inpaint_mask_pil": list_mask_inpaint_resized[list_idx],
                    "depth_origin": list_depth_origin_resized[list_idx],
                    f"depth_inpainted_{suffix}": list_depth_inpainted[list_idx],
                }
                visualize_depth_inpainting(
                    **kwargs,
                    save_path=save_path_fn(i) / viz_filename,
                )
                my_utils.save_rgbd_ldi_pano(
                    pano_rgb_bg=list_inpaint_pano_pil[list_idx],
                    depth_bg=list_depth_inpainted[list_idx],
                    mask_bg=list_mask_inpaint_resized[list_idx],
                    dream=i,
                    save_dir_=save_dir_,
                    phase=phase_tag,
                )
            my_utils.printc(f"PHASE {phase_tag} SUCCESSFULLY COMPLETED!", color='green')

        else:
            my_utils.printc(f"SKIPPING PHASE {phase_tag}: LDI INPAINTING", color='magenta')
            my_utils.printc(f"Loading instead from {load_phase_from}", color='magenta')
            my_utils.copy_phase_folders(
                source_dir=Path(config.save_dir) / load_phase_from,
                dest_dir=Path(save_dir_),
                phase=phase_tag,
            )

    else:
        my_utils.printc(f"PHASE {phase_tag}: LDI INPAINTING NOT APPLIED AS PER CONFIGURATION", color='magenta')