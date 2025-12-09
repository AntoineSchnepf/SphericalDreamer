
import numpy as np
from sphericaldreamer import SphericalDreamer
import my_utils
from PIL import Image
import matplotlib.pyplot as plt


def low_freq_vertical(img, cutoff_frac=0.1):
    """
    Extract a vertically low-frequency version of an image.
    Frequencies are filtered ONLY along the vertical axis (rows).

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
        Low-frequency image (same shape, float32).
    """
    img = np.asarray(img)
    is_uint8 = img.dtype == np.uint8
    img_f = img.astype(np.float32)

    # If grayscale, add a fake channel dimension
    if img_f.ndim == 2:
        img_f = img_f[..., None]  # (H, W, 1)

    H, W, C = img_f.shape

    # 1D FFT along vertical axis (axis=0)
    F = np.fft.fft(img_f, axis=0)               # shape (H, W, C)
    F_shift = np.fft.fftshift(F, axes=0)        # shift zero-freq to center

    # Build vertical low-pass mask
    k_max = int(cutoff_frac * (H / 2.0))        # how many low-freq bins to keep
    k_max = max(1, k_max)

    mask = np.zeros((H,), dtype=bool)
    center = H // 2
    start = max(0, center - k_max)
    end   = min(H, center + k_max + 1)
    mask[start:end] = True                      # keep band around DC

    # Broadcast mask across width and channels
    mask_3d = mask[:, None, None]               # (H, 1, 1)

    # Zero out high vertical frequencies only
    F_shift_filtered = np.zeros_like(F_shift)
    F_shift_filtered[mask, :, :] = F_shift[mask, :, :]

    # Inverse FFT to go back to image space
    F_filtered = np.fft.ifftshift(F_shift_filtered, axes=0)
    low_img = np.fft.ifft(F_filtered, axis=0).real  # (H, W, C)

    # Drop fake channel if grayscale
    if img.ndim == 2:
        low_img = low_img[..., 0]

    if is_uint8:
        low_img = np.clip(low_img, 0, 255).astype(np.uint8)

    return low_img



def low_freq_vertical_row_constant(img, cutoff_frac=0.1):
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




if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=True, 
        debug_parser_override=["--config", "Antoine/F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    img_name = "FD1"
    data_dir= "/home/a.schnepf/phd/SphericalDreamer/OUTPUTS"
    depth_path = f"{data_dir}/gen_depths_bckp/{img_name}.npy"
    image_path = f"{data_dir}/gen_images_bckp/{img_name}.png"  # or .jpg
    threshold = -0.02
    # img: numpy array (H, W, 3) or (H, W)

    depth = np.load(depth_path)
    img = Image.open(image_path)

    low = low_freq_vertical_row_constant(depth, cutoff_frac=0.1)  # keep lowest 10% vertical frequencies
    mask = (depth - low) < threshold
    dilated_mask = my_utils.dilate_mask(mask.astype(np.uint8)*255, pixels=3)


    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.imshow(depth, cmap="gray" if depth.ndim==2 else None)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(low, cmap="gray" if low.ndim==2 else None)
    plt.title("Vertical low-frequency")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(dilated_mask, cmap="gray")
    plt.title(f"Mask (thresh={threshold})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # inpainting
    inpainted_img = spherical_dreamer.inpaint_pano(
        prompt=f"background of {config.prompt}. Only the background.",
        pano_rgb=img,
        mask=my_utils.numpy_bool_to_pil_mask(dilated_mask),  # mask: uint8, 0 or 255
    )

    plt.figure(figsize=(10,20))
    plt.subplot(2,1,1)
    plt.imshow(my_utils.overlay_mask(my_utils.PIL_to_numpy(img), dilated_mask), alpha=0.9)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2,1,2)
    plt.imshow(my_utils.overlay_mask(my_utils.PIL_to_numpy(inpainted_img), dilated_mask), alpha=0.9)
    plt.title("Inpainted Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"inpainted_{img_name}.png")
    plt.show()