import numpy as np
import matplotlib.pyplot as plt

# ---- Your mask_resize wrapper ----
def numpy_resize(img, new_h, new_w, mode="bilinear"):
    """
    Resize a NumPy image using 'nearest' or 'bilinear' interpolation.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W) or (H, W, C).
    new_h : int
        Output height.
    new_w : int
        Output width.
    mode : str
        Interpolation mode: "nearest" or "bilinear".

    Returns
    -------
    out : np.ndarray
        Resized image of shape (new_h, new_w) or (new_h, new_w, C).
    """
    img = np.asarray(img)
    H, W = img.shape[:2]
    has_channels = (img.ndim == 3)

    # Coordinates in the output image
    ys = np.linspace(0, H - 1, new_h)
    xs = np.linspace(0, W - 1, new_w)
    xs, ys = np.meshgrid(xs, ys)

    if mode == "nearest":
        xi = np.rint(xs).astype(int)
        yi = np.rint(ys).astype(int)
        out = img[yi, xi]
        return out.astype(img.dtype)

    elif mode == "bilinear":
        # Neighbor indices
        x0 = np.floor(xs).astype(int)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y0 = np.floor(ys).astype(int)
        y1 = np.clip(y0 + 1, 0, H - 1)

        # Distances
        dx = xs - x0   # (new_h, new_w)
        dy = ys - y0   # (new_h, new_w)

        # For broadcasting with channels, add a trailing axis if needed
        if has_channels:
            dx = dx[..., None]      # (new_h, new_w, 1)
            dy = dy[..., None]      # (new_h, new_w, 1)

        # Neighbor pixels
        Ia = img[y0, x0]   # top-left
        Ib = img[y0, x1]   # top-right
        Ic = img[y1, x0]   # bottom-left
        Id = img[y1, x1]   # bottom-right

        # Bilinear interpolation
        out = (Ia * (1 - dx) * (1 - dy) +
               Ib * (dx)     * (1 - dy) +
               Ic * (1 - dx) * (dy) +
               Id * (dx)     * (dy))

        return out.astype(img.dtype)

    else:
        raise ValueError("Unsupported mode: choose 'nearest' or 'bilinear'")

def opencv_resize(img, new_w, new_h, mode='bilinear'):
    import cv2
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
    return opencv_resize(mask.astype(np.uint8), new_w, new_h, mode="nearest").astype(bool)



# ---- Create 3 different test masks ----
H, W = 720, 1440
upsample_factor = 2

# 1) Random sparse mask
mask_random = (np.random.rand(H, W) > 0.95)

# 2) Circular mask
Y, X = np.ogrid[:H, :W]
cy, cx = H//2, W//2
r = min(H, W) // 4
mask_circle = (X - cx)**2 + (Y - cy)**2 < r**2

# 3) Horizontal stripes
mask_stripes = np.zeros((H, W), bool)
mask_stripes[::20] = True


# ---- Resize all masks ----
def up(mask):
    return mask_resize(mask, H * upsample_factor, W * upsample_factor)

mask_random_up = up(mask_random)
mask_circle_up = up(mask_circle)
mask_stripes_up = up(mask_stripes)


# ---- Visualization ----
plt.figure(figsize=(14, 10))

def show_pair(idx, original, resized, title):
    plt.subplot(3, 2, idx*2 + 1)
    plt.imshow(original, cmap="gray")
    plt.title(f"{title} - original ({H}×{W})")
    plt.axis("off")

    plt.subplot(3, 2, idx*2 + 2)
    plt.imshow(resized, cmap="gray")
    plt.title(f"{title} - upsampled ({H*2}×{W*2})")
    plt.axis("off")

show_pair(0, mask_random,   mask_random_up,   "Random mask")
show_pair(1, mask_circle,   mask_circle_up,   "Circular mask")
show_pair(2, mask_stripes,  mask_stripes_up,  "Striped mask")

plt.tight_layout()
plt.show()