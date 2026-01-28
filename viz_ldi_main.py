from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.cm as cm

# ------------------ Paths ------------------
TARGET_SIZE = (1440, 720)  # (width, height)

exp_path = Path("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/Forest_v2")
original_dir = exp_path / "dream_01/1a/.cache"
original_image_name = "pano_rgb.png"
original_depth_name = "depth.npy"

ldi_dir = exp_path / "dream_01/1b/.cache"
ldi_img_name = "pano_lama.png"
ldi_mask_name = "ldi_mask.npy"
ldi_depth_name = "depth_inpainted_trick.npy"

save_dir = Path("./Figures/ldi_main_paper")
save_dir.mkdir(parents=True, exist_ok=True)

# ------------------ Utils ------------------
def load_depth_as_pil(depth_path, cmap_name="plasma", vmin=0.1, vmax=1.8):
    depth = np.load(depth_path)

    # Identify invalid / zero depth
    invalid = (~np.isfinite(depth)) | (depth <= 0)

    # Normalize only valid depth
    depth_valid = depth.astype(np.float32).copy()
    depth_valid[invalid] = np.nan

    # vmin = np.nanmin(depth_valid)
    # vmax = np.nanmax(depth_valid)

    depth_norm = np.zeros_like(depth_valid, dtype=np.float32)
    good = ~invalid
    depth_norm[good] = (depth_valid[good] - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    # Apply colormap
    depth_rgb = cm.get_cmap(cmap_name)(depth_norm)[..., :3]

    # Force invalid pixels to black
    depth_rgb[invalid] = 0.0

    img = Image.fromarray((depth_rgb * 255).astype(np.uint8))
    return img.resize(TARGET_SIZE, Image.BILINEAR)


def overlay_mask_purple_numpy(img_rgb, mask, alpha=0.35):
    """
    img_rgb : (H, W, 3) uint8 RGB image
    mask    : (H, W) binary or [0,1] mask (1 = overlay)
    alpha   : transparency strength
    """
    img = img_rgb.astype(np.float32) / 255.0

    # Purple color in RGB (normalized)
    purple = np.array([180, 130, 255], dtype=np.float32) / 255.0

    # Ensure mask is float in [0,1]
    mask_f = mask.astype(np.float32)
    if mask_f.max() > 1.0:
        mask_f /= 255.0
    mask_f = np.clip(mask_f, 0.0, 1.0)

    # Per-pixel alpha
    alpha_map = alpha * mask_f[..., None]  # (H,W,1)

    # Alpha blending
    out = img * (1.0 - alpha_map) + purple * alpha_map
    out = np.clip(out, 0.0, 1.0)

    return (out * 255).astype(np.uint8)


def overlay_and_save(pil_img_rgb, mask_np, out_path, alpha=0.35):
    """PIL RGB -> NumPy overlay -> save PIL RGB."""
    img_np = np.array(pil_img_rgb)  # (H,W,3) uint8
    over_np = overlay_mask_purple_numpy(img_np, mask_np, alpha=alpha)
    Image.fromarray(over_np).save(out_path)

# ------------------ Load & resize RGB ------------------
ldi_img = Image.open(os.path.join(ldi_dir, ldi_img_name)).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
original_img = Image.open(os.path.join(original_dir, original_image_name)).convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)

# ------------------ Load & resize depth (as RGB PIL) ------------------
ldi_depth = load_depth_as_pil(os.path.join(ldi_dir, ldi_depth_name))
original_depth = load_depth_as_pil(os.path.join(original_dir, original_depth_name))

# ------------------ Load & resize mask (NumPy) ------------------
mask = np.load(os.path.join(ldi_dir, ldi_mask_name))
# ensure binary float mask in {0,1}
mask = (mask > 0).astype(np.float32)
mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(TARGET_SIZE, Image.NEAREST)
mask = (np.array(mask).astype(np.float32) / 255.0)  # (H,W) in [0,1]

# ------------------ Save raw images ------------------
ldi_img.save(save_dir / "ldi_rgb.png")
ldi_depth.save(save_dir / "ldi_depth.png")
original_img.save(save_dir / "original_rgb.png")
original_depth.save(save_dir / "original_depth.png")

# ------------------ Save overlay images (NumPy alpha blending) ------------------
overlay_and_save(ldi_img, mask, save_dir / "ldi_rgb_mask_overlay.png")
# overlay_and_save(ldi_depth, mask, save_dir / "ldi_depth_mask_overlay.png")
overlay_and_save(original_img, mask, save_dir / "original_rgb_mask_overlay.png")
overlay_and_save(original_depth, mask, save_dir / "original_depth_mask_overlay.png")

print(f"Saved resized images to {save_dir}")

# ------------------ EXTRA: compose LDI depth over original depth (no mask) ------------------

def resize_depth_nearest(depth, target_size):
    """Resize a float depth map to (W,H) using nearest (keeps zeros/NaNs clean)."""
    W, H = target_size
    depth_img = Image.fromarray(depth.astype(np.float32), mode="F")
    depth_img = depth_img.resize((W, H), Image.NEAREST)
    return np.array(depth_img, dtype=np.float32)

# Load raw depth arrays
ldi_depth_raw = np.load(os.path.join(ldi_dir, ldi_depth_name)).astype(np.float32)
orig_depth_raw = np.load(os.path.join(original_dir, original_depth_name)).astype(np.float32)

# Resize to TARGET_SIZE
ldi_depth_rs = resize_depth_nearest(ldi_depth_raw, TARGET_SIZE)
orig_depth_rs = resize_depth_nearest(orig_depth_raw, TARGET_SIZE)

# Define validity from the depth itself (NO mask): valid if finite and > 0
ldi_valid = np.isfinite(ldi_depth_rs) & (ldi_depth_rs > 0)

# Compose: wherever LDI depth is valid, overwrite original depth
depth_composed = orig_depth_rs.copy()
depth_composed[ldi_valid] = ldi_depth_rs[ldi_valid]

# Save visualization with the same colormap rule (invalid/zero -> black)
np.save(save_dir / "depth_composed_ldi_over_original.npy", depth_composed)  # optional, handy
depth_composed_pil = load_depth_as_pil(save_dir / "depth_composed_ldi_over_original.npy", cmap_name="plasma")
depth_composed_pil.save(save_dir / "depth_composed_ldi_over_original.png")

print("Saved composed depth visualization: depth_composed_ldi_over_original.png")