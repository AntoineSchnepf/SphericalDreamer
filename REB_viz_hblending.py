import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from src.Infusion.depth_inpainting.utils.seed_all import seed_all
from src.Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half import (
    DepthEstimationInpaintPipeline,
)


def load_depth_inpaint_pipeline(
    model_path="checkpoints/Infusion",
    device="cuda",
    dtype=torch.float16,
):
    """
    Load the Infusion depth-inpainting pipeline.

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


def make_square_mask(H, W, rel_size=0.3):
    """Create a centered square boolean mask covering rel_size of each axis."""
    cy, cx = H // 2, W // 2
    half_h, half_w = int(H * rel_size / 2), int(W * rel_size / 2)
    mask = np.zeros((H, W), dtype=bool)
    mask[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = True
    return mask


def run_all_methods(depth_gt, rgb_gt, mask, pipe_dp, sd):
    """Run Infusion, harmonic blending and bilinear+NN for a given mask.
    Returns dict with depth results for each method."""
    from harmonic_blending import harmonic_blend_of_depths
    from ldi_inpaiting import interpolate_depth_bilinear_plus_nn

    H, W = depth_gt.shape
    eps = 1e-6

    # Infusion
    depth_holed = depth_gt.copy()
    depth_holed[mask] = 0.0
    disparity = 1.0 / (depth_holed + eps)
    known = ~mask
    disp_min = float(disparity[known].min())
    disp_max = float(disparity[known].max())
    disparity_norm = (disparity - disp_min) / (disp_max - disp_min + eps)
    disparity_norm[mask] = 0.0
    pipe_out = pipe_dp(input_image=rgb_gt, depth_numpy=disparity_norm, mask=mask.astype(np.float32))
    disparity_pred = pipe_out.depth_np.astype(np.float32) * (disp_max - disp_min) + disp_min
    depth_infusion = 1.0 / (disparity_pred + eps)

    # Harmonic blending
    depth_estimated = sd.estimate_pano_depth(rgb_gt)
    _, _, _, depth_hblend = harmonic_blend_of_depths(
        colors=rgb_gt.astype(np.float32) / 255.0,
        warped_depth_interp=depth_gt,
        depth_estimated=depth_estimated,
        missing_info_mask=mask,
        pose=np.eye(4, dtype=np.float32),
        sphere_radius=1.0,
        height=H, width=W,
        phase="1", logging=False, where_save=Path("/tmp"),
    )

    # Bilinear + NN
    depth_holed_nan = depth_gt.copy()
    depth_holed_nan[mask] = np.nan
    depth_bilinear_nn = interpolate_depth_bilinear_plus_nn(
        depth=depth_holed_nan, bg_mask=mask, pad_width=15,
    )

    return {
        "infusion": depth_infusion,
        "hblend": depth_hblend,
        "bilinear_nn": depth_bilinear_nn,
    }


if __name__ == "__main__":
    import cv2
    import argparse
    import contextlib
    from io import StringIO
    with contextlib.redirect_stdout(StringIO()):
        from sphericaldreamer import SphericalDreamer

    _default_rgb = [
        "OUTPUTS/gen_images_bckp/FD0.png",
        "OUTPUTS/gen_images_bckp/CD0.png",
        "OUTPUTS/gen_images_bckp/SD0.png",
    ]
    _default_depth = [
        "OUTPUTS/gen_depths_bckp/FD0.npy",
        "OUTPUTS/gen_depths_bckp/CD0.npy",
        "OUTPUTS/gen_depths_bckp/SD0.npy",
    ]
    _default_mask = [
        "OUTPUTS/SphericalDreamerRecurse/Forest/align_01/2a/.cache/other_data.npy",
        "OUTPUTS/SphericalDreamerRecurse/Forest/align_02/2a/.cache/other_data.npy",
        "OUTPUTS/SphericalDreamerRecurse/Forest/align_03/2a/.cache/other_data.npy",
    ]

    parser = argparse.ArgumentParser(description="Depth inpainting comparison across 3 samples")
    parser.add_argument("--rgb", nargs=3, default=_default_rgb, help="Paths to 3 RGB images (.png)")
    parser.add_argument("--depth", nargs=3, default=_default_depth, help="Paths to 3 depth maps (.npy)")
    parser.add_argument("--mask", nargs=3, default=_default_mask, help="Paths to 3 masks (.npy, bool arrays or dicts with 'missing_info_mask')")
    parser.add_argument("--out", default="OUTPUTS/depth_inpainting_demo", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = 3

    # --- 1. Load all data ---
    rgbs, depths, masks = [], [], []
    for i in range(N):
        rgb = np.array(Image.open(args.rgb[i]).convert("RGB"))
        depth = np.load(args.depth[i], allow_pickle=True)
        if isinstance(depth, np.ndarray) and depth.ndim == 0:
            depth = depth.item()
        depth = np.asarray(depth, dtype=np.float32)

        mask_data = np.load(args.mask[i], allow_pickle=True)
        if isinstance(mask_data, np.ndarray) and mask_data.ndim == 0:
            mask_data = mask_data.item()
        if isinstance(mask_data, dict):
            mask = mask_data["missing_info_mask"]
        else:
            mask = np.asarray(mask_data)
        mask = mask.astype(bool)

        H, W = depth.shape
        if mask.shape != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)

    # --- 2. Load models once ---
    pipe_dp = load_depth_inpaint_pipeline()
    H0, W0 = depths[0].shape
    sd = SphericalDreamer(
        pano_width=W0, pano_height=H0,
        pano_depth_temp_dir="/tmp/pano_depth_temp",
        depth_model="360mono",
    )

    # --- 3. Run all methods on each sample ---
    all_results = []
    for i in range(N):
        print(f"Processing sample {i+1}/{N}: {args.rgb[i]}")
        res = run_all_methods(depths[i], rgbs[i], masks[i], pipe_dp, sd)
        all_results.append(res)

    del pipe_dp
    torch.cuda.empty_cache()

    # --- 4. Plot (6 rows x 3 columns) ---
    # Row 0: RGB GT
    # Row 1: Depth GT
    # Row 2: Depth holed
    # Row 3: Bilinear + NN
    # Row 4: Infusion
    # Row 5: Harmonic blending
    vmin = min(float(d.min()) for d in depths)
    vmax = max(float(d.max()) for d in depths)

    fig, axes = plt.subplots(6, N, figsize=(7 * N, 28))

    row_specs = [
        ("RGB GT",            "rgb"),
        ("Depth GT",          "depth_gt"),
        ("Depth (holed)",     "depth_holed"),
        ("Bilinear + NN",     "bilinear_nn"),
        ("Infusion",          "infusion"),
        ("Harmonic blended",  "hblend"),
    ]

    for col in range(N):
        depth_gt = depths[col]
        rgb_gt = rgbs[col]
        mask = masks[col]
        res = all_results[col]

        depth_holed = depth_gt.copy()
        depth_holed[mask] = np.nan

        for row, (label, key) in enumerate(row_specs):
            ax = axes[row, col]

            if key == "rgb":
                ax.imshow(rgb_gt)
            elif key == "depth_gt":
                ax.imshow(depth_gt, cmap="inferno", vmin=vmin, vmax=vmax)
            elif key == "depth_holed":
                ax.imshow(depth_holed, cmap="inferno", vmin=vmin, vmax=vmax)
            else:
                ax.imshow(res[key], cmap="inferno", vmin=vmin, vmax=vmax)
                mae = float(np.abs(depth_gt - res[key]).mean())
                label = f"{label}  MAE={mae:.4f}"

            if col == 0:
                ax.set_ylabel(label, fontsize=13, rotation=90, labelpad=10)
            if row == 0:
                ax.set_title(Path(args.rgb[col]).stem, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Depth inpainting comparison", fontsize=18, y=0.995)
    fig.tight_layout()

    save_path = out_dir / "comparison_overview.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overview to {save_path}")

    # --- Save individual images ---
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    for col in range(N):
        depth_gt = depths[col]
        rgb_gt = rgbs[col]
        mask = masks[col]
        res = all_results[col]

        depth_holed = depth_gt.copy()
        depth_holed[mask] = np.nan

        images_to_save = {
            "rgb_gt":        ("rgb",   rgb_gt),
            "depth_gt":      ("depth", depth_gt),
            "depth_holed":   ("depth", depth_holed),
            "bilinear_nn":   ("depth", res["bilinear_nn"]),
            "infusion":      ("depth", res["infusion"]),
            "hblend":        ("depth", res["hblend"]),
        }

        for row, (key, (kind, data)) in enumerate(images_to_save.items()):
            fname = tiles_dir / f"{row}{col}_{key}.png"
            if kind == "rgb":
                Image.fromarray(data).save(fname)
            else:
                fig_t, ax_t = plt.subplots(1, 1, figsize=(10, 5))
                ax_t.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax)
                ax_t.axis("off")
                fig_t.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0)
                plt.close(fig_t)

        print(f"Saved tiles for column {col} to {tiles_dir}")
