from pathlib import Path
import numpy as np
import my_utils
from PIL import Image
import matplotlib.pyplot as plt



def overlay_purple_and_save(pano, mask, out_path, alpha=0.35):
    """
    pano : (H,W,3) float32 in [0,1]
    mask : (H,W) binary (1 = overlay)
    """
    pano = pano.astype(np.float32)
    mask = mask.astype(np.float32)

    # Purple color (normalized RGB)
    purple = np.array([180, 130, 255], dtype=np.float32) / 255.0

    # Expand mask to (H,W,1)
    alpha_map = alpha * mask[..., None]

    # Alpha blending
    out = pano * (1.0 - alpha_map) + purple * alpha_map
    out = np.clip(out, 0.0, 1.0)

    # Save as PNG
    Image.fromarray((out * 255).astype(np.uint8)).save(out_path)


if __name__ == "__main__":

    save_dir = Path("OUTPUTS/SphericalDreamerRecurse")
    save_dir__ = save_dir / "forest_v3"


    figdir = Path("/home/a.schnepf/phd/SphericalDreamer/Figures")
    _phase_current = "1a"




    # visualize dream
    # for i, dream_iter in enumerate([0, 1]):

    #     depth = np.load(save_dir__ / f"dream_{dream_iter:02d}" / _phase_current / ".cache"/ "depth.npy")
    #     pano = Image.open(save_dir__ / f"dream_{dream_iter:02d}" / _phase_current / ".cache"/ "pano_rgb.png")

    #     depth_pil = my_utils.depth_to_pil(depth, cmap_name="plasma", vmin=0.1, vmax=1.0)
    #     pano.save(figdir / f"main_fig_01_pano_{i}.png")
    #     depth_pil.save(figdir / f"main_fig_02_depth_{i}.png")

    # vizualize inpaint
    align_iter = 1
    _phase_current = "2a"
    data = np.load(save_dir__ / f"align_{align_iter:02d}" / _phase_current / ".cache"/ "other_data.npy", allow_pickle=True).item( )
    mask = missing_info_mask = data["missing_info_mask"]
    depth = data['warped_depth_interp']
    # pano = data['warped_img_interp']
    pano = Image.open(save_dir__ / f"align_{align_iter:02d}" / _phase_current / ".cache"/ "pano_rgb_inpainted_raw.png")

    import numpy as np
    from PIL import Image



    # ---------------- Usage ----------------
    overlay_purple_and_save(
        pano=np.array(pano.resize((1440, 720))).astype(np.float32) / 255.0,
        mask=data["missing_info_mask"],
        out_path=figdir / "main_fig_inpainting_w_overlay.png",
    )