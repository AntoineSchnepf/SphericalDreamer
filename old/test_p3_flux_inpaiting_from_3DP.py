# inspired from layerpano3D
# copyright to them

from copyreg import pickle
import numpy as np
import torch
from sphericaldreamer import SphericalDreamer
import my_utils
from PIL import Image
import matplotlib.pyplot as plt
import os


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
    lama = False
    img_name = "FD1"
    strengths = [0.6]
    _3DP_svedir = "/home/a.schnepf/phd/3d-photo-inpainting/Midas+3DP_quick_test/"
    data_path = os.path.join(_3DP_svedir, f"{img_name}_extrafine_data.npz")
    data = np.load(data_path)


    img = data['my_original_image'] / 255.0
    depth = data['my_original_depth']

    mask_bg = data['my_new_bg_mask']
    img_bg = data['my_new_bg']/ 255.0
    depth_bg = data['depth_3DP']

    dilated_mask = my_utils.dilate_mask(mask_bg, pixels=3)

    # double inpainting
    all_inpainted_imgs = []
    new_prompt = "Background"
    for strength in strengths:
        if lama:
            inpainted_img = spherical_dreamer.lama_inpaint(
                image=my_utils.numpy_to_PIL(img_bg),
                mask=my_utils.numpy_bool_to_pil_mask(dilated_mask),  # mask: uint8, 0 or 255
            )
        else:
            inpainted_img = spherical_dreamer.inpaint_pano(
                prompt=new_prompt,
                pano_rgb=my_utils.numpy_to_PIL(img_bg),
                mask=my_utils.numpy_bool_to_pil_mask(dilated_mask),  # mask: uint8, 0 or 255
                strength=strength,
                seed_override=794,
            )
        all_inpainted_imgs.append(my_utils.PIL_to_numpy(inpainted_img))

    
    rows = 1 + len(strengths)   # 1 row for original, then 1 per strength
    cols = 2                    # plain + overlay

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    # MAIN TITLE
    fig.suptitle(f"Inpainting Results for {img_name}. Prompt='{new_prompt}'\n Inpainting model = { 'Lama' if lama else 'Flux' }", fontsize=20, y=0.95)


    # Row 0: original image
    axes[0].imshow(img_bg)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(my_utils.overlay_mask(img_bg, dilated_mask))
    axes[1].set_title("Original Image (with mask overlay)")
    axes[1].axis("off")

    # Rows 1..N: inpainted images
    for i, (strength, inpainted_img) in enumerate(zip(strengths, all_inpainted_imgs)):
        row = i + 1
        idx_plain = 2 * row
        idx_overlay = 2 * row + 1

        # Inpainted image
        axes[idx_plain].imshow(inpainted_img)
        axes[idx_plain].set_title(f"Inpainted Image [strength={strength}]")
        axes[idx_plain].axis("off")

        # Inpainted image with mask overlay
        axes[idx_overlay].imshow(my_utils.overlay_mask(inpainted_img, dilated_mask))
        axes[idx_overlay].set_title(f"Inpainted Image [strength={strength}] (with mask overlay)")
        axes[idx_overlay].axis("off")

    plt.tight_layout()
    plt.savefig(f"inpainted_{img_name}.png", dpi=150)
    plt.show()

