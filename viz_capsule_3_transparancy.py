from my_utils import PointCloud, Sphere
import numpy as mp
import open3d as o3d
from pathlib import Path
import pickle
import os
from PIL import Image 
import numpy as np


expname = "forest_v3"


which_ply_list = [
    "sphere1_closed",
    "sphere1_right_opened",
    "sphere1_left_opened",
    "sphere1_both_opened",
    "hollow_capsule",
    "filled_capsule",
    "filled_capsule_colored",
    "Forest_full_pcd_main_fig",
    "Forest_partial_pcd_appendix_fig",
#     "single_sphere_no_ldi",
#     "single_sphere_ldi"
]

figures_dir = "/home/a.schnepf/phd/SphericalDreamer/Figures/main_fig_assets"

for which_ply in which_ply_list:
    output_dir = os.path.join("/home/a.schnepf/phd/SphericalDreamer/Figures/viz_paper_capsule", which_ply)

    files = os.listdir(os.path.join(output_dir, "rgb"))
    files = [f for f in files if f.endswith(".png")]

    for file in files:
        img = Image.open(os.path.join(output_dir, "rgb", file)).convert("RGB")
        mask = Image.open( os.path.join(output_dir, "mask", file)).convert("L")

        # Ensure mask is binary or properly scaled (0–255)
        # If mask is {0,1}, scale it
        mask_np = np.array(mask)
        if mask_np.max() <= 1:
            mask_np = mask_np * 255
        mask = Image.fromarray(mask_np.astype("uint8"), mode="L")

        # Convert RGB → RGBA
        img_rgba = img.convert("RGBA")

        # Use mask as alpha channel (0 = transparent)
        img_rgba.putalpha(mask)

        # Optional: save
        newfilename = f"{which_ply}__{file}"
        img_rgba.save(
            os.path.join(figures_dir, newfilename))
        print(f"Saved RGBA image to {os.path.join(figures_dir, newfilename)}")