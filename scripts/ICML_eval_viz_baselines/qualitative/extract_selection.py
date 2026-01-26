import os
from pathlib import Path
from PIL import Image

selected_x={
    "scenescape": ["x=0.00", "x=-0.09"],
    "wonderjourney": ["x=0.00", "x=-0.19"],
    "sphericaldreamer": ["x=0.05", "x=3.09"],
}

methods=[
    "scenescape", "wonderjourney", "sphericaldreamer"
]

expnames=[
    "dense_rainforest_understory",
    "coral_reef_canyon",
    "martian_badlands_2",
    "phantom_opera_cave_river",
    "sound_of_music_grass_field",
    "upside_down_stranger_things",
]

CROP_TOP = 0.1
CROP_BOT = 0.1

BASE_PATH = Path("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/X_ICML_RENDERS/qualitative")
selected_images = []
for expname in expnames:
    for method in methods:
        
        for ind, x_dir_out in zip([0,1], ["x0", "x1"]):
            x_dir_in = selected_x[method][ind]

            eqr_path = BASE_PATH / expname / method / "rgb_eqr" / x_dir_in / "azi=0.00.png"
            normal_path_front = BASE_PATH / expname / method / "rgb" / x_dir_in / "azi=0.00.png"
            normal_path_left = BASE_PATH / expname / method / "rgb" / x_dir_in / "azi=90.00.png"
            normal_path_back = BASE_PATH / expname / method / "rgb" / x_dir_in / "azi=180.00.png"
            normal_path_right = BASE_PATH / expname / method / "rgb" / x_dir_in / "azi=270.00.png"

            eqr = Image.open(eqr_path)
            normal_front = Image.open(normal_path_front)
            normal_left = Image.open(normal_path_left)
            normal_back = Image.open(normal_path_back)
            normal_right = Image.open(normal_path_right)

            where_save=Path("/home/a.schnepf/phd/SphericalDreamer/Figures/exp_main") / expname / method
            os.makedirs(where_save, exist_ok=True)

            filename_eqr = where_save / f"eqr_{x_dir_out}.png"
            filename_normal_front = where_save / f"normal_{x_dir_out}_front.png"
            filename_normal_left = where_save / f"normal_{x_dir_out}_left.png"
            filename_normal_back = where_save / f"normal_{x_dir_out}_back.png"
            filename_normal_right = where_save / f"normal_{x_dir_out}_right.png"



            eqr = eqr.crop((
                0,
                int(CROP_TOP * eqr.height),
                eqr.width,
                eqr.height - int(CROP_BOT * eqr.height)
            ))

            eqr.save(filename_eqr)
            
            normal_front.save(filename_normal_front)
            normal_left.save(filename_normal_left)
            normal_back.save(filename_normal_back)
            normal_right.save(filename_normal_right)

            print("Saved images for ", expname, method)

