import os
import warnings
import logging
import contextlib
from io import StringIO

# Disabling some warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["GLOG_logtostderr"] = "0"
os.environ["CERES_MINIMIZER_PROGRESS_TO_STDOUT"] = "0"
logging.disable(logging.CRITICAL + 1)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer
import my_utils
from my_utils import printc



if __name__ == "__main__":
    # TODO karim: add override functionality and save config
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )


    # ----------------------------------------------------------------- #
    # ---- PHASE 1. GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH ----- #
    # ----------------------------------------------------------------- #
    printc(f"=== [PHASE 1-A] EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase1_from:
        printc("=== PHASE 1-A: GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH ===", color='green')
        for i in range(config.num_dreams):
            printc(f"--- 1-A: Dreaming {i:02d} / {config.num_dreams} ---", color='yellow')
            # Generate panorama & Estimate Depth
            pano_rgb = spherical_dreamer.gen_pano(prompt=config.prompt, override_with_inpaint=config.phase1.override_with_inpaint, seed_override=seeds[i])
            depth = spherical_dreamer.estimate_pano_depth(pano_rgb=np.array(pano_rgb))
            my_utils.save_rgbd_pano(
                pano_rgb=pano_rgb,
                depth=depth,
                dream=i,
                save_dir_=save_dir_
            )
        printc("PHASE 1-A SUCCESSFULLY COMPLETED!", color='green')
    else:
        printc("SKIPPING PHASE 1-A: GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH", color='magenta')
        printc(f"Loading instead from {config.load_phase1_from}", color='magenta')
        source_phase1_path = Path(config.save_dir) / config.load_phase1_from
        dest_phase1_path = Path(save_dir_)
        my_utils.copy_phase_folders(
            folder_start_with="dream_",
            item_start_with="",
            source_dir=source_phase1_path,
            dest_dir=dest_phase1_path
        )