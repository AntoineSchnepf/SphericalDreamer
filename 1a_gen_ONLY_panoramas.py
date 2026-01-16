import os
import warnings
import logging
import contextlib
from io import StringIO

from cv2 import phase

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
from sky_segmentation import SkyMaskDetector

_phase_1a = "1a"
_phase_1b = "1b"
_phase_2a = "2a"
_phase_2b = "2b"
_phase_2c = "2c"

_phase_current = _phase_1a


if __name__ == "__main__":
    # TODO karim: add override functionality and save config
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    if len(seeds) < config.num_dreams:
        seeds = list(range(config.num_dreams))


    printc(f"=== [PHASE 1-A GEN ONLY] EXPERIMENT: {config.expname} ===", color='cyan')


    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    printc("=== PHASE 1-A GEN ONLY: GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH ===", color='green')
    for i in range(config.num_dreams):
        printc(f"--- 1-A: Dreaming {i:02d} / {config.num_dreams} ---", color='yellow')
        # Generate panorama & Estimate Depth
        pano_rgb = spherical_dreamer.gen_pano(prompt=config.prompt, override_with_inpaint=config.phase1.override_with_inpaint, seed_override=seeds[i])
        pano_rgb.save(save_dir_ / f"pano_rgb{i:02d}_seed={seeds[i]}.png")

    printc("PHASE 1-A GEN-ONLY SUCCESSFULLY COMPLETED!", color='green')