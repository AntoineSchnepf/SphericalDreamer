import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageOps
import copy
from functools import partial
from skimage.segmentation import find_boundaries
from scipy.ndimage import maximum_filter, minimum_filter
import logging
import matplotlib.pyplot as plt
import time
import pickle as pkl
from prodict import Prodict
import pyfiglet
import argparse
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from render_pcd import render_v2
from sphericaldreamer import SphericalDreamer
import my_utils

logging.disable(logging.CRITICAL + 1)



    
if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "forest.yaml"]
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
    print(f"=== EXPERIMENT: {config.expname} ===")
    if not config.skip_phase1:
        print("=== PHASE 1: GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH ===")
        for i in range(config.num_dreams):
            print(f"--- Dreaming Phase {i:02d} / {config.num_dreams} ---")

            # Generate panorama & Estimate Depth
            pano_rgb = spherical_dreamer.gen_pano(prompt=config.prompt, override_with_inpaint=config.phase1.override_with_inpaint, seed_override=seeds[i])
            depth = spherical_dreamer.estimate_pano_depth(pano_rgb=np.array(pano_rgb))
            my_utils.save_rgbd_pano(
                pano_rgb=pano_rgb,
                depth=depth,
                dream=i,
                save_dir_=save_dir_
            )
        print("PHASE 1 SUCCESSFULLY COMPLETED!")
    else:
        print("=== SKIPPING PHASE 1: GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH ===")