import os
import sys
import cv2
from matplotlib import image
from src.pipeline_flux import FluxPipeline
from src.pipeline_flux_fill import FluxFillPipeline
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import torch
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
import argparse
from prodict import Prodict
import pyfiglet
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
from render_pcd import render_v2
import my_utils
from sphericaldreamer import SphericalDreamer

logging.disable(logging.CRITICAL + 1)




if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=True, 
        debug_parser_override=["--config", "forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    # -------------------------------------------------------------------------------- # 
    # ----- PHASE III. POST PROCESSING OF THE FINAL POINTCLOUD WITH HOLE FILLING ----- #
    # -------------------------------------------------------------------------------- #
    print(f"=== EXPERIMENT: {config.expname} ===")
    if not config.skip_phase3:
        print("=== PHASE III : GET FINAL POINTCLOUD WITH HOLE FILLING ===")


        all_pts_world, all_colors_world = my_utils.run_corrective_pipeline_on_world(
            pts=all_pts_world,
            colors=all_colors_world,
            pose_left=pose_init,
            pose_right=pose_end,
            translation_direction=translation_direction,
            verbose=True,
            plot=True,
            **config.geometry_correction.world
        )

        # TODO: Antoine, 16 Oct: REDO EVERYtHING BELOW. CURRENTLY IT;S BAD
        for i, cam_pose in enumerate(my_utils.get_intermediate_camera_poses(
            start_pose=pose_init,
            end_pose=pose_end,
            num_steps=10,
            perturb_y=0.0,
            perturb_z=0.0, 
            perturb_x=0.0,
        )): #TODO: this function does not really do what I currently want, as pertub is added randomly to each indermediate camera. Ideally I would want something dense.
            save_dir__ = os.path.join(save_dir_, f"final_filling_{i:03d}")
            os.makedirs(save_dir__, exist_ok=True)
            print(f"--- Final Filling from new camera pose ---")
            new_pts, new_colors, pcd_naive, pcd_harmonic = generate_missing_points_from_pose(
                all_pts_world, 
                all_colors_world, 
                my_utils.camera_translation(cam_pose, 0.0 * np.array([0, 0, 1])), # when correcting the floor enforcing z=0, you want to raise the camera a bit
                height,
                width,
                upsampling_factor=1,
                prompt=config.prompt,
                skip_inpainting=config.phase3.skip_inpainting, 
                where_save=save_dir__
            )
            all_pts_world = np.concatenate((all_pts_world, new_pts), axis=0)
            all_colors_world = np.concatenate((all_colors_world, new_colors), axis=0)


        final_pcd = my_utils.PointCloud(
            pts=all_pts_world,
            colors=all_colors_world
        )
        with open(os.path.join(save_dir_, "final_dream_pcd.pkl"), 'wb') as f:
            pkl.dump(final_pcd, f)