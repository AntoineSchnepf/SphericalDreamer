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
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
import time
import pickle as pkl
from pathlib import Path
import argparse
from prodict import Prodict
import pyfiglet
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from render_pcd import render_v2
from harmonic_blending import harmonic_blend_of_depths_ldi, naive_blend_of_depths
import my_utils
from my_utils import printc
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer
    from utils.depth_alignment import Pano_depth_estimation

_phase_1a = "1a"
_phase_1b = "1b"
_phase_2a = "2a"
_phase_2b = "2b"
_phase_2c = "2c"

_phase_current = _phase_2c

def align_new_points(
        warped_img_interp,
        warped_depth_interp,
        pano_rgb_inpainted,
        depth_estimated,
        missing_info_mask,
        camera_pose, 
        height,
        width,
        sphere_radius,
        upsampling_factor,
        where_save=None,
        ldi_depth=None,
        ldi_colors=None,
        ldi_mask=None,
):
    # 9. Blend depth
    new_colors = (np.array(pano_rgb_inpainted)/255.0)

    # Optional upsampling to improve pcd density
    if upsampling_factor > 1:
        new_colors = my_utils.opencv_resize(new_colors, height*upsampling_factor, width*upsampling_factor, mode="bilinear")
        warped_depth_interp = my_utils.opencv_resize(warped_depth_interp, height*upsampling_factor, width*upsampling_factor, mode="bilinear")
        depth_estimated = my_utils.opencv_resize(depth_estimated, height*upsampling_factor, width*upsampling_factor, mode="bilinear")
        missing_info_mask = my_utils.mask_resize(missing_info_mask, height*upsampling_factor, width*upsampling_factor)

        # sanity check
        where_depth_nan_resized = np.isnan(warped_depth_interp)
        if np.any(where_depth_nan_resized & (~missing_info_mask)):
            print("WARNING: resized depth has NaNs in inpainted info regions!")
            print(f"Percent of NaNs: {np.mean(where_depth_nan_resized & (~missing_info_mask))*100:.2f}%")
            print("Expanding missing info mask to include these regions.")
            # we want all nans in missing info mask
            missing_info_mask = missing_info_mask | where_depth_nan_resized
            
        if ldi_depth is not None or ldi_colors is not None or ldi_mask is not None:
            assert ldi_depth is not None and ldi_colors is not None and ldi_mask is not None, "If one of ldi_depth, ldi_colors, ldi_mask is provided, all must be provided."
            ldi_depth = my_utils.opencv_resize(ldi_depth, height*upsampling_factor, width*upsampling_factor, mode="bilinear") 
            ldi_colors = my_utils.opencv_resize(ldi_colors, height*upsampling_factor, width*upsampling_factor, mode="bilinear")
            ldi_mask = my_utils.mask_resize(ldi_mask, height*upsampling_factor, width*upsampling_factor)

            where_ldi_depth_nan_resized = np.isnan(ldi_depth)
            if np.any(where_ldi_depth_nan_resized & (ldi_mask)):
                print("WARNING: resized ldi depth has NaNs in ldi regions!")
                print(f"Percent of NaNs: {np.mean(where_ldi_depth_nan_resized & (ldi_mask))*100:.2f}%")
                print("Expanding ldi mask to include these regions.")
                # we want zeros nans in ldi mask
                ldi_mask = ldi_mask & ~where_ldi_depth_nan_resized

    # (Naive blending)
    # TODO: (Antoine): I think the variable below should be inpainting_mask instead of missing_info_mask
    pcd_naive, blended_depth_naive = naive_blend_of_depths(
        colors=new_colors,
        warped_depth_interp=warped_depth_interp,
        depth_estimated=depth_estimated,
        missing_info_mask=missing_info_mask,
        pose=camera_pose,
        sphere_radius=sphere_radius,
        height=height*upsampling_factor,
        width=width*upsampling_factor,
        phase=_phase_current,
        logging=True,
        where_save=where_save
    )

    # (Harmonic blending)
    res = harmonic_blend_of_depths_ldi(
        colors=new_colors,
        warped_depth_interp=warped_depth_interp,
        depth_estimated=depth_estimated,
        missing_info_mask=missing_info_mask,
        pose=camera_pose,
        sphere_radius=sphere_radius,
        height=height*upsampling_factor,
        width=width*upsampling_factor,
        phase=_phase_current,
        logging=True,
        where_save=where_save,

        #ldi args
        ldi_depth=ldi_depth,
        ldi_colors=ldi_colors, 
        ldi_mask=ldi_mask,
    )

    res['pcd_naive'] = pcd_naive
    res['blended_depth_naive'] = blended_depth_naive

    return res

def split_new_points(pts, colors, pose1, pose2, forward):
    # (Antoine, 16 Oct) This function will pose problems if we want to do anything different than a straight line path.
    """
    Split points between points belonging to sphere1, sphere2, and neutral points.
    Points are distrbuted as follows:
        - pts on the left side of cam1 belongs to sphere 1
        - pts on the right side of cam2 belongs to sphere 2
        - pts in between are neutral points
    """
    cam_loc_1 = pose1[:3, 3]
    cam_loc_2 = pose2[:3, 3]
    where_sphere1 = is_point_in_camera_forward_space(pts, cam_loc_1, -forward)  # left of cam1
    where_sphere2 = is_point_in_camera_forward_space(pts, cam_loc_2, forward)   # right of cam2
    where_neutral = ~(where_sphere1 | where_sphere2)
    pts1, colors1 = pts[where_sphere1], colors[where_sphere1]
    pts2, colors2 = pts[where_sphere2], colors[where_sphere2]
    pts_neutral, colors_neutral = pts[where_neutral], colors[where_neutral]
    return (pts1, colors1), (pts2, colors2), (pts_neutral, colors_neutral)

def is_point_in_camera_forward_space(point_positions,
                                    camera_position,
                                    forward_vector,
                                    tolerance=1e-12):
    """
    Determine whether one or more 3D points lie in the half-space
    in front of the plane orthogonal to `forward_vector`
    passing through `camera_position`.

    Parameters
    ----------
    point_positions : array-like, shape (..., 3)
        One or more 3D points. Supports arbitrary leading batch dimensions.
    camera_position : array-like, shape (3,)
        The 3D location of the camera.
    forward_vector : array-like, shape (3,)
        The camera's forward direction vector (does not need to be normalized).
    tolerance : float, optional
        Numerical tolerance for deciding whether a point on the plane counts as "in front".

    Returns
    -------
    np.ndarray of bool
        Boolean array of shape (...) — True for points in the camera’s forward half-space,
        False for points behind it.
    """

    # Convert to arrays
    point_positions = np.asarray(point_positions, dtype=float)
    camera_position = np.asarray(camera_position, dtype=float)
    forward_vector = np.asarray(forward_vector, dtype=float)

    # Check that the forward vector is valid
    if np.allclose(forward_vector, 0):
        raise ValueError("forward_vector must be a non-zero vector.")

    # Vector(s) from camera to point(s) – broadcasting works automatically
    vectors_camera_to_points = point_positions - camera_position

    # Signed distance(s) along the forward direction
    signed_distances = np.sum(vectors_camera_to_points * forward_vector, axis=-1)

    # True if in or beyond the forward half-space
    return signed_distances >= -tolerance


if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "Antoine/F0_forest.yaml"]
    )
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    # -------------------------------------------------------------------- #
    # ---- PHASE 2-C ALIGN PAIRS OF SPHERES WITH HARMONIC BLENDING  ----- #
    # -------------------------------------------------------------------- #
    printc(f"=== [PHASE {_phase_current}]  EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase2c_from:
        printc(f"=== PHASE {_phase_current} : ALIGN PAIRS OF SPHERES WITH HARMONIC BLENDING ===", color='green')

        all_pts_world = np.array([]).reshape(0, 3)
        all_colors_world = np.array([]).reshape(0, 3)

        for i in range(1, config.num_dreams):
            print(f"--- {_phase_current}: Inpainting+Alignment {i:02d} / {config.num_dreams-1} ---")

            pointcloud_zoo = {}
            save_dir__ = save_dir_ / f"align_{i:02d}"
            os.makedirs(save_dir__ / _phase_current / ".cache", exist_ok=True)

            sphere1=my_utils.Sphere.instanciate_from_saved_dict(save_dir__ / _phase_2a / ".cache" / "sphere1.pkl")
            sphere2=my_utils.Sphere.instanciate_from_saved_dict(save_dir__ / _phase_2a / ".cache" / "sphere2.pkl")

            pose1=sphere1.pose
            pose2=sphere2.pose

            data =  np.load(save_dir__ / _phase_2a / ".cache" / "other_data.npy", allow_pickle=True).item()

            depth_estimated       = data['depth_estimated']
            pose_intermediate     = data['pose_intermediate']
            warped_img_interp     = data['warped_img_interp']
            warped_depth_interp   = data['warped_depth_interp']
            pano_rgb_inpainted    = data['pano_rgb_inpainted']
            missing_info_mask     = data['missing_info_mask']

            if config.phase2.apply_ldi:

                colors_bg, depth_bg, mask_bg = my_utils.load_rgbd_ldi_pano(
                    dream=i,
                    save_dir_=save_dir_,
                    phase=_phase_2b,
                )
                ldi_colors =   colors_bg
                ldi_depth  =   depth_bg
                ldi_mask   =   mask_bg
            
            else:
                ldi_colors   = None
                ldi_depth = None
                ldi_mask  = None

            res = align_new_points(
                warped_img_interp=warped_img_interp,
                warped_depth_interp=warped_depth_interp,
                pano_rgb_inpainted=pano_rgb_inpainted,
                depth_estimated=depth_estimated,
                missing_info_mask=missing_info_mask,
                camera_pose=pose_intermediate, 
                height=height,
                width=width,
                sphere_radius=config.sphere_radius,
                upsampling_factor=config.pcd_upsampling_factor,
                where_save=save_dir__,
                ldi_depth=ldi_depth,
                ldi_colors=ldi_colors,
                ldi_mask=ldi_mask,
            )
            new_pts=res['pts_out']
            new_colors=res['colors_out']

            if config.phase2.excessive_pcd_logging:
                pointcloud_zoo['blended_naive_w_excess'] = res['pcd_naive']
                pointcloud_zoo['blended_harmonic_w_excess'] = res['pcd_harmonic']
                pointcloud_zoo["blended_harmonic"] = my_utils.PointCloud(
                    pts=new_pts,
                    colors=new_colors
                )

            # (Optional) Remove outliers
            if config.phase2.outliers_removal.apply_on_fg:
                new_pts, new_colors = my_utils.GeometryTransforms.remove_statistical_outliers(
                    new_pts,
                    new_colors,
                    **config.phase2.outliers_removal.options
                )
                if config.phase2.excessive_pcd_logging:
                    pointcloud_zoo['blended_harmonic_outlier_removed'] = my_utils.PointCloud(
                        pts=new_pts,
                        colors=new_colors
                    )

            # 10. Add new points to their corresponding spheres.
            (new_pts1, new_colors1), (new_pts2, new_colors2), (new_pts_neutral, new_colors_neutral) = split_new_points(
                new_pts, new_colors, pose1, pose2, translation_direction
            )
            
            if config.phase2.apply_ldi:
                new_pts_ldi=res['pts_out_ldi']
                new_colors_ldi=res['colors_out_ldi']

                if config.phase2.excessive_pcd_logging:
                    pointcloud_zoo['blended_harmonic_ldi'] = my_utils.PointCloud(
                        pts=new_pts_ldi,
                        colors=new_colors_ldi
                    )

                # (Optional) Remove outliers for LDI points
                if config.phase2.outliers_removal.apply_on_ldi:
                    new_pts_ldi, new_colors_ldi = my_utils.GeometryTransforms.remove_statistical_outliers(
                        new_pts_ldi,
                        new_colors_ldi,
                        **config.phase2.outliers_removal.options
                    )

                    if config.phase2.excessive_pcd_logging:
                        pointcloud_zoo['blended_harmonic_ldi_outlier_removed'] = my_utils.PointCloud(
                            pts=new_pts_ldi,
                            colors=new_colors_ldi
                        )

                    

                (new_pts_1_ldi, new_colors_1_ldi), (new_pts_2_ldi, new_colors_2_ldi), (new_pts_neutral_ldi, new_colors_neutral_ldi) = split_new_points(
                    new_pts_ldi, new_colors_ldi, pose1, pose2, translation_direction
                )

                new_pts1           = np.concatenate((new_pts1,           new_pts_1_ldi), axis=0)
                new_colors1        = np.concatenate((new_colors1,        new_colors_1_ldi), axis=0)
                new_pts2           = np.concatenate((new_pts2,           new_pts_2_ldi), axis=0)
                new_colors2        = np.concatenate((new_colors2,        new_colors_2_ldi), axis=0)
                new_pts_neutral    = np.concatenate((new_pts_neutral,    new_pts_neutral_ldi), axis=0)
                new_colors_neutral = np.concatenate((new_colors_neutral, new_colors_neutral_ldi), axis=0)


            sphere1.add_new_points(my_utils.world2cam_carte_3D(new_pts1, pose1), new_colors1)
            sphere2.add_new_points(my_utils.world2cam_carte_3D(new_pts2, pose2), new_colors2)

            # Add all new points to world points, including inpainted+deformed points and points from the current dream.
            
            if config.phase2.excessive_pcd_logging:
                pointcloud_zoo['sphere1_init'] = sphere1.closed.get_world_pcd()
                pointcloud_zoo['sphere2_init'] = sphere2.closed.get_world_pcd()
            
            #10.a Points from sphere1
            if i == 1: # first iteration: sphere1 only has right opened
                if config.phase2.excessive_pcd_logging: pointcloud_zoo['sphere1_open'] = sphere1.right_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere1.right_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere1.right_opened.get_world_pcd().colors), axis=0)

            else: # later iterations: sphere1 has both opened
                if config.phase2.excessive_pcd_logging: pointcloud_zoo['sphere1_open'] = sphere1.both_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere1.both_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere1.both_opened.get_world_pcd().colors), axis=0)

            #10.b Neutral points
            all_pts_world = np.concatenate((all_pts_world, new_pts_neutral), axis=0)
            all_colors_world = np.concatenate((all_colors_world, new_colors_neutral), axis=0)

            #10.c Points from sphere2 (only last iter)
            if i == config.num_dreams - 1: 
                if config.phase2.excessive_pcd_logging: pointcloud_zoo['sphere2_open'] = sphere2.left_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere2.left_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere2.left_opened.get_world_pcd().colors), axis=0)
                assert np.allclose(pose2, pose_end), "Error in final camera pose computation"

            # save pcd
            with open(save_dir__  / f"{_phase_current}_pointclouds_zoo.pkl", 'wb') as f:
                pkl.dump(pointcloud_zoo, f)

        # END OF PHASE 2: final pcd save
        with open(save_dir_  / f"{_phase_current}_raw_dream_pcd.pkl", 'wb') as f:
            pkl.dump(
                my_utils.PointCloud(
                    pts=all_pts_world,
                    colors=all_colors_world
                ), f)

        print(f"PHASE {_phase_current} SUCCESSFULLY COMPLETED!")
    else:
        printc(f"SKIPPING PHASE {_phase_current}: ALIGN PAIRS OF SPHERES WITH HARMONIC BLENDING", color='magenta')
        printc(f"Loading instead from {config.load_phase2c_from}", color='magenta')

        source_phase2c_path = Path(config.save_dir) / config.load_phase2c_from
        dest_phase2c_path = Path(save_dir_)

        my_utils.copy_phase_folders(
            source_dir=source_phase2c_path,
            dest_dir=dest_phase2c_path,
            phase=_phase_current,
        )