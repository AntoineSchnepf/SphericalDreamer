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
from PIL import Image, ImageOps
import copy
from functools import partial
from skimage.segmentation import find_boundaries
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib.pyplot as plt
import time
import pickle as pkl
from prodict import Prodict
import pyfiglet
import argparse
from pathlib import Path
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 

import my_utils
from my_utils import printc
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer
    from utils.depth_alignment import Pano_depth_estimation
from render_pcd import render_v2, render_v1, render_v0

output_prefix = "02a_"


def get_missing_info_mask(operations, visited_pixels):
    missing_info_masks = [~visited_pixels]
    for op in operations:
        missing_info_masks.append(op(missing_info_masks[-1]))
    missing_info_mask = missing_info_masks[-1]
    missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
    return missing_info_mask, missing_info_masks_tile

def render_and_inpaint_from_pose(
        current_points, 
        current_colors, 
        camera_pose, 
        height,
        width,
        spherical_dreamer, 
        skip_inpainting, 
        prompt,
        masking_operations,
        rendering_version,
        where_save
    ):

    res = {}

    # 4. Render points from sphere2 (opened right) + sphere2 (opened left), from the intermediate camera
    if rendering_version==0:
        render_fn = render_v0
    elif rendering_version==1:
        render_fn = render_v1
    elif rendering_version==2:
        render_fn = render_v2
    else:
        raise ValueError(f"rendering_version {rendering_version} not recognized!")
    t0 = time.time()
    warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels = render_fn(
        all_pts_world=current_points, 
        all_colors_world=current_colors, 
        pose=camera_pose,
        height=height,
        width=width
    )
    print(f"Rendered all points from intermediate camera in {time.time()-t0:.1f} seconds (Render v{rendering_version})!")


    # 5. Get missing info mask
    missing_info_mask, missing_info_masks_tile = get_missing_info_mask(masking_operations, visited_pixels) 
    where_depth_nan = np.isnan(warped_depth_interp)
    missing_info_mask = missing_info_mask | where_depth_nan
    inpainting_mask = missing_info_mask # TODO: (Antoine, 14 oct) The inpainting mask is currently composed of both <<large missing regions due to limited covering of the main spheres>> and <<small holes due to occlusions>>. We could separate these two cases and do something neater?.

    if np.any(where_depth_nan & (~missing_info_mask)):
        print("IMPORTANT WARNING: depth has NaNs in non-missing info regions!")
    
    warped_img_interp[missing_info_mask] = np.nan
    warped_depth_interp[missing_info_mask] = np.nan
    
    # 6. Inpainting
    overlay_before_inpainting = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img_interp, inpainting_mask, alpha=0.5)) 

    if not skip_inpainting: 
        pano_inpainted_raw = spherical_dreamer.inpaint_pano(
            prompt=prompt, 
            pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
            mask=my_utils.numpy_to_PIL(inpainting_mask),
            width=config.phase2.inpainting_resolution.width,
            height=config.phase2.inpainting_resolution.height,
        ).resize((width, height), resample=Image.LANCZOS)
        # blending
        pano_rgb_inpainted = spherical_dreamer.blend(
            pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
            pano_inpainted_raw=pano_inpainted_raw,
            missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
            blending_mode='compose', #TODO: add to cfg
        ) #TODO: Check the blending strategy again. Maybe seamless is better ? 
        pano_inpainted_raw.save(os.path.join(where_save, output_prefix+"XX_pano_rgb_inpainted_raw.png"))
        pano_rgb_inpainted.save(os.path.join(where_save, output_prefix+"XX_pano_rgb_inpainted.png"))
    else:
        pano_inpainted_raw = Image.open(os.path.join(where_save, output_prefix+"XX_pano_rgb_inpainted_raw.png"))
        pano_rgb_inpainted = Image.open(os.path.join(where_save, output_prefix+"XX_pano_rgb_inpainted.png"))



    # 7. Estimate depth
    # TODO: (Antoine, 16 OCT) LayerPANO3D Has a depth inpainting model, which may be better than this + harmonic blending. Worth testing.
    if not skip_inpainting:
        depth_estimated = spherical_dreamer.estimate_pano_depth(
            pano_rgb=np.array(pano_rgb_inpainted)
        )
        np.save(os.path.join(where_save, output_prefix+"XX_estimated_depth.npy"), depth_estimated)
    else:
        depth_estimated = np.load(os.path.join(where_save, output_prefix+"XX_estimated_depth.npy"))
    # depth_estimated=np.ones_like(depth_estimated) * sphere_radius  
    # print("WARNING: estimated depth override to ones for debugging purposes")
    
    res['warped_img'] = warped_img
    res['warped_depth'] = warped_depth
    res['warped_img_interp'] = warped_img_interp
    res['warped_depth_interp'] = warped_depth_interp
    res['visited_pixels'] = visited_pixels
    res['missing_info_mask'] = missing_info_mask
    res['missing_info_masks_tile'] = missing_info_masks_tile
    res['inpainting_mask'] = inpainting_mask
    res['overlay_before_inpainting'] = overlay_before_inpainting
    res['pano_inpainted_raw'] = pano_inpainted_raw
    res['pano_rgb_inpainted'] = pano_rgb_inpainted
    res['depth_estimated'] = depth_estimated
    
    return res

def get_sphere(dream, save_dir_, config, height, width): 

    colors, depth = my_utils.load_rgbd_pano(
        dream=dream,
        save_dir_=save_dir_
    )
    colors_bg, depth_bg, mask_bg = my_utils.load_rgbd_ldi_pano(
            dream=dream,
            save_dir_=save_dir_,
            phase=1
    )
        
    # optionnal upsampling
    if config.pcd_upsampling_factor>1:
        colors = my_utils.opencv_resize(colors, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        depth = my_utils.opencv_resize(depth, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
    
        colors_bg = my_utils.opencv_resize(colors_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        depth_bg = my_utils.opencv_resize(depth_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        mask_bg = my_utils.mask_resize(mask_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor)
    
    pts_carte = my_utils.depth2cam_carte(
        depth=depth,
        sphere_radius=config.sphere_radius,
        height=height*config.pcd_upsampling_factor,
        width=width*config.pcd_upsampling_factor,
    )

    pts_carte_bg = my_utils.depth2cam_carte(
        depth=depth_bg,
        sphere_radius=config.sphere_radius,
        height=height*config.pcd_upsampling_factor,
        width=width*config.pcd_upsampling_factor,
    )
    pts_carte_bg = pts_carte_bg[mask_bg]
    colors_bg = colors_bg[mask_bg]
    pts_carte = np.concatenate((pts_carte.reshape(-1, 3), pts_carte_bg.reshape(-1, 3)), axis=0)
    colors =  np.concatenate((colors.reshape(-1, 3), colors_bg.reshape(-1, 3)), axis=0)


    # correction pipeline
    pts_carte_corrected, colors_corrected = my_utils.run_corrective_pipeline_on_sphere(
        pts_carte, # in cartesian coordinates
        colors, 
        height*config.pcd_upsampling_factor,
        width*config.pcd_upsampling_factor, 
        **config.geometry_correction.sphere
    )
    sphere = my_utils.Sphere(
        None, pts_carte_corrected, colors_corrected, 
        forward_carte=translation_direction,
        opening_kwargs=config.world_opening,
    )

    return sphere

    
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

    # ------------------------------------------------------------ #
    # ---- PHASE 2-A ALIGN PAIRS OF SPHERES WITH INPAINTING  ---- #
    # ------------------------------------------------------------ #
    printc(f"=== [PHASE 2-A]  EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase2a_from:
        printc(f"=== PHASE 2-A : ALIGN PAIRS OF SPHERES WITH INPAINTING ===", color='green')
        
        # INIT: load data for sphere1
        sphere1 = get_sphere(
            dream=0,
            save_dir_=save_dir_,
            config=config,
            height=height,
            width=width
        )
        pose1 = pose_init
        sphere1.update_pose(pose1)

        # LOOP
        for i in range(1, config.num_dreams):
            printc(f"--- 2-A: Inpainting+Alignment Phase {i:02d} / {config.num_dreams-1} ---", color='yellow')
            save_dir__ = os.path.join(save_dir_, f"align_{i:02d}")
            os.makedirs(save_dir__, exist_ok=True)

            # 1. Load data for sphere2
            sphere2 = get_sphere(
                dream=i,
                save_dir_=save_dir_,
                config=config,
                height=height,
                width=width
            )

            # 2. Move camera
            pose2 = my_utils.camera_translation(pose1, config.delta_walk * translation_direction)
            sphere2.update_pose(pose2)


            # 3. Go to intermediate camera (between cam1 and cam2)
            pose_intermediate = my_utils.camera_translation(pose2, -config.delta_walk/2 * translation_direction)
            pose_intermediate_bis = my_utils.camera_translation(pose1, config.delta_walk/2 * translation_direction)
            
            assert np.allclose(pose_intermediate, pose_intermediate_bis), "Error in camera intermediate pose computation"
            pose_intermediate = my_utils.camera_translation(pose_intermediate, np.array([0, 0, config.raise_intermediate_camera_by_z]))
            rotation_before_inpainting = my_utils.rotation_matrix_z(config.phase2.rotate_intermediate_camera_by_deg * np.pi / 180) 
            pose_intermediate[:3, :3] = rotation_before_inpainting @ pose_intermediate[:3, :3]


            # 4. Generate missing points from pose, inpaint, estimate depth (inside function below)
            current_points=np.concatenate((
                sphere1.right_opened.get_world_pcd().pts, sphere2.left_opened.get_world_pcd().pts
            ), axis=0)
            current_colors=np.concatenate((
                sphere1.right_opened.get_world_pcd().colors, sphere2.left_opened.get_world_pcd().colors
            ), axis=0)
            masking_operations = [
                partial(minimum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                # partial(maximum_filter, size=(8, 8), axes=(0,1)),
            ]
            res_render_inpaint = render_and_inpaint_from_pose(
                current_points=current_points, 
                current_colors=current_colors, 
                camera_pose=pose_intermediate,
                height=height,
                width=width,
                spherical_dreamer=spherical_dreamer, 
                skip_inpainting=config.phase2.skip_inpainting, 
                prompt=config.prompt,
                masking_operations=masking_operations,
                rendering_version=config.phase2.rendering_version,
                where_save=save_dir__,
            )

            warped_img                = res_render_inpaint['warped_img']
            warped_depth              = res_render_inpaint['warped_depth']
            warped_img_interp         = res_render_inpaint['warped_img_interp']
            warped_depth_interp       = res_render_inpaint['warped_depth_interp']
            visited_pixels            = res_render_inpaint['visited_pixels']
            missing_info_mask         = res_render_inpaint['missing_info_mask']
            missing_info_masks_tile   = res_render_inpaint['missing_info_masks_tile']
            inpainting_mask           = res_render_inpaint['inpainting_mask']
            overlay_before_inpainting = res_render_inpaint['overlay_before_inpainting']
            pano_inpainted_raw        = res_render_inpaint['pano_inpainted_raw']
            pano_rgb_inpainted        = res_render_inpaint['pano_rgb_inpainted']
            depth_estimated           = res_render_inpaint['depth_estimated']

            # 5. Save YY data reserved for phase 2.B
            sphere1.save_dict(os.path.join(save_dir__, output_prefix+"YY_sphere1.pkl"))
            sphere2.save_dict(os.path.join(save_dir__, output_prefix+"YY_sphere2.pkl"))
            np.save(f"{save_dir__}/{output_prefix}YY_other.npy", {
                'pose_intermediate'    : pose_intermediate,
                'warped_img_interp'    : warped_img_interp,
                'warped_depth_interp'  : warped_depth_interp,
                'missing_info_mask'    : missing_info_mask,
                'pano_rgb_inpainted'   : pano_rgb_inpainted,
                'depth_estimated'      : depth_estimated,
            })

            # 6. Save debug images
            my_utils.numpy_to_PIL(warped_img)           .save(os.path.join(save_dir__, output_prefix+"01_warped_img.png"))
            my_utils.numpy_to_PIL(warped_img_interp)    .save(os.path.join(save_dir__, output_prefix+"01_warped_img_interp.png"))

            my_utils.depth_numpy_to_figure(warped_depth)        .savefig(os.path.join(save_dir__, output_prefix+"02_warped_depth.png"))
            my_utils.depth_numpy_to_figure(warped_depth_interp) .savefig(os.path.join(save_dir__, output_prefix+"02_warped_depth_interp.png"))
            missing_info_masks_tile                      .save(   os.path.join(save_dir__, output_prefix+"03_missing_info_masks_tile.png"))
            overlay_before_inpainting                    .save(   os.path.join(save_dir__, output_prefix+"04_overlay_before_inpainting.png"))
            pano_inpainted_raw                           .save(   os.path.join(save_dir__, output_prefix+"05_pano_rgb_inpainted_raw.png"))
            pano_rgb_inpainted                           .save(   os.path.join(save_dir__, output_prefix+"06_pano_rgb_inpainted.png"))
            my_utils.depth_numpy_to_figure(depth_estimated) .savefig(os.path.join(save_dir__, output_prefix+"07_estimated_depth.png"))


            # 7. END: Adjust sphere1 to be sphere2 for next iteration
            sphere1 = sphere2
            pose1 = pose2


        printc("=== PHASE 2-A SUCCESSFULLY COMPLETED! ===", color='green')
    else:
        printc("SKIPPING PHASE 2-A: ALIGN PAIRS + INPAINT", color='magenta')
        printc(f"Loading instead from {config.load_phase2a_from}", color='magenta')
        source_phase2a_path = Path(config.save_dir) / config.load_phase2a_from
        dest_phase2a_path = Path(save_dir_)
        my_utils.copy_phase_folders(
            folder_start_with="align_",
            item_start_with=output_prefix,
            source_dir=source_phase2a_path,
            dest_dir=dest_phase2a_path
        )