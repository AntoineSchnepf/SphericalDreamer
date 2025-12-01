import os
import sys
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
from utils.depth_alignment import Pano_depth_estimation
from render_pcd import render_v2
import my_utils
from sphericaldreamer import SphericalDreamer

logging.disable(logging.CRITICAL + 1)

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
        skip_inpainting, 
        prompt,
        masking_operations,
        where_save
    ):

    res = {}

    # 4. Render points from sphere2 (opened right) + sphere2 (opened left), from the intermediate camera
    warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels = render_v2(
        all_pts_world=current_points, 
        all_colors_world=current_colors, 
        pose=camera_pose,
        height=height,
        width=width
    )
    print("Rendered all points from intermediate camera!")


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
            mask=my_utils.numpy_to_PIL(inpainting_mask)
        )
        pano_inpainted_raw.save(os.path.join(where_save, "XX_pano_rgb_inpainted_raw.png"))
    else:
        pano_inpainted_raw = Image.open(os.path.join(where_save, "XX_pano_rgb_inpainted_raw.png"))


    # 7. Inpainting seamless blending
    pano_rgb_inpainted = spherical_dreamer.blend(
        pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
        pano_inpainted_raw=pano_inpainted_raw,
        missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
        blending_mode='compose', #TODO: add to cfg
    ) #TODO: Check the blending strategy again. Maybe seamless is better ? 


    # 8. Estimate depth
    # TODO: (Antoine, 16 OCT) LayerPANO3D Has a depth inpainting model, which may be better than this + harmonic blending. Worth testing.
    if not skip_inpainting:
        depth_estimated = spherical_dreamer.estimate_pano_depth(
            pano_rgb=np.array(pano_rgb_inpainted)
        )
        np.save(os.path.join(where_save, "XX_estimated_depth.npy"), depth_estimated)
    else:
        depth_estimated = np.load(os.path.join(where_save, "XX_estimated_depth.npy"))
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

    # ------------------------------------------------------------ #
    # ---- PHASE II.a ALIGN PAIRS OF SPHERES WITH INPAINTING  ---- #
    # ------------------------------------------------------------ #
    print(f"=== EXPERIMENT: {config.expname} ===")
    if not config.skip_phase2a:
        print(f"=== {config.expname}: PHASE II.a : ALIGN PAIRS OF SPHERES WITH INPAINTING ===")
        # PHASE II.a: INIT

        colors1, depth1 = my_utils.load_rgbd_pano(
            dream=0,
            save_dir_=save_dir_
        )

        # Optional upsampling step to increase pointcloud density
        if config.pcd_upsampling_factor>1:
            colors1 = my_utils.opencv_resize(colors1, height*2, width*2, mode='bilinear')
            depth1 = my_utils.opencv_resize(depth1, height*2, width*2, mode='bilinear')

        pts1_carte = my_utils.depth2cam_carte(
            depth=depth1,
            sphere_radius=config.sphere_radius,
            height=height*config.pcd_upsampling_factor,
            width=width*config.pcd_upsampling_factor,
        )
        pts1_carte_corrected, colors1_corrected = my_utils.run_corrective_pipeline_on_sphere(
            pts1_carte, # in cartesian coordinates
            colors1, 
            height*config.pcd_upsampling_factor,
            width*config.pcd_upsampling_factor, 
            **config.geometry_correction.sphere
        )
        sphere1 = my_utils.Sphere(
            None, pts1_carte_corrected, colors1_corrected, 
            forward_carte=translation_direction,
            opening_kwargs=config.world_opening,
        )
        pose1 = pose_init
        sphere1.update_pose(pose1)

        # PHASE II.a: LOOP
        for i in range(1, config.num_dreams):
            print(f"--- Inpainting+Alignment Phase {i:02d} / {config.num_dreams-1} ---")
            save_dir__ = os.path.join(save_dir_, f"align_{i:02d}")
            os.makedirs(save_dir__, exist_ok=True)

            # 1. Load new sphere and open it (left)
            colors2, depth2 = my_utils.load_rgbd_pano(
                dream=i,
                save_dir_=save_dir_
            )
            # Optional upsampling step to increase pointcloud density
            if config.pcd_upsampling_factor>1:
                colors2 = my_utils.opencv_resize(colors2, height*2, width*2, mode='bilinear')
                depth2 = my_utils.opencv_resize(depth2, height*2, width*2, mode='bilinear')

            pts2_carte = my_utils.depth2cam_carte(
                depth=depth2,
                sphere_radius=config.sphere_radius,
                height=height*config.pcd_upsampling_factor,
                width=width*config.pcd_upsampling_factor,
            ) 
            pts2_carte_corrected, colors2_corrected = my_utils.run_corrective_pipeline_on_sphere(
                pts2_carte, 
                colors2, 
                height*config.pcd_upsampling_factor, 
                width*config.pcd_upsampling_factor, 
                **config.geometry_correction.sphere
            )
            
            sphere2 = my_utils.Sphere(
                None, pts2_carte_corrected, colors2_corrected, 
                forward_carte=translation_direction,
                opening_kwargs=config.world_opening,
            )

            # 2. Move camera
            pose2 = my_utils.camera_translation(pose1, config.delta_walk * translation_direction)
            sphere2.update_pose(pose2)


            # 3. Go to intermediate camera (between cam1 and cam2)
            pose_intermediate = my_utils.camera_translation(pose2, -config.delta_walk/2 * translation_direction)
            pose_intermediate_bis = my_utils.camera_translation(pose1, config.delta_walk/2 * translation_direction)
            
            assert np.allclose(pose_intermediate, pose_intermediate_bis), "Error in camera intermediate pose computation"
            pose_intermediate = my_utils.camera_translation(pose_intermediate, np.array([0, 0, config.raise_intermediate_camera_by_z]))

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
                skip_inpainting=config.phase2.skip_inpainting, 
                prompt=config.prompt,
                masking_operations=masking_operations,
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
            sphere1.save_dict(os.path.join(save_dir__, "YY_sphere1.pkl"))
            sphere2.save_dict(os.path.join(save_dir__, "YY_sphere2.pkl"))
            np.save(f"{save_dir__}/YY_other.npy", {
                'pose_intermediate'    : pose_intermediate,
                'warped_img_interp'    : warped_img_interp,
                'warped_depth_interp'  : warped_depth_interp,
                'missing_info_mask'    : missing_info_mask,
                'pano_rgb_inpainted'   : pano_rgb_inpainted,
                'depth_estimated'      : depth_estimated,
            })

            # 6. Save debug images
            my_utils.numpy_to_PIL(warped_img)           .save(os.path.join(save_dir__, "01_warped_img.png"))
            my_utils.numpy_to_PIL(warped_img_interp)    .save(os.path.join(save_dir__, "01_warped_img_interp.png"))

            my_utils.depth_numpy_to_figure(warped_depth)        .savefig(os.path.join(save_dir__, "02_warped_depth.png"))
            my_utils.depth_numpy_to_figure(warped_depth_interp) .savefig(os.path.join(save_dir__, "02_warped_depth_interp.png"))

            missing_info_masks_tile                      .save(   os.path.join(save_dir__, "03_missing_info_masks_tile.png"))
            overlay_before_inpainting                    .save(   os.path.join(save_dir__, "04_overlay_before_inpainting.png"))
            pano_inpainted_raw                           .save(   os.path.join(save_dir__, "05_pano_rgb_inpainted_raw.png"))
            pano_rgb_inpainted                           .save(   os.path.join(save_dir__, "06_pano_rgb_inpainted.png"))

            my_utils.depth_numpy_to_figure(depth_estimated) .savefig(os.path.join(save_dir__, "07_estimated_depth.png"))


            # 7. END: Adjust sphere1 to be sphere2 for next iteration
            sphere1 = sphere2
            pose1 = pose2


        print("=== PHASE II.a SUCCESSFULLY COMPLETED! ===")
    else:
        print("=== SKIPPING PHASE II.a: ALIGN PAIRS + INPAINT ===")
