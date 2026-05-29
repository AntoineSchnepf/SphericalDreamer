import os
import contextlib
from io import StringIO
import pipeline.bootstrap  # noqa: F401 - sets GLOG env vars and filters warnings

import sys
import numpy as np
from PIL import Image
from functools import partial
from scipy.ndimage import maximum_filter, minimum_filter
from scipy import ndimage as ndi
import time
import pickle as pkl
from pathlib import Path

# local imports
import my_utils
from my_utils import printc
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer
from render_pcd import render_v0

from pipeline.phases import PHASE_1A, PHASE_1B, PHASE_2A

_phase_1a = PHASE_1A
_phase_1b = PHASE_1B
_phase_current = PHASE_2A


def largest_connected_component(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """
    Return a binary mask containing only the largest connected component.

    Parameters
    ----------
    mask : np.ndarray
        2D (or ND) binary mask. Non-zero values are treated as foreground.
    connectivity : int
        For 2D: 1 => 4-connected, 2 => 8-connected.
        For 3D: 1 => 6-connected, 2 => 18-connected, 3 => 26-connected, etc.

    Returns
    -------
    np.ndarray
        Binary mask of the largest connected component (same shape as input).
    """
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = ndi.generate_binary_structure(mask.ndim, connectivity)
    labels, num = ndi.label(mask, structure=structure)

    # Count pixels per component (excluding background label 0)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = counts.argmax()

    return labels == largest_label

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
        current_ldi_mask,
        camera_pose, 
        height,
        width,
        spherical_dreamer, 
        skip_inpainting, 
        prompt,
        masking_operations,
        rendering_version,
        blending_mode,
        where_save
    ):

    res = {}

    # 4. Render points from sphere2 (opened right) + sphere2 (opened left), from the intermediate camera
    if rendering_version == 0:
        render_fn = render_v0
    elif rendering_version in (1, 2):
        raise NotImplementedError(f"Rendering version {rendering_version} is deprecated since 17/12/2025 for ldi mask rendering. Please use version 0.")
    else:
        raise ValueError(f"rendering_version {rendering_version} not recognized!")
    t0 = time.time()
    warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels, is_visited_ldi = render_fn(
        all_pts_world=current_points, 
        all_colors_world=current_colors, 
        all_ldi_mask=current_ldi_mask,
        pose=camera_pose,
        height=height,
        width=width
    )
    print(f"Rendered all points from intermediate camera in {time.time()-t0:.1f} seconds (Render v{rendering_version})!")

    # largest connected component on visited pixels minus LDI visited pixels
    visited_pixels_no_ldi = visited_pixels & (~is_visited_ldi)
    visited_pixels_no_ldi_only_inner = ~largest_connected_component(~visited_pixels_no_ldi)
    missing_info_mask = visited_pixels & visited_pixels_no_ldi_only_inner

    # 5. Get missing info mask
    missing_info_mask, missing_info_masks_tile = get_missing_info_mask(masking_operations, missing_info_mask) 
    where_depth_nan = np.isnan(warped_depth_interp)
    missing_info_mask = missing_info_mask | where_depth_nan
    inpainting_mask = missing_info_mask 
    
    if np.any(where_depth_nan & (~missing_info_mask)):
        print("IMPORTANT WARNING: depth has NaNs in non-missing info regions!")
    
    warped_img_interp[missing_info_mask] = np.nan
    warped_depth_interp[missing_info_mask] = np.nan
    
    # 6. Inpainting
    overlay_before_inpainting = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img_interp, inpainting_mask, alpha=0.5)) 

    in_height, in_width = config.phase2.inpainting_resolution.height, config.phase2.inpainting_resolution.width
    if not skip_inpainting: 
        pano_inpainted_raw = spherical_dreamer.inpaint_pano(
            prompt=prompt, 
            pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
            mask=my_utils.numpy_to_PIL(inpainting_mask),
            width=in_width,
            height=in_height,
        )
        # blending
        pano_rgb_hr = my_utils.opencv_resize(warped_img_interp, in_height, in_width, mode='bilinear')
        missing_info_mask_hr = my_utils.mask_resize(missing_info_mask, in_height, in_width)
        where_nan_mask = np.isnan(pano_rgb_hr).any(axis=-1)
        missing_info_mask_hr = missing_info_mask_hr | where_nan_mask

        pano_rgb_inpainted = spherical_dreamer.blend(
            pano_rgb=my_utils.numpy_to_PIL(pano_rgb_hr),
            pano_inpainted_raw=pano_inpainted_raw,
            missing_info_mask=my_utils.numpy_bool_to_pil_mask(missing_info_mask_hr),
            blending_mode=blending_mode, 
        ) 

        # save cache
        pano_inpainted_raw.save(where_save / _phase_current / ".cache" / "pano_rgb_inpainted_raw.png")
        pano_rgb_inpainted.save(where_save / _phase_current / ".cache" / "pano_rgb_inpainted.png")
    else:
        pano_inpainted_raw = Image.open(where_save / _phase_current / ".cache" / "pano_rgb_inpainted_raw.png")
        pano_rgb_inpainted = Image.open(where_save / _phase_current / ".cache" / "pano_rgb_inpainted.png")


    # 7. Estimate depth
    if not skip_inpainting:
        depth_estimated = spherical_dreamer.estimate_pano_depth(
            pano_rgb=np.array(pano_inpainted_raw)
        )
        np.save(where_save / _phase_current / ".cache" / "estimated_depth.npy", depth_estimated)
    else:
        depth_estimated = np.load(where_save / _phase_current / ".cache" / "estimated_depth.npy")
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

    # 1. Load RGBD pano
    colors, depth = my_utils.load_rgbd_pano(
        dream=dream,
        save_dir_=save_dir_,
        phase=_phase_1a
    )

    # 2. (Optional) upsampling
    if config.pcd_upsampling_factor>1:
        colors = my_utils.opencv_resize(colors, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        depth = my_utils.opencv_resize(depth, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
    
    pts_carte = my_utils.depth2cam_carte(
        depth=depth,
        sphere_radius=config.sphere_radius,
        height=height*config.pcd_upsampling_factor,
        width=width*config.pcd_upsampling_factor,
    )
    # 3. Outliers removal
    if config.phase1.outliers_removal.apply_on_fg:
        pts_carte, colors = my_utils.GeometryTransforms.remove_statistical_outliers(
            pts_carte,
            colors,
            **config.phase1.outliers_removal.options
        )

    _mask_fg = np.zeros(pts_carte.shape[:-1], dtype=bool)  # no LDI points here
    to_cat = [pts_carte.reshape(-1, 3)]
    to_cat_colors = [colors.reshape(-1, 3)]
    to_cat_ldi_mask = [_mask_fg.reshape(-1)] 

    # 4. (Optional) Load LDI background points and merge with foreground points
    if config.phase1.apply_ldi:
        colors_bg, depth_bg, mask_bg = my_utils.load_rgbd_ldi_pano(
            dream=dream,
            save_dir_=save_dir_,
            phase=_phase_1b
        )

        # resizing (upsampling or downsampling) is mandatory for LDI to make it match non-ldi images
        colors_bg = my_utils.opencv_resize(colors_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        depth_bg = my_utils.opencv_resize(depth_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor, mode='bilinear')
        mask_bg = my_utils.mask_resize(mask_bg, height*config.pcd_upsampling_factor, width*config.pcd_upsampling_factor)
                
        pts_carte_bg = my_utils.depth2cam_carte(
            depth=depth_bg,
            sphere_radius=config.sphere_radius,
            height=height*config.pcd_upsampling_factor,
            width=width*config.pcd_upsampling_factor,
        )
        valid_bg = ~np.isnan(pts_carte_bg).any(axis=-1)
        mask_bg = mask_bg & valid_bg
        pts_carte_bg = pts_carte_bg[mask_bg]
        colors_bg = colors_bg[mask_bg]

        # (optional) outliers removal on background points
        if config.phase1.outliers_removal.apply_on_ldi:
            pts_carte_bg, colors_bg = my_utils.GeometryTransforms.remove_statistical_outliers(
                pts_carte_bg,
                colors_bg,
                **config.phase1.outliers_removal.options
            )

        _mask_ldi = np.ones(pts_carte_bg.shape[:-1], dtype=bool)
        to_cat.append(pts_carte_bg.reshape(-1, 3))
        to_cat_colors.append(colors_bg.reshape(-1, 3))
        to_cat_ldi_mask.append(_mask_ldi.reshape(-1))

    # concatenate fg and bg points
    pts_carte, cat_meta = my_utils.concat_with_meta(*to_cat)    
    colors, _ = my_utils.concat_with_meta(*to_cat_colors)
    ldi_mask, _ = my_utils.concat_with_meta(*to_cat_ldi_mask)


    # 5. Correction pipeline
    pts_carte_corrected, colors_corrected, ldi_mask_corrected = my_utils.run_corrective_pipeline_on_sphere(
        pts_carte, # in cartesian coordinates
        colors, 
        ldi_mask, 
        **config.geometry_correction.sphere
    )
    sphere = my_utils.Sphere(
        None, pts_carte_corrected, colors_corrected, 
        ldi_mask=ldi_mask_corrected,
        forward_carte=translation_direction,
        opening_kwargs=config.world_opening,
    )

    return sphere

# this scripts:
# - creates high resolution sphere with optional LDI points and saves them in the cache for the final pcd
# - aligns pairs of spheres with inpainting at intermediate views
# - saves inpaintings and their estimated depth in the cache for phase 2.B and 2.C
    
if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(debug=False)
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
    printc(f"=== [PHASE {_phase_current}]  EXPERIMENT: {config.expname} ===", color='cyan')
    if not config.load_phase2a_from:
        printc(f"=== PHASE {_phase_current} : ALIGN PAIRS OF SPHERES WITH INPAINTING ===", color='green')
        
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
            printc(f"--- {_phase_current}: Inpainting Phase {i:02d} / {config.num_dreams-1} ---", color='yellow')
            save_dir__ = save_dir_ / f"align_{i:02d}"
            os.makedirs(save_dir__ / _phase_current / ".cache", exist_ok=True)


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
            s1_pts = sphere1.right_opened.get_world_pcd().pts
            s1_colors = sphere1.right_opened.get_world_pcd().colors
            s1_ldi_mask = sphere1.right_opened.get_world_pcd().ldi_mask

            # same for sphere2
            s2_pts = sphere2.left_opened.get_world_pcd().pts
            s2_colors = sphere2.left_opened.get_world_pcd().colors
            s2_ldi_mask = sphere2.left_opened.get_world_pcd().ldi_mask


            current_points=np.concatenate((
                s1_pts, s2_pts
            ), axis=0)
            current_colors=np.concatenate((
                s1_colors, s2_colors
            ), axis=0)
            current_ldi_mask=np.concatenate((
                s1_ldi_mask, s2_ldi_mask
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
                current_ldi_mask=current_ldi_mask,
                camera_pose=pose_intermediate,
                height=height,
                width=width,
                spherical_dreamer=spherical_dreamer, 
                skip_inpainting=config.phase2.skip_inpainting, 
                prompt=config.prompt,
                masking_operations=masking_operations,
                rendering_version=config.phase2.rendering_version,
                blending_mode=config.phase2.inpainting_blend_mode,
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

            # 5. Save cache data for phase 2.B and 2.C
            sphere1.save_dict(save_dir__ / _phase_current / ".cache"/ "sphere1.pkl")
            sphere2.save_dict(save_dir__ / _phase_current / ".cache"/ "sphere2.pkl")
            np.save(save_dir__ / _phase_current / ".cache"/ "other_data.npy", {
                'pose_intermediate'    : pose_intermediate,
                'warped_img_interp'    : warped_img_interp,
                'warped_depth_interp'  : warped_depth_interp,
                'missing_info_mask'    : missing_info_mask,
                'pano_rgb_inpainted'   : pano_rgb_inpainted,
                'depth_estimated'      : depth_estimated,
            })

            # 6. Save viz images
            my_utils.numpy_to_PIL(warped_img)                      .save(save_dir__ / _phase_current / "01_warped_img.png")
            my_utils.numpy_to_PIL(warped_img_interp)               .save(save_dir__ / _phase_current / "01_warped_img_interp.png")

            my_utils.depth_numpy_to_figure(warped_depth)        .savefig(save_dir__ / _phase_current / "02_warped_depth.png")
            my_utils.depth_numpy_to_figure(warped_depth_interp) .savefig(save_dir__ / _phase_current / "02_warped_depth_interp.png")

            missing_info_masks_tile                                .save(save_dir__ / _phase_current / "03_missing_info_masks_tile.png")
            overlay_before_inpainting                              .save(save_dir__ / _phase_current / "04_overlay_before_inpainting.png")
            pano_inpainted_raw                                     .save(save_dir__ / _phase_current / "05_pano_rgb_inpainted_raw.png")
            pano_rgb_inpainted                                     .save(save_dir__ / _phase_current / "06_pano_rgb_inpainted.png")
            my_utils.depth_numpy_to_figure(depth_estimated)     .savefig(save_dir__ / _phase_current / "07_estimated_depth.png")


            # 7. END: Adjust sphere1 to be sphere2 for next iteration
            sphere1 = sphere2
            pose1 = pose2


        printc(f"=== PHASE {_phase_current} SUCCESSFULLY COMPLETED! ===", color='green')
    else:
        printc(f"SKIPPING PHASE {_phase_current}: ALIGN PAIRS + INPAINT", color='magenta')
        printc(f"Loading instead from {config.load_phase2a_from}", color='magenta')
        
        source_phase2a_path = Path(config.save_dir) / config.load_phase2a_from
        dest_phase2a_path = Path(save_dir_)

        my_utils.copy_phase_folders(
            source_dir=source_phase2a_path,
            dest_dir=dest_phase2a_path,
            phase=_phase_current,
        )