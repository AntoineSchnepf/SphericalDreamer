# in this script, I want to start with a unit sphere and then do the unfolding, caera transolation and warping, to see what kind of mask I wnd up with


# load generated panorama + estimated depth map
import os
import sys
import cv2
import matplotlib
from src.lama.saicinpainting.training.data import masks
from src.pipeline_flux import FluxPipeline
from src.pipeline_flux_fill import FluxFillPipeline
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import torch
import numpy as np
from PIL import Image, ImageOps
import copy
from functools import partial
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.segmentation import find_boundaries
import time
import pickle as pkl
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
from spherical_dreamer_recurse import get_norm_vector, unfold_points, SphericalDreamer
import my_utils


logging.disable(logging.CRITICAL + 1)

class PointCloud:
    def __init__(self, pts, colors):
        """
        pts: np.array of shape [N, 3]
        colors: np.array of shape [N, 3] with values in [0-1]
        """
        self.pts = pts
        self.colors = colors

def get_pointcloud(colors, points_3D_world_carte):

    if isinstance(colors, Image.Image):
        colors = np.array(colors).reshape(-1, 3)/255.0

    pcd = PointCloud(
        pts=points_3D_world_carte.reshape(-1, 3),
        colors=colors.reshape(-1, 3)
    )
    return pcd

if __name__ == "__main__":

    # ---- args ----
    save_dir = "OUTPUTS/SphericalDreamerRecurseExploration"
    expname = "test_unit_circle_dream_w_blend"
    remove_forward_horizon = False
    do_inpainting  = False
    # dreaming args
    num_dreams = 2
    translation_direction = get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    sphere_radius = 1.0
    delta_walk = sphere_radius * np.pi / 2
    width = 1440
    height = 720
    # ---------------


    # 0. Initialization
    pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    pointclouds = {}
    i=1
    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir = '/tmp/pano_depth_temp'
    )

    save_dir_ = f"{save_dir}/{expname}"

    pts_world = []
    colors_world = []
    # forward alignment

    os.makedirs(os.path.join(save_dir_), exist_ok=True)

    # EACH DREAM RESULTS IN 1 IMAGE + 1 DEPTH


    # Init already done
    # 1/2. Generate initial panorama + depth
    pano_rgb = my_utils.unique_gradient_image(width, height)  # each pixel has a unique color
    pano_rgb.save(f"{save_dir_}/00_initial_pano.png")
    depth1 = np.ones((height, width), dtype=np.float32) * sphere_radius  # unit sphere
    my_utils.depth_numpy_to_PIL(depth1).save(f"{save_dir_}/00_initial_depth.png")
    my_utils.depth_numpy_to_figure(depth1).savefig(f"{save_dir_}/00_depth_figure.png")

    # 4. Apply geometrical transformation to the world points 
    points_2D_cam1_erp = np.stack((np.meshgrid(range(width), range(height))), axis=-1) 
    points_3D_cam1_sph = my_utils.cam_erp_to_cam_sph_3D(
        points_2D_cam1_erp, height, width, depth1, sphere_radius=sphere_radius
    )
    translation_direction_sph = my_utils.carte2sph_3D(translation_direction)
    points_3D_cam1_sph_unfolded = unfold_points(
        forward_sph=translation_direction_sph,
        delta=np.deg2rad(180),
        pts_sph=points_3D_cam1_sph
    )

    # 4. Project to the World and save point cloud
    points_3D_world_carte = my_utils.cam_sph_to_world_3D(
        points_3D_cam1_sph_unfolded, pose
    )
    pc1 = PointCloud(
        pts=points_3D_world_carte.reshape(-1, 3),
        colors=np.array(pano_rgb).reshape(-1, 3)/255.0
    )
    pointclouds[f"dream_{i-1:02d}"] = pc1

    # 5. Reproject to new camera pose (cam2)
    pose2 = pose.copy()
    pose2[:3, 3] += delta_walk * translation_direction  
    points_3D_cam2_carte = np.einsum(
        'ij,...j->...i', 
        np.linalg.inv(pose2), 
        my_utils.cat_ones(points_3D_world_carte)
    )[..., :3]  
    points_3D_cam2_sph = my_utils.carte2sph_3D(points_3D_cam2_carte)  
    points_2D_cam2_erp = my_utils.sph2erp_2D(points_3D_cam2_sph, height, width)  # [H, W, 2]
    depth_cam2 = points_3D_cam2_sph[..., 2]

    # 6. Depth aware splatting
    # TODO: make this better splatting. It should do the bilinear interpolation of neighbouring pixel (i.e. use a filter.)
    warped_img, warped_depth, flow, visited_pixels = my_utils.depth_aware_naive_splatting_vectorized(
        colors1=np.array(pano_rgb)/255.0,  # normalize to [0,1]
        coord_cam1=points_2D_cam1_erp,
        coord_cam2=points_2D_cam2_erp,
        depth_cam2=depth_cam2,
        height=height,
        width=width,
    )
    my_utils.numpy_to_PIL(warped_img).save(f"{save_dir_}/04_warped_img.png")
    warped_depth[~visited_pixels] = 0 
    my_utils.depth_numpy_to_PIL(warped_depth).save(f"{save_dir_}/04_warped_depth.png")

    # 6 bis (optional) Retrieving forward horizon mask in the warped image
    if remove_forward_horizon:
        print("Removing forward horizon")
        horizon_mask_params = {
            "height_p": 0.5,
            "width_p": 0.25
        }
        horizon_mask = np.zeros(shape=(height, width), dtype=bool) 
        h0 = height * (1 - horizon_mask_params['height_p'])/2
        h1 = height * (1 + horizon_mask_params['height_p'])/2
        w0 = width * (1 - horizon_mask_params['width_p'])/2
        w1 = width * (1 + horizon_mask_params['width_p'])/2

        horizon_mask[int(h0):int(h1), int(w0):int(w1)] = True
    else: 
        horizon_mask = np.zeros(shape=(height, width), dtype=bool)

    horizon_mask_warped = np.zeros(shape=(height, width), dtype=bool)
    horizon_points_2D_cam2_erp = np.round(points_2D_cam2_erp[horizon_mask]).astype("int")
    horizon_mask_warped[
        horizon_points_2D_cam2_erp[:, 1], horizon_points_2D_cam2_erp[:, 0]
    ] = True
    closed_horizon_mask_warped = my_utils.close_mask(my_utils.fill_mask(horizon_mask_warped), size=10)

    # 7. Interpolate 
    warped_img_interp, warped_depth_interp = my_utils.interpolate_with_flow(
        colors=np.array(pano_rgb)/255.0, 
        depths=depth_cam2, 
        flow=flow,
        mode='rounded'
    )

    # 8. Obtain mask
    operations = [
        partial(my_utils.fill_mask, flip=True), 
        # partial(my_utils.close_mask, size=10, flip=True),
    ]
    missing_info_masks = [~visited_pixels]
    for op in operations:
        missing_info_masks.append(op(missing_info_masks[-1]))
    missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
    missing_info_masks_tile.save(f"{save_dir_}/06_missing_info_masks_tile.png")
    missing_info_mask = missing_info_masks[-1]
    inpainting_mask = missing_info_mask | closed_horizon_mask_warped

    # 7 (end) Remove pixels that should not be interpolated
    warped_img_interp[missing_info_mask] = np.nan
    warped_depth_interp[missing_info_mask] = np.nan
    my_utils.numpy_to_PIL(warped_img_interp).save(f"{save_dir_}/05_warped_img_interp.png")
    my_utils.depth_numpy_to_PIL(warped_depth_interp).save(f"{save_dir_}/05_warped_depth_interp.png")
    my_utils.depth_numpy_to_figure(warped_depth_interp).savefig(f"{save_dir_}/05_warped_depth_interp_figure.png")
    np.save(f"{save_dir_}/05_warped_depth_interp.npy", warped_depth_interp)

    

    # 9. Inpaint panorama: RGB pixels with different hue
    if do_inpainting:
        overlay_before = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img, inpainting_mask, alpha=0.5))
        overlay_before.save(f"{save_dir_}/07_overlay_before_inpainting.png")
        pano_inpainted_raw = spherical_dreamer.inpaint_pano(
                prompt='Checkerboard-style color gradient pattern with smooth transitions of blue and green', 
                pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
                mask=my_utils.numpy_to_PIL(inpainting_mask)
            )
        pano_inpainted_raw.save(f"{save_dir_}/08_pano_inpainted_raw.png")
    pano_inpainted_raw = Image.open(f"{save_dir_}/08_pano_inpainted_raw.png")  # re-load to avoid out of memory

    # 10. Blend inpainted panorama and source panorama
    pano_blend1, pano_blend2, mask_blend1, mask_blend2 = spherical_dreamer.blend(
        pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
        pano_inpainted_raw=pano_inpainted_raw,
        missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
        horizon_mask=my_utils.numpy_to_PIL(closed_horizon_mask_warped),
    )

    mask_blend1.save(f"{save_dir_}/09_blend1_mask.png")
    mask_blend2.save(f"{save_dir_}/09_blend2_mask.png")
    pano_blend1.save(f"{save_dir_}/09_blend1_pano_rgb_inpainted.png")
    pano_blend2.save(f"{save_dir_}/09_blend2_pano_rgb_inpainted.png")

    pano_rgb_inpainted = pano_blend2
    pano_rgb_inpainted.save(f"{save_dir_}/XX_pano_rgb.png") 

    # 11. New depth estimation
    depth2 = np.ones((height, width), dtype=np.float32) * sphere_radius  # unit sphere
    my_utils.depth_numpy_to_figure(depth2).savefig(f"{save_dir_}/XX_depth_figure.png")

    # visualize depth1 and depth2 on the same figure
    vmin = np.nanmin((np.nanmin(warped_depth_interp), np.nanmin(depth2)))
    vmax = np.nanmax((np.nanmax(warped_depth_interp), np.nanmax(depth2)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    im1 = ax[0].imshow(warped_depth_interp, cmap='plasma', norm=norm)
    ax[0].set_title('Depth 1 interp')
    im2 = ax[1].imshow(depth2, cmap='plasma', norm=norm)
    ax[1].set_title('Depth 2')
    cbar = fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label('Depth')
    plt.show()



    # 13. Blend depth together
    #TODO: see if this can be coded better without relying on the naive blending to to the proejction into the world. The projection should already be done earlier.

    # a. Naive blending.
    blended_depth = np.zeros_like(warped_depth_interp)
    blended_depth[missing_info_mask] = depth2[missing_info_mask]
    blended_depth[~missing_info_mask] = warped_depth_interp[~missing_info_mask]

    # visalization & pointcloud
    pointclouds[f"dream_{i:02d}_naive_blending"] = get_pointcloud(
        colors=np.array(pano_rgb_inpainted)/255.0,
        points_3D_world_carte = my_utils.depth2world(
            depth=blended_depth, pose=pose2, sphere_radius=sphere_radius, height=height, width=width
        )
    )

    plt.imshow(blended_depth, cmap='plasma')
    plt.colorbar()
    plt.title('Blended Depth Naive')
    plt.show()

    # b. Harmonic blending
    def check_partition(*masks):
        """Return True if masks are disjoint and cover the full image."""
        # disjointness
        total = np.zeros_like(masks[0], dtype=bool)
        for m in masks:
            if np.any(total & m):
                return False
            total |= m
        # full coverage
        return np.all(total)
    
    def get_harmonic_blending_mask(missing_info_mask):
        mask1 = ~missing_info_mask
        mask2 = missing_info_mask
        boundary = find_boundaries(mask1, mode='inner', background=False)  # [H, W]
        mask1 = mask1 & (~boundary)
        mask2 = mask2 & (~boundary)
        assert check_partition(mask1, mask2, boundary), "Masks are not a valid partition of the image"
        return mask1, mask2, boundary
    
    def get_mask_fixed(forward, pts):
        # point should be on cartesian coordinates in camera frame
        cosine_similarity = pts @ forward / (np.linalg.norm(forward) * np.linalg.norm(pts, axis=-1) + 1e-8)
        mask_fixed = cosine_similarity >= 0
        return mask_fixed
    
    def verify_mask_fixed(forward, pano_rgb_inpainted):
        plt.figure()
        mask_fixed_ = get_mask_fixed(forward, my_utils.world_to_cam_carte_3D(
            all_pts2_world, pose2)
        )
        img_arr = np.array(pano_rgb_inpainted)/255.0
        img_arr[mask_fixed_] = np.array([1, 0, 0], dtype=np.float32)
        plt.imshow(img_arr)
        plt.title("Fixed points (red) should be in the forward hemisphere")
        plt.show()

    mask1, mask2, mask_boundary = get_harmonic_blending_mask(missing_info_mask)
    all_pts1_world = my_utils.depth2world(
       depth=warped_depth_interp, pose=pose2, sphere_radius=sphere_radius, height=height, width=width
       )
    all_pts2_world = my_utils.depth2world(
         depth=depth2, pose=pose2, sphere_radius=sphere_radius, height=height, width=width
    )
    pts_target_boundary = all_pts1_world[mask_boundary] 
    pts1_world = all_pts1_world[mask1] # these are already good
    pts2_world_exb = all_pts2_world[mask2] # these need to be deformed by mooving the boundary points to the target boundary points
    pts2_boundary = all_pts2_world[mask_boundary]
    pts2_world = np.concatenate((pts2_world_exb, pts2_boundary), axis=0)
    _mask_boundary = np.concatenate((np.zeros(pts2_world_exb.shape[0], dtype=bool), np.ones(pts2_boundary.shape[0], dtype=bool)), axis=0)
    # mask_fixed = np.zeros(pts2_world.shape[0], dtype=bool)
    mask_fixed = get_mask_fixed(translation_direction, my_utils.world_to_cam_carte_3D(pts2_world, pose2))
    verify_mask_fixed(translation_direction, pano_rgb_inpainted)

    # Deformation time!
    t0 = time.time()
    # the following is FALSE!!
    pts2_deformed_world, _ = my_utils.harmonic_deform_pipeline(
        P=pts2_world,
        mask_fixed=mask_fixed,
        mask_boundary=_mask_boundary,
        target_boundary=pts_target_boundary,
        n_coarse=10000,
        every=5,
        max_fixed=2000,
        k=10, m=3
    )
    t1 = time.time()
    print(f"Harmonic deformation took {t1 - t0:.1f}s")

    pts2_deformed_world_exb, pts2_deformed_boundary = np.split(pts2_deformed_world, [pts2_world_exb.shape[0]], axis=0)
    pts_3D_world_carte_new = np.zeros((height, width, 3), dtype=np.float32)
    pts_3D_world_carte_new[mask1] = pts1_world
    pts_3D_world_carte_new[mask2] = pts2_deformed_world_exb
    pts_3D_world_carte_new[mask_boundary] = pts2_deformed_boundary

    # convert back to equirectangular depth
    blended_depth_harmonic = my_utils.world_to_cam_sph_3D(pts_3D_world_carte_new, pose2)[..., 2]
    plt.figure()
    plt.imshow(blended_depth_harmonic, cmap='plasma')
    plt.colorbar()
    plt.title('Blended Depth Harmonic')
    plt.savefig(f"{save_dir_}/12_blended_depth_harmonic.png")
    plt.show()


    pointclouds[f"dream_{i:02d}_harmonic_blending"] = PointCloud(
        pts=pts_3D_world_carte_new,
        colors=np.array(pano_rgb_inpainted)/255.0
    )

    os.makedirs(f"{save_dir_}/dream_{i:02d}", exist_ok=True)
    with open(f"{save_dir_}/dream_{i:02d}/12_pointclouds.pkl", 'wb') as f:
        pkl.dump(pointclouds, f)