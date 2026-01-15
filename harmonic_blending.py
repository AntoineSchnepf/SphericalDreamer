import numpy as np
import my_utils
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import os
import time


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
    """
    missing_info_mask: np.array of shape [H, W] with dtype bool. True where info is missing i.e. where we inpainted
    """
    # TODO: I think we should erode mask1 a bit. Make the boundary more "inside" mask1
    missing_info_mask = my_utils.dilate_mask(missing_info_mask, pixels=3)
    mask1 = ~missing_info_mask
    mask2 = missing_info_mask
    boundary = find_boundaries(mask1, mode='inner', background=False)  # [H, W]
    mask1 = mask1 & (~boundary)
    mask2 = mask2 & (~boundary)
    assert check_partition(mask1, mask2, boundary), "Masks are not a valid partition of the image"
    return mask1, mask2, boundary

def harmonic_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, phase, logging=False, where_save=None):
    """ Inputs are in HxW format except colors which is HxWx3 
    Given the two depth map (interpolated and estimated), it merges with the following constraints:
        - points in the good region of warped_depth_interp stay unchanged
        - points in the missing region of warped_depth_interp are moved as little as possible to make it both continious and close to depth_estimated
    Returns:
        - pts_deformed: np.array of shape [N, 3] in world coordinates of the points coming from depth_estimated, withing the inpainted region, after harmonic deformation
        - colors_out: np.array of shape [N, 3] with values in [0-1] corresponding to pts_deformed
        - pcd_harmonic: PointCloud object with the full blended pointcloud (More points than pts_deformed, repetition with existing points)
        - blended_depth_harmonic: np.array of shape [H, W] with the blended depth
    """

    def _log_masks(mask1, mask2, mask_boundary):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(mask1, cmap='gray')
        plt.title("Mask 1 (good points)")
        plt.subplot(1,3,2)
        plt.imshow(mask2, cmap='gray')
        plt.title("Mask 2 (to be deformed)")
        plt.subplot(1,3,3)
        plt.imshow(mask_boundary, cmap='gray')
        plt.title("Mask boundary")
        plt.savefig(where_save / phase / "07_harmonic_blending_masks.png")
        plt.show()
    
    mask_keep, mask_deform, mask_boundary = get_harmonic_blending_mask(missing_info_mask)

    all_pts_keep = my_utils.depth2world(
        depth=warped_depth_interp, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    ) # here camera pose is not good maybe ??
    all_pts_deform = my_utils.depth2world(
        depth=depth_estimated, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    )
    pts_keep = all_pts_keep[mask_keep] # these are already good
    pts_target_boundary = all_pts_keep[mask_boundary] 
    pts_deform_exb = all_pts_deform[mask_deform] # these need to be deformed by mooving the boundary points to the target boundary points
    pts_deform_boundary = all_pts_deform[mask_boundary]
    pts_deform = np.concatenate((pts_deform_exb, pts_deform_boundary), axis=0)
    _mask_boundary = np.concatenate((np.zeros(pts_deform_exb.shape[0], dtype=bool), np.ones(pts_deform_boundary.shape[0], dtype=bool)), axis=0)

    # Deformation
    assert np.any(np.isnan(pts_deform)) == False, "Error: pts_deform contains NaNs"
    assert np.any(np.isnan(pts_target_boundary)) == False, "Error: pts_target_boundary contains NaNs"
    t0 = time.time()
    pts_deformed, _ = my_utils.harmonic_deform_pipeline(
        P=pts_deform,
        mask_fixed=np.zeros(pts_deform.shape[0], dtype=bool),
        mask_boundary=_mask_boundary,
        target_boundary=pts_target_boundary,
        n_coarse=10000,
        every=5,
        max_fixed=2000,
        k=10, m=3
    )
    t1 = time.time()
    print(f"Harmonic deformation took {t1 - t0:.1f}s")

    pts_deformed_exb, pts_deformed_boundary = np.split(pts_deformed, [pts_deform_exb.shape[0]], axis=0)
    pts_deformed = np.concatenate((pts_deformed_exb, pts_deformed_boundary), axis=0)
    colors_out_exb = colors[mask_deform]
    colors_out_boundary = colors[mask_boundary]
    colors_out = np.concatenate((colors_out_exb, colors_out_boundary), axis=0)

    # Visualization & pointcloud
    pts_3D_carte_new = np.zeros((height, width, 3), dtype=np.float32)
    pts_3D_carte_new[mask_keep] = pts_keep
    pts_3D_carte_new[mask_deform] = pts_deformed_exb
    pts_3D_carte_new[mask_boundary] = pts_deformed_boundary
    blended_depth_harmonic = my_utils.world2cam_sph_3D(pts_3D_carte_new, pose)[..., 2]
    pcd_harmonic = my_utils.PointCloud(
        pts=pts_3D_carte_new,
        colors=colors
    )
    if logging:
        _log_masks(mask_keep, mask_deform, mask_boundary)
        
        # visualize blended depth and pointcloud from current camera
        plt.figure()
        plt.imshow(blended_depth_harmonic, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Harmonic')
        plt.savefig(where_save / phase / "08_blended_depth_harmonic.png")
        plt.show()

    return pts_deformed, colors_out, pcd_harmonic, blended_depth_harmonic

def harmonic_blend_of_depths_ldi(
        colors, 
        warped_depth_interp, 
        depth_estimated, 
        missing_info_mask, 
        sky_mask_inpainted,
        pose, 
        sphere_radius, 
        height, 
        width, 
        phase,
        logging=False, 
        where_save=None,

        # ldi args
        ldi_depth=None,
        ldi_colors=None, 
        ldi_mask=None,
        ldi_sky_mask=None,
    ):
    """ Inputs are in HxW format except colors which is HxWx3 
    Given the two depth map (interpolated and estimated), it merges with the following constraints:
        - points in the good region of warped_depth_interp stay unchanged
        - points in the missing region of warped_depth_interp are moved as little as possible to make it both continious and close to depth_estimated
    Returns:
        - pts_deformed: np.array of shape [N, 3] in world coordinates of the points coming from depth_estimated, withing the inpainted region, after harmonic deformation
        - colors_out: np.array of shape [N, 3] with values in [0-1] corresponding to pts_deformed
        - pcd_harmonic: PointCloud object with the full blended pointcloud (More points than pts_deformed, repetition with existing points)
        - blended_depth_harmonic: np.array of shape [H, W] with the blended depth
    """

    def _log_masks(mask1, mask2, mask_boundary):
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(mask1, cmap='gray')
        plt.title("Mask 1 (good points)")
        plt.subplot(1,3,2)
        plt.imshow(mask2, cmap='gray')
        plt.title("Mask 2 (to be deformed)")
        plt.subplot(1,3,3)
        plt.imshow(mask_boundary, cmap='gray')
        plt.title("Mask boundary")
        plt.savefig(where_save / phase / "07_harmonic_blending_masks.png")
        plt.show()
    
    mask_keep, mask_deform, mask_boundary = get_harmonic_blending_mask(missing_info_mask)

    # for normal layer
    all_pts_keep = my_utils.depth2world(
        depth=warped_depth_interp, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    ) # here camera pose is not good maybe ??
    all_pts_deform = my_utils.depth2world(
        depth=depth_estimated, pose=pose, sphere_radius=sphere_radius, height=height, width=width
    )
    pts_keep = all_pts_keep[mask_keep] # these are already good
    pts_target_boundary = all_pts_keep[mask_boundary] 
    pts_deform_exb = all_pts_deform[mask_deform] # these need to be deformed by mooving the boundary points to the target boundary points
    pts_deform_boundary = all_pts_deform[mask_boundary]
    
    # prepare concatenations
    _mask_deform=np.zeros(pts_deform_exb.shape[0], dtype=bool)
    _mask_boundary=np.ones(pts_deform_boundary.shape[0], dtype=bool)
    to_cat = [pts_deform_exb, pts_deform_boundary]
    to_cat_mask = [_mask_deform, _mask_boundary]

    whether_ldi = (ldi_depth is not None)
    # optional: add ldi points to the deformation pipilines
    if whether_ldi:
        assert ldi_colors is not None and ldi_mask is not None and ldi_sky_mask is not None, "If ldi_depth is provided, ldi_colors, ldi_mask, and ldi_sky_mask must also be provided"
        mask_deform_ldi = (mask_boundary | mask_deform) & ldi_mask # only consider points that are in the deformable region and in ldi mask

        all_pts_deform_ldi = my_utils.depth2world(
            depth=ldi_depth, pose=pose, sphere_radius=sphere_radius, height=height, width=width
        )
        pts_deform_ldi = all_pts_deform_ldi[mask_deform_ldi]

        to_cat.append(pts_deform_ldi)
        _mask_ldi = np.zeros(pts_deform_ldi.shape[0], dtype=bool) 
        to_cat_mask.append(_mask_ldi)


    pts_deform, cat_meta = my_utils.concat_with_meta(*to_cat)
    _mask_boundary, _ = my_utils.concat_with_meta(*to_cat_mask)

    # Deformation
    assert np.any(np.isnan(pts_deform)) == False, "Error: pts_deform contains NaNs"
    assert np.any(np.isnan(pts_target_boundary)) == False, "Error: pts_target_boundary contains NaNs"
    t0 = time.time()
    pts_deformED, _ = my_utils.harmonic_deform_pipeline(
        P=pts_deform,
        mask_fixed=np.zeros(pts_deform.shape[0], dtype=bool),
        mask_boundary=_mask_boundary,
        target_boundary=pts_target_boundary,
        n_coarse=10000,
        every=5,
        max_fixed=2000,
        k=10, m=3
    )
    t1 = time.time()
    print(f"Harmonic deformation took {t1 - t0:.1f}s")

    undid_cat = my_utils.undo_concat(pts_deformED, cat_meta)

    pts_deformED_exb = undid_cat[0]
    pts_deformED_boundary = undid_cat[1]
    pts_out = np.concatenate((pts_deformED_exb, pts_deformED_boundary), axis=0)

    colors_out_exb = colors[mask_deform]
    colors_out_boundary = colors[mask_boundary]
    colors_out = np.concatenate((colors_out_exb, colors_out_boundary), axis=0)

    sky_mask_out_exb = sky_mask_inpainted[mask_deform]
    sky_mask_out_boundary = sky_mask_inpainted[mask_boundary]
    sky_mask_out = np.concatenate((sky_mask_out_exb, sky_mask_out_boundary), axis=0)

    # Visualization & pointcloud
    pts_3D_carte_new = np.zeros((height, width, 3), dtype=np.float32)
    pts_3D_carte_new[mask_keep] = pts_keep
    pts_3D_carte_new[mask_deform] = pts_deformED_exb
    pts_3D_carte_new[mask_boundary] = pts_deformED_boundary
    blended_depth_harmonic = my_utils.world2cam_sph_3D(pts_3D_carte_new, pose)[..., 2]
    pcd_harmonic = my_utils.PointCloud(
        pts=pts_3D_carte_new,
        colors=colors,
        sky_mask=sky_mask_inpainted
    )
    if logging:
        _log_masks(mask_keep, mask_deform, mask_boundary)
        
        # visualize blended depth and pointcloud from current camera
        plt.figure()
        plt.imshow(blended_depth_harmonic, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Harmonic')
        plt.savefig(where_save / phase / "08_blended_depth_harmonic.png")
        plt.show()

        depth_pil = my_utils.depth_to_pil(blended_depth_harmonic, cmap_name="plasma", vmin=0.1, vmax=1.8)
        depth_pil.save(where_save / phase / "08_blended_depth_harmonic_pil.png")

    res = {
        "pts_out": pts_out,
        "colors_out": colors_out,
        "sky_mask_out": sky_mask_out,
        "pcd_harmonic": pcd_harmonic,
        "blended_depth_harmonic": blended_depth_harmonic
    }

    if whether_ldi:
        # extract the deformed ldi points
        pts_out_ldi = undid_cat[2]
        colors_out_ldi = ldi_colors[mask_deform_ldi]
        sky_mask_out_ldi = ldi_sky_mask[mask_deform_ldi]

        res["pts_out_ldi"] = pts_out_ldi
        res["colors_out_ldi"] = colors_out_ldi
        res["sky_mask_out_ldi"] = sky_mask_out_ldi

    return res

def naive_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, phase, logging=False, where_save=None):
    blended_depth = np.zeros_like(warped_depth_interp)
    blended_depth[missing_info_mask] = depth_estimated[missing_info_mask]
    blended_depth[~missing_info_mask] = warped_depth_interp[~missing_info_mask]

    pcd_naive = my_utils.PointCloud(
        pts=my_utils.depth2world(
            depth=blended_depth, pose=pose, sphere_radius=sphere_radius, height=height, width=width
        ),
        colors=colors
    )
    if logging:
        plt.figure()
        plt.imshow(blended_depth, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Naive')
        plt.savefig(where_save / phase / "08_blended_depth_naive.png")
        plt.show()

        depth_pil = my_utils.depth_to_pil(blended_depth, cmap_name="plasma", vmin=0.1, vmax=1.8)
        depth_pil.save(where_save / phase / "08_blended_depth_naive_pil.png")

    return pcd_naive, blended_depth
