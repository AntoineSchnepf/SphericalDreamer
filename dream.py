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
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
from render_pcd import render_v2
import my_utils

logging.disable(logging.CRITICAL + 1)


class SphericalDreamer:

    def __init__(self, pano_depth_temp_dir, pano_width=1440, pano_height=720, depth_model='360mono', seed=119223):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.depth_model = depth_model
        self.seed = seed
        self.pano_depth_temp_dir = pano_depth_temp_dir
        self.flux_lora_pano_path = 'checkpoints/pano_lora_720*1440_v1.safetensors'
        self.is_pano_generator_init = False
        self.is_inpainting_model_init = False
        self.is_improve_resolution_model_init = False
        self.is_lama_init = False

    def init_pano_generator(self):
        self.pano_gen_pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        self.pano_gen_pipeline.load_lora_weights(self.flux_lora_pano_path) 
        self.pano_gen_pipeline.enable_model_cpu_offload()  
        self.pano_gen_pipeline.enable_vae_tiling()
        
    @torch.no_grad()
    def gen_pano(self, prompt, override_with_inpaint=False, seed_override=None):

        if override_with_inpaint:
            return self.inpaint_pano(
                prompt=prompt,
                pano_rgb=Image.new('RGB', (self.pano_width, self.pano_height), (127,127,127)),
                mask=Image.new('L', (self.pano_width, self.pano_height), 255)
            )

        if not self.is_pano_generator_init:
            self.init_pano_generator()
            self.is_pano_generator_init = True

        seed = self.seed if seed_override is None else seed_override
        pano_rgb = self.pano_gen_pipeline(
            prompt, 
            height=self.pano_height,
            width=self.pano_width,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=50, 
            blend_extend=2,
            guidance_scale=7).images[0]

        # image = image.resize((2048,1024))

        return pano_rgb
    
    def estimate_pano_depth(self, pano_rgb:np.array):
        if self .depth_model == '360mono':
            return self.estimate_pano_depth_360mono(pano_rgb)
        elif self.depth_model == 'egformer':
            return self.estimate_pano_depth_egformer(pano_rgb)
        else:
            raise ValueError(f"Unknown depth model: {self.depth_model}. Should be either '360mono' or 'egformer'.")
        
    @torch.no_grad()
    def estimate_pano_depth_360mono(self, pano_rgb:np.array):
        """
        args:
            `pano_rgb`: np.array of shape [pano_h,pano_w,3] and values in [0-255]        
        returns:
            pano_depth: np.array of shape [pano_h,pano_w] and values in [0-1]
        """
        self.depth_estimator = Pano_depth_estimation(
            self.pano_height, 
            self.pano_width, 
            self.pano_depth_temp_dir, 
            self.device, 
            depth_model="DepthAnythingv2"
        )
        pano_depth = self.depth_estimator.get_panodepth(pano_rgb)  #[0-1] 
        return pano_depth  

    @torch.no_grad()        
    def estimate_pano_depth_egformer(self, pano_rgb:np.array):  
        """
        args:
            `pano_rgb`: np.array of shape [pano_h,pano_w,3] and values in [0-255]       
        returns:
            pano_depth: np.array of shape [pano_h,pano_w] and values in [0-1] 
        """
        from egformer import get_egformer_depth
        pano_rgb_pil = Image.fromarray(pano_rgb.astype(np.uint8))
        pano_depth_pil = get_egformer_depth([pano_rgb_pil])[0]
        pano_depth = np.array(pano_depth_pil.convert("L")).astype(np.float32) / 255.0
        return pano_depth

    def init_inpainting_model(self):

        self.pano_inpaint_pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        # self.pano_inpaint_pipeline.load_lora_weights(self.flux_lora_pano_path) # Antoine: Do not use the lora for inpainting, it yields worse results. TODO: maybe verify this further
        self.pano_inpaint_pipeline.enable_model_cpu_offload()
        # pipe.enable_vae_tiling() #todo test with or without this?

    @torch.no_grad()   
    def inpaint_pano(self, prompt, pano_rgb, mask, seed_override=None):
        "pano_rgb, mask: PIL.Image"

        if not self.is_inpainting_model_init:
            self.init_inpainting_model()
            self.is_inpainting_model_init = True

        # i. inpainting
        seed = self.seed if seed_override is None else seed_override
        mask = mask.convert("L")
        pano_inpainted_raw = self.pano_inpaint_pipeline(
            prompt=prompt,
            image=pano_rgb,  
            mask_image=mask, 
            strength=1.0,
            height=self.pano_height,
            width=self.pano_width,
            guidance_scale=30.0,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),  
        ).images[0]

        return pano_inpainted_raw

    def blend(self, pano_rgb, pano_inpainted_raw, missing_info_mask, horizon_mask):

        #ii. compose blending
        mask_blend1 = missing_info_mask
        pano_blend1 = self._blend(
            pano_inpainted_raw, 
            pano_rgb, 
            mask_blend1, 
            mode='compose'
        )

        # iii. seamless blending
        mask_blend2=horizon_mask
        pano_blend2 = self._blend(
            pano_inpainted_raw, 
            pano_blend1, 
            mask_blend2,
            mode='seamless'
        )

        return pano_blend1, pano_blend2, mask_blend1, mask_blend2 #TODO: only pano_blend1 is needed

    def _blend(self, src, dst, mask, mode):
        "Blends two images together, guided by mask. All arguments should be PIL.Image"

        # Naive blending. Just compose the images
        if mode == 'compose':
            pano_blended = Image.composite(src, dst, mask)

        # Seamless blending, with smoothing along the mask edges
        elif mode == 'seamless':
            pano_blended = my_utils.seamless_blend(src, dst, mask)
        else:
            raise ValueError(f"Unknown blending mode: {mode}. Mode should either be 'seamless' or 'compose'.")

        return pano_blended

    def init_improve_resolution_model(self):

        controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler",
            torch_dtype=torch.bfloat16
        )
        self.improve_resolution_pipeline = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.bfloat16
        )
        # self.improve_resolution_pipeline.load_lora_weights(self.flux_lora_pano_path)  # change this.
        self.improve_resolution_pipeline.enable_model_cpu_offload()

    @torch.no_grad()   
    def improve_pano_resolution(self, pano_rgb, prompt, controlnet_conditioning_scale=0.2):

        if not self.is_improve_resolution_model_init:
            self.init_improve_resolution_model()
            self.is_improve_resolution_model_init = True

        image = self.improve_resolution_pipeline(
            prompt=prompts, 
            control_image=pano_rgb,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=50, 
            guidance_scale=3.5,
            height=pano_rgb.size[1],
            width=pano_rgb.size[0],
            generator=torch.Generator("cpu").manual_seed(self.seed) 
        ).images[0]
        return image
    
    def init_lama(self):
        from src.lama import LamaInpainting
        self.lama_model = LamaInpainting()

    @torch.no_grad()   
    def lama_inpaint(self, image:Image, mask:Image):
        """
        image: PIL.Image (RGB)
        mask: PIL.Image (L)
        """
        if not self.is_lama_init:
            self.init_lama()
            self.is_lama_init = True

        return Image.fromarray(self.lama_model(image, mask))


def get_missing_info_mask(operations, visited_pixels, log_mask=True, where_save=None):
    missing_info_masks = [~visited_pixels]
    for op in operations:
        missing_info_masks.append(op(missing_info_masks[-1]))
    if log_mask:
        missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
        missing_info_masks_tile.save(f"{where_save}/02_missing_info_masks_tile.png")
    missing_info_mask = missing_info_masks[-1]
    return missing_info_mask

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

def harmonic_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, logging=True, where_save=None):
    """ Inputs are in HxW format except colors which is HxWx3 
    Given the two depth map (interpolated and estimated), it merges with the following constraints:
        - points in the good region of warped_depth_interp stay unchanged
        - points in the missing region of warped_depth_interp are moved as little as possible to make it both continious and close to depth_estimated
    Returns:
        - pts2_deformed: np.array of shape [N, 3] in world coordinates of the points coming from depth_estimated, withing the inpainted region, after harmonic deformation
        - colors2: np.array of shape [N, 3] with values in [0-1] corresponding to pts2_deformed
        - pcd_harmonic: PointCloud object with the full blended pointcloud (More points than pts2_deformed, repetition with existing points)
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
        plt.savefig(os.path.join(where_save, "07_harmonic_blending_masks.png"))
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
    # mask_fixed = get_mask_fixed(translation_direction, my_utils.world2cam_carte_3D(pts_deform, pose))
    # verify_mask_fixed(translation_direction, pano_rgb_inpainted)

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
    colors2_exb = colors[mask_deform]
    colors2_boundary = colors[mask_boundary]
    colors2 = np.concatenate((colors2_exb, colors2_boundary), axis=0)

    # Visualization & pointcloud
    if logging:
        _log_masks(mask_keep, mask_deform, mask_boundary)
        # TODO: What does the new spherical image looks like from pose ? With deformed points ?

        # visualize blended depth and pointcloud from current camera
        pts_3D_carte_new = np.zeros((height, width, 3), dtype=np.float32)
        pts_3D_carte_new[mask_keep] = pts_keep
        pts_3D_carte_new[mask_deform] = pts_deformed_exb
        pts_3D_carte_new[mask_boundary] = pts_deformed_boundary
        blended_depth_harmonic = my_utils.world2cam_sph_3D(pts_3D_carte_new, pose)[..., 2]
        pcd_harmonic = my_utils.PointCloud(
            pts=pts_3D_carte_new,
            colors=colors
        )

        plt.figure()
        plt.imshow(blended_depth_harmonic, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Harmonic')
        plt.savefig(os.path.join(where_save, "08_blended_depth_harmonic.png"))
        plt.show()

        return pts_deformed, colors2, pcd_harmonic, blended_depth_harmonic

    return pts_deformed, colors2

def naive_blend_of_depths(colors, warped_depth_interp, depth_estimated, missing_info_mask, pose, sphere_radius, height, width, logging=True, where_save=None):

    if logging:

        blended_depth = np.zeros_like(warped_depth_interp)
        blended_depth[missing_info_mask] = depth_estimated[missing_info_mask]
        blended_depth[~missing_info_mask] = warped_depth_interp[~missing_info_mask]

        pcd_naive = my_utils.PointCloud(
            pts=my_utils.depth2world(
                depth=blended_depth, pose=pose, sphere_radius=sphere_radius, height=height, width=width
            ),
            colors=colors
        )

        plt.figure()
        plt.imshow(blended_depth, cmap='plasma')
        plt.colorbar()
        plt.title('Blended Depth Naive')
        plt.savefig(os.path.join(where_save, "08_blended_depth_naive.png"))
        plt.show()

    return pcd_naive, blended_depth

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

def correct_walls_lp(x, y, p=6.0):
    mask = y > 0
    x_corr = x.copy()
    y_corr = y.copy()

    theta = np.atan2(y, x)
    r = np.sqrt(x**2 + y**2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rho = 1.0 / (np.abs(cos_t)**p + np.abs(sin_t)**p)**(1.0/p)
    x = r * rho * cos_t
    y = r * rho * sin_t

    x_corr[mask] = x[mask]
    y_corr[mask] = y[mask]

    return x_corr, y_corr

def generate_missing_points_from_pose(
        current_points, 
        current_colors, 
        camera_pose, 
        height,
        width,
        skip_inpainting, 
        where_save=None
    ):

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
    operations = [
        partial(minimum_filter, size=(3,3), axes=(0,1)),
        partial(maximum_filter, size=(3,3), axes=(0,1)),
        partial(maximum_filter, size=(3,3), axes=(0,1)),
        partial(maximum_filter, size=(3,3), axes=(0,1)),
        # partial(maximum_filter, size=(8, 8), axes=(0,1)),
    ]
    missing_info_mask = get_missing_info_mask(operations, visited_pixels, log_mask=True, where_save=where_save) 
    where_depth_nan = np.isnan(warped_depth_interp)
    missing_info_mask = missing_info_mask | where_depth_nan
    inpainting_mask = missing_info_mask # TODO: (Antoine, 14 oct) The inpainting mask is currently composed of both <<large missing regions due to limited covering of the main spheres>> and <<small holes due to occlusions>>. We could separate these two cases and do something neater?.

    warped_img_interp[missing_info_mask] = np.nan
    warped_depth_interp[missing_info_mask] = np.nan
    my_utils.numpy_to_PIL(warped_img).save(os.path.join(where_save, "01_warped_img.png"))
    my_utils.depth_numpy_to_PIL(warped_depth).save(os.path.join(where_save, "01_warped_depth.png")    )
    my_utils.numpy_to_PIL(warped_img_interp).save(os.path.join(where_save, "03_warped_img_interp.png"))
    my_utils.depth_numpy_to_PIL(warped_depth_interp).save(os.path.join(where_save, "03_warped_depth_interp.png"))
    my_utils.depth_numpy_to_figure(warped_depth_interp).savefig(os.path.join(where_save, "03_warped_depth_interp_figure.png"))
    # np.save(f"{save_dir_}/{where_save}/03_warped_depth_interp.npy", warped_depth_interp)
    
    # 6. Inpainting
    overlay_before = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img_interp, inpainting_mask, alpha=0.5)) 
    overlay_before.save(os.path.join(where_save, "04_overlay_before_inpainting.png"))
    if not skip_inpainting: 
        pano_inpainted_raw = spherical_dreamer.inpaint_pano(
            prompt=prompt, 
            pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
            mask=my_utils.numpy_to_PIL(inpainting_mask)
        )
        pano_inpainted_raw.save(os.path.join(where_save, "XX_pano_rgb_inpainted_raw.png"))
    else:
        pano_inpainted_raw = Image.open(os.path.join(where_save, "XX_pano_rgb_inpainted_raw.png"))
    pano_inpainted_raw.save(os.path.join(where_save, "04_pano_rgb_inpainted_raw.png"))

    # 7. Inpainting seamless blending
    pano_blend1, pano_blend2, mask_blend1, mask_blend2 = spherical_dreamer.blend(
        pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
        pano_inpainted_raw=pano_inpainted_raw,
        missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
        horizon_mask=my_utils.numpy_to_PIL(np.zeros_like(missing_info_mask).astype('bool')),
    )
    #TODO: since we removed horizon, check the blending strategy again. It is `compose` everywhere now. Should be seamless for the large inapainted part
    #TODO: Also, Check if we need both blend1 and blend2
    
    mask_blend1.save(os.path.join(where_save, "05_blend1_mask.png"))
    mask_blend2.save(os.path.join(where_save, "05_blend2_mask.png"))
    pano_blend1.save(os.path.join(where_save, "05_blend1_pano_rgb_inpainted.png"))
    pano_blend2.save(os.path.join(where_save, "05_blend2_pano_rgb_inpainted.png"))

    pano_rgb_inpainted = pano_blend2
    pano_rgb_inpainted.save(os.path.join(where_save, "06_pano_rgb_inpainted.png")) #TODO: this is the same as blend2. Remove repetition

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
    my_utils.depth_numpy_to_PIL(depth_estimated).save(os.path.join(where_save, "07_estimated_depth.png"))
    my_utils.depth_numpy_to_figure(depth_estimated).savefig(os.path.join(where_save, "07_estimated_depth_figure.png"))


    # 9. Blend depth
    new_colors = (np.array(pano_rgb_inpainted)/255.0)

    # (Naive blending)
    # TODO: (Antoine): I think the variable below should be inpainting_mask instead of missing_info_mask
    pcd_naive, blended_depth_naive = naive_blend_of_depths(
        colors=new_colors,
        warped_depth_interp=warped_depth_interp,
        depth_estimated=depth_estimated,
        missing_info_mask=missing_info_mask,
        pose=camera_pose,
        sphere_radius=sphere_radius,
        height=height,
        width=width,
        logging=True,
        where_save=where_save
    )

    # (Harmonic blending)
    pts_deformed_world, new_colors, pcd_harmonic, blended_depth_harmonic = harmonic_blend_of_depths(
        colors=new_colors,
        warped_depth_interp=warped_depth_interp,
        depth_estimated=depth_estimated,
        missing_info_mask=missing_info_mask,
        pose=camera_pose,
        sphere_radius=sphere_radius,
        height=height,
        width=width,
        logging=True,
        where_save=where_save
    )

    return pts_deformed_world, new_colors, pcd_naive, pcd_harmonic


if __name__ == "__main__":
    # ---- args ----
    debug = False
    skip_phase0 = True
    skip_inpainting = False
    skip_phase1 = False
    skip_filling=False
    save_dir = "OUTPUTS/SphericalDreamerRecurse"

    # dreaming args
    num_dreams = 3
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    sphere_radius = 1.0
    depth_model = "360mono"  # "360mono" or "egformer"
    FAR=1.0
    NEAR=0.01
    delta_walk = FAR * np.pi / 2
    raise_intermediate_camera_by_z = 0.3    
    override_with_inpaint=False
    if depth_model == 'egformer':
        width = 1024
        height = 512
    else:
        width = 1440
        height = 720
    seeds = [119224, 119224+9, 119224+20, 119224+33, 119224+45]
    prompts = [
        "A realistic illustration of a college campus. In the middle ground, several academic buildings with brick facades and large windows stand prominently. In the background, a bright blue sky with scattered clouds stretches across the scene. In the foreground, a few elements commonly found on campus, such as students walking, bicycles parked along a path, and a grassy lawn with trees, add depth and life to the scene.",
        "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail.",
        "A serene forest scene with a small stream, dappled sunlight filtering through the leaves, realism style.",
        "A bustling city street at night, neon lights reflecting on wet pavement, realism style.",
        # "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.",
        # "A wide field under daylight, covered in lush green grass with worn paths where the grass has been trampled by many footsteps. In the center of the field stands a large concert stage, decorated with bold triangular patterns. On the stage rests a single guitar, but no performers are present. In front of the stage, a lively crowd gathers, waiting for the show to begin."
    ]
    expnames=[
        "31_campus",
        "31_seaside",
        "31_forest",
        "31_city",
    ]
    indoor_or_outdoor_list = [
        'outdoor',
        'outdoor',
        'indoor',
        'outdoor',
    ]
    # ---------------

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=int, help='Experiment ID to run (0-4)', default=2)
    if True:
        args = parser.parse_args([
            '--exp_id', '2'
        ])
        for _ in range(10):
            print(f"/!\ DEBUG MODE IS ON. Running exp {args.exp_id}/!\ ")
    else:
        args = parser.parse_args()

    expname, prompt, indoor_or_outdoor = expnames[args.exp_id], prompts[args.exp_id], indoor_or_outdoor_list[args.exp_id]

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=depth_model,
    )
    save_dir_ = f"{save_dir}/{expname}"
    
    pose_init = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    pose_end = my_utils.camera_translation(pose_init, delta_walk * translation_direction * (num_dreams-1))


    # ---- PHASE 0. ----- GENERATE INDEPENDENT SPHERICAL IMAGES + DEPTH
    if not skip_phase0:
        for i in range(num_dreams):
            print(f"--- Dreaming Phase {i:02d} / {num_dreams} ---")

            # Generate panorama & Estimate Depth
            pano_rgb = spherical_dreamer.gen_pano(prompt=prompt, override_with_inpaint=override_with_inpaint, seed_override=seeds[i])
            depth = spherical_dreamer.estimate_pano_depth(pano_rgb=np.array(pano_rgb))
            my_utils.save_rgbd_pano(
                pano_rgb=pano_rgb,
                depth=depth,
                dream=i,
                save_dir_=save_dir_
            )
    # --- END PHASE 0. ---


    # ----- args PHASES I -----
    opening_kwargs = {
        'opening_mode': 'cut+cylinder',
        'delta_cut': 2*np.pi/3,
    }
    sphere_correction_kwargs = {
        "correct_depth": False,
        "near": NEAR,
        "far": FAR,
        "correct_walls": False,
        "correct_floor": False,
        "depth_threshold_for_floor_correction": 0.6,
        "remove_sky": False,
        "indoor_or_outdoor": None,
        "remove_outliers": True,
        "verbose": False,
        "plot": True,
    }
    # --- end args PHASES I ---

    print(f"--- Opening first sphere ---")
    # PHASE I. ALIGN PAIRS OF SPHERES WITH INPAINTING + HARMONIC BLENDING
    if not skip_phase1:
        # PHASE 1: INIT
        pointclouds = {}
        all_pts_world = np.array([]).reshape(0, 3)
        all_colors_world = np.array([]).reshape(0, 3)

        colors1, depth1 = my_utils.load_rgbd_pano(
            dream=0,
            save_dir_=save_dir_
        )
        pts1_carte = my_utils.depth2cam_carte(
            depth=depth1,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
        )
        pts1_carte_corrected, colors1_corrected = my_utils.run_corrective_pipeline_on_sphere(
            pts1_carte, # in cartesian coordinates
            colors1, 
            height, width, 
            **sphere_correction_kwargs
        )
        sphere1 = my_utils.Sphere(
            None, pts1_carte_corrected, colors1_corrected, 
            forward_carte=translation_direction,
            opening_kwargs=opening_kwargs,
        )
        pose1 = pose_init
        sphere1.update_pose(pose1)

        # PHASE 1. INPAINTING + HARMONIC BLENDING
        for i in range(1, num_dreams):
            print(f"--- Inpainting+Alignment Phase {i:02d} / {num_dreams-1} ---")
            save_dir__ = os.path.join(save_dir_, f"align_{i:02d}")
            os.makedirs(save_dir__, exist_ok=True)

            # 1. Load new sphere and open it (left)
            colors2, depth2 = my_utils.load_rgbd_pano(
                dream=i,
                save_dir_=save_dir_
            )
            pts2_carte = my_utils.depth2cam_carte(
                depth=depth2,
                sphere_radius=sphere_radius,
                height=height,
                width=width,
            ) 
            pts2_carte_corrected, colors2_corrected = my_utils.run_corrective_pipeline_on_sphere(
                pts2_carte, 
                colors2, 
                height, width, 
                **sphere_correction_kwargs
            )
            
            sphere2 = my_utils.Sphere(
                None, pts2_carte_corrected, colors2_corrected, 
                forward_carte=translation_direction,
                opening_kwargs=opening_kwargs,
            )

            # 2. Move camera
            pose2 = my_utils.camera_translation(pose1, delta_walk * translation_direction)
            sphere2.update_pose(pose2)
            
            print('Loaded and opened sphere!')

            # 3. Go to intermediate camera (between cam1 and cam2)
            pose_intermediate = my_utils.camera_translation(pose2, -delta_walk/2 * translation_direction)
            pose_intermediate_bis = my_utils.camera_translation(pose1, delta_walk/2 * translation_direction)
            
            assert np.allclose(pose_intermediate, pose_intermediate_bis), "Error in camera intermediate pose computation"
            pose_intermediate = my_utils.camera_translation(pose_intermediate, np.array([0, 0, raise_intermediate_camera_by_z]))

            # 4, 5, 6, 7, 8, 9: Generate missing points from pose, inpaint, estimate depth (inside function below)
            new_pts, new_colors, pcd_naive, pcd_harmonic = generate_missing_points_from_pose(
                current_points=np.concatenate((
                    sphere1.right_opened.get_world_pcd().pts, sphere2.left_opened.get_world_pcd().pts
                ), axis=0),
                current_colors=np.concatenate((
                    sphere1.right_opened.get_world_pcd().colors, sphere2.left_opened.get_world_pcd().colors
                ), axis=0),
                camera_pose=pose_intermediate,
                skip_inpainting=skip_inpainting,
                height=height,
                width=width,
                where_save=save_dir__,
            )
            pointclouds[f"inpaint_{i:02d}"] = {}
            pointclouds[f"inpaint_{i:02d}"]['blended_naive_w_excess'] = pcd_naive
            pointclouds[f"inpaint_{i:02d}"]['blended_harmonic_w_excess'] = pcd_harmonic
            pointclouds[f"inpaint_{i:02d}"]["blended_harmonic"] = my_utils.PointCloud(
                pts=new_pts,
                colors=new_colors
            )

            # 10. Add new points to their corresponding spheres.
            (new_pts1, new_colors1), (new_pts2, new_colors2), (new_pts_neutral, new_colors_neutral) = split_new_points(
                new_pts, new_colors, pose1, pose2, translation_direction
            )
            sphere1.add_new_points(my_utils.world2cam_carte_3D(new_pts1, pose1), new_colors1)
            sphere2.add_new_points(my_utils.world2cam_carte_3D(new_pts2, pose2), new_colors2)

            # Add all new points to world points, including inpainted+deformed points and points from the current dream.
            pointclouds[f'dream_{i:02d}'] = {}
            pointclouds[f"dream_{i:02d}"]['sphere1_init'] = sphere1.closed.get_world_pcd()
            pointclouds[f"dream_{i:02d}"]['sphere2_init'] = sphere2.closed.get_world_pcd()
            
            #10.a Points from sphere1
            if i == 1: # first iteration: sphere1 only has right opened
                pointclouds[f"dream_{i:02d}"]['sphere1_open'] = sphere1.right_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere1.right_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere1.right_opened.get_world_pcd().colors), axis=0)
            else: # later iterations: sphere1 has both opened
                pointclouds[f"dream_{i:02d}"]['sphere1_open'] = sphere1.both_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere1.both_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere1.both_opened.get_world_pcd().colors), axis=0)
            #10.b Neutral points
            all_pts_world = np.concatenate((all_pts_world, new_pts_neutral), axis=0)
            all_colors_world = np.concatenate((all_colors_world, new_colors_neutral), axis=0)
            #10.c Points from sphere2 (only last iter)
            if i == num_dreams - 1: 
                pointclouds[f"dream_{i:02d}"]['sphere2_open'] = sphere2.left_opened.get_world_pcd()
                all_pts_world = np.concatenate((all_pts_world, sphere2.left_opened.get_world_pcd().pts), axis=0)
                all_colors_world = np.concatenate((all_colors_world, sphere2.left_opened.get_world_pcd().colors), axis=0)
                assert np.allclose(pose2, pose_end), "Error in final camera pose computation"

            # 11. Log final pointcloud
            pointclouds[f"dream_{i:02d}"][f"total"] = my_utils.PointCloud(
                pts=all_pts_world,
                colors=all_colors_world
            )

            # 12. Adjust sphere1 to be sphere2 for next iteration
            sphere1 = sphere2
            pose1 = pose2

            # save pcd
            with open(os.path.join(save_dir_, "pointclouds_zoo.pkl"), 'wb') as f:
                pkl.dump(pointclouds, f)

        # END OF PHASE II: final pcd save
        with open(os.path.join(save_dir_, "raw_dream_pcd.pkl"), 'wb') as f:
            pkl.dump(
                my_utils.PointCloud(
                    pts=all_pts_world,
                    colors=all_colors_world
                ), f)
    else:
        with open(os.path.join(save_dir_, "raw_dream_pcd.pkl"), 'rb') as f:
            raw_pcd = pkl.load(f)
        all_pts_world = raw_pcd.pts
        all_colors_world = raw_pcd.colors
        
    # --- args PHASE III ---
    world_correction_kwargs = {
        "correct_depth": False,
        "near": NEAR*2,
        "far": FAR*2,
        "correct_walls": False,
        "correct_floor": True,
        "depth_threshold_for_floor_correction": 1.0,
        "remove_outliers": False,
    }
    # --- end args PHASE III ---

    # PHASE III. POST PROCESSING OF THE FINAL POINTCLOUD WITH WORLD CORRECTION + HOLE FILLING
    all_pts_world, all_colors_world = my_utils.run_corrective_pipeline_on_world(
        pts=all_pts_world,
        colors=all_colors_world,
        pose_left=pose_init,
        pose_right=pose_end,
        translation_direction=translation_direction,
        verbose=True,
        plot=True,
        **world_correction_kwargs
    )


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
            skip_inpainting=skip_filling, 
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

    print("PYTHON SCRIPT SUCCESSFULLY RUN TO THE END !")