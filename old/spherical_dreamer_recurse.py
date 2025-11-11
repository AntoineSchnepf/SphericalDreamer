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
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
import my_utils

logging.disable(logging.CRITICAL + 1)

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
    mask_fixed_ = get_mask_fixed(forward, my_utils.world2cam_carte_3D(
        all_pts2_world, pose2)
    )
    img_arr = np.array(pano_rgb_inpainted)/255.0
    img_arr[mask_fixed_] = np.array([1, 0, 0], dtype=np.float32)
    plt.imshow(img_arr)
    plt.title("Fixed points (red) should be in the forward hemisphere")
    plt.show()

def visualized_masks(mask1, mask2, mask_boundary):
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
    plt.savefig(f"{save_dir_}/dream_{i:02d}/07_harmonic_blending_masks.png")
    plt.show()

class PointCloud:
    def __init__(self, pts, colors):
        """
        pts: np.array of shape [..., 3]
        colors: np.array of shape [..., 3] with values in [0-1]
        """
        self.pts = pts.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)
        assert self.pts.shape[0] == self.colors.shape[0], "Error: pts and colors must have the same number of points"

    def get_o3d_pointcloud(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pts)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        return pcd

def get_pointcloud(colors, points_3D_world_carte):

    if isinstance(colors, Image.Image):
        colors = np.array(colors).reshape(-1, 3)/255.0

    pcd = PointCloud(
        pts=points_3D_world_carte.reshape(-1, 3),
        colors=colors.reshape(-1, 3)
    )
    return pcd

class SphericalDreamer:

    def __init__(self, pano_depth_temp_dir, pano_width=1440, pano_height=720):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.seed = 119223
        self.seed_inpaint = 119224 #TODO: remove this
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
    def gen_pano(self, prompt, override_with_inpaint=False):

        if override_with_inpaint:
            return self.inpaint_pano(
                prompt=prompt,
                pano_rgb=Image.new('RGB', (self.pano_width, self.pano_height), (127,127,127)),
                mask=Image.new('L', (self.pano_width, self.pano_height), 255)
            )

        if not self.is_pano_generator_init:
            self.init_pano_generator()
            self.is_pano_generator_init = True

        pano_rgb = self.pano_gen_pipeline(
            prompt, 
            height=self.pano_height,
            width=self.pano_width,
            generator=torch.Generator("cpu").manual_seed(self.seed),
            num_inference_steps=50, 
            blend_extend=2,
            guidance_scale=7).images[0]

        # image = image.resize((2048,1024))

        return pano_rgb
    
    @torch.no_grad()
    def estimate_pano_depth(self, pano_rgb):
        """
        args:
            `pano_rgb`: np.array of shape [pano_h,pano_w,3] and values in [0-255]        
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

    def init_inpainting_model(self):

        self.pano_inpaint_pipeline = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
        # self.pano_inpaint_pipeline.load_lora_weights(self.flux_lora_pano_path) # Antoine: Do not use the lora for inpainting, it yields worse results. TODO: maybe verify this further
        self.pano_inpaint_pipeline.enable_model_cpu_offload()
        # pipe.enable_vae_tiling() #todo test with or without this?

    def inpaint_pano(self, prompt, pano_rgb, mask):
        "pano_rgb, mask: PIL.Image"

        if not self.is_inpainting_model_init:
            self.init_inpainting_model()
            self.is_inpainting_model_init = True

        # i. inpainting
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
            generator=torch.Generator("cpu").manual_seed(self.seed_inpaint),  
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

    def lama_inpaint(self, image:Image, mask:Image):
        """
        image: PIL.Image (RGB)
        mask: PIL.Image (L)
        """
        if not self.is_lama_init:
            self.init_lama()
            self.is_lama_init = True

        return Image.fromarray(self.lama_model(image, mask))

def camera_translation(pose, translation):
    """
    pose: np.array of shape [4,4]
    translation: np.array of shape [3,] in world coordinates
    """
    pose2 = pose.copy()
    pose2[:3, 3] += translation
    return pose2

if __name__ == "__main__":

    # ---- args ----
    skip_init = False
    debug = False
    log_pcds = True
    save_dir = "OUTPUTS/SphericalDreamerRecurse"

    # dreaming args
    num_dreams = 3
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    sphere_radius = 1.0
    delta_walk = sphere_radius * np.pi / 2
    opening_mode = 'cut+cylinder' # 'wall', 'cut+wall', 'cut+cylinder'
    override_with_inpaint=True
    width = 1440
    height = 720
    prompts = [
        "360A realistic illustration of a college campus. In the middle ground, several academic buildings with brick facades and large windows stand prominently. In the background, a bright blue sky with scattered clouds stretches across the scene. In the foreground, a few elements commonly found on campus, such as students walking, bicycles parked along a path, and a grassy lawn with trees, add depth and life to the scene"
        # "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail."
        # "A serene forest scene with a small stream, dappled sunlight filtering through the leaves, realism style.",
        # "A bustling city street at night, neon lights reflecting on wet pavement, realism style.",
        # "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.",
        # "A wide field under daylight, covered in lush green grass with worn paths where the grass has been trampled by many footsteps. In the center of the field stands a large concert stage, decorated with bold triangular patterns. On the stage rests a single guitar, but no performers are present. In front of the stage, a lively crowd gathers, waiting for the show to begin."
    ]
    expnames=[
        "15_campus_override_w_inpaint"
        # "09_bali_aligned", 
        # "forest", 
        # "city", 
        # "beach", 
        # "the_stage",
    ]
    # ---------------

    if debug:
        for _ in range(10):
            print("/!\ Debug mode is on /!\ ")

    # 0. Initialization

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir = '/tmp/pano_depth_temp'
    )
    pointclouds = {}
    for expname, prompt in zip(expnames, prompts):
        save_dir_ = f"{save_dir}/{expname}"

        pts_world = np.array([]).reshape(0, 3)
        colors_world = np.array([]).reshape(0, 3)
        # forward alignment
        for i in range(0, num_dreams):
            
            print(f"--- Dream {i:02d} / {num_dreams} ---")
            
            os.makedirs(os.path.join(save_dir_, f"dream_{i:02d}"), exist_ok=True)
            pointclouds[f'dream_{i:02d}'] = {}

            if i == 0:
                # Init: 0. Identity pose
                pose = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                    
                # Init: 1. Generate panorama & Estimate Depth
                if not skip_init and not debug:
                    pano_rgb = spherical_dreamer.gen_pano(prompt=prompt, override_with_inpaint=override_with_inpaint)
                    depth2 = spherical_dreamer.estimate_pano_depth(pano_rgb=np.array(pano_rgb))
                    pano_rgb.save(f"{save_dir_}/dream_{i:02d}/XX_pano_rgb.png")
                    np.save(f"{save_dir_}/dream_{i:02d}/XX_depth.npy", depth2)
                else:
                    pano_rgb = Image.open(f"{save_dir_}/dream_{i:02d}/XX_pano_rgb.png")
                    depth2 = np.load(f"{save_dir_}/dream_{i:02d}/XX_depth.npy")

                # Logging: Initial Sphere
                colors = np.array(pano_rgb)/255.0
                if log_pcds:
                    pcd1 = PointCloud(
                        pts=my_utils.depth2world(
                            depth=depth2, pose=pose, sphere_radius=sphere_radius, height=height, width=width
                        ),
                        colors=colors
                    )
                    pointclouds[f"dream_{i:02d}"]["initial"] = pcd1

                # Init: 2. Apply geometrical transformation to the world points 
                points_2D_cam1_erp = np.stack((np.meshgrid(range(width), range(height))), axis=-1) 
                points_3D_cam1_sph = my_utils.cam_erp2cam_sph_3D(
                    points_2D_cam1_erp, height, width, depth2, sphere_radius=sphere_radius
                )
                translation_direction_sph = my_utils.carte2sph_3D(translation_direction)
                _, points_3D_cam1_sph_unfolded, mask_opening = my_utils.open_world(
                    forward_sph=translation_direction_sph,
                    pts_sph=points_3D_cam1_sph,
                    mode=opening_mode,
                    delta_cut=2*np.pi/3
                )
                points_3D_cam1_sph_unfolded = points_3D_cam1_sph_unfolded[mask_opening]
                colors = colors[mask_opening]

                # Init: 3. Project to the World and save point cloud
                points_3D_world_carte = my_utils.cam_sph2world_3D(
                    points_3D_cam1_sph_unfolded, pose
                )

                # Init: 4. Add new colored points and save data
                if log_pcds:
                    pcd2 = PointCloud(
                        pts=points_3D_world_carte,
                        colors=colors
                    )
                    pointclouds[f"dream_{i:02d}"]["final"] = pcd2 
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_initial.pkl", 'wb') as f:
                        pkl.dump(pcd1, f)
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_final.pkl", 'wb') as f:
                        pkl.dump(pcd2, f)

                pts_world = np.concatenate((pts_world, pcd2.pts), axis=0)
                colors_world = np.concatenate((colors_world, pcd2.colors), axis=0)

                my_utils.depth_numpy_to_PIL(depth2).save(f"{save_dir_}/dream_{i:02d}/XX_depth.png")
                my_utils.depth_numpy_to_figure(depth2).savefig(f"{save_dir_}/dream_{i:02d}/XX_depth_figure.png")

            else:
                
                # Init already done in previous iteration
                pose = camera_translation(pose, delta_walk * translation_direction)

                # 2. Reproject to new camera pose (cam2)
                points_3D_cam2_carte = np.einsum(
                    'ij,...j->...i', 
                    np.linalg.inv(pose), 
                    my_utils.cat_ones(pts_world)
                )[..., :3]
                points_3D_cam2_sph = my_utils.carte2sph_3D(points_3D_cam2_carte)  
                points_2D_cam2_erp = my_utils.sph2erp_2D(points_3D_cam2_sph, height, width)  # [N, 2]
                depth_cam2 = points_3D_cam2_sph[..., 2] # [N,] 

                # 3. Splatting + Interpolation
                warped_img, warped_depth, warped_img_interp, warped_depth_interp, visited_pixels = my_utils.splatting_and_interpolation(
                    colors=colors_world,
                    depth_cam2=depth_cam2,
                    coord_cam2=points_2D_cam2_erp,
                    height=height,
                    width=width,
                    interpolation_mode='original',
                )
                my_utils.numpy_to_PIL(warped_img).save(f"{save_dir_}/dream_{i:02d}/01_warped_img.png")
                warped_depth[~visited_pixels] = 0 
                my_utils.depth_numpy_to_PIL(warped_depth).save(f"{save_dir_}/dream_{i:02d}/01_warped_depth.png")


                # 4. Obtain inpainting mask 
                #TODO (MASKING): Check the mask here. It needs to be separated into blending / inpainting
                operations = [
                    partial(minimum_filter, size=(3,3), axes=(0,1)),
                    partial(maximum_filter, size=(3,3), axes=(0,1)),
                    partial(maximum_filter, size=(3,3), axes=(0,1)),
                    partial(maximum_filter, size=(3,3), axes=(0,1)),
                    # partial(maximum_filter, size=(8, 8), axes=(0,1)),
                ]
                missing_info_masks = [~visited_pixels]
                for op in operations:
                    missing_info_masks.append(op(missing_info_masks[-1]))
                missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
                missing_info_masks_tile.save(f"{save_dir_}/dream_{i:02d}/02_missing_info_masks_tile.png")
                missing_info_mask = missing_info_masks[-1]
                inpainting_mask = missing_info_mask # TODO: separate the mask from "outside the eyes" and the one from "inside the eyes"

                # 3 (end) Remove pixels that should not be interpolated
                warped_img_interp[missing_info_mask] = np.nan
                warped_depth_interp[missing_info_mask] = np.nan
                my_utils.numpy_to_PIL(warped_img_interp).save(f"{save_dir_}/dream_{i:02d}/03_warped_img_interp.png")
                my_utils.depth_numpy_to_PIL(warped_depth_interp).save(f"{save_dir_}/dream_{i:02d}/03_warped_depth_interp.png")
                my_utils.depth_numpy_to_figure(warped_depth_interp).savefig(f"{save_dir_}/dream_{i:02d}/03_warped_depth_interp_figure.png")
                # np.save(f"{save_dir_}/dream_{i:02d}/03_warped_depth_interp.npy", warped_depth_interp)

                # 5. Inpaint panorama
                if not debug: 
                    pano_inpainted_raw = spherical_dreamer.inpaint_pano(
                        prompt=prompt, 
                        pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
                        mask=my_utils.numpy_to_PIL(inpainting_mask)
                    )
                    pano_inpainted_raw.save(f"{save_dir_}/dream_{i:02d}/XX_pano_rgb_inpainted_raw.png")
                else:
                    pano_inpainted_raw = Image.open(f"{save_dir_}/dream_{i:02d}/XX_pano_rgb_inpainted_raw.png")
                overlay_before = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img, inpainting_mask, alpha=0.5)) 
                overlay_before.save(f"{save_dir_}/dream_{i:02d}/04_overlay_before_inpainting.png")
                pano_inpainted_raw.save(f"{save_dir_}/dream_{i:02d}/04_pano_rgb_inpainted_raw.png")

                # 6. Blend inpainted panorama and source panorama
                pano_blend1, pano_blend2, mask_blend1, mask_blend2 = spherical_dreamer.blend(
                    pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
                    pano_inpainted_raw=pano_inpainted_raw,
                    missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
                    horizon_mask=my_utils.numpy_to_PIL(np.zeros_like(missing_info_mask).astype('bool')),
                ) 
                #TODO: since we removed horizon, check the blending strategy again. It is `compose` everywhere now. 
                #TODO: Also, Check if we need both blend1 and blend2
                
                mask_blend1.save(f"{save_dir_}/dream_{i:02d}/05_blend1_mask.png")
                mask_blend2.save(f"{save_dir_}/dream_{i:02d}/05_blend2_mask.png")
                pano_blend1.save(f"{save_dir_}/dream_{i:02d}/05_blend1_pano_rgb_inpainted.png")
                pano_blend2.save(f"{save_dir_}/dream_{i:02d}/05_blend2_pano_rgb_inpainted.png")

                pano_rgb_inpainted = pano_blend2
                pano_rgb_inpainted.save(f"{save_dir_}/dream_{i:02d}/06_pano_rgb_inpainted.png") #TODO: this is the same as blend2. Remove repetition

                # 7. Estimate depth
                if not debug:
                    depth2 = spherical_dreamer.estimate_pano_depth(
                        pano_rgb=np.array(pano_rgb_inpainted)
                    )
                    np.save(f"{save_dir_}/dream_{i:02d}/XX_estimated_depth.npy", depth2)
                else:
                    depth2 = np.load(f"{save_dir_}/dream_{i:02d}/XX_estimated_depth.npy")
                my_utils.depth_numpy_to_PIL(depth2).save(f"{save_dir_}/dream_{i:02d}/07_estimated_depth.png")
                my_utils.depth_numpy_to_figure(depth2).savefig(f"{save_dir_}/dream_{i:02d}/07_estimated_depth_figure.png")


                # 8. Blend depth

                # 8.a. Naive blending.
                blended_depth = np.zeros_like(warped_depth_interp)
                blended_depth[missing_info_mask] = depth2[missing_info_mask]
                blended_depth[~missing_info_mask] = warped_depth_interp[~missing_info_mask]

                # visalization & pointcloud
                if log_pcds:
                    pcd1_naive = PointCloud(
                        pts=my_utils.depth2world(
                            depth=blended_depth, pose=pose, sphere_radius=sphere_radius, height=height, width=width
                        ),
                        colors=(np.array(pano_rgb_inpainted)/255.0)
                    )
                    pointclouds[f"dream_{i:02d}"]["naive_blending"] = pcd1_naive
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_naive_blending.pkl", 'wb') as f:
                        pkl.dump(pcd1_naive, f)
                    plt.figure()
                    plt.imshow(blended_depth, cmap='plasma')
                    plt.colorbar()
                    plt.title('Blended Depth Naive')
                    plt.savefig(f"{save_dir_}/dream_{i:02d}/08_blended_depth_naive.png")
                    plt.show()

                # 8.b. Harmonic blending
                mask1, mask2, mask_boundary = get_harmonic_blending_mask(missing_info_mask)
                visualized_masks(mask1, mask2, mask_boundary)
                all_pts1_world = my_utils.depth2world(
                    depth=warped_depth_interp, pose=pose, sphere_radius=sphere_radius, height=height, width=width
                )
                all_pts2_world = my_utils.depth2world(
                    depth=depth2, pose=pose, sphere_radius=sphere_radius, height=height, width=width
                )
                pts1_world = all_pts1_world[mask1] # these are already good
                pts_target_boundary = all_pts1_world[mask_boundary] 
                pts2_world_exb = all_pts2_world[mask2] # these need to be deformed by mooving the boundary points to the target boundary points
                pts2_boundary = all_pts2_world[mask_boundary]
                pts2_world = np.concatenate((pts2_world_exb, pts2_boundary), axis=0)
                _mask_boundary = np.concatenate((np.zeros(pts2_world_exb.shape[0], dtype=bool), np.ones(pts2_boundary.shape[0], dtype=bool)), axis=0)
                mask_fixed = get_mask_fixed(translation_direction, my_utils.world2cam_carte_3D(pts2_world, pose))
                # verify_mask_fixed(translation_direction, pano_rgb_inpainted)

                # Deformation
                t0 = time.time()
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
                pts2_deformed_world = np.concatenate((pts2_deformed_world_exb, pts2_deformed_boundary), axis=0)
                colors2_exb = (np.array(pano_rgb_inpainted)/255.0)[mask2]
                colors2_boundary = (np.array(pano_rgb_inpainted)/255.0)[mask_boundary]
                colors2 = np.concatenate((colors2_exb, colors2_boundary), axis=0)

                # Visualization & pointcloud
                if log_pcds:
                    pts_3D_world_carte_new = np.zeros((height, width, 3), dtype=np.float32)
                    pts_3D_world_carte_new[mask1] = pts1_world
                    pts_3D_world_carte_new[mask2] = pts2_deformed_world_exb
                    pts_3D_world_carte_new[mask_boundary] = pts2_deformed_boundary
                    blended_depth_harmonic = my_utils.world2cam_sph_3D(pts_3D_world_carte_new, pose)[..., 2]
                    pcd1_harmonic = PointCloud(
                        pts=pts_3D_world_carte_new,
                        colors=np.array(pano_rgb_inpainted)/255.0
                    )
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_harmonic_blending.pkl", 'wb') as f:
                        pkl.dump(pcd1_harmonic, f)
                    pointclouds[f"dream_{i:02d}"]["harmonic_blending"] = pcd1_harmonic
                    plt.figure()
                    plt.imshow(blended_depth_harmonic, cmap='plasma')
                    plt.colorbar()
                    plt.title('Blended Depth Harmonic')
                    plt.savefig(f"{save_dir_}/dream_{i:02d}/08_blended_depth_harmonic.png")
                    plt.show()

                    # TODO: What does the new spherical image looks like from pose ? With deformed points ?

                
                # 9. Open World (But not at last iteration)
                if i < num_dreams - 1:
                    pts2_opened_cam_sph, _, mask_opened = my_utils.open_world(
                        forward_sph=translation_direction_sph,
                        pts_sph=my_utils.world2cam_sph_3D(pts2_deformed_world, pose),
                        mode=opening_mode,
                        delta_cut=2*np.pi/3
                    )
                    colors2 = colors2[mask_opened]
                    pts2_opened_world = my_utils.cam_sph2world_3D(
                        pts2_opened_cam_sph,
                        pose
                    )
                else:
                    pts2_opened_world = pts2_deformed_world

                if log_pcds:
                    pointclouds[f"dream_{i:02d}"]["final"] = PointCloud(
                        pts=pts2_opened_world,
                        colors=colors2
                    )
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_final.pkl", 'wb') as f:
                        pkl.dump(pointclouds[f"dream_{i:02d}"]["final"], f)

                # 10. Save new points
                pts_world = np.concatenate((pts_world, pts2_opened_world), axis=0)
                colors_world = np.concatenate((colors_world, colors2), axis=0)

                if log_pcds:
                    pointclouds[f"dream_{i:02d}"][f"total"] = PointCloud(
                        pts=pts_world,
                        colors=colors_world
                    )
                    with open(f"{save_dir_}/dream_{i:02d}/pcd_total.pkl", 'wb') as f:
                        pkl.dump(pointclouds[f"dream_{i:02d}"]["total"], f)

        # save pcd
        with open(f"{save_dir_}/pointclouds.pkl", 'wb') as f:
            pkl.dump(pointclouds, f)


    print("PYTHON SCRIPT SUCCESSFULLY RUN TO THE END !")