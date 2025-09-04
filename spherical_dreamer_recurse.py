# load generated panorama + estimated depth map
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
from scipy.ndimage import maximum_filter, minimum_filter
import logging
from scipy.interpolate import griddata as interp_grid
import matplotlib.pyplot as plt
import time
# local imports
_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir) 
from utils.depth_alignment import Pano_depth_estimation
import my_utils 

logging.disable(logging.CRITICAL + 1)

def pil_mask_to_numpy_bool(pil_mask):
    return np.array(pil_mask.convert("L")) > 0

def numpy_bool_to_pil_mask(mask):
    return Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

def show_masks(masks, alpha=0.5, background=None):
    """
    Visualize several boolean masks on the same image with color overlaps.

    Parameters
    ----------
    masks : list of np.ndarray
        List of boolean arrays (all same shape).
    alpha : float
        Transparency for overlays.
    background : np.ndarray or None
        Optional grayscale/RGB image to show under masks.
    """
    H, W = masks[0].shape
    # Assign distinct colors (cycle through tab colormap)
    cmap = plt.cm.get_cmap("tab10", len(masks))
    colors = [np.array(cmap(i)[:3]) for i in range(len(masks))]

    # Background (default = white)
    if background is None:
        img = np.ones((H, W, 3), dtype=float)
    else:
        # Normalize background to 0-1 RGB
        bg = np.array(background, dtype=float)
        if bg.ndim == 2:
            bg = np.stack([bg]*3, axis=-1)
        bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        img = bg

    # Blend each mask color
    for m, col in zip(masks, colors):
        m3 = np.stack([m]*3, axis=-1)
        img = np.where(m3, (1-alpha)*img + alpha*col, img)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

def seamless_blend(src, dst, mask):
    """
    Blend src into dst guided by mask (all PIL.Image objects).
    src and dst must be the same size.
    Returns a PIL.Image with seamless blending.
    """
    # Convert to OpenCV format
    src_cv  = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
    dst_cv  = cv2.cvtColor(np.array(dst), cv2.COLOR_RGB2BGR)
    mask_cv = np.array(mask.convert("L"))

    # Compute center of panoramic image
    height, width = mask_cv.shape

    # Blend
    # -- v2 --
    br = cv2.boundingRect(mask_cv) # bounding rect (x,y,width,height)
    centerOfBR = (br[0] + br[2] // 2, br[1] + br[3] // 2)
    blended_cv = cv2.seamlessClone(src_cv, dst_cv, mask_cv, centerOfBR, cv2.NORMAL_CLONE)

    # -- v1 --
    # center = (width//2, height//2)
    # blended_cv = cv2.seamlessClone(src_cv, dst_cv, mask_cv, center, cv2.NORMAL_CLONE)


    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(blended_cv, cv2.COLOR_BGR2RGB))

def interpolate_with_flow(colors, depths, flow, mode='original'):
    """
    args:
        colors: np.array with values in [0,1]
        depths: np.array with float values 
        flow: dict with keys as pixel coordinates in source image and values as pixels coordinates in target (warped) image.
        It looks like {(u_, v_) : (i, j)}

    """
    width = colors.shape[1]
    height = colors.shape[0]

    grid = np.stack((np.meshgrid(range(width), range(height))), axis=-1).reshape(-1, 2).astype(np.float32)
    rgb_values = []
    depth_values = []
    points = []
    points_rounded = []
    for (u_, v_), (a,b) in flow.items():
        rgb_values.append(
            colors[b,a]
        )
        depth_values.append(
            depths[b,a]
        )
        points.append((u_, v_))
        points_rounded.append((round(u_), round(v_)))

    points = np.array(points)
    rgb_values = np.array(rgb_values)
    points_rounded = np.array(points_rounded)

    if mode == 'original':
        points_ = points
    elif mode == 'rounded':
        points_ = points_rounded
    else:
        raise ValueError("Mode must be 'rounded' or 'original'.")
    
    image_interp = interp_grid(
        points_,
        rgb_values,
        grid, 
        method='linear', 
        # fill_value=0
    ).reshape(height,width,3)

    depth_interp = interp_grid(
        points_,
        depth_values,
        grid,
        method='linear',
        # fill_value=0
    ).reshape(height,width)

    return image_interp, depth_interp

def erp_to_world(points_2D_cam_erp, height, width, depth, pose, sphere_radius=1.0):
    """
    Convert Equirectangular coordinates to world coordinates.
    
    Args:
        points_2D_cam_erp (np.array): Equirectangular coordinates of shape [..., 2].
        depth (np.array): Depth map of shape [...].
        pose (np.array): Camera pose matrix of shape [4, 4].
        sphere_radius (float): Radius of the sphere.
    
    Returns:
       points_3D_world_carte: np.array w. shape [..., 3]. World coordinates. Convention X, Y, Z.
    """
    assert np.all(points_2D_cam_erp.shape[:-1] == depth.shape)
    points_2D_cam_sph = my_utils.erp2sph_2D(points_2D_cam_erp, erp_image_height=height, erp_image_width=width)
    r = depth * sphere_radius
    points_3D_cam_sph = np.concatenate((points_2D_cam_sph, np.expand_dims(r, axis=-1)), axis=-1)
    points_3D_cam_carte = my_utils.sph2carte_3D(points_3D_cam_sph)
    points_3D_world_carte = np.einsum('ij,...j->...i', pose, my_utils.cat_ones(points_3D_cam_carte))[..., :3]
    return points_3D_world_carte

def depth_aware_naive_splatting_vectorized(colors1, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    Vectorized depth-aware splatting with a z-buffer.
    Supports inputs shaped [H, W, *] or flattened [HW, *].
    """
    # Basic checks
    assert colors1.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert coord_cam1.shape[-1] == 2
    assert colors1.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    # Flatten to [N, ...]
    colors1   = colors1.reshape((-1, 3))
    coord_cam1 = coord_cam1.reshape((-1, 2))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    # Round target coordinates to nearest integer pixel (u -> x/col, v -> y/row)
    u = coord_cam2[:, 0]
    v = coord_cam2[:, 1]
    u_r = np.rint(u).astype(np.int32)
    v_r = np.rint(v).astype(np.int32)

    # Keep only those that fall inside the target frame
    in_bounds = (u_r >= 0) & (u_r < width) & (v_r >= 0) & (v_r < height)
    if not np.any(in_bounds):
        warped_img   = np.zeros((height, width, 3), dtype=np.float32)
        warped_depth = np.full((height, width), np.inf, dtype=np.float32)
        visited      = np.zeros((height, width), dtype=bool)
        return warped_img, warped_depth, {}, visited

    # Restrict to valid points
    u_r = u_r[in_bounds]
    v_r = v_r[in_bounds]
    depths = depth_cam2[in_bounds].astype(np.float32)
    colors = colors1[in_bounds]
    coord1 = coord_cam1[in_bounds]
    u_float = u[in_bounds]
    v_float = v[in_bounds]

    # Linearized target indices (row-major)
    tgt_lin = (v_r.astype(np.int64) * width + u_r.astype(np.int64))

    # Resolve collisions per target pixel with z-buffer: keep the *nearest* depth
    order = np.lexsort((depths, tgt_lin))        # primary: tgt_lin, secondary: depth (ascending)
    tgt_sorted = tgt_lin[order]
    _, first_idx = np.unique(tgt_sorted, return_index=True)
    winners = order[first_idx]

    # Winners' data
    u_win_r = u_r[winners]
    v_win_r = v_r[winners]
    depths_win = depths[winners]
    colors_win = colors[winners]
    coord1_win = coord1[winners]      # (a, b) source coordinates
    u_win_f = u_float[winners]        # unrounded u for flow key
    v_win_f = v_float[winners]        # unrounded v for flow key

    # Allocate outputs
    warped_img   = np.zeros((height, width, 3), dtype=np.float32)
    warped_depth = np.full((height, width), np.inf, dtype=np.float32)
    visited      = np.zeros((height, width), dtype=bool)

    # Scatter winners into outputs
    warped_img[v_win_r, u_win_r]   = colors_win
    warped_depth[v_win_r, u_win_r] = depths_win
    visited[v_win_r, u_win_r]      = True

    # Flow mapping: map exact (float u, float v) -> (a, b) from coord_cam1
    # Convert to native Python floats for dict keys/values
    flow = {(float(uf), float(vf)): (ab[0], ab[1])
            for uf, vf, ab in zip(u_win_f, v_win_f, coord1_win)}

    return warped_img, warped_depth, flow, visited

def depth_aware_naive_splatting(colors1, coord_cam1, coord_cam2, depth_cam2, height, width):
    """
    This functions computes a new image, at a new camera location, based on:
        (i) a set coordinates of colored points in the new image: `coord_cam2` (float values)
        (ii) corresponding colors: `colors1`
        (iii) corresponding depths: `depth_cam2`
    This functions rounds the float values, to obtain proper pixel location. If there are multiple pixels for a location, 
    the point that is the nearest to the camera is chosen. This models occlusions. 

    It is a simple form of splatting, using a z-buffer. 

    args:
        img: np.array with values in [0,1]. Shape [H, W, 3] or [HW, 3].Colors of the pixels at canonical locations(
            [[[0,0], ..., [0,W]],
            ...
            [H,0], ..., [H,W]]]
        ) in the source image, i.e. the source image itself
        coord_cam2: pixel coordinates in the new image (float values). Shape [H, W, 2] or [HW, 2].
        depth_cam2: np.array. Depth at new camera location (float values). Shape [H, W] or [HW].

        Modif 20 Aug: img, coord_cam2 and depth_cam2 can be flattened
    """
    assert colors1.shape[-1] == 3
    assert coord_cam2.shape[-1] == 2
    assert colors1.shape[:-1] == coord_cam2.shape[:-1] == depth_cam2.shape == coord_cam1.shape[:-1]

    colors1 = colors1.reshape((-1, 3))
    coord_cam1 = coord_cam1.reshape((-1, 2))
    coord_cam2 = coord_cam2.reshape((-1, 2))
    depth_cam2 = depth_cam2.reshape((-1,))

    warped_img = np.zeros(shape=(height, width, 3), dtype=np.float32)
    warped_depth = np.full(shape=(height, width) , fill_value=np.inf, dtype=np.float32)
    visited_pixels = np.zeros(shape=(height, width), dtype=bool)  # keep track of visited pixels
    # more_than_once_visited = np.zero(shape=(height, width), dtype=bool)  # keep track of pixels visited more than once
    # visited_count = 0


    flow_mapping = {}
    # Iterate over all the 3D points 
    for k in range(len(coord_cam2)):
        (a,b) = coord_cam1[k] # (a, b) represent the coordinates of the current point in the source image
        (u,v) = coord_cam2[k] # (u, v) represent the coordinates of the current point in the target image

        u_ = int(round(u))  
        v_ = int(round(v))

        color = colors1[k]
        depth2 = depth_cam2[k]
        
        if 0 <= u_ < width and 0 <= v_ < height:

            # /!\ Reference frame for an image inverts horizontal and vertical axis
            if warped_depth[v_, u_] > depth2 : # If this points is closer than previous
                warped_depth[v_, u_] = depth2 
                warped_img[v_, u_] = color 
                visited_pixels[v_, u_] = True
                flow_mapping[(u_, v_)] = ((a, b), (u, v))

    flow = {}
    for ((a, b), (u, v)) in flow_mapping.values():
        flow[(float(u), float(v))] = (a, b)

    return warped_img, warped_depth, flow, visited_pixels
                        
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
        self.pano_gen_pipeline.load_lora_weights(self.flux_lora_pano_path) # change this.
        self.pano_gen_pipeline.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
        self.pano_gen_pipeline.enable_vae_tiling()
        
    @torch.no_grad()
    def gen_pano(self, prompt):
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
        self.pano_inpaint_pipeline.load_lora_weights(self.flux_lora_pano_path) # change this.
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
        mask_blend1.save(f"{save_dir_}/dream_{i:02d}/09_blend1_mask.png")
        pano_blend1 = self._blend(
            pano_inpainted_raw, 
            pano_rgb, 
            mask_blend1, 
            mode='compose'
        )

        # iii. seamless blending
        mask_blend2=horizon_mask
        mask_blend2.save(f"{save_dir_}/dream_{i:02d}/09_blend2_mask.png")
        pano_blend2 = self._blend(
            pano_inpainted_raw, 
            pano_blend1, 
            mask_blend2,
            mode='seamless'
        )

        return pano_blend1, pano_blend2 #TODO: only pano_blend1 is needed

    def _blend(self, src, dst, mask, mode):
        "Blends two images together, guided by mask. All arguments should be PIL.Image"

        # Naive blending. Just compose the images
        if mode == 'compose':
            pano_blended = Image.composite(src, dst, mask)

        # Seamless blending, with smoothing along the mask edges
        elif mode == 'seamless':
            pano_blended = seamless_blend(src, dst, mask)
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



if __name__ == "__main__":

    # ---- args ----
    debug = True
    auto_resolution = False
    use_lama = False
    remove_forward_horizon = True
    save_dir = "SphericalDreamerRecurse"
    num_dreams = 5
    if debug:
        num_dreams = 1
    delta_walk = 0.1
    prompts = [
        "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail."
        # "A serene forest scene with a small stream, dappled sunlight filtering through the leaves, realism style.",
        # "A bustling city street at night, neon lights reflecting on wet pavement, realism style.",
        # "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.",
        # "A wide field under daylight, covered in lush green grass with worn paths where the grass has been trampled by many footsteps. In the center of the field stands a large concert stage, decorated with bold triangular patterns. On the stage rests a single guitar, but no performers are present. In front of the stage, a lively crowd gathers, waiting for the show to begin."
    ]
    expnames=[
        # "09_bali_RFW_warped_depth", 
        # "forest", 
        # "city", 
        "beach", 
        # "the_stage",
    ]
    # ---------------

    if debug:
        for _ in range(10):
            print("/!\ Debug mode is on /!\ ")

    # 0. Initialization
    width = 1440
    height = 720
    pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    sphere_radius = 1.0

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir = '/tmp/pano_depth_temp'
    )

    for expname, prompt in zip(expnames, prompts):
        save_dir_ = f"{save_dir}/{expname}"

        for i in range(0, num_dreams):
            os.makedirs(os.path.join(save_dir_, f"dream_{i:02d}"), exist_ok=True)
            if i == 0 :
                # 1. Generate panorama
                if not debug:
                    pano_rgb = spherical_dreamer.gen_pano(
                        prompt=prompt,
                    )
                    pano_rgb.save(f"{save_dir_}/dream_{i:02d}/01_pano_rgb.png")
                pano_rgb = Image.open(f"{save_dir_}/dream_{i:02d}/01_pano_rgb.png")

            else:
                pano_rgb = Image.open(f"{save_dir_}/dream_{i-1:02d}/10_pano_rgb_inpainted.png")
                pano_rgb.save(f"{save_dir_}/dream_{i:02d}/01_pano_rgb.png")

                if auto_resolution:
                    pano_rgb = Image.open(f"{save_dir_}/dream_{i-1:02d}/12_pano_rgb_inpainted_autores.png")
                    pano_rgb.save(f"{save_dir_}/dream_{i:02d}/02_pano_rgb_autores.png")

            # 3. Estimate depth
            if not debug:
                depth = spherical_dreamer.estimate_pano_depth(
                    pano_rgb=np.array(pano_rgb)
                )
                my_utils.depth_numpy_to_PIL(depth).save(f"{save_dir_}/dream_{i:02d}/03_depth.png")
                np.save(f"{save_dir_}/dream_{i:02d}/03_depth.npy", depth)
            depth = np.load(f"{save_dir_}/dream_{i:02d}/03_depth.npy")  # load depth

            # 3.5 (Optional) Remove Forward Horizon
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

            # 4. Project in the world
            points_2D_cam1_erp = np.stack((np.meshgrid(range(width), range(height))), axis=-1) 
            points_3D_world_carte = erp_to_world(
                points_2D_cam1_erp,
                height=spherical_dreamer.pano_height,
                width=spherical_dreamer.pano_width,
                depth=depth,
                pose=pose,
                sphere_radius=sphere_radius,
            ) # [H, W, 3]

            # 5. Reproject to new camera pose (cam2)
            translation_direction = np.array([1, 0, 0], dtype=np.float32) 
            translation_direction /= np.linalg.norm(translation_direction)  
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
            # t = time.time()
            warped_img, warped_depth, flow, visited_pixels = depth_aware_naive_splatting_vectorized(
                colors1=np.array(pano_rgb)/255.0,  # normalize to [0,1]
                coord_cam1=points_2D_cam1_erp,
                coord_cam2=points_2D_cam2_erp,
                depth_cam2=depth_cam2,
                height=spherical_dreamer.pano_height,
                width=spherical_dreamer.pano_width,
            )

            # print("Vectorized splatting time:", time.time() - t)
            my_utils.numpy_to_PIL(warped_img).save(f"{save_dir_}/dream_{i:02d}/04_warped_img.png")
            warped_depth[~visited_pixels] = 0 
            my_utils.depth_numpy_to_PIL(warped_depth).save(f"{save_dir_}/dream_{i:02d}/04_warped_depth.png")


            # 6 bis (optional) Retrieving forward horizon mask in the warped image
            if remove_forward_horizon:
                horizon_mask_warped = np.zeros(shape=(height, width), dtype=bool)
                horizon_points_2D_cam2_erp = np.round(points_2D_cam2_erp[horizon_mask]).astype("int")
                horizon_mask_warped[
                    horizon_points_2D_cam2_erp[:, 1], horizon_points_2D_cam2_erp[:, 0]
                ] = True

                closed_horizon_mask_warped = my_utils.close_mask(my_utils.fill_mask(horizon_mask_warped), size=10)

            # 7. Interpolate OR Lama inpaint
            if use_lama:
                raise NotImplementedError("lama inpainting is depreciated")
                print("lama inpainting")
                mask_lama = visited_pixels 
                mask_lama = copy.deepcopy(visited_pixels)
                operations = [
                    # partial(minimum_filter, size=(3,3), axes=(0,1)),
                ]
                for op in operations:
                    mask_lama = op(mask_lama)
                warped_img_interp = np.array(spherical_dreamer.lama_inpaint(
                    my_utils.numpy_to_PIL(warped_img), 
                    my_utils.numpy_to_PIL(~mask_lama).convert('L')
                ))/255.0

            else:
                warped_img_interp, warped_depth_interp = interpolate_with_flow(
                    colors=np.array(pano_rgb)/255.0, 
                    depths=depth_cam2, 
                    flow=flow,
                    mode='rounded'
                )
                my_utils.numpy_to_PIL(warped_img_interp).save(f"{save_dir_}/dream_{i:02d}/05_warped_img_interp.png")
                my_utils.depth_numpy_to_PIL(warped_depth_interp).save(f"{save_dir_}/dream_{i:02d}/05_warped_depth_interp.png")
                np.save(f"{save_dir_}/dream_{i:02d}/05_warped_depth_interp.npy", warped_depth_interp)

            # 8. Obtain mask    
            operations = [
                partial(minimum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                partial(maximum_filter, size=(3,3), axes=(0,1)),
                # partial(maximum_filter, size=(8, 8), axes=(0,1)),
            ]
            missing_info_masks = [(1 - visited_pixels)]
            for op in operations:
                missing_info_masks.append(op(missing_info_masks[-1]))
            missing_info_masks_tile = my_utils.tile_image([my_utils.numpy_to_PIL(m) for m in missing_info_masks])
            missing_info_masks_tile.save(f"{save_dir_}/dream_{i:02d}/06_missing_info_masks_tile.png")
            missing_info_mask = missing_info_masks[-1]
            inpainting_mask = missing_info_mask | closed_horizon_mask_warped


            # 9. Inpaint panorama
            if not debug: 
                overlay_before = my_utils.numpy_to_PIL(my_utils.overlay_mask(warped_img, inpainting_mask, alpha=0.5)) 
                overlay_before.save(f"{save_dir_}/dream_{i:02d}/07_overlay_before_inpainting.png")
                pano_inpainted_raw = spherical_dreamer.inpaint_pano(
                    prompt=prompt, 
                    pano_rgb=my_utils.numpy_to_PIL(warped_img_interp), 
                    mask=my_utils.numpy_to_PIL(inpainting_mask)
                )
                pano_inpainted_raw.save(f"{save_dir_}/dream_{i:02d}/08_pano_rgb_inpainted_raw.png")
            pano_inpainted_raw = Image.open(f"{save_dir_}/dream_{i:02d}/08_pano_rgb_inpainted_raw.png")

            # 10. Blend inpainted panorama and source panorama
            pano_blend1, pano_blend2 = spherical_dreamer.blend(
                pano_rgb=my_utils.numpy_to_PIL(warped_img_interp),
                pano_inpainted_raw=pano_inpainted_raw,
                missing_info_mask=my_utils.numpy_to_PIL(missing_info_mask),
                horizon_mask=my_utils.numpy_to_PIL(closed_horizon_mask_warped)
            )

            pano_blend1.save(f"{save_dir_}/dream_{i:02d}/09_blend1_pano_rgb_inpainted.png")
            pano_blend2.save(f"{save_dir_}/dream_{i:02d}/09_blend2_pano_rgb_inpainted.png")

            pano_rgb_inpainted = pano_blend2
            pano_rgb_inpainted.save(f"{save_dir_}/dream_{i:02d}/10_pano_rgb_inpainted.png") #TODO: this is the same as blend2. Remove repetition

            # 11. Auto-resolution on inpainted image
            if auto_resolution:
                if not debug:
                    pano_rgb_inpainted_autores = spherical_dreamer.improve_pano_resolution(pano_rgb_inpainted, prompt)
                    pano_rgb_inpainted_autores.save(f"{save_dir_}/dream_{i:02d}/11_pano_rgb_inpainted_autores_raw.png")
                pano_rgb_inpainted_autores = Image.open(f"{save_dir_}/dream_{i:02d}/11_pano_rgb_inpainted_autores_raw.png")
                # cast new autores pixels where we used interpolation before
                # composite_mask =  pass

                m1 = (missing_info_masks[0]).astype('bool') # all pixels to be inpainted or interpolated
                m2 = (missing_info_masks[-1]).astype('bool') # all pixels to be inpainted
                composite_mask= m1 & ~m2 # only interpolated pixels

                composite_mask = my_utils.numpy_to_PIL(composite_mask)
                composite_mask = ImageOps.invert(composite_mask.convert("L"))

                pano_inpainted_autores = Image.composite(pano_rgb_inpainted, pano_rgb_inpainted_autores, composite_mask)
                pano_inpainted_autores.save(f"{save_dir_}/dream_{i:02d}/12_pano_rgb_inpainted_autores.png")


    print("PYTHON SCRIPT SUCCESSFULLY RUN TO THE END !")