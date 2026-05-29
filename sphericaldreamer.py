
from src.pipeline_flux import FluxPipeline
from src.pipeline_flux_fill import FluxFillPipeline
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import torch
import numpy as np
from PIL import Image
import logging
import contextlib
from io import StringIO
# local imports
from utils.depth_alignment import Pano_depth_estimation
from render_pcd import render_v2
import my_utils
from typing import Union

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
    
    def estimate_pano_depth(self, pano_rgb:Union[Image.Image, np.array]):
        """
        args:
            `pano_rgb`: PIL/np.array of shape [pano_h,pano_w,3] and values in [0-255]        
        returns:
            pano_depth: np.array of shape [pano_h,pano_w] and values in [0-1]
        """
        # --- Convert PIL → numpy ---
        if isinstance(pano_rgb, Image.Image):
            pano_rgb = np.array(pano_rgb)

        # --- Ensure uint8 RGB in [0, 255] ---
        if pano_rgb.dtype != np.uint8:
            # If float in [0,1], scale up
            if pano_rgb.max() <= 1.0:
                pano_rgb = (pano_rgb * 255).clip(0, 255).astype(np.uint8)
            else:
                raise ValueError(f"pano_rgb has dtype {pano_rgb.dtype} with max value {pano_rgb.max()}. Expected uint8 in [0,255] or float in [0,1].")

        if self .depth_model == '360mono':
            return self.estimate_pano_depth_360mono(pano_rgb)
        elif self.depth_model == 'egformer':
            return self.estimate_pano_depth_egformer(pano_rgb)
        else:
            raise ValueError(f"Unknown depth model: {self.depth_model}. Should be either '360mono' or 'egformer'.")
        
    @torch.no_grad()
    def estimate_pano_depth_360mono(self, pano_rgb:Union[Image.Image, np.array]):
        """
        args:
            `pano_rgb`: PIL/np.array of shape [pano_h,pano_w,3] and values in [0-255]        
        returns:
            pano_depth: np.array of shape [pano_h,pano_w] and values in [0-1]
        """
        with contextlib.redirect_stdout(StringIO()):
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
        # self.pano_inpaint_pipeline.load_lora_weights(self.flux_lora_pano_path) # Antoine: Do not use the lora for inpainting, it yields worse results. 
        self.pano_inpaint_pipeline.enable_model_cpu_offload()
        self.pano_inpaint_pipeline.enable_vae_tiling() 

    @torch.no_grad()   
    def inpaint_pano(self, prompt, pano_rgb, mask, strength=1.0, height=None, width=None, seed_override=None):
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
            strength=strength,
            height=height if height else self.pano_height,
            width=width if width else self.pano_width,
            guidance_scale=30.0,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),  
        ).images[0]

        return pano_inpainted_raw

    def blend(self, pano_rgb, pano_inpainted_raw, missing_info_mask, blending_mode='compose'):

        #ii. compose blending
        mask_blend = missing_info_mask
        pano_blend = self._blend(
            pano_inpainted_raw, 
            pano_rgb, 
            mask_blend, 
            mode=blending_mode
        )

        return pano_blend

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
            prompt=prompt, 
            control_image=pano_rgb,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
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

    def _release_lama_memory(self):
        if self.is_lama_init:
            del self.lama_model
            self.is_lama_init = False
    
    def _release_flux_memory(self):
        if self.is_pano_generator_init:
            del self.pano_gen_pipeline
            self.is_pano_generator_init = False

    def _release_flux_inpainting_memory(self):
        if self.is_inpainting_model_init:
            del self.pano_inpaint_pipeline
            self.is_inpainting_model_init = False

    def release_all_memory(self):
        self._release_flux_memory()
        self._release_flux_inpainting_memory()
        self._release_lama_memory()