# load generated panorama + estimated depth map
import os
from pathlib import Path

# --- Removed problematic/unused imports ---
# from diffusers import FluxControlNetModel
# from diffusers.pipelines import FluxControlNetPipeline

import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from src.pipeline_flux import FluxPipeline
from src.pipeline_flux_fill import FluxFillPipeline

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

import my_utils  # assuming it's in PYTHONPATH

import argparse

# run this using:
# python test_flux_gen_and_inpaint.py --sdxl; python test_flux_gen_and_inpaint.py --flux


parser = argparse.ArgumentParser()
parser.add_argument("--sdxl", action="store_true", help="Run SDXL tests")
parser.add_argument("--flux", action="store_true", help="Run FLUX tests")
args = parser.parse_args()

def process_filename(prompt, seed):
    return f"{seed}_" + prompt.replace(",", "_").replace(" ", "_")[:100] + ".png"


prompts = [
    "ultra-detailed 360 panorama of a modern interior, realistic lighting",
    "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail."
]

save_dir = Path("OUTPUTS/ImageGen")
seed = 1234

# ---- FLUX ----- #
gen_guidance_scale = 7.5
inpaint_guidance_scale = 30.0
width = 1440
height = 720
flux_lora_pano_path = 'checkpoints/pano_lora_720*1440_v1.safetensors'
# ---------------- #

# ----- SD (SDXL) ------- #
device = "cuda" if torch.cuda.is_available() else "cpu"
negative_prompt = "lowres, noisy, distorted"
sd_lora_repo = "artificialguybr/360Redmond"
WIDTH, HEIGHT = 1600, 800
dtype = torch.float16 if device == "cuda" else torch.float32
# ----------------------- #


# ============ FLUX pipelines ============
if args.flux:
    flux_inpaint_pipeline = FluxFillPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        torch_dtype=torch.bfloat16,
    )
    flux_inpaint_pipeline.enable_model_cpu_offload()
    flux_inpaint_pipeline.enable_vae_tiling()

    flux_gen_pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    flux_gen_pipeline.enable_model_cpu_offload()
    flux_gen_pipeline.enable_vae_tiling()


    @torch.inference_mode()
    def generate_flux(prompt, seed=seed):
        pano_generated = flux_gen_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=gen_guidance_scale,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        return pano_generated


    @torch.inference_mode()
    def inpaint_flux(prompt, image, mask_pil, composition=True, seed=seed):
        # Ensure mask is single-channel L with 0/255
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        pano_inpainted = flux_inpaint_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_pil,
            strength=1.0,
            height=height,
            width=width,
            guidance_scale=inpaint_guidance_scale,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        if composition:
            composite_mask = ImageOps.invert(mask_pil)  # white=keep from original
            pano_inpainted_composite = Image.composite(image, pano_inpainted, composite_mask)
            return pano_inpainted, pano_inpainted_composite

        return pano_inpainted

if args.sdxl:
    # ============ SDXL pipelines ============
    # SDXL base for generation
    sd_gen_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    sd_gen_pipe.enable_attention_slicing()

    # FIX: use the proper SDXL inpainting checkpoint (not the base) for inpainting
    sd_inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    sd_inpaint_pipe.enable_attention_slicing()


    @torch.inference_mode()
    def generate_sd(prompt, seed=seed):
        generator = torch.Generator(device=device).manual_seed(seed)
        pano_generated = sd_gen_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=5.5,
            cross_attention_kwargs={"scale": 0.85},
            num_inference_steps=30,
            generator=generator,
        ).images[0]
        return pano_generated


    @torch.inference_mode()
    def inpaint_sd(prompt, image, mask_pil, composition=True, seed=seed):
        # Ensure mask is single-channel L with 0/255 (white=inpaint)
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        generator = torch.Generator(device=device).manual_seed(seed)
        pano_inpainted = sd_inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_pil,
            width=WIDTH,
            height=HEIGHT,
            guidance_scale=5.0,  # SDXL inpaint tends to like lower CFG than txt2img
            cross_attention_kwargs={"scale": 0.85},
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        if composition:
            composite_mask = ImageOps.invert(mask_pil)  # white=keep from original
            pano_inpainted_composite = Image.composite(image, pano_inpainted, composite_mask)
            return pano_inpainted, pano_inpainted_composite

        return pano_inpainted


# ===== LORA toggle & main loop =====
for lora in [False, True]:

    if lora:
        # FLUX LoRA
        if args.flux:
            try:
                flux_gen_pipeline.load_lora_weights(flux_lora_pano_path)
            except Exception as e:
                print(f"[WARN] Could not load FLUX LoRA for gen: {e}")

            try:
                flux_inpaint_pipeline.load_lora_weights(
                    flux_lora_pano_path, weight_dtype=torch.bfloat16
                )
            except Exception as e:
                print(f"[WARN] Could not load FLUX LoRA for inpaint: {e}")

        # SDXL LoRA
        if args.sdxl:
            try:
                sd_gen_pipe.load_lora_weights(sd_lora_repo)
                sd_inpaint_pipe.load_lora_weights(sd_lora_repo)
            except Exception as e:
                print(f"[WARN] Could not load SDXL LoRA: {e}")

    else:
        # Safer unload (guard against older diffusers without unload)
        if args.flux:
            for p in (flux_gen_pipeline, flux_inpaint_pipeline):
                if hasattr(p, "unload_lora_weights"):
                    try:
                        p.unload_lora_weights()
                    except Exception as e:
                        print(f"[WARN] unload_lora_weights failed: {e}")
        if args.sdxl:
            for p in (sd_gen_pipe, sd_inpaint_pipe):
                if hasattr(p, "unload_lora_weights"):
                    try:
                        p.unload_lora_weights()
                    except Exception as e:
                        print(f"[WARN] unload_lora_weights failed: {e}")


    for delta_seed in [0, 3296, 9174]:
        this_seed = seed + delta_seed

        # --- Inpainting test on a warped image (ensure output dir exists) ---
        prompt = (
            """A realistic illustration of a college campus. In the middle ground, several academic buildings with brick facades and large windows stand prominently. In the background, a bright blue sky with scattered clouds stretches across the scene. In the foreground, a few elements commonly found on campus, such as students walking, bicycles parked along a path, and a grassy lawn with trees, add depth and life to the scene"""
        )

        img_path = "OUTPUTS/SphericalDreamerRecurse/14_campus_cylinder_opening/dream_01/03_warped_img_interp.png"
        msk_path = "OUTPUTS/SphericalDreamerRecurse/14_campus_cylinder_opening/dream_01/05_blend1_mask.png"

        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(msk_path).convert("L")

        # FLUX inpaint
        if args.flux:
            pano_inpainted, pano_inpainted_composite = inpaint_flux(
                prompt, image_pil, mask_pil, composition=True, seed=this_seed
            )
            out_path = save_dir / f"FLUX{'_lora' if lora else ''}" / "inpainting" / process_filename(prompt, this_seed)
            os.makedirs(out_path.parent, exist_ok=True)
            pano_inpainted_composite.save(out_path)

        # SDXL inpaint
        if args.sdxl:
            pano_inpainted, pano_inpainted_composite = inpaint_sd(
                prompt, image_pil, mask_pil, composition=True, seed=this_seed
            )
            out_path = save_dir / f"SDXL{'_lora' if lora else ''}" / "inpainting" / process_filename(prompt, this_seed)
            os.makedirs(out_path.parent, exist_ok=True)
            pano_inpainted_composite.save(out_path)

        # --- Generation & "inpaint-as-generation" for all prompts ---
        for pmt in prompts:
            print(f"--- lora: {lora}, seed: {this_seed} ---")

            # FLUX generate
            if args.flux:
                pano_generated = generate_flux(pmt, seed=this_seed)
                out_path = save_dir / f"FLUX{'_lora' if lora else ''}" / "generation" / process_filename(pmt, this_seed)
                os.makedirs(out_path.parent, exist_ok=True)
                pano_generated.save(out_path)

            # SDXL generate
            if args.sdxl:
                pano_generation = generate_sd(pmt, seed=this_seed)
                out_path = save_dir / f"SDXL{'_lora' if lora else ''}" / "generation" / process_filename(pmt, this_seed)
                os.makedirs(out_path.parent, exist_ok=True)
                pano_generation.save(out_path)

            # FLUX inpaint as generation
            if args.flux:
                img = my_utils.numpy_to_PIL((np.ones((height, width, 3)) * 127).astype(np.uint8))  # neutral gray
                msk = Image.new("L", (width, height), 255)  # white = paint all
                pano_inpainted, pano_inpainted_composite = inpaint_flux(pmt, img, msk, composition=True, seed=this_seed)
                out_path = save_dir / f"FLUX{'_lora' if lora else ''}" / "inpainting_as_generation" / process_filename(pmt, this_seed)
                os.makedirs(out_path.parent, exist_ok=True)
                pano_inpainted_composite.save(out_path)

            # SDXL inpaint as generation
            if args.sdxl:
                img = Image.new("RGB", (WIDTH, HEIGHT), (127, 127, 127))
                msk = Image.new("L", (WIDTH, HEIGHT), 255)  # white=paint, black=keep
                pano_inpainted, pano_inpainted_composite = inpaint_sd(pmt, img, msk, composition=True, seed=this_seed)
                out_path = save_dir / f"SDXL{'_lora' if lora else ''}" / "inpainting_as_generation" / process_filename(pmt, this_seed)
                os.makedirs(out_path.parent, exist_ok=True)
                pano_inpainted_composite.save(out_path)