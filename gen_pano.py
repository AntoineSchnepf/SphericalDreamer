import torch
import os
import random
import argparse
from src.pipeline_flux import FluxPipeline


save_dir = "OUTPUTS/gen_pano"
prompt = "A wide panoramic landscape with a bright blue sky, majestic mountains in the background, a calm turquoise sea in the foreground, and lush greenery along the shore. The scene should feel vibrant, sunny, and relaxing, like a holiday postcard photograph, with realistic lighting and high detail."
lora_path = 'checkpoints/pano_lora_720*1440_v1.safetensors' 
seed = 119223


os.makedirs(f"{save_dir}", exist_ok=True)


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(lora_path) # change this.
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU




pipe.enable_vae_tiling()
    
image = pipe(prompt, 
            height=720,
            width=1440,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=50, 
            blend_extend=2,
            guidance_scale=7).images[0]

image = image.resize((2048,1024))

image.save(f"{save_dir}/rgb.png")
