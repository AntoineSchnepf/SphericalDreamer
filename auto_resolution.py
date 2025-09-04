import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import os
import numpy as np
import matplotlib.pyplot as plt

controlnet_conditioning_scale = 0.6
num_inference_steps = 50
guidance_scale = 3.5
prompt = "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style."




# Load a control image
control_image = load_image(
  "SphericalDreamerRecurse_outputs/city/dream_00/01_pano_rgb.png"  # Path to your control image
)
# Define box (left, upper, right, lower)

# left=70
left = int(4* 1440/9)
w = int(1440/9)
upper=int(4*720/9)-25
h = int(720/9)
box = (left, upper, left + w, upper + h)
control_image_crop = control_image.crop(box)
factor=9
control_image_crop_upsampled = control_image_crop.resize((w*factor, h*factor))


# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
  "jasperai/Flux.1-dev-Controlnet-Upscaler",
  torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
  "black-forest-labs/FLUX.1-dev",
  controlnet=controlnet,
  torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
image = pipe(
    prompt = "A bustling city street at night, neon lights reflecting on wet pavement, realism style.",
    control_image=control_image_crop_upsampled,
    controlnet_conditioning_scale = controlnet_conditioning_scale,
    num_inference_steps = num_inference_steps,
    guidance_scale = guidance_scale,
    height=control_image_crop_upsampled.size[1],
    width=control_image_crop_upsampled.size[0]
).images[0]
image