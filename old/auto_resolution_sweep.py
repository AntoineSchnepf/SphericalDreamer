import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import os
import datetime
import argparse
import wandb



parser = argparse.ArgumentParser()
parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.6)
parser.add_argument("--num_inference_steps", type=int, default=28)
parser.add_argument("--guidance_scale", type=float, default=3.5)
parser.add_argument("--prompt", type=str, default="Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.")
args = parser.parse_args()

# wandb.init(
#     project="auto_resolution",
#     config={
#         "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
#         "num_inference_steps": args.num_inference_steps,
#         "guidance_scale": args.guidance_scale,
#         "prompt": args.prompt
#     },
#     name=f"CCS={args.controlnet_conditioning_scale}_prompt_on={int(not args.prompt=='')}"
# )

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"auto_resolution/{now}"

os.makedirs(save_dir, exist_ok=True)

# Load a control image
control_image = load_image(
  "dream_explore/inpainted_image.png"  # Path to your control image
)

control_image.save(os.path.join(save_dir, "control_image.png"))  # Save the control image for reference
wandb.log({"control_image": wandb.Image(control_image)})


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
#args

image = pipe(
    prompt=args.prompt,
    control_image=control_image,
    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    height=control_image.size[1],
    width=control_image.size[0]
).images[0]

image.save(os.path.join(save_dir, "output_image.png"))  # Save the output image
# wandb.log({"output_image": wandb.Image(image)})
# save hyperparameters
with open(os.path.join(save_dir, "hyperparameters.txt"), "w") as f:
    f.write(f"controlnet_conditioning_scale: {args.controlnet_conditioning_scale}\n")
    f.write(f"num_inference_steps: {args.num_inference_steps}\n")
    f.write(f"guidance_scale: {args.guidance_scale}\n")
    f.write(f"height: {control_image.size[1]}\n")
    f.write(f"width: {control_image.size[0]}\n")
    f.write(f"prompt: {args.prompt}\n")
