import torch
from diffusers import FluxPipeline
from typing import Optional


class FluxImageGenerator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
        )

        if self.device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_tiling()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        out_path: str,
        seed: int = 0,
        steps: int = 50,
        guidance_scale: float = 7.0,
        size: int = 1024,
    ):
        generator = torch.Generator("cpu").manual_seed(seed)

        image = self.pipe(
            prompt,
            height=size,
            width=size,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).images[0]

        image.save(out_path)
        return image


# ---------------- Example usage ----------------
OUT_DIR = "BASELINES_INPUT_IMGS"
PROMPTS = {
    "dense_rainforest_understory": (
        "A dense rainforest understory extending forward and behind the observer, thick "
        "overlapping vegetation with no sharp edges, large leaves in deep green tones, moist "
        "ground covered in roots and organic debris, overcast sky filtering light through "
        "the canopy, soft diffused illumination, humid and lush atmosphere, immersive tropical "
        "environment."
    ),
    "coral_reef_canyon": (
        "A wide coral reef canyon extending forward and behind the observer, smooth "
        "rock walls covered in coral and marine growth, vibrant yet softened colors including "
        "turquoise, coral pink, and sandy beige, filtered underwater lighting with soft light "
        "rays, floating particles in the water, calm immersive oceanic environment."
    ),
    "martian_badlands_2": (
        "A Martian landscape of rust-red and ochre soil, scattered with dark basalt "
        "rocks and patches of muted green and purple alien vegetation, under a dusty salmon-colored "
        "sky."
    ),
    "phantom_opera_cave_river": (
        "A large-scale 3D subterranean cave environment inspired by the Phantom of "
        "the Opera underground river setting, with no visible people or animals, an irregular "
        "cavern corridor extending forward and continuing behind the observer, no sharp edges "
        "or well-defined geometry, rounded organic rock formations softened by moisture and "
        "time, a dark slow-moving river running across the ground with reflective surface "
        "and subtle ripples, wet rocky banks with scattered stones and damp sediment, stalactites "
        "and stalagmites smoothed into natural shapes, faint mist hovering above water, dim "
        "atmospheric lighting as if from distant unseen lanterns creating soft warm reflections, "
        "deep shadows fading into darkness, cinematic gothic mood, immersive enclosed underground "
        "atmosphere with realistic rock and water textures and strong spatial depth."
    ),
    "sound_of_music_grass_field": (
        "A wide open rolling field of lush green grass inspired by The Sound of Music, "
        "gentle natural hills extending forward and continuing behind the observer with no "
        "harsh edges or defined geometry, thick healthy grass blades forming smooth wind-like "
        "patterns, scattered wildflower hints subtle and sparse, cloudy sky overhead with "
        "no visible sun, soft diffused daylight creating even illumination and minimal harsh "
        "shadows, distant tree line barely visible on the horizon, peaceful cinematic pastoral "
        "mood, realistic vegetation textures, immersive open countryside atmosphere."
    ),
    "upside_down_stranger_things": (
        "A desolate Upside Down-inspired landscape with cloudy oppressive skies and "
        "no visible sun, the environment stretching forward and behind the observer with "
        "organic uneven terrain, dark damp ground covered in tangled root-like growth and "
        "soft alien debris, floating ash-like particles suspended in the air, twisted vegetation "
        "silhouettes without sharp geometry, murky fog reducing visibility in the distance, "
        "muted blue-gray lighting with eerie contrast, wet reflective patches and slimy textures, "
        "cinematic horror mood, immersive otherworldly atmosphere."
    ),
}


generator = FluxImageGenerator()

for i, (name, prompt) in enumerate(PROMPTS.items()):
    generator.generate(
        prompt=prompt,
        out_path=f"{OUT_DIR}/{name}.png",
        seed=13 + i,
    )