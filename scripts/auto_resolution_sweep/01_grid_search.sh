#!/bin/bash

# Grid values
CONTROLNET_SCALES=(0.8 0.85 0.9 0.95 0.99)
INFERENCE_STEPS=(28)
GUIDANCE_SCALES=(3.5)

# List of prompts
PROMPTS=(
  "Sandy beach, large driftwood in the foreground, calm sea beyond, realism style."
    ""
)

# Loop over all combinations
for prompt in "${PROMPTS[@]}"; do
  for c_scale in "${CONTROLNET_SCALES[@]}"; do
    for steps in "${INFERENCE_STEPS[@]}"; do
      for g_scale in "${GUIDANCE_SCALES[@]}"; do
        echo "Running with prompt=\"$prompt\", controlnet_conditioning_scale=$c_scale, num_inference_steps=$steps, guidance_scale=$g_scale"
        python test_auto_resolution.py \
          --controlnet_conditioning_scale "$c_scale" \
          --num_inference_steps "$steps" \
          --guidance_scale "$g_scale" \
          --prompt "$prompt"
      done
    done
  done
done