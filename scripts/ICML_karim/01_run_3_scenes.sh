#!/bin/bash

configs=(
    configs/expC/phantom_opera_cave_river.yaml
    # configs/expB/sound_of_music_grass_field.yaml
    # configs/expB/upside_down_stranger_things.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running blender renders with config: $config"
    blender --background --python 5v2_render_blender.py -- --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done