#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"
script="$repo_dir/dream.sh"

# Ensure the script exists
if [ ! -f "$script" ]; then
    echo "❌ Error: Python script not found at $script"
    exit 1
fi

# Define array of configs
configs=(
    exp2/desert.yaml
    exp2/plains.yaml
    exp2/seaside.yaml
    exp2/tundra.yaml
    exp2/volcanic.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    python 3_fix_world_geometry.py --config $config
    python 4_render_video.py --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."