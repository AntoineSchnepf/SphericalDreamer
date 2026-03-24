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
    exp0/caverns.yaml
    exp0/cityscape.yaml
    exp0/desert.yaml
    exp0/highland.yaml
    exp0/plains.yaml
    exp0/ruins.yaml
    exp0/stronghold.yaml
    exp0/tundra.yaml
    exp0/volcanic.yaml
    exp0/wetland.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    bash "$script" "$config"

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."