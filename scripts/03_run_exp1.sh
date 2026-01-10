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
    exp1/caverns.yaml
    exp1/cityscape.yaml
    exp1/desert.yaml
    exp1/highland.yaml
    exp1/plains.yaml
    exp1/ruins.yaml
    exp1/stronghold.yaml
    exp1/tundra.yaml
    exp1/volcanic.yaml
    exp1/wetland.yaml
    exp1/forest.yaml
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