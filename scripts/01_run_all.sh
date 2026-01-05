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
    # exp0/0_caverns.yaml
    # exp0/0_cityscape.yaml
    # exp0/0_desert.yaml
    # exp0/0_highland.yaml
    # exp0/0_plains.yaml
    # exp0/0_ruins.yaml
    # exp0/0_stronghold.yaml
    exp0/0_tundra.yaml
    exp0/0_volcanic.yaml
    exp0/0_wetland.yaml
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