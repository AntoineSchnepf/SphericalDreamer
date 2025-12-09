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
    "Antoine/C0_city.yaml"
    "Antoine/C1_city_corr.yaml"
    "Antoine/F0_forest.yaml"
    "Antoine/S0_seaside.yaml"
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    bash "$script" "$config"

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
        exit 1
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."