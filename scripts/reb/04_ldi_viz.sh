



#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"



# Define array of configs
configs=(
    rebuttal/ldi_viz/fungal_underground_colony.yaml
    rebuttal/ldi_viz/misty_bamboo_forest.yaml
    rebuttal/ldi_viz/stone_pillar_desert_2.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    python 1b_ldi.py --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."
