#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
exp4/bioluminescent_forest_2.yaml
exp4/crystal_tundra_2.yaml
exp4/martian_badlands_2.yaml
exp4/stone_pillar_desert_2.yaml
exp4/coral_abyss_2.yaml
exp4/infinite_library_2.yaml
exp4/catacomb_network_2.yaml
exp4/futuristic_datacenter_2.yaml
exp4/alien_hive_2.yaml
exp4/circular_metro_2.yaml
exp4/bioluminescent_forest.yaml
exp4/crystal_tundra.yaml
exp4/martian_badlands.yaml
exp4/stone_pillar_desert.yaml
exp4/coral_abyss.yaml
exp4/infinite_library.yaml
exp4/catacomb_network.yaml
exp4/futuristic_datacenter.yaml
exp4/alien_hive.yaml
exp4/circular_metro.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    bash dream.sh $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."