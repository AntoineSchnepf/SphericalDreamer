#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
exp4_1a_only/bioluminescent_forest.yaml
exp4_1a_only/crystal_tundra.yaml
exp4_1a_only/martian_badlands.yaml
exp4_1a_only/stone_pillar_desert.yaml
exp4_1a_only/coral_abyss.yaml
exp4_1a_only/infinite_library.yaml
exp4_1a_only/catacomb_network.yaml
exp4_1a_only/futuristic_datacenter.yaml
exp4_1a_only/alien_hive.yaml
exp4_1a_only/circular_metro.yaml
exp4_1a_only/bioluminescent_forest_2.yaml
exp4_1a_only/crystal_tundra_2.yaml
exp4_1a_only/martian_badlands_2.yaml
exp4_1a_only/stone_pillar_desert_2.yaml
exp4_1a_only/coral_abyss_2.yaml
exp4_1a_only/infinite_library_2.yaml
exp4_1a_only/catacomb_network_2.yaml
exp4_1a_only/futuristic_datacenter_2.yaml
exp4_1a_only/alien_hive_2.yaml
exp4_1a_only/circular_metro_2.yaml

)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    python 1a_gen_ONLY_panoramas.py --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."