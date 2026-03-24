#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    "rebuttal/ablation_N/N=5/martian_badlands_2.yaml"
    "rebuttal/ablation_N/N=5/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_N/N=5/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_N/N=5/sound_of_music_grass_field.yaml"
    "rebuttal/ablation_N/N=7/martian_badlands_2.yaml"
    "rebuttal/ablation_N/N=7/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_N/N=7/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_N/N=7/sound_of_music_grass_field.yaml"
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