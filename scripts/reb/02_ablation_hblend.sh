#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"



# Define array of configs
configs=(
    "rebuttal/ablation_hblend_ldi/ldi/martian_badlands_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_hblend_ldi/ldi/sound_of_music_grass_field.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=naive/martian_badlands_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=naive/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=naive/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=naive/sound_of_music_grass_field.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=interp_bilinear_plus_nn/martian_badlands_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=interp_bilinear_plus_nn/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=interp_bilinear_plus_nn/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=interp_bilinear_plus_nn/sound_of_music_grass_field.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=inpaint/martian_badlands_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=inpaint/bioluminescent_forest_2.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=inpaint/phantom_opera_cave_river.yaml"
    "rebuttal/ablation_hblend_ldi/ldi+hblend=inpaint/sound_of_music_grass_field.yaml"
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