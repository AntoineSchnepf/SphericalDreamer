#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--bioluminescent_forest_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--martian_badlands_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--martian_badlands_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--martian_badlands_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--martian_badlands_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--phantom_opera_cave_river--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--sound_of_music_grass_field--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--bioluminescent_forest_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--martian_badlands_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--phantom_opera_cave_river--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--sound_of_music_grass_field--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--bioluminescent_forest_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--bioluminescent_forest_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--martian_badlands_2--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--martian_badlands_2--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--phantom_opera_cave_river--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--phantom_opera_cave_river--ldi+hblend=naive.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--sound_of_music_grass_field--ldi.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=inpaint.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=interp_bilinear_plus_nn.yaml"
    "/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif_ABL_hblend_ldi/sphericaldreamer/rot+trans/ABL_HblendLdi--sound_of_music_grass_field--ldi+hblend=naive.yaml"
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    blender --background --python viz_blender_ICML_RENDERS.py -- --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."
