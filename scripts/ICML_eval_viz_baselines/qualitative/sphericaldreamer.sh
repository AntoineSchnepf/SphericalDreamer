#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    # sphericaldreamer configs
    
    _blender/sphericaldreamer/dense_rainforest_understory.yaml
    _blender/sphericaldreamer/coral_reef_canyon.yaml
    _blender/sphericaldreamer/martian_badlands_2.yaml
    _blender/sphericaldreamer/phantom_opera_cave_river.yaml
    _blender/sphericaldreamer/sound_of_music_grass_field.yaml
    _blender/sphericaldreamer/upside_down_stranger_things.yaml
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