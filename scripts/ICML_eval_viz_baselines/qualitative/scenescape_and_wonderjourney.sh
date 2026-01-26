#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    # wonderjourney configs
    _blender/wonderjourney/dense_rainforest_understory.yaml
    _blender/wonderjourney/coral_reef_canyon.yaml
    _blender/wonderjourney/martian_badlands_2.yaml
    _blender/wonderjourney/phantom_opera_cave_river.yaml
    _blender/wonderjourney/sound_of_music_grass_field.yaml
    _blender/wonderjourney/upside_down_stranger_things.yaml

    # scenescape configs
    # _blender/scenescape/dense_rainforest_understory.yaml
    # _blender/scenescape/coral_reef_canyon.yaml
    # _blender/scenescape/martian_badlands_2.yaml
    # _blender/scenescape/phantom_opera_cave_river.yaml
    # _blender/scenescape/sound_of_music_grass_field.yaml
    # _blender/scenescape/upside_down_stranger_things.yaml
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
