#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    # FOR dense_rainforest_understory_outside

    _blender/outside/wonderjourney/phantom_opera_cave_river_outside.yaml
    _blender/outside/scenescape/phantom_opera_cave_river_outside.yaml
    _blender/outside/sphericaldreamer/phantom_opera_cave_river_outside.yaml

    # BACKUP

    # _blender/outside/sphericaldreamer/coral_reef_canyon_outside.yaml
    # _blender/outside/sphericaldreamer/martian_badlands_2_outside.yaml
    # _blender/outside/sphericaldreamer/phantom_opera_cave_river_outside.yaml
    # _blender/outside/sphericaldreamer/sound_of_music_grass_field_outside.yaml
    # _blender/outside/sphericaldreamer/upside_down_stranger_things_outside.yaml
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