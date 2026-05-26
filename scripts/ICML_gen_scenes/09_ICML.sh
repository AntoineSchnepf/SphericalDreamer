#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    # exp5/dense_rainforest_understory.yaml
    # exp5/coral_reef_canyon.yaml
    exp4/martian_badlands_2.yaml
    # expC/phantom_opera_cave_river.yaml
    # expA/sound_of_music_grass_field.yaml
    # expA/upside_down_stranger_things.yaml
)

# Run experiments
for config in "${configs[@]}"; do
    echo "🚀 Running experiment with config: $config"
    blender --background --python 5_render_blender.py -- --config $config

    if [ $? -ne 0 ]; then
        echo "❌ Experiment with config $config failed."
    fi

    echo "✅ Experiment with config $config completed successfully."
done

echo "🎉 All experiments completed successfully."