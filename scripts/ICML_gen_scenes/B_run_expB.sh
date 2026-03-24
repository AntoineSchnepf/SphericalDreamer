#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/k.kassab/panorama/SphericalDreamer"


# Define array of configs
configs=(
    expB/mars_surface.yaml
    expB/salt_mine_interior.yaml
    expB/sound_of_music_grass_field.yaml
    expB/upside_down_stranger_things.yaml
    expB/dune_desert.yaml
    expB/botw_large_plain_grass.yaml
    expB/teletubbies_landscape.yaml
    expB/ski_snowfield_trees_far.yaml
    expB/zelda_cave_organic.yaml
    expB/venus_surface.yaml
    expB/foreign_planet_surface.yaml
    expB/tomato_field_large.yaml
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