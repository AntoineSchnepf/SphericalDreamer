#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/k.kassab/panorama/SphericalDreamer"


# Define array of configs
configs=(
    expC/lotr_battlefield_empty.yaml
    expC/ghost_of_tsushima_green_field.yaml
    expC/ghost_of_tsushima_cherry_blossom_field.yaml
    expC/windows_xp_wallpaper_field.yaml
    expC/dystopian_neon_tron_world.yaml
    expC/phantom_opera_cave_river.yaml
    expC/roman_theatre_interior.yaml
    expC/large_cyberpunk_tron_environment.yaml
    expC/alien_tron_apocalyptic_world.yaml
    expC/purple_sunset_lalaland_city_overlook.yaml
    expC/dystopian_tron_cave.yaml
    expC/post_apocalyptic_lastofus_cave.yaml
    expC/disney_insideout_cave.yaml
    expC/insideout_towers_far_environment.yaml
    expC/zelda_botw_world_empty.yaml
    expC/utopic_chromatica_environment.yaml
    expC/wizard_of_oz_yellow_brick_road.yaml
    expC/wizard_of_oz_yellow_brick_road_endless.yaml
    expC/aerial_city_view_high_sky.yaml
    expC/dystopian_neon_lab_westworld.yaml
    expC/dystopian_neon_lab_cyberpunk.yaml
    expC/wicked_emerald_city_far.yaml
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