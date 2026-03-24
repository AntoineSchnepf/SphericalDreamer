#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
exp5/misty_bamboo_forest.yaml
exp5/endless_lavender_fields.yaml
exp5/glacial_ice_tunnel.yaml
exp5/floating_stone_valley.yaml
exp5/submerged_ancient_city.yaml
exp5/infinite_wheat_plains.yaml
exp5/crystal_salt_flats.yaml
exp5/fungal_underground_colony.yaml
exp5/abandoned_subway_tunnel.yaml
exp5/endless_marble_colonnade.yaml
exp5/dense_rainforest_understory.yaml
exp5/alien_silicon_desert.yaml
exp5/underground_water_reservoir.yaml
exp5/rolling_moss_hills.yaml
exp5/deserted_meditation_temple.yaml
exp5/coral_reef_canyon.yaml
exp5/wind_eroded_plateau.yaml
exp5/infinite_greenhouse_corridor.yaml
exp5/subarctic_tundra_plain.yaml

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