#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/k.kassab/panorama/SphericalDreamer"


# Define array of configs
configs=(
    expA/train.yaml
    expA/airplane.yaml
    expA/tunnel.yaml
    expA/caves.yaml
    expA/ferry.yaml
    expA/ship_cabin.yaml
    expA/hospital.yaml
    expA/mall.yaml
    expA/mine.yaml
    expA/bunker.yaml
    expA/catacombs.yaml
    expA/factory.yaml
    expA/powerplant.yaml
    expA/ice_cave.yaml
    expA/canyon.yaml
    expA/rock_gallery.yaml
    expA/space_station.yaml
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