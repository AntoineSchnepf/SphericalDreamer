#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
    exp3/train.yaml
    exp3/airplane.yaml
    exp3/tunnel.yaml
    exp3/caves.yaml
    exp3/ferry.yaml
    exp3/ship_cabin.yaml
    exp3/hospital.yaml
    exp3/mall.yaml
    exp3/mine.yaml
    exp3/bunker.yaml
    exp3/catacombs.yaml
    exp3/factory.yaml
    exp3/powerplant.yaml
    exp3/ice_cave.yaml
    exp3/canyon.yaml
    exp3/rock_gallery.yaml
    exp3/space_station.yaml
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