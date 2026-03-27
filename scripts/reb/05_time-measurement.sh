#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"



# Define array of configs
configs=(
'rebuttal/time-measurment/sound_of_music_grass_field-N=2.yaml'
'rebuttal/time-measurment/sound_of_music_grass_field-N=3.yaml'
'rebuttal/time-measurment/sound_of_music_grass_field-N=4.yaml'
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