#!/bin/bash
# Run multiple experiments for SphericalDreamer

repo_dir="/home/a.schnepf/phd/SphericalDreamer"


# Define array of configs
configs=(
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/trans/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/wonderjourney/rot+trans/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/trans/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/scenescape/rot+trans/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot/upside_down_stranger_things.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/dense_rainforest_understory.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/coral_reef_canyon.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/martian_badlands_2.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/phantom_opera_cave_river.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/sound_of_music_grass_field.yaml
# /home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/trans/upside_down_stranger_things.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/dense_rainforest_understory.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/coral_reef_canyon.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/martian_badlands_2.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/phantom_opera_cave_river.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/sound_of_music_grass_field.yaml
/home/a.schnepf/phd/SphericalDreamer/configs/_blender_quantitatif/sphericaldreamer/rot+trans/upside_down_stranger_things.yaml
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
