#!/usr/bin/env bash
set -euo pipefail

exp_savedirs=(
    #   _exp5/dense_rainforest_understory
#     _exp5/coral_reef_canyon
#     _exp4/martian_badlands_2
    # expB/sound_of_music_grass_field
    # expB/upside_down_stranger_things
    expC/phantom_opera_cave_river
)

exp_names=(
#   dense_rainforest_understory
#     coral_reef_canyon
#     martian_badlands_2
    # sound_of_music_grass_field
    # upside_down_stranger_things
    phantom_opera_cave_river
)

BASE_DIR="/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse"
OUT_BASE="/home/a.schnepf/phd/experiments"

# sanity check
if [ "${#exp_savedirs[@]}" -ne "${#exp_names[@]}" ]; then
  echo "ERROR: exp_savedirs and exp_names must have the same length" >&2
  exit 1
fi

for i in "${!exp_savedirs[@]}"; do
  exp_savedir="${exp_savedirs[i]}"
  exp_name="${exp_names[i]}"

  src="${BASE_DIR}/${exp_savedir}/3_final_dream_pcd_unfiltered.ply"
  dst_dir="${OUT_BASE}/${exp_name}/sphericaldreamer"
  dst="${dst_dir}/pcd.ply"

  mkdir -p "${dst_dir}"
  cp "${src}" "${dst}"
done