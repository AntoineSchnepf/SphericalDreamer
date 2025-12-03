if [ "$1" = "" ]; then
    config_name=_default.yaml
else
    config_name=$1
fi

echo Running 01_gen_panoramas.py...
python3 01_gen_panoramas.py --config $config_name
echo Running 02a_align_pairs_inpainting.py...
python3 02a_align_pairs_inpainting.py --config $config_name
echo Running 02b_align_pairs_harmonic_blending.py...
python3 02b_align_pairs_harmonic_blending.py --config $config_name
echo Running 03_final_filling.py...
# phase III is under implementation