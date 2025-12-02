if [ "$1" = "" ]; then
    config_name=_default.yaml
else
    config_name=$1
fi

echo Running phase I...
python3 01_gen_panoramas.py --config $config_name
echo Running phase IIa...
python3 02a_align_pairs_inpainting.py --config $config_name
echo Running phase IIb...
python3 02b_align_pairs_harmonic_blending.py --config $config_name
# echo Running phase III...
# phase III is under implementation

