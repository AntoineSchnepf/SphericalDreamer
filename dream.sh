if [ "$1" = "" ]; then
    config_name=_default.yaml
else
    config_name=$1
fi

#1a Generate panoramas
echo Running 01a_gen_panoramas.py...
python3 01a_gen_panoramas.py --config $config_name

#1b LDI generation
echo Running 01b_ldi.py...
python3 01b_ldi.py --config $config_name

#2a Align pairs with inpainting
echo Running 02a_align_pairs_inpainting.py...
python3 02a_align_pairs_inpainting.py --config $config_name

#2b. LDI generation
echo Running 02b_ldi.py...
python3 02b_ldi.py --config $config_name

#2c Align pairs with harmonic blending
echo Running 02c_align_pairs_harmonic_blending.py...
python3 02c_align_pairs_harmonic_blending.py --config $config_name

#3. Gaussian Splat #TODO


# 4. Render video
echo Running 04_render_video.py...
python3 04_render_video.py --config $config_name