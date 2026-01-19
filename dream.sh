if [ "$1" = "" ]; then
    config_name=_default.yaml
else
    config_name=$1
fi

#1a Generate panoramas
echo Running 1a_gen_panoramas.py...
python3 1a_gen_panoramas.py --config $config_name

#1b LDI generation
echo Running 1b_ldi.py...
python3 1b_ldi.py --config $config_name

#2a Align pairs with inpainting
echo Running 2a_align_pairs_inpainting.py...
python3 2a_align_pairs_inpainting.py --config $config_name

#2b. LDI generation
echo Running 2b_ldi.py...
python3 2b_ldi.py --config $config_name

#2c Align pairs with harmonic blending
echo Running 2c_align_pairs_harmonic_blending.py...
python3 2c_align_pairs_harmonic_blending.py --config $config_name

#3. Fix world geometry
echo Running 3_fix_world_geometry.py...
python3 3_fix_world_geometry.py --config $config_name

# 4. Render video
echo Running 4_render_video.py...
python3 4_render_video.py --config $config_name

# 5. Render blender
echo Running 5_render_blender.py...
blender --background --python 5v2_render_blender.py -- --config $config_name

echo Running 6_train_splatfacto.py...
python3 6_train_splatfacto.py --config $config_name