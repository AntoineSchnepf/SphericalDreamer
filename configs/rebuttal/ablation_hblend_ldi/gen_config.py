from pathlib import Path
load_from=Path("/home/a.schnepf/phd/SphericalDreamer/configs/paper")
save_at=Path("/home/a.schnepf/phd/SphericalDreamer/configs/rebuttal/ablation_hblend_ldi")

ablations = ["ldi", "ldi+hblend=naive", "ldi+hblend=interp_bilinear_plus_nn", "ldi+hblend=inpaint"]

import sys
import os
sys.path.append("/home/a.schnepf/phd/SphericalDreamer")
from my_utils import yaml_load, save_config

CFG_NAMES = [
    'martian_badlands_2.yaml',
    # 'dense_rainforest_understory.yaml',
    'bioluminescent_forest_2.yaml',
    'phantom_opera_cave_river.yaml',
    # 'upside_down_stranger_things.yaml',
    # 'coral_reef_canyon.yaml',
    'sound_of_music_grass_field.yaml'
 ]

cfg_list = []
for ablation in ablations:
    subdir = ablation
    os.system(f"rm -rf {save_at / subdir}")
    os.makedirs(save_at / subdir, exist_ok=True)
    for cfg_name in CFG_NAMES:
        if cfg_name.endswith(".yaml"):
            config = yaml_load(cfg_name, load_from)
            config['expname'] = f"ABL_HblendLdi--{config['expname']}--{ablation}"
            config['save_dir'] = 'OUTPUTS/SphericalDreamerRecurse/ablation_hblend_ldi'

            config['phase1'] = {}
            config['phase1']['apply_ldi'] = False

            config['phase2'] = {}
            config['phase2']['apply_ldi'] = False

            if ablation == "ldi":
                config['phase2']['ablate_hblending'] = {'apply': False}
            else:
                hblend_mode = ablation.split("=", 1)[1]
                config['phase2']['ablate_hblending'] = {
                    'apply': True,
                    'depth_blending_mode': hblend_mode,
                }

            save_config(config, cfg_name, save_at / subdir)
            print(f"Saved config for {ablation} at {save_at / subdir / cfg_name}")
            cfg_list.append(save_at / subdir / cfg_name)

with open(save_at / "config_paths.txt", "w") as f:
    for cfg_path in cfg_list:
        f.write(str(cfg_path) + "\n")