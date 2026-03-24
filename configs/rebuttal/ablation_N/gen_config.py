from pathlib import Path
load_from=Path("/home/a.schnepf/phd/SphericalDreamer/configs/paper")
save_at=Path("/home/a.schnepf/phd/SphericalDreamer/configs/rebuttal/ablation_N")

N_values = [5, 7]
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
for N in N_values:
    os.system(f"rm -rf {save_at / f'N={N}'}")
    os.makedirs(save_at / f"N={N}", exist_ok=True)
    for cfg_name in CFG_NAMES:
        if cfg_name.endswith(".yaml"):
            config = yaml_load(cfg_name, load_from)
            config["num_dreams"] = N
            config['expname'] = f"ABL_N--{config['expname']}--N={N}"
            config['save_dir'] = 'OUTPUTS/SphericalDreamerRecurse/ablation_N'

            save_config(config, cfg_name, save_at / f"N={N}")


            print(f"Saved config for N={N} at {save_at / f'N={N}' / cfg_name}")

            cfg_list.append(save_at / f"N={N}" / cfg_name)

with open(save_at / "config_paths.txt", "w") as f:    
    for cfg_path in cfg_list:
        f.write(str(cfg_path) + "\n")