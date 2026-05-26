import os
import warnings
import logging
import contextlib
from io import StringIO
from pathlib import Path
from glob import glob

os.environ["GLOG_minloglevel"] = "2"
os.environ["GLOG_logtostderr"] = "0"
os.environ["CERES_MINIMIZER_PROGRESS_TO_STDOUT"] = "0"
logging.disable(logging.CRITICAL + 1)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", FutureWarning)

import sys

_360monodepth_install_dir = "/home/a.schnepf/phd/LayerPano3D/submodules/360monodepth/code/python/src/"
sys.path.append(_360monodepth_install_dir)
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer
import my_utils
from my_utils import printc

NUM_IMAGES = 20
OUTPUT_ROOT = Path("OUTPUTS/cond_gen_variability")
CONFIG_DIR = Path("configs/paper")

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.realpath(__file__))
    default_config_dir = os.path.join(repo_path, "configs")
    paper_config_dir = os.path.join(repo_path, "configs/paper")
    config_files = sorted(glob(os.path.join(paper_config_dir, "*.yaml")))

    if not config_files:
        raise FileNotFoundError(f"No config files found in {paper_config_dir}")

    printc(f"Found {len(config_files)} configs in {CONFIG_DIR}", color='cyan')

    default_config = my_utils.yaml_load("_default.yaml", default_config_dir)
    width = default_config.get("width", 1440)
    height = default_config.get("height", 720)
    depth_model = default_config.get("depth_model", "360mono")

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir="/tmp/pano_depth_temp",
        depth_model=depth_model,
    )

    seeds = list(range(NUM_IMAGES))

    for cfg_path in config_files:
        cfg_name = os.path.basename(cfg_path)
        scene_config = my_utils.yaml_load(cfg_name, paper_config_dir)
        config = my_utils.deep_update(dict(default_config), scene_config)

        expname = config["expname"]
        prompt = config["prompt"]
        save_dir = OUTPUT_ROOT / expname
        os.makedirs(save_dir, exist_ok=True)

        printc(f"=== Generating {NUM_IMAGES} panoramas for: {expname} ===", color='green')
        for i in range(NUM_IMAGES):
            printc(f"  [{expname}] image {i+1:02d}/{NUM_IMAGES}  (seed={seeds[i]})", color='yellow')
            pano_rgb = spherical_dreamer.gen_pano(prompt=prompt, seed_override=seeds[i])
            pano_rgb.save(save_dir / f"image{i:02d}.png")

        printc(f"  Done: {expname} -> {save_dir}", color='green')

    printc("All configs processed!", color='cyan')
