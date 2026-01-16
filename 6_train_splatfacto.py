import os, sys
os.environ.setdefault("OPEN3D_HEADLESS", "1") # for open3d headless rendering
import numpy as np
import open3d as o3d
import pickle
import json
import my_utils
import numpy as np
import open3d as o3d
import time
from my_utils import PointCloud
from my_utils import set_camera_from_elev_azim, printc
from tqdm import tqdm
from pathlib import Path
import textwrap


config = my_utils.fetch_config_via_parser(
    debug=False, 
    debug_parser_override=["--config", "Karim/forest.yaml"]
)
repo_path = os.path.dirname(os.path.realpath(__file__))
repo_path = Path(repo_path)

data = repo_path / config.save_dir / config.expname / "nfs_dataset"
output_dir = repo_path / config.save_dir / config.expname / "nerfstudio_chkpt"

command = f"""ns-train splatfacto \
    --data {data} \
    --output-dir {output_dir} \
    --experiment-name {config.expname} \
    --pipeline.model.collider-params near_plane {config.phase6.near_plane} far_plane {config.phase6.far_plane} \
    --pipeline.model.background-color {config.phase6.background_color} \
    nerfstudio-data
"""

printc("Launching nerfstudio with command:", "gray")
printc(command.replace(" --", "\\\n   --"), "gray")
os.system(command)

# TODO: Apply later
"""
ns-render camera-path \
    --load-config /home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/F1_forest/nerfstudio_chkpt/forest/splatfacto/2025-12-30_085122/config.yml \
    --camera-path-filename /home/a.schnepf/phd/SphericalDreamer/OUTPUTS/camera_paths/walk_path.json \
    --output-path /home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/F1_forest/nerfstudio_chkpt/render/walk_path.mp4
"""

--output-dir /home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/F1_forest/gsplat.ply
