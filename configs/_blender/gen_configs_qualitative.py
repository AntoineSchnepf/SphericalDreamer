# wonder journeyconfig
import sys
from pathlib import Path
import yaml
import numpy as np

def save_config(cfg, cfg_name, save_dir):
    """
    Save a configuration dictionary as a YAML file.

    Parameters
    ----------
    cfg : dict
        Configuration data.
    cfg_name : str
        Name of the YAML file (e.g. 'forest.yaml').
    save_dir : str or Path
        Directory where the config will be saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / cfg_name
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_path

def yaml_load(path):
    """
    Load a YAML file and return its contents as a Python dict.
    """
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)
    
PROMPTS = {
    "dense_rainforest_understory": (
        "A dense rainforest understory extending forward and behind the observer, thick "
        "overlapping vegetation with no sharp edges, large leaves in deep green tones, moist "
        "ground covered in roots and organic debris, overcast sky filtering light through "
        "the canopy, soft diffused illumination, humid and lush atmosphere, immersive tropical "
        "environment."
    ),
    "coral_reef_canyon": (
        "A wide coral reef canyon extending forward and behind the observer, smooth "
        "rock walls covered in coral and marine growth, vibrant yet softened colors including "
        "turquoise, coral pink, and sandy beige, filtered underwater lighting with soft light "
        "rays, floating particles in the water, calm immersive oceanic environment."
    ),
    "martian_badlands_2": (
        "A Martian landscape of rust-red and ochre soil, scattered with dark basalt "
        "rocks and patches of muted green and purple alien vegetation, under a dusty salmon-colored "
        "sky."
    ),
    "phantom_opera_cave_river": (
        "A large-scale 3D subterranean cave environment inspired by the Phantom of "
        "the Opera underground river setting, with no visible people or animals, an irregular "
        "cavern corridor extending forward and continuing behind the observer, no sharp edges "
        "or well-defined geometry, rounded organic rock formations softened by moisture and "
        "time, a dark slow-moving river running across the ground with reflective surface "
        "and subtle ripples, wet rocky banks with scattered stones and damp sediment, stalactites "
        "and stalagmites smoothed into natural shapes, faint mist hovering above water, dim "
        "atmospheric lighting as if from distant unseen lanterns creating soft warm reflections, "
        "deep shadows fading into darkness, cinematic gothic mood, immersive enclosed underground "
        "atmosphere with realistic rock and water textures and strong spatial depth."
    ),
    "sound_of_music_grass_field": (
        "A wide open rolling field of lush green grass inspired by The Sound of Music, "
        "gentle natural hills extending forward and continuing behind the observer with no "
        "harsh edges or defined geometry, thick healthy grass blades forming smooth wind-like "
        "patterns, scattered wildflower hints subtle and sparse, cloudy sky overhead with "
        "no visible sun, soft diffused daylight creating even illumination and minimal harsh "
        "shadows, distant tree line barely visible on the horizon, peaceful cinematic pastoral "
        "mood, realistic vegetation textures, immersive open countryside atmosphere."
    ),
    "upside_down_stranger_things": (
        "A desolate Upside Down-inspired landscape with cloudy oppressive skies and "
        "no visible sun, the environment stretching forward and behind the observer with "
        "organic uneven terrain, dark damp ground covered in tangled root-like growth and "
        "soft alien debris, floating ash-like particles suspended in the air, twisted vegetation "
        "silhouettes without sharp geometry, murky fog reducing visibility in the distance, "
        "muted blue-gray lighting with eerie contrast, wet reflective patches and slimy textures, "
        "cinematic horror mood, immersive otherworldly atmosphere."
    ),
}


def make_position_product_with_azis(base_positions):
    "base_positions: list of [x,y,z,elev,azi]"
    positions = []
    for pos in base_positions:
        for azi in [0, 90, 180, 270]:   
            pos_copy = pos.copy()
            pos_copy[4] = azi
            positions.append(pos_copy)
    return positions

def get_scenescape_positions(expname):
    scenescape_base_positions = yaml_load(f"/home/a.schnepf/phd/experiments/{expname}/scenescape/camera_poses_xyz_elev_azi.yaml")['positions']
    return make_position_product_with_azis(scenescape_base_positions)

WONDERJOURNEY_BASE_POSITIONS = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.1875, 0.0, 0.0, 0.0, 0.0],
    [-0.375, 0.0, 0.0, 0.0, 0.0],
]
def get_wonderjourney_positions(expname):
    return make_position_product_with_azis(WONDERJOURNEY_BASE_POSITIONS)

def get_sphericaldreamer_positions(num_dreams = 3, n_x=4):
    min_x = 0
    max_x = 1.57 * (num_dreams-1)
    les_x = [0.05, 3.14-0.05]
    positions = []
    for x in les_x:
        for azi in [0, 90, 180, 270]:
            positions.append([float(x), 0.0, 0.05, 0.0, azi])
    return positions
    

SPHERICAL_DREAMER_DOWNSAMPLE = True
for scene_type in ["wonderjourney", "scenescape", "sphericaldreamer"]:



    CFG_DIR = Path(f"/home/a.schnepf/phd/SphericalDreamer/configs/_blender/{scene_type}")
    CONFIG_IN = CFG_DIR / ".." / "example.yaml"


    cfg_names = []  # <-- collect generated configs



    def get_world_save_name(scene_type):
        if scene_type == "scenescape":
            return "mesh.obj"
        elif scene_type == "wonderjourney":
            return "pcd.ply"
        elif scene_type == "sphericaldreamer":
            return "pcd.ply"

    # load from yaml

    for expname, prompt in PROMPTS.items():
        if scene_type == "sphericaldreamer":
            positions = get_sphericaldreamer_positions(num_dreams=3, n_x=4)
            word_save_name = "pcd.ply"
        elif scene_type == "scenescape":
            positions = get_scenescape_positions(expname)
            word_save_name = "mesh.obj"
        elif scene_type == "wonderjourney":
            positions = get_wonderjourney_positions(expname)
            word_save_name = "pcd.ply"

        UPDATES = {
            'positions' : positions,
            "expname": expname,
            "scene_type": scene_type,
            "world_path": f"/home/a.schnepf/phd/experiments/{expname}/{scene_type}/{word_save_name}",   
            "keep_ratio": 
            0.9 if (scene_type == "sphericaldreamer" and SPHERICAL_DREAMER_DOWNSAMPLE) else 1.0,
        }

        cfg = yaml_load(CONFIG_IN)

        cfg['phase5v2']['custom_trajectory']['positions'] = UPDATES['positions']
        cfg['expname'] = UPDATES['expname']
        cfg['phase5v2']['custom_world']['scene_type'] = UPDATES['scene_type']
        cfg['phase5v2']['custom_world']['world_path'] = UPDATES['world_path']
        cfg['phase5v2']['render_settings']['keep_ratio'] = UPDATES['keep_ratio']
        cfg['phase5v2']['render_settings']['point_size'] = 0.0022 *  15 if scene_type == 'wonderjourney' else 0.0022 
        cfg['save_dir'] = f"OUTPUTS/X_ICML_RENDERS/qualitative"

        save_config(cfg, cfg_name=f"{expname}.yaml", save_dir=CFG_DIR)
        cfg_names.append(f"{expname}.yaml")
        print(f"Saved config for {expname}")


    # ---- write config list file ----
    cfg_list_path = CFG_DIR / "config_list.txt"
    with cfg_list_path.open("w") as f:
        for name in cfg_names:
            f.write(f"{name}\n")


