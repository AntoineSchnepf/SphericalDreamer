# wonder journeyconfig
import sys
from pathlib import Path
import yaml
import numpy as np
import random

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

def _interp_angle_deg(a0, a1, t):
    """
    Interpolate angles in degrees along the shortest path (wrap-aware).
    Returns value in [0, 360).
    """
    # shortest signed delta in [-180, 180)
    delta = (a1 - a0 + 180.0) % 360.0 - 180.0
    return (a0 + t * delta) % 360.0

def _resample_positions_linear(positions, n_out):
    """
    Linear resampling over the polyline defined by the input sequence.
    Uses cumulative chord length in (x,y,z,elev) space to distribute samples.
    Azimuth is interpolated wrap-aware.
    """
    P = np.asarray(positions, dtype=float)
    if len(P) == 0:
        raise ValueError("No positions to resample.")
    if len(P) == 1:
        # repeat the single pose
        x, y, z, elev, azi = P[0]
        return [[float(x), float(y), float(z), float(elev), float(azi % 360.0)] for _ in range(n_out)]

    # chord-length parameterization (ignore azi for distance)
    Q = P[:, :4]  # x,y,z,elev
    seg = np.linalg.norm(Q[1:] - Q[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    # if all identical (zero length), just repeat first
    if total == 0.0:
        x, y, z, elev, azi = P[0]
        return [[float(x), float(y), float(z), float(elev), float(azi % 360.0)] for _ in range(n_out)]

    targets = np.linspace(0.0, total, n_out)

    out = []
    j = 0
    for tt in targets:
        while j < len(s) - 2 and tt > s[j + 1]:
            j += 1
        s0, s1 = s[j], s[j + 1]
        t = 0.0 if s1 == s0 else (tt - s0) / (s1 - s0)

        # linear interp for x,y,z,elev
        v0 = P[j, :4]
        v1 = P[j + 1, :4]
        v = (1.0 - t) * v0 + t * v1

        # wrap-aware interp for azimuth
        a0 = P[j, 4]
        a1 = P[j + 1, 4]
        azi = _interp_angle_deg(a0, a1, t)

        out.append([float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(azi)])
    return out


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
NUM_TRANS = 20
NUM_ROT = 20
NUM_ROT_AND_TRANS = 20


def OLD_combine_translations_rotations(translations, rotations, wrap_azim=True):
    """
    translations, rotations: list of [x, y, z, elev, azim]
    output: cartesian product where x,y,z,elev come from translation
            and azim = translation_azim + rotation_azim
    """
    out = []
    for t in translations:
        x, y, z, elev, az_t = t
        for r in rotations:
            az_r = r[4]
            az = az_t + az_r
            if wrap_azim:
                az = az % 360.0
            out.append([x, y, z, elev, az])
    return out


def combine_translations_rotations_random(
    translations,
    rotations,
    K,
    seed=0,
    wrap_azim=True,
):
    """
    Deterministic random sampling of K poses from the Cartesian product
    of translations x rotations.
    Same seed => same subset + same order.
    """
    T, R = len(translations), len(rotations)
    N = T * R

    if K > N:
        raise ValueError(f"K={K} larger than total combinations N={N}")

    rng = random.Random(seed)

    # sample K distinct flat indices deterministically
    flat_indices = rng.sample(range(N), K)

    out = []
    for idx in flat_indices:
        i = idx // R
        j = idx % R

        x, y, z, elev, az_t = translations[i]
        az = az_t + rotations[j][4]
        if wrap_azim:
            az %= 360.0

        out.append([x, y, z, elev, az])

    return out

# WONDERJOURNEY
WONDERJOURNEY_X_INIT = 0.0
WNDERJOUNREY_X_END = -0.19
def get_wonderjourney_trans_positions(expname):
    les_x = np.linspace(WONDERJOURNEY_X_INIT, WNDERJOUNREY_X_END, num=NUM_TRANS)
    positions = []
    for x in les_x:
        positions.append([float(x), 0.0, 0.0, 0.0, 0.0])
    return positions

def get_wonderjourney_rot_position(expname):
    les_theta = np.linspace(0, 360, num=NUM_ROT, endpoint=False)
    positions = []
    for theta in les_theta:
        positions.append([-0.0, 0.0, 0.0, 0.0, float(theta)])
    return positions

def get_wonderjourney_rot_trans_positions(expname):
    translations = get_wonderjourney_trans_positions(expname)
    rotations = get_wonderjourney_rot_position(expname)
    return combine_translations_rotations_random(translations, rotations, K=NUM_ROT_AND_TRANS)

# SCENESCAPE
import numpy as np

SPHERICALDREAMER_X_INIT = 0.0
SPHERICALDREAMER_X_END = -0.09
def get_scenescape_trans_positions(expname,
                                   x_init=SPHERICALDREAMER_X_INIT,
                                   x_end=SPHERICALDREAMER_X_END,
                                   num_trans=NUM_TRANS):
    path = f"/home/a.schnepf/phd/experiments/{expname}/scenescape/camera_poses_xyz_elev_azi.yaml"
    data = yaml_load(path)
    positions = data["positions"]  # list of [x,y,z,elev,azi]

    # keep x between init and end (works whether end < init or not)
    lo, hi = (x_end, x_init) if x_end < x_init else (x_init, x_end)
    filtered = [p for p in positions if lo <= float(p[0]) <= hi]

    if len(filtered) == 0:
        raise ValueError(f"No positions found with x in [{lo}, {hi}] from {path}")

    # Ensure sorted by x in the direction from init -> end (optional but usually desired)
    # If moving from 0.0 down to -0.09, sort descending.
    descending = x_end < x_init
    filtered.sort(key=lambda p: float(p[0]), reverse=descending)

    # resample to exactly num_trans
    return _resample_positions_linear(filtered, num_trans)


def get_scenescape_rot_position(expname):
    les_theta = np.linspace(0, 360, num=NUM_ROT, endpoint=False)
    positions = []
    for theta in les_theta:
        positions.append([0.0, 0.0, 0.0, 0.0, float(theta)])
    return positions

def get_scenescape_rot_trans_positions(expname):
    translations = get_scenescape_trans_positions(expname)
    rotations = get_scenescape_rot_position(expname)
    return combine_translations_rotations_random(translations, rotations, K=NUM_ROT_AND_TRANS)


# SPHERICAL DREAMER
SPHERICALDREAMER_X_INIT = 0.0
SPHERICALDREAMER_X_END = 3.14-0.05
def get_spherical_dreamer_trans_positions(expname):
    les_x = np.linspace(SPHERICALDREAMER_X_INIT, SPHERICALDREAMER_X_END, num=NUM_ROT_AND_TRANS)
    positions = []
    for x in les_x:
        positions.append([float(x), 0.0, 0.0, 0.0, 0.0])
    return positions

def get_spherical_dreamer_rot_position(expname):
    les_theta = np.linspace(0, 360, num=NUM_ROT, endpoint=False)
    positions = []
    for theta in les_theta:
        positions.append([0.05, 0.0, 0.0, 0.0, float(theta)])
    return positions


def get_sphericaldreamer_rot_trans_positions(expname):
    translations = get_spherical_dreamer_trans_positions(expname)
    rotations = get_spherical_dreamer_rot_position(expname)
    return combine_translations_rotations_random(translations, rotations, K=NUM_ROT_AND_TRANS)
    
def get_positions(expname, scene_type, traj_type):
    if traj_type == 'rot':
        get_positions_func = {
            "wonderjourney": get_wonderjourney_rot_position,
            "scenescape": get_scenescape_rot_position,
            "sphericaldreamer": get_spherical_dreamer_rot_position,
        }
    elif traj_type == 'trans':
        get_positions_func = {
            "wonderjourney": get_wonderjourney_trans_positions,
            "scenescape": get_scenescape_trans_positions,
            "sphericaldreamer": get_spherical_dreamer_trans_positions,
        }
    elif traj_type == 'rot+trans':
        get_positions_func = {
            "wonderjourney": get_wonderjourney_rot_trans_positions,
            "scenescape": get_scenescape_rot_trans_positions,
            "sphericaldreamer": get_sphericaldreamer_rot_trans_positions,
        }
    else:
        raise ValueError(f"Unknown traj_type: {traj_type}")

    return get_positions_func[scene_type](expname)

def get_world_save_name(scene_type):
    if scene_type == "scenescape":
        return "mesh.obj"
    elif scene_type == "wonderjourney":
        return "pcd.ply"
    elif scene_type == "sphericaldreamer":
        return "pcd.ply"


BASE_PATH = Path(f"/home/a.schnepf/phd/SphericalDreamer/configs/blender/_blender_quantitatif")
SPHERICAL_DREAMER_DOWNSAMPLE = False
TRAJ_TYPES = ['rot', 'trans', 'rot+trans']
HEIGHT = 800
WIDTH = 1600 

cfg_paths= []  # <-- collect generated configs
for scene_type in ["wonderjourney", "scenescape", "sphericaldreamer"]:
    for traj_type in TRAJ_TYPES:

        CFG_DIR = BASE_PATH / scene_type / traj_type
        CONFIG_IN = BASE_PATH / "example.yaml"


        for expname, prompt in PROMPTS.items():

            UPDATES = {
                'positions' : get_positions(expname, scene_type, traj_type),
                "expname": expname,
                "scene_type": scene_type,
                "world_path": f"/home/a.schnepf/phd/experiments/{expname}/{scene_type}/{get_world_save_name(scene_type)}",   
                "keep_ratio": 
                0.9 if (scene_type == "sphericaldreamer" and SPHERICAL_DREAMER_DOWNSAMPLE) else 1.0,
            }

            cfg = yaml_load(CONFIG_IN)

            cfg['phase5']['custom_trajectory']['positions'] = UPDATES['positions']
            cfg['expname'] = UPDATES['expname']
            cfg['phase5']['custom_world']['scene_type'] = UPDATES['scene_type']
            cfg['phase5']['custom_world']['world_path'] = UPDATES['world_path']
            cfg['phase5']['render_settings']['keep_ratio'] = UPDATES['keep_ratio']
            cfg['phase5']['custom_trajectory']['render_eqr_too'] = False
            cfg['phase5']['render_settings']['width'] = WIDTH
            cfg['phase5']['render_settings']['height'] = HEIGHT
            cfg['phase5']['nfs_dataset']['bg_color'] = [0,0,0,0]
            # cfg['phase5']['advanced_render_settings']['use_distance_based_point_size'] = True
            cfg['phase5']['render_settings']['point_size'] = 0.0022 *  15 if scene_type == 'wonderjourney' else 0.0022 
            cfg['save_dir'] = f"OUTPUTS/X_ICML_RENDERS/quantitative/{traj_type}"

            save_config(cfg, cfg_name=f"{expname}.yaml", save_dir=CFG_DIR)
            cfg_paths.append(CFG_DIR / f"{expname}.yaml")
            print(f"Saved config for {expname}")

# ---- write config list file ----
cfg_list_path = BASE_PATH / "config_list.txt"
with cfg_list_path.open("w") as f:
    for name in cfg_paths:
        f.write(f"{name}\n")


