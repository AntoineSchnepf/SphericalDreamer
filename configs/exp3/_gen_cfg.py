PROMPTS = {
    "train": "An empty train interior with long straight corridors extending forward and continuing behind the observer, rows of seats, windows reflecting dim exterior light, cinematic lighting, realistic textures, immersive enclosed space.",
    "airplane": "A wide-body airplane cabin interior with a straight central aisle extending forward and behind the observer, overhead bins, rows of seats, soft ambient lighting, detailed materials, realistic perspective.",
    "tunnel": "A long underground tunnel stretching straight ahead and continuing behind the observer, concrete walls, dim industrial lights, strong depth cues, moody atmosphere.",
    "caves": "A natural cave interior forming a straight passage extending forward and behind the observer, rough rock walls, stalactites, uneven ground, low light, mysterious and immersive subterranean environment.",
    "ferry": "A narrow ferry corridor forming a straight passage extending forward and behind the observer, metal walls, handrails, utilitarian lighting, slight curvature, enclosed maritime atmosphere.",
    "ship_cabin": "A compact ship cabin aligned along a straight axis extending forward and behind the observer, wooden and metal elements, small portholes, nautical details, warm yet confined interior.",
    "hospital": "An abandoned hospital interior with long straight corridors extending forward and behind the observer, tiled floors, flickering fluorescent lights, medical equipment, eerie atmosphere.",
    "mall": "A closed shopping mall interior with wide straight corridors extending forward and behind the observer, empty storefronts, escalators, artificial lighting, quiet and abandoned mood.",
    "mine": "An abandoned mine tunnel forming a straight passage extending forward and behind the observer, wooden supports, rocky walls, dust in the air, sparse lighting, deep underground feeling.",
    "bunker": "An underground nuclear bunker with straight reinforced corridors extending forward and behind the observer, concrete walls, heavy doors, industrial lighting, claustrophobic enclosed space.",
    "catacombs": "A network of catacombs forming straight stone corridors extending forward and behind the observer, arched ceilings, skull-lined walls, torch-like lighting, ancient and ominous mood.",
    "factory": "An abandoned factory interior with large straight halls extending forward and behind the observer, rusted machinery, broken windows, shafts of light, industrial decay.",
    "powerplant": "An industrial power plant interior organized along a straight axis extending forward and behind the observer, massive turbines, pipes, control panels, metallic textures, dramatic scale.",
    "ice_cave": "An ice cave interior forming a straight frozen passage extending forward and behind the observer, translucent blue ice walls, frozen textures, soft diffused light, cold and otherworldly atmosphere.",
    "canyon": "A narrow canyon passage with towering rock walls forming a straight path extending forward and behind the observer, limited sky visibility, strong vertical scale, natural erosion details.",
    "rock_gallery": "A man-made rock tunnel carved into stone, forming a straight gallery extending forward and behind the observer, visible chisel marks, dim lighting, ancient or industrial excavation feel.",
    "space_station": "A futuristic space station interior with straight circular corridors extending forward and behind the observer, smooth metallic surfaces, artificial lighting, immersive sci-fi environment.",
}

import sys
sys.path.append("/home/a.schnepf/phd/SphericalDreamer/")
from my_utils import yaml_load, save_config


NUM_DREAMS = 5
EXP_ID = 3

base_config = {
    "expname": "",
    "prompt": "",
    "num_dreams": NUM_DREAMS,

    "load_phase1a_from": None,
    "load_phase1b_from": None,
    "load_phase2a_from": None,
    "load_phase2b_from": None,
    "load_phase2c_from": None,
    "load_phase3_from": None,  


    "save_dir": f"OUTPUTS/SphericalDreamerRecurse/_exp{EXP_ID}/",
}

cfg_list = []
for key in PROMPTS.keys():
    cfg = base_config.copy()
    cfg["expname"] = key
    cfg["prompt"] = PROMPTS[key]

    save_config(cfg, cfg_name=f"{key}.yaml", save_dir=f"/home/a.schnepf/phd/SphericalDreamer/configs/exp{EXP_ID}/")
    cfg_list.append(key)

with open(f"/home/a.schnepf/phd/SphericalDreamer/configs/exp{EXP_ID}/_cfg_list.txt", "w") as f:
    for item in cfg_list:
        f.write(f"exp{EXP_ID}/{item}.yaml\n")