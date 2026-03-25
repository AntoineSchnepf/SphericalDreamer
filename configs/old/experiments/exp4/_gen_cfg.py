PROMPTS = {
    "bioluminescent_forest": (
        "Towering, softly glowing plants with repeating organic structures in all directions. "
        "The density and scale remain consistent, creating an endless, luminous canopy with no dominant orientation."
    ),

    "crystal_tundra": (
        "A flat, icy plain densely populated with translucent crystal spires of varying heights, "
        "refracting light uniformly in every direction under a pale sky."
    ),

    "martian_badlands": (
        "Rust-colored terrain covered with evenly distributed, low alien flora and mineral formations, "
        "extending homogeneously across the horizon."
    ),

    "stone_pillar_desert": (
        "A vast arid landscape filled with naturally eroded rock columns arranged quasi-periodically, "
        "forming an omnidirectional maze-like environment."
    ),

    "coral_abyss": (
        "A deep-sea environment with large coral structures and floating biotic forms surrounding the observer uniformly, "
        "with no clear up or down orientation beyond light falloff."
    ),

    "infinite_library": (
        "A circular or repeating architectural space lined with identical bookshelves and arches, "
        "extending endlessly in all horizontal directions."
    ),

    "catacomb_network": (
        "Stone corridors branching symmetrically, with repeating alcoves, pillars, and arches, "
        "creating uniform visual content regardless of viewing direction."
    ),

    "futuristic_datacenter": (
        "Rows of identical server racks and glowing conduits arranged radially or in a grid, "
        "forming a technologically dense but directionally uniform interior."
    ),

    "alien_hive": (
        "Organic, ribbed walls and repeating chambers grown rather than built, "
        "with tunnels radiating in all directions and consistent structural motifs."
    ),

    "circular_metro": (
        "A large ring-shaped underground station with repeating platforms, pillars, and signage, "
        "giving the impression of continuity in every direction."
    ),

    "bioluminescent_forest_2": (
        "An alien forest illuminated by bioluminescence, with tall plant trunks in deep indigo and emerald tones, "
        "broad leaves glowing in cyan and turquoise, and soft violet light diffusing through a dense canopy, "
        "creating a luminous, otherworldly atmosphere."
    ),

    "crystal_tundra_2": (
        "A frozen tundra covered in translucent crystal spires tinted with pale blue, icy white, and hints of lavender, "
        "set on a snow-covered ground under a desaturated gray-blue sky, with cold light refracting through the crystals."
    ),

    "martian_badlands_2": (
        "A Martian landscape of rust-red and ochre soil, scattered with dark basalt rocks and patches of muted green and purple alien vegetation, "
        "under a dusty salmon-colored sky."
    ),

    "stone_pillar_desert_2": (
        "An arid desert filled with tall stone pillars in warm sandstone, amber, and reddish-brown hues, "
        "standing on a pale beige ground beneath a bright, sun-bleached sky."
    ),

    "coral_abyss_2": (
        "A deep underwater environment with large coral formations in shades of crimson, orange, and violet, "
        "surrounded by dark blue water, floating particles, and soft cyan light filtering from above."
    ),

    "infinite_library_2": (
        "A grand interior library space with dark wooden bookshelves, brass details, cream-colored stone arches, "
        "and warm golden lighting reflecting off polished floors and leather-bound books."
    ),

    "catacomb_network_2": (
        "An underground stone catacomb with walls and arches in weathered limestone gray, dusty beige floors, "
        "and dim torchlight casting warm orange highlights and deep shadows."
    ),

    "futuristic_datacenter_2": (
        "A futuristic data center interior dominated by matte black and dark gray server racks, "
        "interlaced with glowing blue, green, and magenta cables under cool white overhead lighting."
    ),

    "alien_hive_2": (
        "An alien hive interior formed from organic structures in dark bronze, obsidian black, and muted amber tones, "
        "with glossy surfaces reflecting a soft internal orange and teal glow."
    ),

    "circular_metro_2": (
        "An abandoned underground metro station with concrete walls in faded gray, tiled surfaces in off-white and pale green, "
        "yellowed signage, and cold fluorescent lighting casting a bluish tint across the space."
    ),
}
import sys
sys.path.append("/home/a.schnepf/phd/SphericalDreamer/")
from my_utils import yaml_load, save_config


NUM_DREAMS = 3
EXP_ID = 4

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