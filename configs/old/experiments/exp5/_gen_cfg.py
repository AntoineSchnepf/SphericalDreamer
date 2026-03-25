PROMPTS  = {
    "misty_bamboo_forest": "A dense bamboo forest with tall slender stalks extending forward and continuing behind the observer, smooth vertical geometry with no sharp edges, pale green and muted yellow bamboo surfaces, soft leafy canopy filtering light from above, overcast sky producing diffused illumination, light mist between the stalks reducing distant visibility, calm and meditative atmosphere, immersive natural woodland environment.",

    "endless_lavender_fields": "An expansive lavender field extending forward and behind the observer with no visible boundaries, long rows of purple lavender plants forming soft repetitive patterns, green stems and rich violet blossoms gently blending together, cloudy sky with soft diffused daylight, minimal shadows, distant low hills barely visible through haze, peaceful rural atmosphere, immersive aromatic countryside landscape.",

    "glacial_ice_tunnel": "A smooth ice tunnel carved through a glacier extending forward and continuing behind the observer, rounded translucent ice walls with layered textures, pale blue and cyan tones throughout, soft light diffusing through the ice with no direct sun, subtle internal reflections and refractions, cold and quiet atmosphere, immersive frozen subterranean environment.",

    "floating_stone_valley": "A surreal valley filled with large smooth stone forms gently floating above the ground, extending forward and continuing behind the observer, rounded monolithic shapes with eroded surfaces, muted gray and sandy color palette, overcast sky with soft diffused light, faint atmospheric haze, dreamlike and tranquil mood, immersive fantastical landscape.",

    "submerged_ancient_city": "A submerged ancient city environment viewed underwater, extending forward and behind the observer, smooth stone structures softened by erosion and coral growth, green-blue lighting filtered through water, sandy seabed with scattered debris, slow floating particles in the water, calm and mysterious atmosphere, immersive aquatic ruins.",

    "infinite_wheat_plains": "A wide open wheat plain extending forward and continuing behind the observer, tall golden wheat stalks forming gentle waves across the landscape, soft rolling terrain with no sharp features, cloudy sky producing even diffused lighting, subtle wind motion implied through texture flow, peaceful agricultural atmosphere, immersive rural environment.",

    "crystal_salt_flats": "An expansive salt flat landscape extending forward and behind the observer, smooth white crystalline ground with subtle geometric cracking patterns, shallow reflective patches scattered across the surface, pale sky with no visible sun, soft ambient lighting creating minimal shadows, serene and minimalist atmosphere, immersive surreal desert environment.",

    "fungal_underground_colony": "An underground environment dominated by large fungal growths extending forward and behind the observer, rounded mushroom forms with soft glowing caps, organic walls covered in spores and moss, warm amber and cool violet bioluminescent lighting, dark earthy ground, quiet and alien atmosphere, immersive subterranean ecosystem.",

    "abandoned_subway_tunnel": "A long abandoned subway tunnel extending forward and continuing behind the observer, curved concrete walls softened by grime and age, muted gray and brown tones, evenly spaced dim industrial lights creating a repeating rhythm, damp ground with subtle reflections, quiet and desolate atmosphere, immersive underground transit space.",

    "endless_marble_colonnade": "A monumental marble colonnade with repeating rounded columns extending forward and behind the observer, smooth polished stone surfaces in pale white and beige tones, soft indirect lighting with no visible source, minimal shadows emphasizing symmetry, calm and timeless atmosphere, immersive classical architectural environment.",

    "dense_rainforest_understory": "A dense rainforest understory extending forward and behind the observer, thick overlapping vegetation with no sharp edges, large leaves in deep green tones, moist ground covered in roots and organic debris, overcast sky filtering light through the canopy, soft diffused illumination, humid and lush atmosphere, immersive tropical environment.",

    "alien_silicon_desert": "An alien desert composed of smooth silicon-like dunes extending forward and behind the observer, reflective pale surfaces with subtle iridescent hues, soft undulating terrain with no sharp ridges, hazy sky with diffused light, faint shimmering heat distortions, quiet and otherworldly atmosphere, immersive extraterrestrial landscape.",

    "underground_water_reservoir": "A vast underground water reservoir extending forward and behind the observer, smooth concrete walls with rounded edges, still water covering the floor creating mirror-like reflections, cool blue-gray lighting evenly illuminating the space, repeating structural pillars fading into the distance, calm and echoing atmosphere, immersive industrial interior.",

    "rolling_moss_hills": "A landscape of gentle moss-covered hills extending forward and behind the observer, smooth organic terrain with no sharp features, vibrant green moss textures covering the ground, soft overcast sky producing even lighting, minimal contrast and shadows, serene and natural atmosphere, immersive fantasy meadow environment.",

    "deserted_meditation_temple": "A deserted meditation temple complex with long covered walkways extending forward and behind the observer, smooth stone floors and rounded columns, neutral sand and stone tones, soft ambient lighting with no visible sources, sparse architectural details repeated rhythmically, peaceful contemplative atmosphere, immersive spiritual environment.",

    "coral_reef_canyon": "A wide coral reef canyon extending forward and behind the observer, smooth rock walls covered in coral and marine growth, vibrant yet softened colors including turquoise, coral pink, and sandy beige, filtered underwater lighting with soft light rays, floating particles in the water, calm immersive oceanic environment.",

    "wind_eroded_plateau": "A high plateau shaped by wind erosion extending forward and behind the observer, smooth layered rock formations with rounded edges, muted earth tones of beige and gray, hazy sky diffusing sunlight evenly, distant features fading into atmospheric haze, quiet and expansive atmosphere, immersive natural landscape.",

    "infinite_greenhouse_corridor": "A massive greenhouse corridor extending forward and continuing behind the observer, curved glass walls and ceiling with smooth metal frames, lush green plants growing uniformly along both sides, soft natural light diffused through cloudy glass panels, warm humid atmosphere, repetitive yet organic structure, immersive botanical interior.",

    "subarctic_tundra_plain": "A wide subarctic tundra plain extending forward and behind the observer, low soft vegetation covering gently rolling ground, muted greens and browns mixed with patches of frost, overcast sky producing flat diffused lighting, minimal shadows, cold and open atmosphere, immersive northern wilderness environment."
}
import sys
sys.path.append("/home/a.schnepf/phd/SphericalDreamer/")
from my_utils import yaml_load, save_config


NUM_DREAMS = 3
EXP_ID = 5

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