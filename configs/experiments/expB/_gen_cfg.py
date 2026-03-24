PROMPTS = {
    "mars_surface": "The surface of Mars under clear empty skies with no visible sun or celestial bodies, a wide open barren landscape extending forward and continuing behind the observer, fine reddish dust and compacted regolith forming gentle natural slopes, sparse scattered rocks and a few small boulders with weathered edges, subtle wind-swept texture lines, distant low hills barely breaking the horizon, dry silent atmosphere, realistic mineral textures, cinematic neutral lighting with soft shadows, immersive alien desert emptiness.",
    "salt_mine_interior": "The inside of a salt mine resembling a natural cave environment with no straight lines or sharp geometries, an irregular hollowed passage opening forward and continuing behind the observer, rounded organic shapes carved by excavation and erosion-like forms, white and pale pink salt walls with crystalline shimmer, rough matte areas mixed with glossy reflective patches, uneven ground with soft salt rubble, low diffused industrial lighting creating gentle highlights on crystals, faint haze in the air, immersive enclosed subterranean atmosphere with realistic salt textures.",
    "sound_of_music_grass_field": "A wide open rolling field of lush green grass inspired by The Sound of Music, gentle natural hills extending forward and continuing behind the observer with no harsh edges or defined geometry, thick healthy grass blades forming smooth wind-like patterns, scattered wildflower hints subtle and sparse, cloudy sky overhead with no visible sun, soft diffused daylight creating even illumination and minimal harsh shadows, distant tree line barely visible on the horizon, peaceful cinematic pastoral mood, realistic vegetation textures, immersive open countryside atmosphere.",
    "upside_down_stranger_things": "A desolate Upside Down-inspired landscape with cloudy oppressive skies and no visible sun, the environment stretching forward and behind the observer with organic uneven terrain, dark damp ground covered in tangled root-like growth and soft alien debris, floating ash-like particles suspended in the air, twisted vegetation silhouettes without sharp geometry, murky fog reducing visibility in the distance, muted blue-gray lighting with eerie contrast, wet reflective patches and slimy textures, cinematic horror mood, immersive otherworldly atmosphere.",
    "dune_desert": "A vast desert environment inspired by Dune with no plants and no visible sun, endless dunes of fine pale sand extending forward and continuing behind the observer, smooth wind-carved ridges and soft flowing slopes with no sharp geometry, subtle ripple patterns and shifting gradients across the sand, cloudy sky muting the light into a soft diffused glow, faint atmospheric haze near the horizon, minimal surface detail besides sand texture, quiet immense scale, cinematic epic emptiness, realistic desert materials and depth cues.",
    "botw_large_plain_grass": "A large open environment inspired by Zelda: Breath of the Wild, a wide grassy plain extending forward and continuing behind the observer, soft natural terrain with gentle slopes and no defined geometric shapes, long grass with varied density and subtle wind flow patterns, occasional small rocks or uneven patches, broad cloudy or softly lit sky depending on mood with no strong direct sun highlight, distant hills fading into atmospheric haze, bright natural color palette, cinematic adventure atmosphere, immersive expansive openness.",
    "teletubbies_landscape": "A Teletubbies-inspired playful outdoor environment with soft rounded hills and no defined geometry, the landscape extending forward and behind the observer with bright smooth grassy terrain, vibrant green ground with gentle rolling curves, simple scattered flowers or tiny bushes kept minimal, sky evenly lit with no visible sun, clean cheerful lighting with soft shadows, saturated friendly colors, whimsical calm atmosphere, minimal detail but realistic grass texture, immersive childlike surreal countryside feeling.",
    "ski_snowfield_trees_far": "An empty snow field made for skiing, a wide open expanse of smooth snow extending forward and continuing behind the observer, subtle groomed ski texture lines or gentle wind-packed patterns, distant line of dark evergreen trees far away near the horizon, clear pale sky overhead with no visible sun, cold crisp lighting with soft shadows and bright snow reflections, minimal objects in the foreground, strong sense of open space and quiet, realistic snow surface texture, immersive alpine winter atmosphere.",
    "zelda_cave_organic": "A Zelda-inspired cave interior with no well-defined or sharp geometries, an irregular cavern passage extending forward and behind the observer, rounded rock formations and natural arches, soft uneven walls with mossy patches and mineral staining, scattered stones and earthy ground, subtle glowing ambient light as if from distant bioluminescent minerals or soft torches out of view, cinematic fantasy mood, realistic rock textures with gentle moisture highlights, immersive adventurous underground environment.",
    "venus_surface": "The surface of Venus with no defined geometry, a wide open alien terrain extending forward and behind the observer, thick cloudy skies overhead with no visible sun, hazy atmosphere reducing distance clarity, yellow-orange tinted light diffused through clouds, ground covered in rocky plains and cracked volcanic-like textures, occasional smooth boulders softened by erosion, heat-like shimmer in the air, muted contrast, realistic mineral textures, oppressive atmospheric density, immersive hostile planet surface mood.",
    "foreign_planet_surface": "The surface of an unknown foreign planet with no defined geometry and no visible sun, a broad open terrain extending forward and continuing behind the observer, smooth natural landforms with gentle rises and dips, strange soil coloration and subtle mineral variation, minimal scattered rocks and sparse alien debris, sky empty or clouded depending on atmosphere but with no celestial objects present, soft diffused lighting, atmospheric haze adding depth, realistic alien material textures, cinematic exploration mood, immersive uncharted world feeling.",
    "tomato_field_large": "A large agricultural field where tomatoes are planted, extending forward and continuing behind the observer with no defined sharp geometries, long organized rows of tomato plants forming repeating patterns across soft soil beds, lush green foliage with clusters of small tomatoes visible intermittently, cloudy sky overhead with no visible sun, soft diffused daylight evenly lighting the scene, moist dark soil texture with small footprints or irrigation traces, distant farm structures barely visible on the horizon, calm realistic rural atmosphere, immersive cultivated landscape.",
}

import sys
sys.path.append("/home/k.kassab/panorama/SphericalDreamer/")
from my_utils import yaml_load, save_config


NUM_DREAMS = 3
EXP_ID = "B"

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

    save_config(cfg, cfg_name=f"{key}.yaml", save_dir=f"/home/k.kassab/panorama/SphericalDreamer/configs/exp{EXP_ID}/")
    cfg_list.append(key)

with open(f"/home/k.kassab/panorama/SphericalDreamer/configs/exp{EXP_ID}/_cfg_list.txt", "w") as f:
    for item in cfg_list:
        f.write(f"exp{EXP_ID}/{item}.yaml\n")