from pathlib import Path
import sys
import os
import shutil

sys.path.append("/home/a.schnepf/phd/SphericalDreamer")
from my_utils import yaml_load, save_config

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "expname.yaml"
PROMPT_PATH = HERE / "prompt.txt"
SAVE_AT = HERE / "cfgs"
SEEDS = [119224, 422911, 731056]

EXPNAMES = [
    "japanese_onsen",
    "soviet_metro",
    "gothic_cathedral",
    "cyberpunk_market",
    "nautilus_shell",
    "victorian_conservatory",
    "cloud_monastery",
    "artdeco_ballroom",
    "roman_bathhouse",
    "pharaoh_tomb",
    "ottoman_hammam",
    "aztec_temple",
    "alchemist_workshop",
    "ming_throne",
    "viking_longhouse",
    "moorish_palace",
    "byzantine_cistern",
    "pompeii_villa",
    "baroque_opera",
    "steel_mill",
    "retro_diner",
    "nuclear_bunker",
    "soviet_submarine",
    "artnouveau_metro",
    "brutalist_pool",
    "movie_palace",
    "victorian_brewery",
    "psychiatric_ward",
    "mission_control",
    "venice_canal",
    "moroccan_medina",
    "havana_street",
    "kowloon_slum",
    "london_alley",
    "tokyo_subway",
    "mumbai_chawl",
    "seoul_backstreet",
    "detroit_factory",
    "hongkong_scaffolding",
    "space_station",
    "asteroid_bazaar",
    "steampunk_cathedral",
    "postapocalyptic_mall",
    "underwater_dome",
    "orbital_habitat",
    "solarpunk_farm",
    "server_cathedral",
    "zerogravity_chapel",
    "giger_corridor",
    "torii_corridor",
    "grand_mosque",
    "tibetan_library",
    "speakeasy",
    "scriptorium",
    "seance_parlor",
    "funhouse",
    "colosseum_hypogeum",
    "railway_terminal",
    "pastel_hotel",
]


def parse_prompts(path):
    text = path.read_text()
    blocks = text.strip().split("\n\n")
    prompts = []
    for block in blocks:
        prompt = " ".join(line.strip() for line in block.strip().splitlines())
        if prompt:
            prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    prompts = parse_prompts(PROMPT_PATH)
    template = yaml_load("expname.yaml", HERE)

    assert len(prompts) == len(EXPNAMES), (
        f"Mismatch: {len(prompts)} prompts vs {len(EXPNAMES)} expnames"
    )

    if SAVE_AT.exists():
        shutil.rmtree(SAVE_AT)
    SAVE_AT.mkdir(parents=True)

    cfg_list = []
    for expname, prompt in zip(EXPNAMES, prompts):
        for seed_idx, seed in enumerate(SEEDS):
            config = dict(template)
            config["expname"] = f"{expname}_s{seed_idx}"
            config["prompt"] = prompt
            config["seed"] = seed

            cfg_name = f"{expname}_s{seed_idx}.yaml"
            save_config(config, cfg_name, SAVE_AT)
            cfg_path = SAVE_AT / cfg_name
            cfg_list.append(cfg_path)
            print(f"  {cfg_name}")

    with open(HERE / "config_paths.txt", "w") as f:
        for cfg_path in cfg_list:
            f.write(str(cfg_path) + "\n")

    print(f"\nGenerated {len(cfg_list)} configs in {SAVE_AT}")
    print(f"Config paths written to {HERE / 'config_paths.txt'}")
