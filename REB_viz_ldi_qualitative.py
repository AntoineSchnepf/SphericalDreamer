"""
Collect specific LDI qualitative images (mask + inpainting) and copy them
into a flat organized output directory.

Source:
  OUTPUTS/SphericalDreamerRecurse/old/experiments/_expN/<expname>/dream_XX/1b/ldi_insights/07_final_selected_mask.png
  OUTPUTS/SphericalDreamerRecurse/old/experiments/_expN/<expname>/dream_XX/1b/ldi_insights/08_lama_flux_double_inpainting.png

Destination:
  OUTPUTS/ldi_qualitative/<expname>/XX_ldi_mask.png
  OUTPUTS/ldi_qualitative/<expname>/XX_ldi_inpaint.png
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "OUTPUTS" / "SphericalDreamerRecurse" / "old" / "experiments"
OUTPUT_DIR = ROOT / "OUTPUTS" / "ldi_qualitative"

# (exp_folder, expname, dream_number)
TARGETS = [
    ("_exp4", "catacomb_network_2",         "01"),
    ("_exp4", "coral_abyss_2",              "02"),
    ("_exp4", "crystal_tundra_2",           "02"),
    ("_exp5", "endless_marble_colonnade",   "00"),
    ("_exp5", "fungal_underground_colony",  "02"),
    ("_exp5", "infinite_greenhouse_corridor","01"),
    ("_exp5", "misty_bamboo_forest",        "01"),
    ("_exp4", "stone_pillar_desert_2",      "02"),
    ("_exp5", "submerged_ancient_city",     "01"),
]

IMAGE_TYPES = {
    "07_final_selected_mask.png": "ldi_mask",
    "08_lama_flux_double_inpainting.png": "ldi_inpaint",
}


def collect_and_copy():
    copied = 0
    missing = 0

    for exp_folder, expname, dream_num in TARGETS:
        insights_dir = (
            EXPERIMENTS_DIR / exp_folder / expname / f"dream_{dream_num}" / "1b" / "ldi_insights"
        )

        for src_filename, dst_tag in IMAGE_TYPES.items():
            src = insights_dir / src_filename
            if not src.is_file():
                print(f"[MISSING] {src}")
                missing += 1
                continue

            dst_dir = OUTPUT_DIR / expname
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{dream_num}_{dst_tag}.png"

            shutil.copy2(src, dst)
            print(f"  {expname}/{dream_num}_{dst_tag}.png")
            copied += 1

    print(f"\nDone. Copied {copied} images ({missing} missing).")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    collect_and_copy()
