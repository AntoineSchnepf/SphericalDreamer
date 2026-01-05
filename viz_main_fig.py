from pathlib import Path
import numpy as np
import my_utils
from PIL import Image

save_dir = Path("OUTPUTS/SphericalDreamerRecurse")
save_dir__ = save_dir / "Caverns"


figdir = Path("/home/a.schnepf/phd/SphericalDreamer/Figures")
_phase_current = "1a"

for i, dream_iter in enumerate([3, 4]):

    depth = np.load(save_dir__ / f"dream_{dream_iter:02d}" / _phase_current / ".cache"/ "depth.npy")
    pano = Image.open(save_dir__ / f"dream_{dream_iter:02d}" / _phase_current / ".cache"/ "pano_rgb.png")

    depth_pil = my_utils.depth_to_pil(depth, cmap_name="plasma", vmin=0.1, vmax=1.0)
    pano.save(figdir / f"main_fig_01_pano_{dream_iter}.png")
    depth_pil.save(figdir / f"main_fig_02_depth_{dream_iter}.png")


