from pathlib import Path
import numpy as np
import my_utils

save_dit = Path("OUTPUTS/SphericalDreamerRecurse/")
save_dir__ = save_dit / "forest_v3/align_01"

_phase_current = "2a"


data = np.load(save_dir__ / _phase_current / ".cache"/ "other_data.npy", allow_pickle=True)
warped_depth_interp = data.item().get("warped_depth_interp")
depth_estimated = data.item().get("depth_estimated")
warped_depth_interp
depth_estimated_pil = my_utils.depth_to_pil(warped_depth_interp, cmap_name="plasma", vmin=0.1, vmax=1.8)
warped_depth_interp_pil =my_utils.depth_to_pil(depth_estimated, cmap_name="plasma", vmin=0.1, vmax=1.8)
depth_estimated_pil.save(save_dir__ / _phase_current / "02_warped_depth_interp_pil.png")
warped_depth_interp_pil.save(save_dir__ / _phase_current / "07_estimated_depth_pil.png")