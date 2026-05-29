import os
import contextlib
from io import StringIO
import pipeline.bootstrap  # noqa: F401 - sets GLOG env vars and filters warnings

import sys
import numpy as np
from pathlib import Path
import my_utils
import ldi_inpaiting as ldi

# local imports
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer

from pipeline.phases import PHASE_2A, PHASE_2B

_phase_2a = PHASE_2A
_phase_current = PHASE_2B


if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(debug=False)
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    def _load_data(i):
        """Load inpainted RGB and estimated depth from the phase-2a cache for dream i."""
        save_dir__ = save_dir_ / f"align_{i:02d}"
        data = np.load(save_dir__ / _phase_2a / ".cache" / "other_data.npy", allow_pickle=True).item()
        img = my_utils.opencv_resize(
            my_utils.PIL_to_numpy(data['pano_rgb_inpainted']),
            config.height, config.width, mode="bilinear",
        )
        depth = my_utils.opencv_resize(
            data['depth_estimated'],
            config.height, config.width, mode="bilinear",
        )
        return img, depth

    ldi.run_ldi_phase(
        config,
        save_dir_,
        spherical_dreamer,
        phase_tag=_phase_current,
        phase_cfg=config.phase2,
        load_phase_from=config.load_phase2b_from,
        dream_indices=range(1, config.num_dreams),
        load_data_fn=_load_data,
        save_path_fn=lambda i: save_dir_ / f"align_{i:02d}" / _phase_current / "ldi_insights",
        viz_filename="10_depth_inpainting_visualization.png",
    )
