import os
import contextlib
from io import StringIO
import pipeline.bootstrap  # noqa: F401 - sets GLOG env vars and filters warnings

import sys
from pathlib import Path
import my_utils
import ldi_inpaiting as ldi

# local imports
with contextlib.redirect_stdout(StringIO()):
    from sphericaldreamer import SphericalDreamer

from pipeline.phases import PHASE_1A, PHASE_1B

_phase_1a = PHASE_1A
_phase_current = PHASE_1B


def _load_data(i, save_dir_, phase):
    return my_utils.load_rgbd_pano(dream=i, save_dir_=save_dir_, phase=phase)


if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(debug=False)
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

    spherical_dreamer = SphericalDreamer(
        pano_width=width,
        pano_height=height,
        pano_depth_temp_dir='/tmp/pano_depth_temp',
        depth_model=config.depth_model,
    )

    ldi.run_ldi_phase(
        config,
        save_dir_,
        spherical_dreamer,
        phase_tag=_phase_current,
        phase_cfg=config.phase1,
        load_phase_from=config.load_phase1b_from,
        dream_indices=range(config.num_dreams),
        load_data_fn=lambda i: _load_data(i, save_dir_, _phase_1a),
        save_path_fn=lambda i: save_dir_ / f"dream_{i:02d}" / _phase_current / "ldi_insights",
        viz_filename="07_depth_inpainting_visualization.png",
    )
