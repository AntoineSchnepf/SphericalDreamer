import os
os.environ.setdefault("OPEN3D_HEADLESS", "1") # for open3d headless rendering
import numpy as np
import open3d as o3d
import pickle
import my_utils
import numpy as np
import open3d as o3d
import time
from my_utils import PointCloud
from my_utils import set_camera_from_elev_azim, printc
from tqdm import tqdm

_phase_1a = "1a"
_phase_1b = "1b"
_phase_2a = "2a"
_phase_2b = "2b"
_phase_2c = "2c"
_phase_3 = "3"
_phase_current = _phase_3

if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "Antoine/debug.yaml"]
    )

    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)
    repo_path = os.path.dirname(os.path.realpath(__file__))

    # --------------------------------------- #
    # ---- PHASE 3 FIX WORLD GEOMETRY   ----- #
    # --------------------------------------- #

    printc(f"=== [PHASE {_phase_current}]  EXPERIMENT: {config.expname} ===", color='cyan')
    printc(f"=== PHASE {_phase_current} : FIX WORLD GEOMETRY ===", color='green')
    # 1. Load pcd from previous phase
    t0 = time.time()
    with open(save_dir_ /f"{_phase_2c}_raw_dream_pcd.pkl", "rb") as f:
        PointCloud_instance = pickle.load(f)

    printc(f"--- {_phase_current}: Loaded raw point cloud in {time.time() - t0:.2f} seconds ---", color='yellow')
    t0 = time.time()

    # 2. Downsample point cloud for faster processing
    n_pts_before = len(PointCloud_instance.pts)
    if config.phase3.pointcloud_downsampling.mode != "deactivated":
        if config.phase3.pointcloud_downsampling.mode == "skip":
            t0 = time.time()
            pts, colors = PointCloud_instance.pts, PointCloud_instance.colors
            stride = config.phase3.pointcloud_downsampling.skip_options.stride
            pts = pts[::stride]
            colors = colors[::stride]
            PointCloud_instance = PointCloud(pts=pts, colors=colors)
            pcd = PointCloud_instance.get_o3d_pointcloud()
            printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(pcd.points)} points in {time.time() - t0:.2f} seconds using skip.", color='yellow')
        elif config.phase3.pointcloud_downsampling.mode == "voxel":
            t0 = time.time()
            pcd = PointCloud_instance.get_o3d_pointcloud()
            voxel_size = config.phase3.pointcloud_downsampling.voxel_options.voxel_size
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(pcd.points)} points in {time.time() - t0:.2f} seconds using voxel downsampling.", color='yellow')
        elif config.phase3.pointcloud_downsampling.mode == "auto":
            t0 = time.time()
            pcd = PointCloud_instance.get_o3d_pointcloud()
            n_target = float(config.phase3.pointcloud_downsampling.auto_options.num_max_points)
            ratio = n_target / n_pts_before
            if ratio >= 1.0:
                printc(f"--- {_phase_current}: Auto downsampling skipped as point cloud has {n_pts_before} points which is less than target {n_target} points.", color='yellow')
            else:
                stride = int(np.ceil(1 / ratio))
                pts, colors = PointCloud_instance.pts, PointCloud_instance.colors
                pts = pts[::stride]
                colors = colors[::stride]
                PointCloud_instance = PointCloud(pts=pts, colors=colors)
                pcd = PointCloud_instance.get_o3d_pointcloud()
                printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(pcd.points)} points in {time.time() - t0:.2f} seconds using skipping (stride={stride}).", color='yellow')
        else: 
            raise ValueError(f"--- {_phase_current}: Unknown downsampling mode: {config.phase3.pointcloud_downsampling.mode}")
    else:
        pcd = PointCloud_instance.get_o3d_pointcloud()
        printc(f"--- {_phase_current}: No downsampling applied to point cloud, keeping {n_pts_before} points.", color='yellow')


    # 3. Fix world geometry
    if config.phase3.world_correction.apply:
        t0 = time.time()
        pose_left = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        delta_walk = config.sphere_radius * config.delta_walk 
        translation_direction = my_utils.get_norm_vector(np.array(config.translation_direction, dtype=np.float32))
        translation = (config.num_dreams-1) * delta_walk * np.array(config.translation_direction)
        pose_right = my_utils.camera_translation(pose_left, translation) 

        # matplotlib visualization of point cloud
        pts=np.asarray(pcd.points)
        colors=np.asarray(pcd.colors)

        # remove points at both ends for now
        pts_corrected, colors = my_utils.run_corrective_pipeline_on_world(
            pts=pts,
            colors=colors,
            pose_left=pose_left,
            pose_right=pose_right,
            translation_direction=translation_direction,
            verbose=False,
            plot=False,
            **config.phase3.world_correction.options
        )

        pcd = my_utils.PointCloud(pts_corrected, colors).get_o3d_pointcloud()
        printc(f"--- {_phase_current}: Corrected world geometry in {time.time() - t0:.2f} seconds.", color='yellow')

    # 4. Remove outliers
    if config.phase3.remove_outliers.apply:
        t0 = time.time()
        n_before = len(pcd.points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config.phase3.remove_outliers.options.nb_neighbors, 
                                                                std_ratio=config.phase3.remove_outliers.options.std_ratio) 

        printc(f"--- {_phase_current}: Removed {n_before - len(pcd.points)} outliers in {time.time() - t0:.2f} seconds.", color='yellow')

    if not np.isfinite(pcd.points).all():
        printc("WARNING: Point cloud contains NaN or infinite values", color='red')

    # 5. Save corrected point cloud
    with open(save_dir_ /f"{_phase_3}_final_dream_pcd.pkl", "wb") as f:
        PointCloud_instance = my_utils.PointCloud(
            pts=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors)
        )
        pickle.dump(PointCloud_instance, f)

    printc(f"--- {_phase_current}: Saved final point cloud to {save_dir_/_phase_3}_final_dream_pcd.pkl", color='yellow')
    printc(f"PHASE {_phase_current} SUCCESSFULLY COMPLETED!", color='green')