import os
os.environ.setdefault("OPEN3D_HEADLESS", "1")  # for open3d headless rendering
import numpy as np
import open3d as o3d
import pickle
import time
import my_utils
from my_utils import PointCloud, printc

from pipeline.phases import PHASE_2C, PHASE_3

_phase_2c = PHASE_2C
_phase_3 = PHASE_3
_phase_current = _phase_3

if __name__ == "__main__":
    config = my_utils.fetch_config_via_parser(debug=False)

    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)
    repo_path = os.path.dirname(os.path.realpath(__file__))

    # --------------------------------------- #
    # ---- PHASE 3 FIX WORLD GEOMETRY   ----- #
    # --------------------------------------- #

    printc(f"=== [PHASE {_phase_current}]  EXPERIMENT: {config.expname} ===", color='cyan')
    printc(f"=== PHASE {_phase_current} : FIX WORLD GEOMETRY ===", color='green')
    # 1. Load pcd from previous phase
    t0 = time.time()
    with open(save_dir_ /f"{_phase_2c}_raw_world_pcd.pkl", "rb") as f:
        PointCloud_instance = pickle.load(f)

    printc(f"--- {_phase_current}: Loaded raw point cloud in {time.time() - t0:.2f} seconds ---", color='yellow')
    t0 = time.time()

    # 2. Fix world geometry
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
        pts = PointCloud_instance.pts
        colors = PointCloud_instance.colors
        ldi_mask = PointCloud_instance.ldi_mask

        # remove points at both ends for now
        pts_corrected, colors_corrected, ldi_mask_corrected = my_utils.run_corrective_pipeline_on_world(
            pts=pts,
            colors=colors,
            ldi_mask=ldi_mask,
            pose_left=pose_left,
            pose_right=pose_right,
            translation_direction=translation_direction,
            verbose=False,
            plot=True,
            **config.phase3.world_correction.options
        )

        PointCloud_instance = my_utils.PointCloud(pts_corrected, colors_corrected, ldi_mask_corrected)
        printc(f"--- {_phase_current}: Corrected world geometry in {time.time() - t0:.2f} seconds.", color='yellow')
    
    # Save pcd as .ply (used by 5_render_blender.py)
    t0 = time.time()
    points = np.asarray(PointCloud_instance.pts, dtype=np.float32)
    colors = np.asarray(PointCloud_instance.colors, dtype=np.float32)
    my_pcd = o3d.geometry.PointCloud()
    my_pcd.points = o3d.utility.Vector3dVector(points)
    my_pcd.colors = o3d.utility.Vector3dVector(colors)
    printc(f"--- {_phase_current}: Converted to o3d point cloud in {time.time() - t0:.2f} seconds.", color='yellow')
    t0 = time.time()
    o3d.io.write_point_cloud(save_dir_ /f"{_phase_3}_world_pcd.ply", my_pcd)
    printc(f"--- {_phase_current}: Saved world point cloud to {save_dir_ /f'{_phase_3}_world_pcd.ply'} in {time.time() - t0:.2f} seconds.", color='yellow')

    # 2. Downsample point cloud for faster processing
    n_pts_before = len(PointCloud_instance.pts)
    if config.phase3.pointcloud_downsampling.mode != "deactivated":
        if config.phase3.pointcloud_downsampling.mode == "skip":
            t0 = time.time()
            pts, colors, ldi_mask = PointCloud_instance.pts, PointCloud_instance.colors, PointCloud_instance.ldi_mask
            stride = config.phase3.pointcloud_downsampling.skip_options.stride
            pts = pts[::stride]
            colors = colors[::stride]
            ldi_mask = ldi_mask[::stride]
            PointCloud_instance = PointCloud(pts=pts, colors=colors, ldi_mask=ldi_mask)
            printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(PointCloud_instance.pts)} points in {time.time() - t0:.2f} seconds using skip.", color='yellow')

        elif config.phase3.pointcloud_downsampling.mode == "voxel":
            raise NotImplementedError("Voxel downsampling with ldi_mask was removed (10/01/2025).")
        
        elif config.phase3.pointcloud_downsampling.mode == "auto":
            t0 = time.time()
            n_target = float(config.phase3.pointcloud_downsampling.auto_options.num_max_points)
            ratio = n_target / n_pts_before
            if ratio >= 1.0:
                printc(f"--- {_phase_current}: Auto downsampling skipped as point cloud has {n_pts_before} points which is less than target {n_target} points.", color='yellow')
            else:
                stride = int(np.ceil(1 / ratio))
                pts, colors, ldi_mask = PointCloud_instance.pts, PointCloud_instance.colors, PointCloud_instance.ldi_mask
                pts = pts[::stride]
                colors = colors[::stride]
                ldi_mask = ldi_mask[::stride]
                PointCloud_instance = PointCloud(pts=pts, colors=colors, ldi_mask=ldi_mask)
                printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(PointCloud_instance.pts)} points in {time.time() - t0:.2f} seconds using skipping (stride={stride}).", color='yellow')
        
        else: 
            raise ValueError(f"--- {_phase_current}: Unknown downsampling mode: {config.phase3.pointcloud_downsampling.mode}")
    
    else:
        printc(f"--- {_phase_current}: No downsampling applied to point cloud, keeping {n_pts_before} points.", color='yellow')



    # 4. Remove outliers
    if config.phase3.remove_outliers.apply:
        t0 = time.time()
        n_before = len(PointCloud_instance.pts)
        pcd = PointCloud_instance.get_o3d_pointcloud()
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=config.phase3.remove_outliers.options.nb_neighbors, 
                                                                std_ratio=config.phase3.remove_outliers.options.std_ratio) 
        ldi_mask = PointCloud_instance.ldi_mask[ind]
        PointCloud_instance = my_utils.PointCloud(
            pts=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors),
            ldi_mask=ldi_mask,
        )
        printc(f"--- {_phase_current}: Removed {n_before - len(PointCloud_instance.pts)} outliers in {time.time() - t0:.2f} seconds.", color='yellow')

    if not np.isfinite(PointCloud_instance.pts).all():
        printc("WARNING: Point cloud contains NaN or infinite values", color='red')

    # 5. Save downsampled point cloud (used by 4_render_video.py)
    with open(save_dir_ /f"{_phase_3}_world_pcd_downsampled.pkl", "wb") as f:
        pickle.dump(PointCloud_instance, f)

    printc(f"--- {_phase_current}: Saved downsampled point cloud to {save_dir_ /f'{_phase_3}_world_pcd_downsampled.pkl'}", color='yellow')
    printc(f"PHASE {_phase_current} SUCCESSFULLY COMPLETED!", color='green')