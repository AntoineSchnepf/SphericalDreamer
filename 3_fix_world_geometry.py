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
from PIL import Image

from typing import Tuple, Union
import numpy as np
import torch

def bounding_box_xy(
    points: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, float, float, float]:
    """
    Compute a 2D bounding box (x_min, x_max, y_min, y_max) from a point cloud,
    intentionally ignoring the Z coordinate.

    Parameters
    ----------
    points : np.ndarray or torch.Tensor, shape (N, 3)
        Point cloud as (X, Y, Z).

    Returns
    -------
    (x_min, x_max, y_min, y_max) : tuple of floats
    """
    if isinstance(points, torch.Tensor):
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("points must have shape (N, 3) or (N, >=2)")
        x = points[:, 0]
        y = points[:, 1]
        x_min = float(x.min().item())
        x_max = float(x.max().item())
        y_min = float(y.min().item())
        y_max = float(y.max().item())
        return x_min, x_max, y_min, y_max

    elif isinstance(points, np.ndarray):
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError("points must have shape (N, 3) or (N, >=2)")
        x = points[:, 0]
        y = points[:, 1]
        x_min = float(x.min())
        x_max = float(x.max())
        y_min = float(y.min())
        y_max = float(y.max())
        return x_min, x_max, y_min, y_max

    else:
        raise TypeError("points must be a numpy array or torch tensor")
    
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
        debug_parser_override=["--config", "exp0/0_caverns.yaml"]
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
    
    # 3. Save corrected point cloud
    t0 = time.time()
    with open(save_dir_ /f"{_phase_3}_final_dream_pcd_unfiltered.pkl", "wb") as f:
        pickle.dump(PointCloud_instance, f)
    printc(f"--- {_phase_current}: Saved unfiltered point cloud to {save_dir_ /f'{_phase_3}_final_dream_pcd_unfiltered.pkl'} in {time.time() - t0:.2f} seconds.", color='yellow')

    # Save pcd as .ply
    t0 = time.time()
    # my_pcd = PointCloud_instance.get_o3d_pointcloud()
    points = np.asarray(PointCloud_instance.pts, dtype=np.float32)
    colors = np.asarray(PointCloud_instance.colors, dtype=np.float32)
    my_pcd = o3d.geometry.PointCloud()
    my_pcd.points = o3d.utility.Vector3dVector(points)
    my_pcd.colors = o3d.utility.Vector3dVector(colors)
    printc(f"--- {_phase_current}: Converted to o3d point cloud in {time.time() - t0:.2f} seconds.", color='yellow')
    t0 = time.time()
    o3d.io.write_point_cloud(save_dir_ /f"{_phase_3}_final_dream_pcd_unfiltered.ply", my_pcd)
    printc(f"--- {_phase_current}: Saved unfiltered point cloud to {save_dir_ /f'{_phase_3}_final_dream_pcd_unfiltered.ply'} in {time.time() - t0:.2f} seconds.", color='yellow')

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
            raise NotImplementedError("Voxel downsampling with ldi_mask was depreciated (10/01/2025).")
            t0 = time.time()
            pcd = PointCloud_instance.get_o3d_pointcloud()
            voxel_size = config.phase3.pointcloud_downsampling.voxel_options.voxel_size
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            printc(f"--- {_phase_current}: Downsampled point cloud from {n_pts_before} to {len(pcd.points)} points in {time.time() - t0:.2f} seconds using voxel downsampling.", color='yellow')
        
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

    # 5. Save corrected point cloud
    with open(save_dir_ /f"{_phase_3}_final_dream_pcd.pkl", "wb") as f:
        pickle.dump(PointCloud_instance, f)

    # Save pcd as .ply
    o3d.io.write_point_cloud(save_dir_ /f"{_phase_3}_final_dream_pcd.ply", PointCloud_instance.get_o3d_pointcloud())

    printc(f"--- {_phase_current}: Saved final point cloud to {save_dir_ /f'{_phase_3}_final_dream_pcd.pkl'}", color='yellow')
    printc(f"PHASE {_phase_current} SUCCESSFULLY COMPLETED!", color='green')