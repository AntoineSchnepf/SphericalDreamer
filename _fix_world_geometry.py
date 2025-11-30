import pickle
from cv2 import threshold
import numpy as np 
import open3d as o3d
from PIL import Image
import os
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
import my_utils
from my_utils import PointCloud
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import os
os.chdir("/Users/a.schnepf/Documents/code/phd/scene_gen/SphericalDreamer")

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# OLD FUNCTIONS KEPT FOR REFERENCE
def unfold_and_plot_cylindrical_point_cloud(pts_cylindrical, colors, skip=100):
    # Plot unfolded cylindrical projection
    x = pts_cylindrical[..., 0]       # axial coordinate along X
    p_r = pts_cylindrical[..., 1]     # radial distance
    theta_c = pts_cylindrical[..., 2] # angle in [-pi,

    fig, axes = plt.subplots(2,1, figsize=(10, 10))
    sc = axes[0].scatter(x[::skip], theta_c[::skip], c=p_r[::skip], s=2, cmap='viridis')
    fig.colorbar(sc, ax=axes[0], label="Radial distance p")
    axes[0].set_xlabel("X (axial direction)")
    axes[0].set_ylabel("θ (angle around cylinder, radians)")
    axes[0].set_title("Unfolded Cylindrical Depth Map")

    # Plot unfolded colors 
    sc2 = axes[1].scatter(x[::skip], theta_c[::skip], c=colors[::skip], s=2)
    axes[1].set_xlabel("X (axial direction)")
    axes[1].set_ylabel("θ (angle around cylinder, radians)")
    axes[1].set_title("Unfolded Cylindrical Color Map")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def correct_walls_lp(x, y, p=6.0):
    mask = y > 0
    x_corr = x.copy()
    y_corr = y.copy()

    theta = np.atan2(y, x)
    r = np.sqrt(x**2 + y**2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    rho = 1.0 / (np.abs(cos_t)**p + np.abs(sin_t)**p)**(1.0/p)
    x = r * rho * cos_t
    y = r * rho * sin_t

    x_corr[mask] = x[mask]
    y_corr[mask] = y[mask]

    return x_corr, y_corr
# ENF OF OLD FUNCTIONS

if __name__ == "__main__":

    # --- parse args: which sphere to load --- #
    expname = "31_forest"
    num_dreams = 5
    # ---------------------------------------- #

    save_dir = "OUTPUTS/SphericalDreamerRecurse"
    save_dir_ = f"{save_dir}/{expname}"

    filename = f"{save_dir_}/pointclouds.pkl"
    with open(filename, 'rb') as f:
        point_clouds = pickle.load(f)
    pcd = point_clouds[f'dream_{num_dreams-1:02d}']['total'].get_o3d_pointcloud()
    old_lp_correct = False

    # --- script --- 
    width = 1440
    height = 720
    NEAR = 0.2
    FAR = 2.0
    
    pose_left = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    world_correction_kwargs = {
        "correct_depth": True,
        "near": NEAR,
        "far": FAR,
        "correct_walls": False,
        "correct_floor": True,
        "depth_threshold_for_floor_correction": 1.0,
    }

    sphere_radius = 1.0
    delta_walk = sphere_radius * np.pi / 2
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    translation = (num_dreams-1) * delta_walk * np.array([1, 0, 0])  # along x axis
    pose_right = my_utils.camera_translation(pose_left, translation) 


    # matplotlib visualization of point cloud
    pts=np.asarray(pcd.points)
    colors=np.asarray(pcd.colors)

    # remove points at both ends for now
    pts_corrected = my_utils.run_corrective_pipeline_on_world(
        pts=pts,
        colors=colors,
        pose_left=pose_left,
        pose_right=pose_right,
        translation_direction=translation_direction,
        verbose=True,
        plot=True,
        **world_correction_kwargs
    )
    
    if old_lp_correct:
        pts_corrected_x, pts_corrected_y, pts_corrected_z = pts_corrected[..., 0], pts_corrected[..., 1], pts_corrected[..., 2]
        pts_corrected_y_v2, pts_corrected_z_v2 = correct_walls_lp(pts_corrected_y, pts_corrected_z, p=6.0)
        pts_corrected =  np.stack((pts_corrected_x, pts_corrected_y_v2, pts_corrected_z_v2), axis=-1)


    final_pcd = my_utils.PointCloud(pts_corrected, colors).get_o3d_pointcloud()

    # remove outliers
    npoints_before = len(final_pcd.points)
    cl, ind = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.2)
    final_pcd = final_pcd.select_by_index(ind)
    npoints_after = len(final_pcd.points)
    print(f"Removed {npoints_before - npoints_after} Outliers Points  ({(npoints_before - npoints_after) / (npoints_before + 1e-8) * 100:.2f}%) ")


    o3d.visualization.draw_geometries([final_pcd])




