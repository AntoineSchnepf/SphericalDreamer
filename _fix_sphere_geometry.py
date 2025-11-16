import pickle
from cv2 import threshold
import numpy as np 
import open3d as o3d
from PIL import Image
import os
import sys
import time
from IPython import get_ipython
import matplotlib.pyplot as plt
import my_utils 

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



if __name__ == "__main__":

    # --- parse args: which sphere to load --- #
    expname = "24_city"
    dream_iter = 0
    open_right = True
    open_left = False
    # ---------------------------------------- #


    save_dir = "OUTPUTS/SphericalDreamerRecurse"
    save_dir_ = f"{save_dir}/{expname}"

    # --- script init --- 
    width = 1440
    height = 720
    sphere_radius = 1.0
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    FAR=2.0
    NEAR=0.2
    opening_kwargs = {
        'opening_mode': 'cut+cylinder',
        'delta_cut': 2*np.pi/3,
    }
    pose1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    # --- script init end --- 


    colors1, depth1 = my_utils.load_rgbd_pano(
        dream=0,
        save_dir_=save_dir_
    )
    # depth1 = np.ones_like(depth1) * 1.0  # dummy depth for testing

    pts1_carte = my_utils.depth2cam_carte(
        depth=depth1,
        sphere_radius=sphere_radius,
        height=height,
        width=width,
    ) 
    pts1_carte_corrected, colors1_corrected = my_utils.run_corrective_pipeline_on_sphere(
        pts1_carte, # in cartesian coordinates
        colors1, 
        height, width, 
        correct_depth=False, 
        near=NEAR, 
        far=FAR, 
        correct_walls=True, 
        correct_floor=True, 
        depth_threshold_for_floor_correction=0.6, 
        remove_sky=False, 
        indoor_or_outdoor=None, 
        remove_outliers=True, 
        verbose=False,
        plot=True,
    )



    sphere1 = my_utils.Sphere(
        pose1, pts1_carte_corrected, colors1_corrected, 
        forward_carte=translation_direction,
        opening_kwargs=opening_kwargs,
    )

    # open 3d viewer
    if open_right and not open_left:
        pcd = sphere1.right_opened.get_world_pcd().get_o3d_pointcloud()
    elif open_left and not open_right:
        pcd = sphere1.left_opened.get_world_pcd().get_o3d_pointcloud()
    elif open_right and open_left:
        pcd = sphere1.both_opened.get_world_pcd().get_o3d_pointcloud()
    else:
        pcd = sphere1.closed.get_world_pcd().get_o3d_pointcloud()

    o3d.visualization.draw_geometries([pcd])




