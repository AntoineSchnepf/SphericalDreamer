import open3d as o3d
import numpy as np
import os
import argparse
import sys
from IPython import get_ipython
import matplotlib.pyplot as plt
import pickle
import my_utils


from PIL import Image
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


class PointCloud:
    
    def __init__(self, pts, colors):
        """
        pts: np.array of shape [..., 3]
        colors: np.array of shape [..., 3] with values in [0-1]
        """
        self.pts = pts.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)
        assert self.pts.shape[0] == self.colors.shape[0], "Error: pts and colors must have the same number of points"

    def get_o3d_pointcloud(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        nan_mask = np.isnan(self.pts).any(axis=1)
        pcd.points = o3d.utility.Vector3dVector(self.pts[~nan_mask])
        pcd.colors = o3d.utility.Vector3dVector(self.colors[~nan_mask])
        return pcd

expname = "26_city"


if __name__ == "__main__":

    # -----------------------------------------
    # -- possible keys for pointclouds dict ---
    to_load = {
        # dream, state


        # ('dream_00', 'init'),

        # ('dream_01', 'init'),
        # ('dream_01', 'open'),
        # ('dream_01', 'blended_naive_w_excess'),
        # ('dream_01', 'blended_harmonic_w_excess'),
        # ('dream_01', 'blended_harmonic'),
        # ('dream_01', 'total'),

        # ('dream_02', 'open'),
        # ('dream_02', 'blended_naive_w_excess'),
        # ('dream_02', 'blended_harmonic_w_excess'),
        # ('dream_02', 'blended_harmonic'),
        # ('dream_02', 'total'),

        ('dream_02', 'total'),
    }

    # -------------------------------------------


    save_dir = "../DATA/SphericalDreamerRecurse"
    save_dir_ = f"{save_dir}/{expname}"

    # open pickle file
    filename = f"{save_dir_}/pointclouds.pkl"
    with open(filename, 'rb') as f:
        point_clouds = pickle.load(f)

    pcds = []
    for dream, state in to_load:
        pcds.append(point_clouds[dream][state].get_o3d_pointcloud())
    o3d.visualization.draw_geometries(pcds)
