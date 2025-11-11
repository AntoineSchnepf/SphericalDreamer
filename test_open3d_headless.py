import os
import numpy as np
import open3d as o3d
import pickle
from my_utils import PointCloud
with open("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/24_forest/pointclouds.pkl", "rb") as f:
    pointclouds = pickle.load(f)

# --- Headless hint (usually not required if you built Open3D headless) ---
# Some environments honor this variable to force headless mode.
os.environ.setdefault("OPEN3D_HEADLESS", "1")

pcd = pointclouds['dream_04']['total'].get_o3d_pointcloud()
pcd_filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print(f"Filtered point cloud: {len(pcd_filtered.points)} points")
print(f"Removed {len(pcd.points) - len(pcd_filtered.points)} outliers")

# -------- Headless rendering (Offscreen) --------
# Color the filtered cloud for visibility
pcd_filtered.paint_uniform_color([0.0, 0.6, 1.0])

# Optional: also visualize removed points in faint gray (comment out if not needed)
# out_idx = np.setdiff1d(np.arange(points.shape[0]), np.asarray(ind))
# pcd_removed = o3d.geometry.PointCloud()
# pcd_removed.points = o3d.utility.Vector3dVector(points[out_idx])
# pcd_removed.paint_uniform_color([0.6, 0.6, 0.6])

# Create an offscreen renderer (no window)
width, height = 1280, 720
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene

# Nice clean white background
scene.set_background([1.0, 1.0, 1.0, 1.0])

# Basic unlit material for points
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 3.0  # make points visible

# Add geometries
scene.add_geometry("filtered", pcd_filtered, mat)
# scene.add_geometry("removed", pcd_removed, mat)  # if you enabled pcd_removed above

# Frame the camera to fit the cloud
bbox = pcd_filtered.get_axis_aligned_bounding_box()
center = bbox.get_center()

# A bit of ambient light helps with meshes, but points are unlit; still set ambient

# Render to image and save
img = renderer.render_to_image()
out_path = "filtered_point_cloud.png"
o3d.io.write_image(out_path, img)
print(f"Saved headless render to {out_path}")