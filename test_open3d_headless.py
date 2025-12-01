import os
import numpy as np
import open3d as o3d
import pickle
from my_utils import PointCloud


import my_utils
import numpy as np
import open3d as o3d

def set_camera_from_elev_azim(scene_camera,
                              cam_pos,
                              elev_deg,
                              azim_deg,
                              fov_deg,
                              width,
                              height,
                              near,
                              far):
    """
    Set Open3D rendering camera from:
      - camera position (world coords),
      - elevation angle (deg) above XY plane,
      - azimuth angle (deg) around Z,
      - perspective intrinsics (fov, near, far).

    Convention:
      - World Z is "up".
      - Azimuth = 0° looks along +X, increases toward +Y.
      - Elevation = 0° in XY plane, +90° straight up (+Z), -90° straight down.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    # Forward direction from spherical angles
    fx = np.cos(elev) * np.cos(azim)
    fy = np.cos(elev) * np.sin(azim)
    fz = np.sin(elev)
    forward = np.array([fx, fy, fz], dtype=float)

    # Default world up
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    # Avoid collinearity between forward and up
    if np.abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)

    # Orthonormal basis: right, up, forward
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Look-at point (along forward)
    lookat = cam_pos + forward

    # Use the correct FovType enum (Vertical or Horizontal)
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical

    scene_camera.set_projection(fov_deg,
                                width / height,
                                near,
                                far,
                                fov_type)
    scene_camera.look_at(lookat, cam_pos, up)


expname = '33_city'
num_dreams = 4
sphere_radius = 1.0
max_x = (num_dreams-1) * sphere_radius * np.pi/2
visualize_removed_points = False
remove_outliers = True
fix_world = True
point_size = 3.0

with open(f"/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/{expname}/raw_dream_pcd.pkl", "rb") as f:
    pcd = pickle.load(f).get_o3d_pointcloud()


if fix_world:
    NEAR = 0.1
    FAR = 1.0

    pose_left = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    world_correction_kwargs = {
        "correct_depth": False,
        "near": NEAR,
        "far": FAR,
        "correct_walls": True,
        "remove_outliers": False,
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
    pts_corrected, colors = my_utils.run_corrective_pipeline_on_world(
        pts=pts,
        colors=colors,
        pose_left=pose_left,
        pose_right=pose_right,
        translation_direction=translation_direction,
        verbose=True,
        plot=False,
        **world_correction_kwargs
    )

    final_pcd = my_utils.PointCloud(pts_corrected, colors).get_o3d_pointcloud()
else:
    final_pcd = pcd

# --- Headless hint (usually not required if you built Open3D headless) ---
# Some environments honor this variable to force headless mode.
os.environ.setdefault("OPEN3D_HEADLESS", "1")

if remove_outliers:
    pcd_filtered, ind = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)

    print(f"Filtered point cloud: {len(pcd_filtered.points)} points")
    print(f"Removed {len(final_pcd.points) - len(pcd_filtered.points)} outliers")
else:
    pcd_filtered = final_pcd
    ind = np.arange(len(final_pcd.points))

# -------- Headless rendering (Offscreen) --------
if visualize_removed_points:
    out_idx = np.setdiff1d(np.arange(np.asarray(final_pcd.points).shape[0]), np.asarray(ind))
    pcd_removed = o3d.geometry.PointCloud()
    pcd_removed.points = o3d.utility.Vector3dVector(np.asarray(final_pcd.points)[out_idx])
    pcd_removed.paint_uniform_color([0.6, 0.6, 0.6])

# Create an offscreen renderer (no window)
width, height = 1280, 720
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene

# black background
scene.set_background([0.0, 0.0, 0.0, 1.0])

# Basic unlit material for points
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = point_size  # make points visible

# Add geometries
scene.add_geometry("filtered", pcd_filtered, mat)
if visualize_removed_points:
    scene.add_geometry("removed", pcd_removed, mat)  # if you enabled pcd_removed above

# -------- Camera control --------
# Compute a basic bounding box and radius for scaling camera parameters
bbox = pcd_filtered.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()

for azim_deg in np.linspace(0, 360, num=20, endpoint=False):
    cam = scene.camera
    x_percent = 0.5
    min_x = 0.0

    x_pos = min_x + x_percent * (max_x - min_x)
    cam_pos = np.array([x_pos, 0.0, 0.0])  # will be set later
    elev_deg = 0.0
    fov_deg = 60.0
    near = 0.01 * np.linalg.norm(extent)
    far = 10.0 * np.linalg.norm(extent)

    set_camera_from_elev_azim(
        cam,
        cam_pos=cam_pos,       # 3D world position of the camera
        elev_deg=elev_deg,     # elevation angle
        azim_deg=azim_deg,     # azimuth angle
        fov_deg=fov_deg,
        width=width,
        height=height,
        near=near,
        far=far,
    ) 


    # A bit of ambient light helps with meshes, but points are unlit; still set ambient

    # Render to image and save
    where_save = os.path.join("OUTPUTS", "test_o3d_rendering", expname)
    os.makedirs(where_save, exist_ok=True)
    img = renderer.render_to_image()
    out_path = f"{where_save}/azim={azim_deg:.1f}__point_size={point_size}__rm_outlier={remove_outliers}.png"
    o3d.io.write_image(out_path, img)
    print(f"Saved headless render to {out_path}")