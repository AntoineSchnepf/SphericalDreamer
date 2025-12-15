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


config = my_utils.fetch_config_via_parser(
    debug=False, 
    debug_parser_override=["--config", "Karim/forest.yaml"]
)

# -------- Import pointcloud --------
# repo_path = os.path.dirname(os.path.realpath(__file__))
# t0 = time.time()
# with open(f"{repo_path}/{config.save_dir}/{config.expname}/02b_raw_dream_pcd.pkl", "rb") as f:
#     pcd = pickle.load(f).get_o3d_pointcloud()
# printc(f"Loaded raw point cloud in {time.time() - t0:.2f} seconds.", color='yellow')
# max_x = (config.num_dreams-1) * config.sphere_radius * config.delta_walk
# printc(f"max_X: {max_x}", color='red')

repo_path = os.path.dirname(os.path.realpath(__file__))
t0 = time.time()
with open(f"{repo_path}/{config.save_dir}/{config.expname}/02c_raw_dream_pcd.pkl", "rb") as f:
    PointCloud_instance = pickle.load(f)

printc(f"Loaded raw point cloud in {time.time() - t0:.2f} seconds.", color='yellow')

t0 = time.time()
pts = PointCloud_instance.pts
colors = PointCloud_instance.colors
skip = 10
pts = pts[::skip]
colors = colors[::skip]
PointCloud_instance = PointCloud(pts=pts, colors=colors)
pcd = PointCloud_instance.get_o3d_pointcloud()

printc(f"Downsampled point cloud (skip = {skip}) to {len(pcd.points)} points in {time.time() - t0:.2f} seconds.", color='yellow')

max_x = (config.num_dreams-1) * config.sphere_radius * config.delta_walk
printc(f"max_X: {max_x}", color='red')


# -------- Fix world geometry --------
if config.phase4.fix_world:
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
        **config.phase4.world
    )

    final_pcd = my_utils.PointCloud(pts_corrected, colors).get_o3d_pointcloud()
else:
    final_pcd = pcd

# -------- Remove outliers --------
if config.phase4.remove_outliers.apply:
    pcd_filtered, ind = final_pcd.remove_statistical_outlier(nb_neighbors=config.phase4.remove_outliers.nb_neighbors, 
                                                            std_ratio=config.phase4.remove_outliers.std_ratio) 

    printc(f"Filtered point cloud: {len(pcd_filtered.points)} points", color='yellow')
    printc(f"Removed {len(final_pcd.points) - len(pcd_filtered.points)} outliers", color='yellow')
else:
    pcd_filtered = final_pcd
    ind = np.arange(len(final_pcd.points))

if not np.isfinite(pcd_filtered.points).all():
    printc("WARNING: Point cloud contains NaN or infinite values", color='red')


# -------- Headless rendering (Offscreen) --------

# Optionally visualize removed points
if config.phase4.visualize_removed_points:
    out_idx = np.setdiff1d(np.arange(np.asarray(final_pcd.points).shape[0]), np.asarray(ind))
    pcd_removed = o3d.geometry.PointCloud()
    pcd_removed.points = o3d.utility.Vector3dVector(np.asarray(final_pcd.points)[out_idx])
    pcd_removed.paint_uniform_color([0.6, 0.6, 0.6])
    if not np.isfinite(pcd_removed.points).all():
        printc("WARNING: Removed point cloud contains NaN or infinite values", color='red')


# Create an offscreen renderer (no window)
width, height = config.phase4.width, config.phase4.height
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background(config.phase4.bg_color) 


# Disabling color processing
view = scene.view  
view.set_post_processing(False)
# Linear tone mapping
cg = o3d.visualization.rendering.ColorGrading(
    o3d.visualization.rendering.ColorGrading.Quality.HIGH,
    o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR,
)
view.set_color_grading(cg)

# Basic unlit material for points
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = config.phase4.point_size 

# Add geometries
scene.add_geometry("filtered", pcd_filtered, mat)
if config.phase4.visualize_removed_points:
    scene.add_geometry("removed", pcd_removed, mat) 

# -------- Camera control --------
# Compute a basic bounding box and radius for scaling camera parameters
bbox = pcd_filtered.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
near = 0.01 * np.linalg.norm(extent)
far = 10.0 * np.linalg.norm(extent)
fov_deg = 60.0

if config.phase4.render_settings.trajectory == 'custom':
    camera_keypoints = config.phase4.render_settings.custom_trajectory
else:
    camera_keypoints = my_utils.get_template_tranjectories(config.phase4.render_settings.trajectory)

min_x = config.phase4.render_settings.ranges.min_x
min_y, max_y = config.phase4.render_settings.ranges.min_y, config.phase4.render_settings.ranges.max_y
mid_y = (min_y + max_y) / 2
min_z, max_z = config.phase4.render_settings.ranges.min_z, config.phase4.render_settings.ranges.max_z
mid_z = config.phase4.render_settings.ranges.mid_z
fpm = config.phase4.render_settings.framerate.fpm
fpd_e = config.phase4.render_settings.framerate.fpd_e
fpd_a = config.phase4.render_settings.framerate.fpd_a

all_images = []
all_cameras = my_utils.interpolate_camera_keypoints(camera_keypoints, fpm, fpd_e, fpd_a, max_x)
pbar = tqdm(total=len(all_cameras), desc="Rendering frames")
printc(f"Rendering frames with {config.phase4.render_settings.trajectory} trajectory...", color='yellow')
for i, camera_pos in enumerate(all_cameras):
    cam = scene.camera
    x_percent, y_percent, z_percent = camera_pos[0], camera_pos[1], camera_pos[2]
    elev_deg, azim_deg = camera_pos[3], camera_pos[4]

    x_pos = min_x + x_percent * (max_x - min_x)
    y_pos = mid_y + y_percent * (max_y - min_y)
    z_pos = mid_z + z_percent * (max_z - min_z)
    cam_pos = np.array([x_pos, y_pos, z_pos])

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

    # Render to image and save
    where_save = os.path.join("OUTPUTS", "test_o3d_rendering", config.expname)
    os.makedirs(where_save, exist_ok=True)
    img = renderer.render_to_image()
    all_images.append(img)
    out_path = f"{where_save}/x_percent={x_percent:.2f}__azim={azim_deg:.1f}__point_size={config.phase4.point_size}__rm_outlier={config.phase4.remove_outliers.apply}.png"
    pbar.update(1)
pbar.close()

output_dir = os.path.join(config.save_dir, config.expname, "rendered_trajectories")
os.makedirs(output_dir, exist_ok=True)
my_utils.save_video_from_o3d_images(
    all_images,
    os.path.join(output_dir, f"{config.phase4.render_settings.trajectory}.mp4"),
    fps=config.phase4.render_settings.framerate.fps
)