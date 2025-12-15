import os
import numpy as np
import open3d as o3d
import pickle
import my_utils
import numpy as np
import open3d as o3d
from my_utils import PointCloud
from my_utils import set_camera_from_elev_azim, printc
from tqdm import tqdm
os.environ.setdefault("OPEN3D_HEADLESS", "1") # for open3d headless rendering


config = my_utils.fetch_config_via_parser(
    debug=False, 
    debug_parser_override=["--config", "forest.yaml"]
)

# -------- Import pointcloud --------
repo_path = os.path.dirname(os.path.realpath(__file__))
with open(f"{repo_path}/{config.save_dir}/{config.expname}/02b_raw_dream_pcd.pkl", "rb") as f:
    pcd = pickle.load(f).get_o3d_pointcloud()
max_x = (config.num_dreams-1) * config.sphere_radius * config.delta_walk


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

# -------- Headless rendering (Offscreen) --------

# Optionally visualize removed points
if config.phase4.visualize_removed_points:
    out_idx = np.setdiff1d(np.arange(np.asarray(final_pcd.points).shape[0]), np.asarray(ind))
    pcd_removed = o3d.geometry.PointCloud()
    pcd_removed.points = o3d.utility.Vector3dVector(np.asarray(final_pcd.points)[out_idx])
    pcd_removed.paint_uniform_color([0.6, 0.6, 0.6])

# Create an offscreen renderer (no window)
width, height = config.phase4.width, config.phase4.height
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background(config.phase4.bg_color) 

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

all_images = []
all_steps = [0, 1] # Independent
num_translate = 400 # Independent
num_rotate = 900 # Independent
pbar = tqdm(total=len(all_steps)*(num_translate+num_rotate) - num_translate, desc="Rendering frames")
for i, x_step in enumerate(all_steps):
    if i != 0:
        for x_percent in np.linspace(all_steps[i-1], all_steps[i], num=num_translate):
            cam = scene.camera
            # x_percent = 0.5
            min_x = 0.0

            x_pos = min_x + x_percent * (max_x - min_x)
            cam_pos = np.array([x_pos, 0.0, 0.0])  # will be set later
            elev_deg = 0.0
            fov_deg = 60.0
            azim_deg = 0.0

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

    for azim_deg in np.linspace(0, 360, num=num_rotate):
        cam = scene.camera
        x_percent = all_steps[i]
        min_x = 0.0

        x_pos = min_x + x_percent * (max_x - min_x)
        cam_pos = np.array([x_pos, 0.0, 0.0])  # will be set later
        elev_deg = 0.0
        fov_deg = 60.0

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
        where_save = os.path.join("OUTPUTS", "test_o3d_rendering", config.expname)
        os.makedirs(where_save, exist_ok=True)
        img = renderer.render_to_image()
        all_images.append(img)
        out_path = f"{where_save}/azim={azim_deg:.1f}__point_size={config.phase4.point_size}__rm_outlier={config.phase4.remove_outliers.apply}.png"
        pbar.update(1)
pbar.close()

my_utils.save_video_from_o3d_images(
    all_images,
    os.path.join("OUTPUTS", "test_o3d_rendering", config.expname, f"test.mp4"),
    fps=90
)