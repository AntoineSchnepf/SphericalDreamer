import os, sys
os.environ.setdefault("OPEN3D_HEADLESS", "1") # for open3d headless rendering
import numpy as np
import open3d as o3d
import pickle
import json
import my_utils
import numpy as np
import open3d as o3d
import time
from my_utils import PointCloud
from my_utils import set_camera_from_elev_azim, printc
from tqdm import tqdm

_phase_3 = "3"
_phase_current = "5"

config = my_utils.fetch_config_via_parser(
        debug=False, 
        debug_parser_override=["--config", "Antoine/debug.yaml"]
)

seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)

repo_path = os.path.dirname(os.path.realpath(__file__))

# -------------------------------- #
# ---- PHASE 5 RENDER NFS  ----- #
# -------------------------------- #
printc(f"=== [PHASE {_phase_current}]  EXPERIMENT: {config.expname} ===", color='cyan')
printc(f"=== PHASE {_phase_current} : RENDER NFS ===", color='green')

# -------- Import pointcloud --------
t0 = time.time()
with open(save_dir_ /f"{_phase_3}_final_dream_pcd.pkl", "rb") as f:
    PointCloud_instance = pickle.load(f)
pcd = PointCloud_instance.get_o3d_pointcloud()
printc(f"--- {_phase_current}: Loaded final point cloud in {time.time() - t0:.2f} seconds.", color='yellow')


max_x = (config.num_dreams-1) * config.sphere_radius * config.delta_walk
printc(f"max_X: {max_x}", color='red')

if not np.isfinite(pcd.points).all():
    printc("WARNING: Point cloud contains NaN or infinite values", color='red')


# -------- Headless rendering (Offscreen) --------


# Save pcd as .ply
where_save_pcd = os.path.join(config.save_dir, config.expname, "nfs_dataset", "pointcloud")
os.makedirs(where_save_pcd, exist_ok=True)
o3d.io.write_point_cloud(f"{where_save_pcd}/05_pcd_filtered.ply", pcd)
printc(f"Saved filtered point cloud to {where_save_pcd}/05_pcd_filtered.ply", color='yellow')

# Create an offscreen renderer (no window)
width, height = config.phase5.width, config.phase5.height
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background(config.phase5.bg_color) 


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
mat.point_size = config.phase5.point_size 

# Add geometries
scene.add_geometry("filtered", pcd, mat, add_downsampled_copy_for_fast_rendering=False)

# -------- Camera control --------
# Compute a basic bounding box and radius for scaling camera parameters
bbox = pcd.get_axis_aligned_bounding_box()
center = bbox.get_center()
extent = bbox.get_extent()
near = 0.01 * np.linalg.norm(extent)
far = 10.0 * np.linalg.norm(extent)
fov_deg = 60.0

printc(f"near: {near}, far: {far}, center: {center}, extent: {extent}", color='red')

min_x = config.phase5.render_settings.ranges.min_x
min_y, max_y = config.phase5.render_settings.ranges.min_y, config.phase5.render_settings.ranges.max_y
min_z, max_z = config.phase5.render_settings.ranges.min_z, config.phase5.render_settings.ranges.max_z

all_cameras = my_utils.sample_cameras(
    min_x=min_x, 
    max_x=max_x, 
    min_y=min_y, 
    max_y=max_y, 
    min_z=min_z, 
    max_z=max_z, 
    nb_points=config.phase5.render_settings.nb_points, 
    nb_samples_per_point=config.phase5.render_settings.nb_samples_per_point, 
    seed=config.seed
)

pbar = tqdm(total=len(all_cameras), desc="Rendering frames")
printc(f"Rendering frames for Nerfstudio...", color='yellow')
transforms = {
    "camera_model": "OPENCV",
    "ply_file_path": "pointcloud/05_pcd_filtered.ply",
    "frames": [],
}
for i, camera_pos in enumerate(all_cameras):
    cam = scene.camera
    cam_pos = np.array([camera_pos[0], camera_pos[1], camera_pos[2]])
    elev_deg, azim_deg = camera_pos[3], camera_pos[4]

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
    output_dir = os.path.join(config.save_dir, config.expname, "nfs_dataset")
    output_dir_ = os.path.join(output_dir, "rgb")
    os.makedirs(output_dir_, exist_ok=True)
    img = renderer.render_to_image()
    out_path = f"{i:04d}.png"
    o3d.io.write_image(os.path.join(output_dir_, out_path), img)

    frame = my_utils.get_nerfstudio_frame(
        cam_pos=cam_pos,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        width=width,
        height=height,
        fov_deg=fov_deg,
        file_path=os.path.join("rgb", out_path)
    )
    transforms['frames'].append(frame)
    pbar.update(1)
pbar.close()

# save transforms json file
with open(os.path.join(output_dir, "transforms.json"), "w") as f:
    json.dump(transforms, f, indent=4)


printc(f"Rendered {len(all_cameras)} frames in {time.time() - t0:.2f} seconds.", color='yellow')
printc(f"Saved NFS dataset to {output_dir}", color='yellow')