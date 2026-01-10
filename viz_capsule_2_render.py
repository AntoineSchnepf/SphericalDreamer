"""
Blender script to render a point cloud from inside using Cycles.
Usage: blender --background --python viz_capsule_2_render.py

To render in series, use (best option so far):
blender --background --python viz_capsule_2_render.py -- --config Antoine/forest_v3.yaml

(debugging) For manual parallel rendering, use --frame-start and --frame-end:
blender --background --python viz_capsule_2_render.py -- --config Karim/forest_v3.yaml --frame-start 0 --frame-end 100

For parallel launcher (parallel rendering always renders on CPU):
python viz_capsule_2_render.py --parallel --num-workers 8 --config Karim/forest_v3.yaml
"""

import math
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
# Add script directory to path BEFORE importing local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Check if running in Blender
try:
    import bpy
    from mathutils import Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    Vector = None  # Will not be used outside Blender

### --- ANTOINE STUFF ---  ###
fov_deg = None
def get_ply_path(expname, which_ply):
    ply_fname = f"{expname}__key={which_ply}.ply"
    return ply_files_save_dir / ply_fname


ply_files_save_dir = Path("/home/a.schnepf/phd/SphericalDreamer/viz_ply_pointclouds")

# which_ply = "sphere1_closed"
# which_ply = "sphere1_right_opened"
# which_ply = "filled_capsule"
# which_ply = "filled_capsule_colored"
# which_ply = "Forest"
# which_ply = "Forest_full_pcd_main_fig"
# which_ply = "Forest_partial_pcd_appendix_fig"
# which_ply = "single_sphere_no_ldi"
which_ply = "single_sphere_ldi"

max_x =  1.57
max_x_forest = 1.57 * (5-1)
all_cameras_per_sphere = {
    # X, Y, Z, ELEV_DEG, AZIM_DEG
    "sphere1_closed": [
        (1.5, -3, 1, -15, 90 + 20),
        (1.5, -3, 1, -15, 90 + 20),
        (2.5, -3, 1, -15, 90 + 30),
        (-2.5, -3, 1, -15, 90 - 30),
    ],
    "sphere1_right_opened": [
        (1.5, -3, 1, -15, 90 + 20),
        (2.5, -3, 1, -15, 90 + 30),
    ],
    "sphere1_left_opened": [
        (-1.5, -3, 1, -15, 90 - 20),
        (-2.5, -3, 1, -15, 90 - 30),
    ],
    "sphere1_both_opened": [
        (1.5, -3, 1, -15, 90 + 20),
        (-1.5, -3, 1, -15, 90 - 20),
        (2.5, -3, 1, -15, 90 + 30),
        (-2.5, -3, 1, -15, 90 - 30),
        (-3, 0.0, 1, -15, 90 - 90),
        (3, 0.0, 1, -15, 180),
    ],
    "hollow_capsule": [ 
        (max_x/2, -4, 1, -15, 90),
    ],
    "filled_capsule": [ 
        (max_x/2, -4, 1, -15, 90),
    ],
    "filled_capsule_colored": [ 
        (max_x/2, -4, 1, -15, 90),
    ],
    "Forest": [
        (max_x_forest/2, -8, 2, -15, 90),
    ],
    "Forest_full_pcd_main_fig": [
        (max_x_forest/2, -8, 2, -15, 90),
    ],
    "Forest_partial_pcd_appendix_fig": [
        (max_x_forest/2, -8, 2, -15, 90),
    ],
    "single_sphere_no_ldi": [
        (max_x, 0, 0.0, 0, 180),
        (max_x, 0.1, 0.0, 0, 180),
        (max_x, 0.2, 0.0, 0, 180),
        (max_x, 0.3, 0.0, 0, 180),
        (max_x, -0.1, 0.0, 0, 180),
        (max_x, -0.2, 0.0, 0, 180),
        (max_x, -0.3, 0.0, 0, 180),

    ],
    "single_sphere_ldi": [
        (max_x, 0, 0.0, 0, 180),
        (max_x, 0.1, 0.0, 0, 180),
        (max_x, 0.2, 0.0, 0, 180),
        (max_x, 0.3, 0.0, 0, 180),
        (max_x, -0.1, 0.0, 0, 180),
        (max_x, -0.2, 0.0, 0, 180),
        (max_x, -0.3, 0.0, 0, 180),
    ],


}
all_cameras = all_cameras_per_sphere[which_ply]
height_ = 1024
width_ = 1024
# which_ply = "Forest_full_pcd_main_fig"
if which_ply == "Forest_partial_pcd_appendix_fig" or which_ply == "Forest_full_pcd_main_fig":
    height_ = 1048
    width_ = 2048


if which_ply == "single_sphere_no_ldi" or which_ply == "single_sphere_ldi":
    fov_deg =  120.0

if IN_BLENDER:
    from blender_utils import (
        setup_render_handlers, cleanup_render_handlers, clear_scene, import_ply,
        export_ply_filtered,
        ensure_camera, set_world_black, bbox_center_and_extent_world, set_camera_like_open3d,
        add_debug_test_sphere, compute_voxel_downsampling, make_point_cloud_geometry_nodes,
        setup_gpu_rendering,
        printc, fetch_config_via_parser, setup,
        sample_cameras, get_nerfstudio_frame
    )

_phase_1a = "1a"
_phase_1b = "1b"
_phase_2a = "2a"
_phase_2b = "2b"
_phase_2c = "2c"
_phase_3 = "3"
_phase_4 = "4"


# -------------------------
# Parse frame range arguments (for parallel rendering)
# -------------------------
def parse_frame_range():
    """Parse --frame-start, --frame-end, and --worker-id from command line arguments."""
    frame_start = None
    frame_end = None
    worker_id = None
    
    # Find arguments after '--' (Blender passes script args after --)
    argv = sys.argv
    if '--' in argv:
        script_args = argv[argv.index('--') + 1:]
    else:
        script_args = []
    
    for i, arg in enumerate(script_args):
        if arg == '--frame-start' and i + 1 < len(script_args):
            frame_start = int(script_args[i + 1])
        elif arg == '--frame-end' and i + 1 < len(script_args):
            frame_end = int(script_args[i + 1])
        elif arg == '--worker-id' and i + 1 < len(script_args):
            worker_id = int(script_args[i + 1])
    
    return frame_start, frame_end, worker_id

# Only load config when running in Blender
if IN_BLENDER:
    FRAME_START, FRAME_END, WORKER_ID = parse_frame_range()
    
    # -------------------------
    # Config
    # -------------------------
    # Filter out parallel-specific args before calling fetch_config_via_parser
    # (it would try to interpret --frame-start, --frame-end, --worker-id as config overrides)
    original_argv = sys.argv.copy()
    filtered_argv = [sys.argv[0]]
    if '--' in sys.argv:
        script_args = sys.argv[sys.argv.index('--') + 1:]
        skip_next = False
        for i, arg in enumerate(script_args):
            if skip_next:
                skip_next = False
                continue
            if arg in ('--frame-start', '--frame-end', '--worker-id'):
                skip_next = True  # Skip this arg and its value
            else:
                filtered_argv.append(arg)
    sys.argv = ['blender', '--'] + filtered_argv[1:] if len(filtered_argv) > 1 else ['blender', '--']
    
    try:
        config = fetch_config_via_parser(debug=False, debug_parser_override=["--config", "exp0/0_caverns.yaml"])
    finally:
        sys.argv = original_argv
    
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = setup(config)
    
    cfg_render = config.phase5v2.render_settings
    cfg_render.height = height_ #TAG
    cfg_render.width = width_ # TAG
    cfg_adv = config.phase5v2.advanced_render_settings
    cfg_nfs = config.phase5v2.nfs_dataset
    
    # render_settings
    POINT_RADIUS, KEEP_RATIO, VOXEL_SIZE = cfg_render.point_size, cfg_render.keep_ratio, cfg_render.voxel_size
    # Use series GPU setting by default; parallel mode will override via command line
    USE_GPU = cfg_render.use_gpu_series
    MIN_NEAR_PLANE, MAX_FAR_NEAR_RATIO = cfg_render.min_near_plane, cfg_render.max_far_near_ratio
    RES_X, RES_Y = cfg_render.width, cfg_render.height
    
    # advanced_render_settings
    USE_FAST_POINTS, USE_DISTANCE_BASED_RADIUS = cfg_adv.use_fast_points, cfg_adv.use_distance_based_point_size
    USE_BACKFACE_CULLING, RENDER_SAMPLES = cfg_adv.use_backface_culling, cfg_adv.render_samples
    MAX_RENDER_DISTANCE = cfg_adv.max_render_distance
    USE_DISTANCE_CULLING = cfg_adv.distance_culling.apply
    CULLING_START_DISTANCE, CULLING_END_DISTANCE = cfg_adv.distance_culling.min_distance, cfg_adv.distance_culling.max_distance
    
    # nfs_dataset settings
    if fov_deg is None:
        FOV_DEG = cfg_nfs.fov_deg
    else:
        FOV_DEG = fov_deg
    NFS_NB_POINTS = cfg_nfs.nb_points
    NFS_NB_SAMPLES_PER_POINT = cfg_nfs.nb_samples_per_point
    NFS_RANGES = cfg_nfs.ranges
    
    # Paths
    if which_ply == "Forest":
        PLY_PATH = "/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/Forest/3_final_dream_pcd_unfiltered.ply"
    else:
        PLY_PATH = get_ply_path(config.expname, which_ply)
    PLY_PATH = str(PLY_PATH)


def main():
    start_time = time.time()
    print("=" * 60 + "\nBlender Point Cloud Renderer - NFS Dataset\n" + "=" * 60)
    
    
    clear_scene()
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import point cloud
    print(f"Importing PLY: {PLY_PATH}")
    pc_obj = import_ply(PLY_PATH)
    if pc_obj is None:
        print("[ERROR] Failed to import PLY file!")
        return
    print(f"Imported: {pc_obj.name} ({len(pc_obj.data.vertices)} vertices)")
    
    # Find color attribute
    color_attr_name = None
    if hasattr(pc_obj.data, "color_attributes") and pc_obj.data.color_attributes:
        color_attr_name = pc_obj.data.color_attributes[0].name
    
    # Compute bounding box
    center, extent = bbox_center_and_extent_world(pc_obj)
    bbox_radius = 0.5 * extent.length
    print(f"BBox center: ({center.x:.3f}, {center.y:.3f}, {center.z:.3f}), radius: {bbox_radius:.3f}")
    
    # Voxel downsampling
    voxel_attr_name = None
    if KEEP_RATIO < 1.0:
        voxel_attr_name, _ = compute_voxel_downsampling(pc_obj, voxel_size=VOXEL_SIZE, target_ratio=KEEP_RATIO)
    
    # Compute clipping planes
    near = MIN_NEAR_PLANE
    max_far = near * MAX_FAR_NEAR_RATIO
    desired_far = bbox_radius * 10.0
    far = min(desired_far, max_far)
    print(f"Clipping: near={near:.4f}, far={far:.4f}")
    
    # Sample cameras (same as 5_render_nfs.py)
    total_cameras = len(all_cameras)
    print(f"Sampled {total_cameras} camera positions")
    
    # Determine frame range for this worker
    frame_start = FRAME_START if FRAME_START is not None else 0
    frame_end = FRAME_END if FRAME_END is not None else total_cameras
    frame_end = min(frame_end, total_cameras)
    is_partial_render = FRAME_START is not None or FRAME_END is not None
    
    if is_partial_render:
        print(f"[Worker {WORKER_ID}] Rendering frames {frame_start} to {frame_end} (of {total_cameras} total)")
    
    # Setup output directories
    output_dir = os.path.join("/home/a.schnepf/phd/SphericalDreamer/Figures/viz_paper_capsule", which_ply)
    output_dir_rgb = os.path.join(output_dir, "rgb")
    output_dir_mask = os.path.join(output_dir, "mask")
    output_dir_pcd = os.path.join(output_dir, "pointcloud")
    os.makedirs(output_dir_rgb, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_pcd, exist_ok=True)
    
    # Export processed point cloud (only if not a partial render or first worker)
    exported_ply_name = "5v2_pcd_filtered.ply"
    ply_dest = os.path.join(output_dir_pcd, exported_ply_name)
    if not is_partial_render or WORKER_ID == 0:
        if KEEP_RATIO >= 1.0:
            # No filtering needed, just copy the original PLY
            import shutil
            shutil.copy2(PLY_PATH, ply_dest)
            print(f"Copied original PLY to: {ply_dest}")
        else:
            export_ply_filtered(pc_obj, ply_dest, voxel_attr_name=voxel_attr_name)
            print(f"Exported processed PLY to: {ply_dest}")
    
    # Initialize transforms dict
    transforms = {
        "camera_model": "OPENCV",
        "ply_file_path": f"pointcloud/{exported_ply_name}",
        "frames": [],
    }
    
    # Configure render settings (once, before the loop)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = False
    scene.render.film_transparent = True  # Enable transparent background for mask extraction
    set_world_black()
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0
    
    # Enable RGBA output for mask extraction
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Cycles settings
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces = scene.cycles.diffuse_bounces = scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = scene.cycles.volume_bounces = scene.cycles.transparent_max_bounces = 0
    scene.cycles.use_fast_gi = scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    
    # GPU/CPU setup
    if USE_GPU:
        setup_gpu_rendering(scene)
    else:
        scene.cycles.device = 'CPU'
        print("  Using CPU rendering")
    
    # Setup camera
    cam = ensure_camera("Cam")
    bpy.context.scene.camera = cam
    cam.data.dof.use_dof = False
    
    # Render loop (only render frames in our range)
    frames_to_render = list(range(frame_start, frame_end))
    print(f"Rendering {len(frames_to_render)} frames for Nerfstudio...")
    for idx, i in enumerate(frames_to_render):
        camera_pos = all_cameras[i]
        cam_x, cam_y, cam_z, elev_deg, azim_deg = camera_pos
        cam_pos_vec = Vector((cam_x, cam_y, cam_z))
        dist = (cam_pos_vec - center).length
        
        # Create geometry nodes (needs to be updated per camera for distance-based effects)
        make_point_cloud_geometry_nodes(
            pc_obj, POINT_RADIUS, keep_ratio=KEEP_RATIO,
            color_attr_name=color_attr_name or "Col", use_voxel_attr=voxel_attr_name,
            use_fast_points=USE_FAST_POINTS, use_distance_based_radius=USE_DISTANCE_BASED_RADIUS,
            use_distance_culling=USE_DISTANCE_CULLING, use_backface_culling=USE_BACKFACE_CULLING,
            camera_position=(cam_x, cam_y, cam_z),
            base_distance=max(0.1, dist) if dist > 0.01 else bbox_radius * 0.5,
            culling_start_distance=CULLING_START_DISTANCE, culling_end_distance=CULLING_END_DISTANCE,
            max_render_distance=MAX_RENDER_DISTANCE,
        )
        
        # Set camera
        set_camera_like_open3d(cam, (cam_x, cam_y, cam_z), elev_deg, azim_deg, FOV_DEG, RES_X, RES_Y, near, far)
        
        # Force scene update (required for geometry nodes and camera changes to take effect)
        bpy.context.view_layer.update()
        
        # Render to image
        out_filename = f"{i:04d}.png"
        scene.render.filepath = os.path.join(output_dir_rgb, out_filename)
        
        if idx == 0:
            setup_render_handlers()
        
        print(f"  Rendering frame {idx+1}/{len(frames_to_render)} (global: {i}): pos=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), elev={elev_deg:.1f}, azim={azim_deg:.1f}")
        sys.stdout.flush()
        bpy.ops.render.render(write_still=True)
        
        # Save mask (areas with no points = black (ignore), areas with points = white (keep))
        # The alpha channel from transparent render: 0 = no geometry, 255 = geometry
        # We keep it as-is: black (0) = background/no points (ignore), white (255) = has points (keep)
        mask_filename = f"{i:04d}.png"
        mask_filepath = os.path.join(output_dir_mask, mask_filename)
        
        # Get the rendered image from Blender's image buffer
        render_result = bpy.data.images.get('Render Result')
        if render_result is not None:
            # Save render result to a temporary location, then extract mask
            temp_rgba_path = os.path.join(output_dir_rgb, f"_temp_{i:04d}.png")
            render_result.save_render(filepath=temp_rgba_path)
            
            # Load the RGBA image, extract alpha, and save as mask
            import cv2
            rgba_img = cv2.imread(temp_rgba_path, cv2.IMREAD_UNCHANGED)
            if rgba_img is not None and rgba_img.shape[2] == 4:
                alpha_channel = rgba_img[:, :, 3]
                # Alpha is already: 0 (transparent/no geometry) = black, 255 (opaque/geometry) = white
                # Use low threshold (1) so only truly empty pixels are marked as holes
                # This avoids anti-aliased edge pixels being incorrectly marked as background
                _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(mask_filepath, mask)
            
            # Also save the RGB version (without alpha) as the final output
            if rgba_img is not None:
                rgb_img = rgba_img[:, :, :3]
                cv2.imwrite(scene.render.filepath, rgb_img)
            
            # Remove temporary file
            if os.path.exists(temp_rgba_path):
                os.remove(temp_rgba_path)
        
        # Add frame to transforms
        frame = get_nerfstudio_frame(
            cam_pos=np.array([cam_x, cam_y, cam_z]),
            elev_deg=elev_deg,
            azim_deg=azim_deg,
            width=RES_X,
            height=RES_Y,
            fov_deg=FOV_DEG,
            file_path=os.path.join("rgb", out_filename)
        )
        frame["mask_path"] = os.path.join("mask", mask_filename)
        transforms['frames'].append(frame)
    
    cleanup_render_handlers()
    
    # Save transforms.json (partial or full)
    if is_partial_render:
        # Save partial transforms for this worker
        transforms_path = os.path.join(output_dir, f"transforms_worker_{WORKER_ID}.json")
    else:
        transforms_path = os.path.join(output_dir, "transforms.json")
    
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)
    
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"Rendered {len(frames_to_render)} frames")
    print(f"Saved NFS dataset to: {output_dir}")
    print(f"Saved transforms to: {transforms_path}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 60)


def merge_transforms(output_dir, num_workers):
    """Merge partial transforms.json files from multiple workers."""
    merged = {
        "camera_model": "OPENCV",
        "ply_file_path": f"pointcloud/{ply_name}",
        "frames": [],
    }
    
    for worker_id in range(num_workers):
        partial_path = os.path.join(output_dir, f"transforms_worker_{worker_id}.json")
        if os.path.exists(partial_path):
            with open(partial_path, "r") as f:
                partial = json.load(f)
                merged["frames"].extend(partial["frames"])
            # Clean up partial file
            os.remove(partial_path)
    
    # Sort frames by file path to ensure correct order
    merged["frames"].sort(key=lambda x: x["file_path"])
    
    # Save merged transforms
    final_path = os.path.join(output_dir, "transforms.json")
    with open(final_path, "w") as f:
        json.dump(merged, f, indent=4)
    
    print(f"Merged {len(merged['frames'])} frames into {final_path}")
    return final_path


def run_parallel(num_workers, config_path):
    """
    Launch multiple Blender processes in parallel to render the dataset.
    This function is called when running the script directly with Python.
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    start_time = time.time()
    
    # We need to compute total frames without Blender
    # Import config parsing utilities
    sys.path.insert(0, script_dir)
    from my_utils import fetch_config_via_parser as my_fetch_config
    from my_utils import setup as my_setup
    
    # Temporarily modify sys.argv to only include config-related args
    # (my_fetch_config parses sys.argv and would fail on --parallel, --num-workers, etc.)
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], "--config", config_path]
    
    try:
        # Parse config to get total frame count
        config = my_fetch_config(debug=False, debug_parser_override=["--config", config_path])
        seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_setup(config)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    cfg_nfs = config.phase5v2.nfs_dataset
    total_frames = cfg_nfs.nb_points * cfg_nfs.nb_samples_per_point
    
    output_dir = os.path.join(config.save_dir, config.expname, "viz_paper_capsule")
    
    # Check GPU settings for parallel mode
    use_gpu_parallel = config.phase5v2.render_settings.use_gpu_parallel
    if use_gpu_parallel:
        print("\033[93m" + "=" * 60)
        print("WARNING: GPU rendering enabled in parallel mode.")
        print("Multiple Blender instances sharing the same GPU may be slower.")
        print("Consider setting use_gpu_parallel: False in config for better performance.")
        print("=" * 60 + "\033[0m")
    
    gpu_mode = "GPU" if use_gpu_parallel else "CPU"
    print("=" * 60)
    print(f"Parallel Blender Rendering ({gpu_mode} mode)")
    print(f"  Total frames: {total_frames}")
    print(f"  Workers: {num_workers}")
    print(f"  Config: {config_path}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # Calculate frame ranges for each worker
    frames_per_worker = total_frames // num_workers
    worker_ranges = []
    for i in range(num_workers):
        start = i * frames_per_worker
        end = (i + 1) * frames_per_worker if i < num_workers - 1 else total_frames
        worker_ranges.append((i, start, end))
    
    # Build Blender commands (use use_gpu_parallel setting)
    script_path = os.path.abspath(__file__)
    gpu_setting = "True" if use_gpu_parallel else "False"
    commands = []
    for worker_id, start, end in worker_ranges:
        cmd = [
            "blender", "--background", "--python", script_path,
            "--", "--config", config_path,
            "--phase5v2.render_settings.use_gpu_series", gpu_setting,  # Override with parallel setting
            "--frame-start", str(start),
            "--frame-end", str(end),
            "--worker-id", str(worker_id)
        ]
        commands.append((worker_id, cmd))
    
    # Run workers in parallel using ThreadPoolExecutor (works better for subprocess calls)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def run_worker(args):
        worker_id, cmd = args
        print(f"[Worker {worker_id}] Starting: frames {worker_ranges[worker_id][1]}-{worker_ranges[worker_id][2]}")
        result = subprocess.run(cmd, capture_output=False)
        return worker_id, result.returncode
    
    print(f"\nLaunching {num_workers} Blender processes...")
    
    # Use ThreadPoolExecutor for parallel execution (better for subprocess calls)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_worker, cmd): cmd[0] for cmd in commands}
        for future in as_completed(futures):
            worker_id, returncode = future.result()
            if returncode == 0:
                print(f"[Worker {worker_id}] Completed successfully")
            else:
                print(f"[Worker {worker_id}] Failed with code {returncode}")
    
    # Merge transforms
    print("\nMerging transforms from all workers...")
    merge_transforms(output_dir, num_workers)
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Parallel rendering complete!")
    print(f"Output: {output_dir}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    if IN_BLENDER:
        main()
    else:
        # Running directly with Python - check for parallel mode
        import argparse
        parser = argparse.ArgumentParser(description="Parallel Blender rendering launcher")
        parser.add_argument("--parallel", action="store_true", help="Run in parallel mode")
        parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
        parser.add_argument("--config", type=str, required=True, help="Config file path")
        parser.add_argument("--merge-only", action="store_true", help="Only merge existing partial transforms")
        args = parser.parse_args()
        
        if args.merge_only:
            from my_utils import fetch_config_via_parser as my_fetch_config
            from my_utils import setup as my_setup
            # Filter sys.argv to only pass --config to the config parser
            original_argv = sys.argv.copy()
            sys.argv = [sys.argv[0], "--config", args.config]
            try:
                config = my_fetch_config(debug=False, debug_parser_override=["--config", args.config])
            finally:
                sys.argv = original_argv
            output_dir = os.path.join(config.save_dir, config.expname, "viz_paper_capsule")
            merge_transforms(output_dir, args.num_workers)
        elif args.parallel:
            run_parallel(args.num_workers, args.config)
        else:
            print("Use --parallel flag to run in parallel mode, or run via Blender:")
            print("  blender --background --python 5v2_render_blender.py -- --config <config.yaml>")
elif IN_BLENDER:
    # Running as a module inside Blender
    main()
