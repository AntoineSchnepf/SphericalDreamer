"""
Blender script to render a point cloud from inside using Cycles.
Usage: blender --background --python viz_blender_ICML_RENDERS.py

To render in series, use (best option so far):
blender --background --python viz_blender_ICML_RENDERS.py -- --config Karim/forest_v3.yaml

To render with a custom world file (PLY or OBJ):
blender --background --python viz_blender_ICML_RENDERS.py --custom-world my_custom_world.ply -- --config Karim/forest_v3.yaml
blender --background --python viz_blender_ICML_RENDERS.py --custom-world my_custom_world.obj -- --config Karim/forest_v3.yaml

(debugging) For manual parallel rendering, use --frame-start and --frame-end:
blender --background --python viz_blender_ICML_RENDERS.py -- --config Karim/forest_v3.yaml --frame-start 0 --frame-end 100

For parallel launcher (parallel rendering always renders on CPU):
python viz_blender_ICML_RENDERS.py --parallel --num-workers 8 --config Karim/forest_v3.yaml
"""

from email.mime import base
import math
import sys
import os
import json
import time
import numpy as np
DEBUG = False
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

if IN_BLENDER:
    from blender_utils import (
        setup_render_handlers, cleanup_render_handlers, clear_scene, import_ply, import_obj,
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



def get_compositor_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    # Ensure compositing is enabled in the scene
    scene.render.use_compositing = True

    # Preferred (when available): scene.node_tree
    # Note: scene.use_nodes is deprecated but still needed in many versions
    try:
        scene.use_nodes = True
    except Exception:
        pass

    if getattr(scene, "node_tree", None) is not None:
        return scene.node_tree

    # Hard fallback: create/find a compositor node tree
    # Blender stores compositor trees in bpy.data.node_groups
    # Create a new compositor tree and attach it
    tree = bpy.data.node_groups.new(name="CompositorTree", type="CompositorNodeTree")

    # Some versions don't let you assign scene.node_tree directly, but scene.use_nodes
    # should pick up the created tree; if not, just return `tree` and use it directly.
    return tree


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


def parse_custom_world():
    """Parse --custom-world argument from command line (appears before '--' separator)."""
    argv = sys.argv
    
    # --custom-world can appear before '--' (Blender args section)
    # or after '--' (script args section)
    for i, arg in enumerate(argv):
        if arg == '--custom-world' and i + 1 < len(argv):
            return argv[i + 1]
    
    return None

# Only load config when running in Blender
if IN_BLENDER:
    FRAME_START, FRAME_END, WORKER_ID = parse_frame_range()
    CUSTOM_WORLD_PLY = parse_custom_world()

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
    FOV_DEG = cfg_nfs.fov_deg
    NFS_NB_POINTS = cfg_nfs.nb_points
    NFS_NB_SAMPLES_PER_POINT = cfg_nfs.nb_samples_per_point
    NFS_RANGES = cfg_nfs.ranges
    
    # Paths - priority: CLI arg > config custom_world > default
    if CUSTOM_WORLD_PLY is not None:
        # CLI argument takes highest priority
        WORLD_PATH = CUSTOM_WORLD_PLY
        print(f"[DEBUG] Overriding WORLD_PATH to: {WORLD_PATH}")
        world_name = os.path.basename(WORLD_PATH)
        print(f"Using custom world (CLI): {WORLD_PATH}")
    elif hasattr(config.phase5v2, 'custom_world') and config.phase5v2.custom_world.enable:
        # Config-based custom world
        WORLD_PATH = config.phase5v2.custom_world.world_path
        if WORLD_PATH is None:
            raise ValueError("phase5v2.custom_world.enable is True but world_path is not set!")
        world_name = os.path.basename(WORLD_PATH)
        print(f"Using custom world (config): {WORLD_PATH}")
    else:
        # Default path (PLY)
        world_name = f"{_phase_3}_final_dream_pcd_unfiltered.ply"
        WORLD_PATH = save_dir_ / world_name
        WORLD_PATH = str(WORLD_PATH)
    
    # Detect world file type
    WORLD_EXT = os.path.splitext(WORLD_PATH)[1].lower()
    IS_MESH_WORLD = WORLD_EXT in ['.obj', '.fbx', '.gltf', '.glb']  # Mesh formats
    IS_POINTCLOUD_WORLD = WORLD_EXT in ['.ply']  # Point cloud formats


def main():
    start_time = time.time()
    print("=" * 60 + "\nBlender Point Cloud Renderer - NFS Dataset\n" + "=" * 60)
    
    # Compute max_x based on config (same as 5_render_nfs.py)
    max_x = (config.num_dreams - 1) * config.sphere_radius * config.delta_walk
    printc(f"max_X: {max_x}", color='red')
    
    clear_scene()
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import world (point cloud or mesh)
    print(f"Importing world: {WORLD_PATH} (type: {'mesh' if IS_MESH_WORLD else 'point cloud'})")
    if IS_MESH_WORLD:
        world_obj = import_obj(WORLD_PATH)
        if world_obj is None:
            print("[ERROR] Failed to import mesh file!")
            return
        print(f"Imported mesh: {world_obj.name} ({len(world_obj.data.vertices)} vertices, {len(world_obj.data.polygons)} faces)")
    else:
        world_obj = import_ply(WORLD_PATH)
        if world_obj is None:
            print("[ERROR] Failed to import PLY file!")
            return
        print(f"Imported point cloud: {world_obj.name} ({len(world_obj.data.vertices)} vertices)")
    
    # Apply scale factor if custom world is enabled
    scale_factor = 1.0
    if hasattr(config.phase5v2, 'custom_world') and config.phase5v2.custom_world.enable:
        if hasattr(config.phase5v2.custom_world, 'scale_factor'):
            scale_factor = config.phase5v2.custom_world.scale_factor
    if scale_factor != 1.0:
        world_obj.scale = (scale_factor, scale_factor, scale_factor)
        # Apply the scale to make it permanent (modifies vertex data)
        bpy.context.view_layer.objects.active = world_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        print(f"Applied scale factor: {scale_factor}")
    
    # Find color attribute (for point clouds)
    color_attr_name = None
    if not IS_MESH_WORLD and hasattr(world_obj.data, "color_attributes") and world_obj.data.color_attributes:
        color_attr_name = world_obj.data.color_attributes[0].name
    
    # Compute bounding box
    center, extent = bbox_center_and_extent_world(world_obj)
    bbox_radius = 0.5 * extent.length
    print(f"BBox center: ({center.x:.3f}, {center.y:.3f}, {center.z:.3f}), radius: {bbox_radius:.3f}")
    
    # Voxel downsampling (only for point clouds)
    voxel_attr_name = None
    if not IS_MESH_WORLD and KEEP_RATIO < 1.0:
        print(f"DEBUG: KEEP_RATIO < 1.0: {KEEP_RATIO < 1.0}")
        voxel_attr_name, _ = compute_voxel_downsampling(world_obj, voxel_size=VOXEL_SIZE, target_ratio=KEEP_RATIO)
    
    # Compute clipping planes
    near = MIN_NEAR_PLANE
    max_far = near * MAX_FAR_NEAR_RATIO
    desired_far = bbox_radius * 10.0
    far = min(desired_far, max_far)
    print(f"Clipping: near={near:.4f}, far={far:.4f}")
    
    # Check for custom trajectory
    cfg_custom_traj = config.phase5v2.custom_trajectory
    use_custom_trajectory = cfg_custom_traj.enable if hasattr(cfg_custom_traj, 'enable') else False
    
    # Check for custom world
    use_custom_world = (CUSTOM_WORLD_PLY is not None) or (hasattr(config.phase5v2, 'custom_world') and config.phase5v2.custom_world.enable)
    
    if use_custom_trajectory:
        # Use custom camera positions from config
        custom_positions = cfg_custom_traj.positions
        if not custom_positions:
            print("[ERROR] Custom trajectory enabled but no positions provided!")
            return
        
        # Convert to list of tuples: (x, y, z, elev_deg, azim_deg)
        all_cameras = [tuple(pos) for pos in custom_positions]
        total_cameras = len(all_cameras)
        print(f"Using custom trajectory with {total_cameras} camera positions")
    else:
        # Sample cameras (same as 5_render_nfs.py)
        all_cameras = sample_cameras(
            min_x=NFS_RANGES.min_x,
            max_x=max_x,
            min_y=NFS_RANGES.min_y,
            max_y=NFS_RANGES.max_y,
            min_z=NFS_RANGES.min_z,
            max_z=NFS_RANGES.max_z,
            nb_points=NFS_NB_POINTS,
            nb_samples_per_point=NFS_NB_SAMPLES_PER_POINT,
            seed=config.seed
        )
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
    # If custom trajectory or custom world, use special output structure
    if use_custom_trajectory or use_custom_world:
        from datetime import datetime
        # Get world name without extension
        world_basename = os.path.splitext(os.path.basename(WORLD_PATH))[0]
        # Create timestamp for trajectory folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = config.phase5v2.custom_world.scene_type
        output_dir = os.path.join(config.save_dir, config.expname, method_name)
        print(f"Using custom output directory: {output_dir}")
    else:
        output_dir = os.path.join(config.save_dir, config.expname, "nfs_dataset")
    
    output_dir_rgb = os.path.join(output_dir, "rgb")
    output_dir_depth = os.path.join(output_dir, "depth")
    output_dir_mask = os.path.join(output_dir, "mask")
    output_dir_pcd = os.path.join(output_dir, "pointcloud")
    os.makedirs(output_dir_rgb, exist_ok=True)
    os.makedirs(output_dir_depth, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_pcd, exist_ok=True)
    
    # Check if we should also render equirectangular images
    render_eqr_too = False
    eqr_width, eqr_height = 2048, 1024  # Default EQR resolution
    if use_custom_trajectory and hasattr(cfg_custom_traj, 'render_eqr_too'):
        render_eqr_too = cfg_custom_traj.render_eqr_too
        if render_eqr_too:
            if hasattr(cfg_custom_traj, 'eqr_resolution'):
                eqr_width = cfg_custom_traj.eqr_resolution.width
                eqr_height = cfg_custom_traj.eqr_resolution.height
            output_dir_eqr = os.path.join(output_dir, "rgb_eqr")
            os.makedirs(output_dir_eqr, exist_ok=True)
            print(f"Will also render equirectangular images ({eqr_width}x{eqr_height}) to: {output_dir_eqr}")
    
    # Save metadata for custom runs
    if use_custom_trajectory or use_custom_world:
        metadata = {
            "world_name": os.path.basename(WORLD_PATH),
            "world_path": WORLD_PATH,
            "world_type": "mesh" if IS_MESH_WORLD else "point_cloud",
            "custom_trajectory_enabled": use_custom_trajectory,
            "custom_world_enabled": use_custom_world,
            "trajectory_positions": [list(pos) for pos in all_cameras],  # List of [x, y, z, elev, azim]
            "num_cameras": len(all_cameras),
            "timestamp": datetime.now().isoformat(),
            "config_expname": config.expname,
            "fov_deg": FOV_DEG,
            "resolution": {"width": RES_X, "height": RES_Y},
            "render_eqr_too": render_eqr_too,
        }
        if render_eqr_too:
            metadata["eqr_resolution"] = {"width": eqr_width, "height": eqr_height}
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved metadata to: {metadata_path}")
    
    # Export processed world file (only if not a partial render or first worker)
    # if IS_MESH_WORLD:
    #     # For meshes, just copy the original file
    #     exported_world_name = os.path.basename(WORLD_PATH)
    #     world_dest = os.path.join(output_dir_pcd, exported_world_name)
    #     if not is_partial_render or WORKER_ID == 0:
    #         import shutil
    #         shutil.copy2(WORLD_PATH, world_dest)
    #         print(f"Copied mesh to: {world_dest}")
    # else:
    #     # For point clouds, export filtered PLY
    #     exported_world_name = "5v2_pcd_filtered.ply"
    #     world_dest = os.path.join(output_dir_pcd, exported_world_name)
    #     if not is_partial_render or WORKER_ID == 0:
    #         if KEEP_RATIO >= 1.0:
    #             # No filtering needed, just copy the original PLY
    #             import shutil
    #             shutil.copy2(WORLD_PATH, world_dest)
    #             print(f"Copied original PLY to: {world_dest}")
    #         else:
    #             export_ply_filtered(world_obj, world_dest, voxel_attr_name=voxel_attr_name)
    #             print(f"Exported processed PLY to: {world_dest}")
    
    
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
    
    # Cycles settings - different for mesh vs point cloud
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_adaptive_sampling = False
    
    # Depth rendering settings

    view_layer = scene.view_layers["ViewLayer"]
    view_layer.use_pass_z = True

    tree = get_compositor_tree(scene)

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    rl = nodes.new("CompositorNodeRLayers")
    depth_out = nodes.new("CompositorNodeOutputFile")

    depth_out.label = "DepthOutput"
    depth_out.format.file_format = "OPEN_EXR_MULTILAYER"
    depth_out.format.color_depth = "32"      # float32
    depth_out.format.color_mode = "RGB"       # single channel

    # Link Z pass to file output
    tree.links.new(rl.outputs["Depth"], depth_out.inputs[0])


    if IS_MESH_WORLD:
        # Use emissive material for meshes - no lighting needed, uniform appearance
        scene.cycles.max_bounces = 0
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 0
        scene.cycles.use_fast_gi = False
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        
        print(f"  [DEBUG] Mesh has {len(world_obj.data.polygons)} faces, {len(world_obj.data.vertices)} vertices")
        print(f"  [DEBUG] Mesh location: {world_obj.location}")
        print(f"  [DEBUG] Mesh scale: {world_obj.scale}")
        
        # Create emissive material for all mesh materials
        def create_emissive_material(name, color_attr_name=None):
            """Create an emissive material that uses vertex colors or texture."""
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            mat.use_backface_culling = False  # Show both sides of faces
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            
            # Clear default nodes
            nodes.clear()
            
            # Create nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            emission = nodes.new('ShaderNodeEmission')
            emission.inputs['Strength'].default_value = 1.0
            
            # Check for vertex colors
            if color_attr_name:
                vertex_color = nodes.new('ShaderNodeVertexColor')
                vertex_color.layer_name = color_attr_name
                links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
            else:
                # Use a bright color for debugging (not gray)
                emission.inputs['Color'].default_value = (1.0, 0.5, 0.2, 1.0)  # Orange for visibility
                print(f"    WARNING!! Using default orange color (no vertex colors)")
            
            links.new(emission.outputs['Emission'], output.inputs['Surface'])
            return mat
        
        # Find vertex color attribute if available
        mesh_color_attr = None
        if hasattr(world_obj.data, "color_attributes") and world_obj.data.color_attributes:
            mesh_color_attr = world_obj.data.color_attributes[0].name
        
        if world_obj.data.materials:
            # Replace existing materials with emissive versions
            print(f"  Replacing {len(world_obj.data.materials)} material(s) with emissive versions")
            for i, mat in enumerate(world_obj.data.materials):
                if mat is not None:
                    # Check if material has a texture we should preserve
                    has_texture = False
                    if mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                has_texture = True
                                # Create emissive material with texture
                                new_mat = bpy.data.materials.new(name=f"{mat.name}_emissive")
                                new_mat.use_nodes = True
                                nodes = new_mat.node_tree.nodes
                                links = new_mat.node_tree.links
                                nodes.clear()
                                
                                output = nodes.new('ShaderNodeOutputMaterial')
                                emission = nodes.new('ShaderNodeEmission')
                                emission.inputs['Strength'].default_value = 1.0
                                tex_node = nodes.new('ShaderNodeTexImage')
                                tex_node.image = node.image
                                
                                links.new(tex_node.outputs['Color'], emission.inputs['Color'])
                                links.new(emission.outputs['Emission'], output.inputs['Surface'])
                                
                                world_obj.data.materials[i] = new_mat
                                print(f"    Material '{mat.name}' -> emissive with texture")
                                break
                    
                    if not has_texture:
                        # Replace with simple emissive material
                        new_mat = create_emissive_material(f"{mat.name}_emissive", mesh_color_attr)
                        world_obj.data.materials[i] = new_mat
                        print(f"    Material '{mat.name}' -> emissive")
        else:
            # Create emissive material
            mat = create_emissive_material("MeshEmissive", mesh_color_attr)
            world_obj.data.materials.append(mat)
            if mesh_color_attr:
                print(f"  Created emissive material with vertex colors: {mesh_color_attr}")
            else:
                print("  Created basic emissive material (gray)")
    else:
        # Point clouds use emissive geometry nodes, no bounces needed
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
    
    # Debug: Print first camera position relative to bounding box
    if frames_to_render and IS_MESH_WORLD:
        first_cam = all_cameras[frames_to_render[0]]
        printc(f"  [INFO] First camera: pos=({first_cam[0]:.2f}, {first_cam[1]:.2f}, {first_cam[2]:.2f}), elev={first_cam[3]:.1f}, azim={first_cam[4]:.1f}", color='yellow')
        printc(f"  [INFO] Mesh center: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f}), bbox_radius={bbox_radius:.2f}", color='yellow')
        printc(f"  [INFO] Near clip: {near:.4f}, Far clip: {far:.4f}", color='yellow')
    
    for idx, i in enumerate(frames_to_render):
        camera_pos = all_cameras[i]
        cam_x, cam_y, cam_z, elev_deg, azim_deg = camera_pos
        cam_pos_vec = Vector((cam_x, cam_y, cam_z))
        dist = (cam_pos_vec - center).length
        
        # Create geometry nodes for point clouds (not needed for meshes)
        if not IS_MESH_WORLD:
            make_point_cloud_geometry_nodes(
                world_obj, POINT_RADIUS, keep_ratio=KEEP_RATIO,
                color_attr_name=color_attr_name or "Col", use_voxel_attr=voxel_attr_name,
                use_fast_points=USE_FAST_POINTS, use_distance_based_radius=USE_DISTANCE_BASED_RADIUS,
                use_distance_culling=USE_DISTANCE_CULLING, use_backface_culling=USE_BACKFACE_CULLING,
                camera_position=(cam_x, cam_y, cam_z),
                base_distance=max(0.1, dist) if dist > 0.01 else bbox_radius * 0.5,
                culling_start_distance=CULLING_START_DISTANCE, culling_end_distance=CULLING_END_DISTANCE,
                max_render_distance=MAX_RENDER_DISTANCE,
                wonderjourney_override=(config.phase5v2.custom_world.scene_type == "wonderjourney") 
            )
        
        # Set camera
        # TAG: CAMERA SET UP
        scene_type = config.phase5v2.custom_world.scene_type
        if scene_type is not None:
            if scene_type == "scenescape":
                scale_factor = 1.0
                R_corr = np.array([
                    [0, -1, 0],
                    [-1, 0, 0],
                    [0, 0, -1],
                ])
                T_corr = R_corr @ np.array([0.0, 0.0, 0])

                world_transform = np.concatenate([R_corr, T_corr[:, None]], axis=1)
                world_transform = np.concatenate([world_transform, np.array([[0, 0, 0, 1]])], axis=0)
                set_camera_like_open3d(cam, (scale_factor*cam_x, -scale_factor*cam_y, scale_factor*cam_z), elev_deg, azim_deg, FOV_DEG, RES_X, RES_Y, near, far, world_transform=world_transform)

            elif scene_type == "wonderjourney":
                scale_factor = 1.0
                R_corr = np.array([
                    [0, -1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ])
                T_corr = R_corr @ np.array([0.0, 0.0, 0])

                world_transform = np.concatenate([R_corr, T_corr[:, None]], axis=1)
                world_transform = np.concatenate([world_transform, np.array([[0, 0, 0, 1]])], axis=0)
                set_camera_like_open3d(cam, (scale_factor*cam_x, scale_factor*cam_y, scale_factor*cam_z), elev_deg, azim_deg, FOV_DEG, RES_X, RES_Y, near, far, world_transform=world_transform)
            
            elif scene_type == "sphericaldreamer":
                set_camera_like_open3d(cam, (cam_x, cam_y, cam_z), elev_deg, azim_deg, FOV_DEG, RES_X, RES_Y, near, far)
                
            else:
                raise ValueError(f"Unsupported scene_type: {scene_type}")


        # set_camera_like_open3d(cam, (cam_x, cam_y, cam_z), elev_deg, azim_deg, FOV_DEG, RES_X, RES_Y, near, far, world_transform=world_transform)
        
        # Force scene update (required for geometry nodes and camera changes to take effect)
        bpy.context.view_layer.update()
        
        # Render to image
        out_filename = f"azi={azim_deg:.02f}.png"
        out_filename_depth = f"azi={azim_deg:.02f}.exr"
        savedir_ = os.path.join(output_dir_rgb, f"x={cam_x:.02f}")
        os.makedirs(savedir_, exist_ok=True) 
        scene.render.filepath = os.path.join(savedir_, out_filename)
        
        if idx == 0:
            setup_render_handlers()
        
        print(f"  Rendering frame {idx+1}/{len(frames_to_render)} (global: {i}): pos=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), elev={elev_deg:.1f}, azim={azim_deg:.1f}")
        sys.stdout.flush()
        bpy.ops.render.render(write_still=True)
        
        # Save mask (areas with no points = black (ignore), areas with points = white (keep))
        # The alpha channel from transparent render: 0 = no geometry, 255 = geometry
        # We keep it as-is: black (0) = background/no points (ignore), white (255) = has points (keep)
        savedir_ = os.path.join(output_dir_mask, f"x={cam_x:.02f}")
        os.makedirs(savedir_, exist_ok=True) 

        mask_filename = f"azi={azim_deg:.02f}.png" #TAG
        mask_filepath = os.path.join(savedir_, mask_filename)
        
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
                if config.phase5v2.render_settings.save_rgba:
                    base, ext = os.path.splitext(scene.render.filepath)
                    cv2.imwrite(base + "_rgba.png", rgba_img)
                if DEBUG:
                    cv2.imwrite(f"render_debug_{i:04d}.png", rgb_img)  # Debug output
            
            # Remove temporary file
            # if os.path.exists(temp_rgba_path):
            #     os.remove(temp_rgba_path)
        
        # Render equirectangular image if enabled
        if render_eqr_too:
            # Save current camera settings
            orig_cam_type = cam.data.type
            orig_resolution_x = scene.render.resolution_x
            orig_resolution_y = scene.render.resolution_y
            
            # Configure camera for equirectangular (panoramic) rendering
            cam.data.type = 'PANO'
            # Set panorama type - try different API locations for compatibility
            if hasattr(cam.data, 'panorama_type'):
                # Blender 4.0+
                orig_panorama_type = cam.data.panorama_type
                cam.data.panorama_type = 'EQUIRECTANGULAR'
            elif hasattr(cam.data, 'cycles') and hasattr(cam.data.cycles, 'panorama_type'):
                # Older Blender with Cycles
                orig_panorama_type = cam.data.cycles.panorama_type
                cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
            else:
                print("    [WARNING] Could not set panorama type - EQR render may not work correctly")
                orig_panorama_type = None
            
            scene.render.resolution_x = eqr_width
            scene.render.resolution_y = eqr_height
            
            # For equirectangular, we just use camera position (no rotation needed, captures full 360°)
            # But we still set orientation so the "front" of the panorama matches the perspective view direction
            
            # Render EQR
            savedir_ = os.path.join(output_dir_eqr, f"x={cam_x:.02f}")
            os.makedirs(savedir_, exist_ok=True) 

            eqr_filename = f"azi={azim_deg:.02f}.png"
            scene.render.filepath = os.path.join(savedir_, eqr_filename)
            
            print(f"    Rendering EQR frame {idx+1}/{len(frames_to_render)}: {eqr_width}x{eqr_height}")
            sys.stdout.flush()
            bpy.ops.render.render(write_still=True)
            
            # Extract RGB from RGBA for EQR
            eqr_render_result = bpy.data.images.get('Render Result')
            if eqr_render_result is not None:
                temp_eqr_path = os.path.join(output_dir_eqr, f"_temp_{i:04d}.png")
                eqr_render_result.save_render(filepath=temp_eqr_path)
                
                eqr_rgba = cv2.imread(temp_eqr_path, cv2.IMREAD_UNCHANGED)
                if eqr_rgba is not None and len(eqr_rgba.shape) == 3 and eqr_rgba.shape[2] == 4:
                    eqr_rgb = eqr_rgba[:, :, :3]
                    cv2.imwrite(scene.render.filepath, eqr_rgb)
                    if config.phase5v2.render_settings.save_rgba:
                        base, ext = os.path.splitext(scene.render.filepath)
                        cv2.imwrite(base + "_rgba.png", eqr_rgba)

                    if DEBUG:
                        cv2.imwrite(f"render_eqr_debug_{i}.png", eqr_rgb) 
                
                if os.path.exists(temp_eqr_path):
                    os.remove(temp_eqr_path)
            
            # Restore camera settings for next perspective render
            cam.data.type = orig_cam_type
            # Restore panorama type if it was set
            if orig_panorama_type is not None:
                if hasattr(cam.data, 'panorama_type'):
                    cam.data.panorama_type = orig_panorama_type
                elif hasattr(cam.data, 'cycles') and hasattr(cam.data.cycles, 'panorama_type'):
                    cam.data.cycles.panorama_type = orig_panorama_type
            scene.render.resolution_x = orig_resolution_x
            scene.render.resolution_y = orig_resolution_y
    
    cleanup_render_handlers()
    
    
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"Rendered {len(frames_to_render)} frames")
    print(f"Saved NFS dataset to: {output_dir}")
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
            output_dir = os.path.join(config.save_dir, config.expname, "nfs_dataset")
            merge_transforms(output_dir, args.num_workers)
        elif args.parallel:
            run_parallel(args.num_workers, args.config)
        else:
            print("Use --parallel flag to run in parallel mode, or run via Blender:")
            print("  blender --background --python 5v2_render_blender.py -- --config <config.yaml>")
elif IN_BLENDER:
    # Running as a module inside Blender
    main()
