"""
Blender script to render a point cloud as an equirectangular (360°) image.
Usage: blender --background --python 5_render_blender.py -- --config <config.yaml>

To render with a custom world file (PLY or OBJ):
blender --background --python 5_render_blender.py --custom-world my_custom_world.ply -- --config Karim/forest_v3.yaml
blender --background --python 5_render_blender.py --custom-world my_custom_world.obj -- --config Karim/forest_v3.yaml

Camera positions are set via phase5.custom_trajectory.positions in the config.
Each position is [x, y, z, elev_deg, azim_deg]. Default: [[0, 0, 0, 0, 0]].
"""

import math
import sys
import os
import json
import time
import numpy as np

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
    Vector = None

if IN_BLENDER:
    from blender_utils import (
        setup_render_handlers, cleanup_render_handlers, clear_scene, import_ply, import_obj,
        export_ply_filtered,
        ensure_camera, set_world_black, bbox_center_and_extent_world, set_camera_like_open3d,
        add_debug_test_sphere, compute_voxel_downsampling, make_point_cloud_geometry_nodes,
        setup_gpu_rendering,
        printc, fetch_config_via_parser, setup,
    )

from pipeline.phases import PHASE_3

_phase_3 = PHASE_3


def parse_custom_world():
    """Parse --custom-world argument from command line (appears before '--' separator)."""
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == '--custom-world' and i + 1 < len(argv):
            return argv[i + 1]
    return None


# Only load config when running in Blender
if IN_BLENDER:
    CUSTOM_WORLD_PLY = parse_custom_world()

    # -------------------------
    # Config
    # -------------------------
    config = fetch_config_via_parser(debug=False)
    seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = setup(config)

    cfg_render = config.phase5.render_settings
    cfg_adv = config.phase5.advanced_render_settings

    # render_settings
    POINT_RADIUS, KEEP_RATIO, VOXEL_SIZE = cfg_render.point_size, cfg_render.keep_ratio, cfg_render.voxel_size
    USE_GPU = cfg_render.use_gpu_series
    MIN_NEAR_PLANE, MAX_FAR_NEAR_RATIO = cfg_render.min_near_plane, cfg_render.max_far_near_ratio
    RES_X, RES_Y = cfg_render.width, cfg_render.height

    # advanced_render_settings
    USE_FAST_POINTS, USE_DISTANCE_BASED_RADIUS = cfg_adv.use_fast_points, cfg_adv.use_distance_based_point_size
    USE_BACKFACE_CULLING, RENDER_SAMPLES = cfg_adv.use_backface_culling, cfg_adv.render_samples
    MAX_RENDER_DISTANCE = cfg_adv.max_render_distance
    USE_DISTANCE_CULLING = cfg_adv.distance_culling.apply
    CULLING_START_DISTANCE, CULLING_END_DISTANCE = cfg_adv.distance_culling.min_distance, cfg_adv.distance_culling.max_distance

    # Paths - priority: CLI arg > config custom_world > default
    if CUSTOM_WORLD_PLY is not None:
        WORLD_PATH = CUSTOM_WORLD_PLY
        world_name = os.path.basename(WORLD_PATH)
        print(f"Using custom world (CLI): {WORLD_PATH}")
    elif hasattr(config.phase5, 'custom_world') and config.phase5.custom_world.enable:
        WORLD_PATH = config.phase5.custom_world.world_path
        if WORLD_PATH is None:
            raise ValueError("phase5.custom_world.enable is True but world_path is not set!")
        world_name = os.path.basename(WORLD_PATH)
        print(f"Using custom world (config): {WORLD_PATH}")
    else:
        world_name = f"{_phase_3}_world_pcd.ply"
        WORLD_PATH = save_dir_ / world_name
        WORLD_PATH = str(WORLD_PATH)

    # Detect world file type
    WORLD_EXT = os.path.splitext(WORLD_PATH)[1].lower()
    IS_MESH_WORLD = WORLD_EXT in ['.obj', '.fbx', '.gltf', '.glb']
    IS_POINTCLOUD_WORLD = WORLD_EXT in ['.ply']


def main():
    start_time = time.time()
    print("=" * 60 + "\nBlender Point Cloud Renderer - EQR\n" + "=" * 60)

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
    if hasattr(config.phase5, 'custom_world') and config.phase5.custom_world.enable:
        if hasattr(config.phase5.custom_world, 'scale_factor'):
            scale_factor = config.phase5.custom_world.scale_factor
    if scale_factor != 1.0:
        world_obj.scale = (scale_factor, scale_factor, scale_factor)
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
        voxel_attr_name, _ = compute_voxel_downsampling(world_obj, voxel_size=VOXEL_SIZE, target_ratio=KEEP_RATIO)

    # Compute clipping planes
    near = MIN_NEAR_PLANE
    max_far = near * MAX_FAR_NEAR_RATIO
    desired_far = bbox_radius * 10.0
    far = min(desired_far, max_far)
    print(f"Clipping: near={near:.4f}, far={far:.4f}")

    # Get camera positions from custom_trajectory config
    cfg_custom_traj = config.phase5.custom_trajectory
    raw_positions = list(cfg_custom_traj.positions) if cfg_custom_traj.positions else [[0, 0, 0, 0, 0]]
    all_cameras = [tuple(pos) for pos in raw_positions]
    print(f"Using {len(all_cameras)} camera position(s) from custom_trajectory config")

    # EQR resolution
    if hasattr(cfg_custom_traj, 'eqr_resolution'):
        eqr_width = cfg_custom_traj.eqr_resolution.width
        eqr_height = cfg_custom_traj.eqr_resolution.height
    else:
        eqr_width, eqr_height = 2048, 1024

    # Setup output directory
    from datetime import datetime
    world_basename = os.path.splitext(os.path.basename(WORLD_PATH))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.save_dir, config.expname, "blender_render", f"trajectory_{timestamp}")
    output_dir_eqr = os.path.join(output_dir, "rgb_eqr")
    os.makedirs(output_dir_eqr, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save metadata
    metadata = {
        "world_name": os.path.basename(WORLD_PATH),
        "world_path": WORLD_PATH,
        "world_type": "mesh" if IS_MESH_WORLD else "point_cloud",
        "trajectory_positions": [list(pos) for pos in all_cameras],
        "num_cameras": len(all_cameras),
        "timestamp": datetime.now().isoformat(),
        "config_expname": config.expname,
        "eqr_resolution": {"width": eqr_width, "height": eqr_height},
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # Configure render settings (once, before the loop)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = False
    scene.render.film_transparent = True
    set_world_black()
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_adaptive_sampling = False

    if IS_MESH_WORLD:
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

        def create_emissive_material(name, color_attr_name=None):
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            mat.use_backface_culling = False
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            output = nodes.new('ShaderNodeOutputMaterial')
            emission = nodes.new('ShaderNodeEmission')
            emission.inputs['Strength'].default_value = 1.0
            if color_attr_name:
                vertex_color = nodes.new('ShaderNodeVertexColor')
                vertex_color.layer_name = color_attr_name
                links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
            else:
                emission.inputs['Color'].default_value = (1.0, 0.5, 0.2, 1.0)
                print(f"    WARNING!! Using default orange color (no vertex colors)")
            links.new(emission.outputs['Emission'], output.inputs['Surface'])
            return mat

        mesh_color_attr = None
        if hasattr(world_obj.data, "color_attributes") and world_obj.data.color_attributes:
            mesh_color_attr = world_obj.data.color_attributes[0].name

        if world_obj.data.materials:
            print(f"  Replacing {len(world_obj.data.materials)} material(s) with emissive versions")
            for i, mat in enumerate(world_obj.data.materials):
                if mat is not None:
                    has_texture = False
                    if mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'TEX_IMAGE' and node.image:
                                has_texture = True
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
                        new_mat = create_emissive_material(f"{mat.name}_emissive", mesh_color_attr)
                        world_obj.data.materials[i] = new_mat
                        print(f"    Material '{mat.name}' -> emissive")
        else:
            mat = create_emissive_material("MeshEmissive", mesh_color_attr)
            world_obj.data.materials.append(mat)
    else:
        scene.cycles.max_bounces = scene.cycles.diffuse_bounces = scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = scene.cycles.volume_bounces = scene.cycles.transparent_max_bounces = 0
        scene.cycles.use_fast_gi = scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False

    if USE_GPU:
        setup_gpu_rendering(scene)
    else:
        scene.cycles.device = 'CPU'
        print("  Using CPU rendering")

    # Setup camera
    cam = ensure_camera("Cam")
    bpy.context.scene.camera = cam
    cam.data.dof.use_dof = False

    import cv2

    for idx, camera_pos in enumerate(all_cameras):
        cam_x, cam_y, cam_z, elev_deg, azim_deg = camera_pos
        cam_pos_vec = Vector((cam_x, cam_y, cam_z))
        dist = (cam_pos_vec - center).length

        # Update geometry nodes for point clouds
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
            )

        # Position the camera (orientation determines the EQR "front" direction)
        set_camera_like_open3d(cam, (cam_x, cam_y, cam_z), elev_deg, azim_deg, 60.0, RES_X, RES_Y, near, far)
        bpy.context.view_layer.update()

        # Switch to panoramic equirectangular
        cam.data.type = 'PANO'
        if hasattr(cam.data, 'panorama_type'):
            cam.data.panorama_type = 'EQUIRECTANGULAR'
        elif hasattr(cam.data, 'cycles') and hasattr(cam.data.cycles, 'panorama_type'):
            cam.data.cycles.panorama_type = 'EQUIRECTANGULAR'
        else:
            print("    [WARNING] Could not set panorama type - EQR render may not work correctly")

        scene.render.resolution_x = eqr_width
        scene.render.resolution_y = eqr_height

        eqr_filename = f"{idx:04d}.png"
        scene.render.filepath = os.path.join(output_dir_eqr, eqr_filename)

        if idx == 0:
            setup_render_handlers()

        print(f"  Rendering EQR {idx + 1}/{len(all_cameras)} ({eqr_width}x{eqr_height}): "
              f"pos=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), elev={elev_deg:.1f}, azim={azim_deg:.1f}")
        sys.stdout.flush()
        bpy.ops.render.render(write_still=True)

        # Extract RGB from RGBA render result
        eqr_render_result = bpy.data.images.get('Render Result')
        if eqr_render_result is not None:
            temp_path = os.path.join(output_dir_eqr, f"_temp_{idx:04d}.png")
            eqr_render_result.save_render(filepath=temp_path)
            eqr_rgba = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
            if eqr_rgba is not None and len(eqr_rgba.shape) == 3 and eqr_rgba.shape[2] == 4:
                cv2.imwrite(scene.render.filepath, eqr_rgba[:, :, :3])
            if os.path.exists(temp_path):
                os.remove(temp_path)

    cleanup_render_handlers()

    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"Rendered {len(all_cameras)} EQR frame(s)")
    print(f"Saved to: {output_dir_eqr}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    if IN_BLENDER:
        main()
    else:
        print("This script must be run via Blender:")
        print("  blender --background --python 5_render_blender.py -- --config <config.yaml>")
elif IN_BLENDER:
    main()
