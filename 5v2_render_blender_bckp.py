"""
Blender 5.0.1 script to render a point cloud from inside.

Usage:
    blender --background --python 05v2_render_blender.py

Or run directly in Blender's Python console / scripting workspace.
"""

import bpy # DO NOT PIP INSTALL
import math
import sys
import os
from mathutils import Vector
import my_utils

# Add script directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import utility functions from blender_utils
from blender_utils import (
    # Render handlers
    setup_render_handlers,
    cleanup_render_handlers,
    # Scene helpers
    clear_scene,
    import_ply,
    find_color_attribute_name,
    bake_color_to_point_domain,
    # Camera helpers
    ensure_camera,
    set_world_black,
    bbox_center_and_extent_world,
    set_camera_like_open3d,
    # Material helpers
    make_unlit_vertexcolor_material,
    add_debug_test_sphere,
    make_simple_emission_material,
    # Point cloud helpers
    compute_voxel_downsampling,
    make_point_cloud_geometry_nodes,
)


# -------------------------
# User params
# -------------------------

config = my_utils.fetch_config_via_parser(
    debug=False, 
    debug_parser_override=["--config", "exp0/0_caverns.yaml"]
)

seeds, width, height, save_dir_, pose_init, pose_end, translation_direction = my_utils.setup(config)
repo_path = os.path.dirname(os.path.realpath(__file__))

# PLY_PATH = "/home/k.kassab/panorama/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/F1_forest/nfs_dataset/pointcloud/05_pcd_filtered_d.ply"
# PLY_PATH = "/home/k.kassab/panorama/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/Caverns/3_final_dream_pcd_unfiltered.ply"
PLY_PATH = "/home/k.kassab/panorama/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/forest_v3/3_final_dream_pcd_unfiltered.ply"
# PLY_PATH = "/home/k.kassab/panorama/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/Desert/3_final_dream_pcd_unfiltered.ply"

# Point size and density settings
# Rule: POINT_RADIUS should be >= VOXEL_SIZE to avoid gaps
# Increase POINT_RADIUS if you see gaps between points on surfaces
POINT_RADIUS = config.phase5v2.render_settings.point_size
KEEP_RATIO   = config.phase5v2.render_settings.keep_ratio
# !!!! KEEP_RATIO is ignored if KEEP_RATIO < 1 and VOXEL_SIZE is not None

# Voxel-based downsampling: keeps more points in sparse regions, fewer in dense regions
# Set to None to use random sampling, or a float for voxel size (e.g., 0.01)
# Smaller voxel = more points kept, larger voxel = more aggressive downsampling
# TIP: Set POINT_RADIUS >= VOXEL_SIZE to ensure points overlap and fill gaps
VOXEL_SIZE = config.phase5v2.render_settings.voxel_size

# Use fast point rendering (no icospheres) - MUCH faster but points are flat discs
# Set to False to use icosphere instances (slower but 3D spheres)
# NOTE: If you see uniform color patches, try setting this to False
USE_FAST_POINTS = config.phase5v2.advanced_render_settings.use_fast_points

# Also increase voxel size for faster rendering with icospheres
# Use distance-based point radius (points closer to camera are larger)
# This can help with perspective rendering issues
# Set to False for uniform point size, True for distance-scaled points
USE_DISTANCE_BASED_RADIUS = config.phase5v2.advanced_render_settings.use_distance_based_point_size

# Distance-based culling: randomly remove more points as distance increases
# This prevents the "wall of points" effect at distance with fast_points
# Points are 100% kept before CULLING_START_DISTANCE, 10% kept after CULLING_END_DISTANCE
USE_DISTANCE_CULLING = config.phase5v2.advanced_render_settings.distance_culling.apply
CULLING_START_DISTANCE = config.phase5v2.advanced_render_settings.distance_culling.min_distance
CULLING_END_DISTANCE = config.phase5v2.advanced_render_settings.distance_culling.max_distance

# Hard distance cutoff: DELETE all points beyond this distance from camera
# This removes the "sky dome" / curtain effect from panorama-derived point clouds
# Set to None to disable, or a float value (e.g., 2.0) to enable
MAX_RENDER_DISTANCE = config.phase5v2.advanced_render_settings.max_render_ditance

# Backface culling: delete points whose vertex normals face AWAY from camera
# This is useful for panorama-derived point clouds where you only want to see
# the "front" of surfaces, not the back of the sphere/dome
# REQUIRES: mesh has valid vertex normals (most PLY files do)
USE_BACKFACE_CULLING = config.phase5v2.advanced_render_settings.use_backface_culling

# Debug: add a test sphere at origin to verify rendering works
DEBUG_ADD_TEST_SPHERE = config.phase5v2.debug_settings.add_test_sphere

# Render engine:
# - "CYCLES" works headless (CPU or GPU) - recommended for cloud instances
# - "BLENDER_EEVEE_NEXT" requires display/OpenGL - won't work headless without EGL setup
RENDER_ENGINE = config.phase5v2.render_settings.render_engine

# For headless GPU rendering, set to True if your instance has a GPU (T4, V100, A100, etc.)
# Set to False to use CPU rendering (slower but always works)
USE_GPU = config.phase5v2.render_settings.USE_GPU

# Camera parameters
CAM_X, CAM_Y, CAM_Z = 0.0, 0.0, 0.0
ELEV_DEG = 0.0
AZIM_DEG = 0.0
FOV_DEG  = 60.0             # Horizontal field of view

# Debug: place camera at the center of the point cloud bounding box
RENDER_FROM_CENTER = config.phase5v2.debug_settings.render_from_center.apply
# Translate camera forward from center (in units of bbox_radius)
# 0.0 = at center, 0.5 = halfway to edge, 1.0 = at edge of bounding box
FORWARD_TRANSLATION_FACTOR = config.phase5v2.debug_settings.render_from_center.forward_translation_factor

# Output
OUT_PATH = "/home/k.kassab/panorama/SphericalDreamer/OUTPUTS/tmp/pointcloud.png"
RES_X, RES_Y = config.phase5v2.render_settings.width, config.phase5v2.render_settings.height

# Anti-aliasing settings (reduce aliasing artifacts on surfaces)
# Higher values = smoother but slower
RENDER_SAMPLES = config.phase5v2.advanced_render_settings.render_samples
PIXEL_FILTER_WIDTH = config.phase5v2.advanced_render_settings.pixel_filter_width

# Clipping plane settings
# IMPORTANT: Large near/far ratio causes Z-buffer precision issues (uniform color artifacts)
# Rule of thumb: far/near ratio should be < 10000 for stable rendering
MIN_NEAR_PLANE = config.phase5v2.render_settings.min_near_plane
MAX_FAR_NEAR_RATIO = config.phase5v2.render_settings.max_far_near_ratio


# -------------------------
# Main
# -------------------------
def main():
    print("=" * 60)
    print("Blender Point Cloud Renderer")
    print("=" * 60)
    
    clear_scene()

    # Clear existing objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Import point cloud
    print(f"Importing PLY: {PLY_PATH}")
    pc_obj = import_ply(PLY_PATH)
    
    if pc_obj is None:
        print("[ERROR] Failed to import PLY file!")
        return
    
    print(f"Imported object: {pc_obj.name}")
    print(f"  Vertices: {len(pc_obj.data.vertices)}")
    
    # Debug: print available color attributes and find the right one
    color_attr_name = None
    if hasattr(pc_obj.data, "color_attributes") and pc_obj.data.color_attributes:
        print(f"  Color attributes: {[a.name for a in pc_obj.data.color_attributes]}")
        for attr in pc_obj.data.color_attributes:
            print(f"    - {attr.name}: domain={attr.domain}, type={attr.data_type}")
            if color_attr_name is None:
                color_attr_name = attr.name
    else:
        print("  No color_attributes on mesh")
    
    # Compute near/far clipping planes from bounding box BEFORE modifying the object
    center, extent = bbox_center_and_extent_world(pc_obj)
    bbox_radius = 0.5 * extent.length
    
    print(f"Point cloud center: ({center.x:.4f}, {center.y:.4f}, {center.z:.4f})")
    print(f"Point cloud extent: ({extent.x:.4f}, {extent.y:.4f}, {extent.z:.4f})")
    print(f"BBox radius: {bbox_radius:.4f}")
    
    # Determine camera position BEFORE creating geometry nodes (needed for distance-based radius)
    if RENDER_FROM_CENTER:
        # Debug mode: place camera at the center of the point cloud
        cam_pos = center.copy()
        
        # Apply forward translation if specified
        if FORWARD_TRANSLATION_FACTOR != 0.0:
            az = math.radians(float(AZIM_DEG))
            el = math.radians(float(ELEV_DEG))
            forward = Vector((
                math.cos(el) * math.cos(az),
                math.cos(el) * math.sin(az),
                math.sin(el),
            )).normalized()
            cam_pos = cam_pos + forward * (bbox_radius * FORWARD_TRANSLATION_FACTOR)
            print(f"RENDER_FROM_CENTER mode: camera at center + {FORWARD_TRANSLATION_FACTOR:.2f} * bbox_radius forward")
        else:
            print(f"RENDER_FROM_CENTER mode: placing camera at bbox center")
    else:
        cam_pos = Vector((CAM_X, CAM_Y, CAM_Z))
    
    dist = (cam_pos - center).length
    
    # Compute voxel-based downsampling if KEEP_RATIO < 1.0
    voxel_attr_name = None
    if KEEP_RATIO < 1.0:
        print("Computing voxel-based downsampling...")
        voxel_attr_name, kept_count = compute_voxel_downsampling(
            pc_obj, 
            voxel_size=VOXEL_SIZE,  # None = auto-compute
            target_ratio=KEEP_RATIO
        )
    
    # Create point cloud using geometry nodes (efficient for large point clouds)
    print("Creating point cloud geometry nodes...")
    mod, mat = make_point_cloud_geometry_nodes(
        pc_obj, 
        POINT_RADIUS, 
        keep_ratio=KEEP_RATIO,
        color_attr_name=color_attr_name if color_attr_name else "Col",
        use_voxel_attr=voxel_attr_name,
        use_fast_points=USE_FAST_POINTS,
        use_distance_based_radius=USE_DISTANCE_BASED_RADIUS,
        use_distance_culling=USE_DISTANCE_CULLING,
        use_backface_culling=USE_BACKFACE_CULLING,
        camera_position=(cam_pos.x, cam_pos.y, cam_pos.z),
        base_distance=max(0.1, dist) if dist > 0.01 else bbox_radius * 0.5,
        culling_start_distance=CULLING_START_DISTANCE,
        culling_end_distance=CULLING_END_DISTANCE,
        max_render_distance=MAX_RENDER_DISTANCE,
    )
    
    # Set near/far based on point cloud size
    # CRITICAL: Maintain reasonable near/far ratio to avoid Z-buffer precision issues
    # Z-buffer precision degrades when far/near ratio is too large (>10000)
    # This causes parts of the scene to render as uniform color blocks
    
    # Near plane is fixed to MIN_NEAR_PLANE, far plane is computed from ratio
    near = MIN_NEAR_PLANE
    max_far = near * MAX_FAR_NEAR_RATIO
    
    if RENDER_FROM_CENTER:
        # When at center, we want to see the full extent of the point cloud
        desired_far = bbox_radius * 10.0
    else:
        # Camera outside - see from camera to far side of scene
        desired_far = dist + bbox_radius * 2.0
    
    # Clamp far plane to maximum allowed by ratio
    far = min(desired_far, max_far)
    
    if desired_far > max_far:
        print(f"  [INFO] Far plane clamped from {desired_far:.4f} to {far:.4f} (max ratio {MAX_FAR_NEAR_RATIO})")
    
    print(f"Camera position: ({cam_pos.x:.4f}, {cam_pos.y:.4f}, {cam_pos.z:.4f})")
    print(f"Camera distance from center: {dist:.4f}")
    print(f"Clipping planes: near={near:.6f}, far={far:.6f} (ratio: {far/near:.1f})")
    
    # Add debug test sphere if enabled
    if DEBUG_ADD_TEST_SPHERE:
        # Place sphere slightly in front of camera
        az = math.radians(float(AZIM_DEG))
        el = math.radians(float(ELEV_DEG))
        forward = Vector((
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        )).normalized()
        sphere_pos = cam_pos + forward * (bbox_radius * 0.3)
        add_debug_test_sphere(sphere_pos, radius=bbox_radius * 0.05)
    
    # Set up camera
    cam = ensure_camera("Cam")
    bpy.context.scene.camera = cam
    cam.data.dof.use_dof = False
    
    set_camera_like_open3d(
        cam_obj=cam,
        cam_pos=(cam_pos.x, cam_pos.y, cam_pos.z),
        elev_deg=ELEV_DEG,
        azim_deg=AZIM_DEG,
        fov_deg=FOV_DEG,
        width=RES_X,
        height=RES_Y,
        near=near,
        far=far,
    )
    
    # Configure render settings
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    scene.render.filepath = OUT_PATH
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False  # Solid black background
    
    # Set black background
    set_world_black()
    
    # Color management
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.exposure = 0.0
    
    # Engine-specific settings
    if RENDER_ENGINE == "CYCLES":
        # Anti-aliasing: more samples = smoother edges
        scene.cycles.samples = RENDER_SAMPLES
        scene.cycles.use_adaptive_sampling = False  # Stop early if converged
        scene.cycles.adaptive_threshold = 0.01     # Lower = higher quality
        
        # Pixel filter for anti-aliasing (smooths jagged edges)
        # scene.cycles.pixel_filter_type = 'BLACKMAN_HARRIS'  # Good balance of sharpness/smoothness
        # scene.cycles.filter_width = PIXEL_FILTER_WIDTH      # 1.5 is a good default
        
        # Disable expensive features we don't need (emission doesn't need bounces)
        scene.cycles.max_bounces = 0
        scene.cycles.diffuse_bounces = 0
        scene.cycles.glossy_bounces = 0
        scene.cycles.transmission_bounces = 0
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 0
        scene.cycles.use_fast_gi = False
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        
        # GPU/CPU setup for headless rendering
        if USE_GPU:
            # Try to enable GPU rendering
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons.get('cycles')
            if cycles_prefs:
                cycles_prefs = cycles_prefs.preferences
                
                # Try CUDA first (NVIDIA), then HIP (AMD), then OPTIX, then CPU
                gpu_types = ['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL']
                gpu_found = False
                
                for gpu_type in gpu_types:
                    try:
                        cycles_prefs.compute_device_type = gpu_type
                        cycles_prefs.get_devices()
                        
                        # Enable all available devices
                        for device in cycles_prefs.devices:
                            device.use = True
                            if device.type != 'CPU':
                                gpu_found = True
                                print(f"  Enabled {gpu_type} device: {device.name}")
                        
                        if gpu_found:
                            scene.cycles.device = 'GPU'
                            print(f"  Using GPU rendering with {gpu_type}")
                            break
                    except Exception as e:
                        continue
                
                if not gpu_found:
                    print("  No GPU found, falling back to CPU rendering")
                    scene.cycles.device = 'CPU'
            else:
                print("  Cycles addon not found, using CPU")
                scene.cycles.device = 'CPU'
        else:
            scene.cycles.device = 'CPU'
            print("  Using CPU rendering (USE_GPU=False)")
        
    elif RENDER_ENGINE in ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"]:
        print("  WARNING: EEVEE may not work on headless instances!")
        print("  If rendering fails, switch to CYCLES")
        # EEVEE is real-time - very fast for emission materials
        scene.eevee.taa_render_samples = 16  # Anti-aliasing samples
        
        # Disable expensive effects we don't need
        if hasattr(scene.eevee, 'use_gtao'):
            scene.eevee.use_gtao = False  # Ambient occlusion
        if hasattr(scene.eevee, 'use_bloom'):
            scene.eevee.use_bloom = False
        if hasattr(scene.eevee, 'use_ssr'):
            scene.eevee.use_ssr = False  # Screen-space reflections
        if hasattr(scene.eevee, 'use_motion_blur'):
            scene.eevee.use_motion_blur = False
        if hasattr(scene.eevee, 'use_volumetric_lights'):
            scene.eevee.use_volumetric_lights = False
        if hasattr(scene.eevee, 'use_shadows'):
            scene.eevee.use_shadows = False  # No shadows needed
    else:
        raise RuntimeError(f"Unsupported render engine: {RENDER_ENGINE}")
    
    # Setup render progress handlers
    setup_render_handlers()
    
    # Render with progress
    print("Rendering...")
    print(f"  Engine: {RENDER_ENGINE}")
    print(f"  Resolution: {RES_X}x{RES_Y}")
    if RENDER_ENGINE == "CYCLES":
        print(f"  Samples: {scene.cycles.samples}")
    sys.stdout.flush()
    
    bpy.ops.render.render(write_still=True)
    
    # Cleanup handlers
    cleanup_render_handlers()
    
    print(f"Saved: {OUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
else:
    # When run as a script in Blender (not imported as module)
    main()
