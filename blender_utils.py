"""
Blender utility functions for point cloud rendering.

This module contains helper functions for:
- PLY import and color attribute handling
- Camera setup (Open3D-style parameters)
- Geometry nodes for point cloud visualization
- Render progress tracking
- Material creation
"""

import bpy
import bpy.app.handlers
import math
import sys
from mathutils import Vector
import os
import argparse
import ast
from prodict import Prodict
import pyfiglet
import collections.abc
from collections.abc import Mapping, Sequence
import yaml
from pathlib import Path
import numpy as np
from mathutils import Matrix, Euler

# -------------------------
# Render Progress Tracking
# -------------------------
_render_start_time = None
_last_progress = -1


def render_init_handler(scene):
    """Called when render starts."""
    global _render_start_time, _last_progress
    import time
    _render_start_time = time.time()
    _last_progress = -1
    print("\n" + "=" * 40)
    print("RENDER STARTED")
    print("=" * 40)
    sys.stdout.flush()


def render_pre_handler(scene):
    """Called before each sample/frame."""
    pass


def render_post_handler(scene):
    """Called after each sample."""
    pass


def render_stats_handler(scene):
    """Called periodically during render with stats."""
    global _last_progress
    # Try to get progress from render result
    # This is called frequently during render
    pass


def render_complete_handler(scene):
    """Called when render completes."""
    global _render_start_time
    import time
    if _render_start_time:
        elapsed = time.time() - _render_start_time
        print(f"\nRender completed in {elapsed:.1f} seconds")
    sys.stdout.flush()


def render_cancel_handler(scene):
    """Called if render is cancelled."""
    print("\nRender CANCELLED")
    sys.stdout.flush()


def setup_render_handlers():
    """Register render progress handlers."""
    # Clear any existing handlers we added
    handlers_to_add = [
        (bpy.app.handlers.render_init, render_init_handler),
        (bpy.app.handlers.render_complete, render_complete_handler),
        (bpy.app.handlers.render_cancel, render_cancel_handler),
    ]
    
    for handler_list, handler_func in handlers_to_add:
        # Remove if already present
        if handler_func in handler_list:
            handler_list.remove(handler_func)
        handler_list.append(handler_func)
    
    print("Render progress handlers registered")


def cleanup_render_handlers():
    """Unregister render progress handlers."""
    handlers_to_remove = [
        (bpy.app.handlers.render_init, render_init_handler),
        (bpy.app.handlers.render_complete, render_complete_handler),
        (bpy.app.handlers.render_cancel, render_cancel_handler),
    ]
    
    for handler_list, handler_func in handlers_to_remove:
        if handler_func in handler_list:
            handler_list.remove(handler_func)


def setup_gpu_rendering(scene):
    """Configure GPU rendering if available, fallback to CPU."""
    prefs = bpy.context.preferences.addons.get('cycles')
    if not prefs:
        print("  Cycles addon not found, using CPU")
        scene.cycles.device = 'CPU'
        return
    
    cycles_prefs = prefs.preferences
    for gpu_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL']:
        try:
            cycles_prefs.compute_device_type = gpu_type
            cycles_prefs.get_devices()
            gpu_found = False
            for device in cycles_prefs.devices:
                device.use = True
                if device.type != 'CPU':
                    gpu_found = True
                    print(f"  Enabled {gpu_type} device: {device.name}")
            if gpu_found:
                scene.cycles.device = 'GPU'
                print(f"  Using GPU rendering with {gpu_type}")
                return
        except:
            continue
    
    print("  No GPU found, falling back to CPU rendering")
    scene.cycles.device = 'CPU'


# -------------------------
# Scene Helpers
# -------------------------
def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_ply(path):
    """Import a PLY file and return the imported object."""
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=path)
    elif hasattr(bpy.ops.import_scene, "ply"):
        bpy.ops.import_scene.ply(filepath=path)
    else:
        raise RuntimeError("No PLY import operator found in this Blender build.")
    return bpy.context.view_layer.objects.active


def import_obj(path):
    """Import an OBJ/FBX/glTF file and return the imported object(s).
    
    Returns a single object if only one was imported, or a list of objects if multiple.
    For multiple objects, also returns a parent empty that contains them all.
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    
    # Track objects before import
    objects_before = set(bpy.data.objects)
    
    if ext == '.obj':
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        elif hasattr(bpy.ops.import_scene, "obj"):
            bpy.ops.import_scene.obj(filepath=path)
        else:
            raise RuntimeError("No OBJ import operator found in this Blender build.")
    elif ext == '.fbx':
        if hasattr(bpy.ops.import_scene, "fbx"):
            bpy.ops.import_scene.fbx(filepath=path)
        else:
            raise RuntimeError("No FBX import operator found in this Blender build.")
    elif ext in ['.gltf', '.glb']:
        if hasattr(bpy.ops.import_scene, "gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            raise RuntimeError("No glTF import operator found in this Blender build.")
    else:
        raise RuntimeError(f"Unsupported mesh format: {ext}")
    
    # Find all newly imported objects
    objects_after = set(bpy.data.objects)
    new_objects = list(objects_after - objects_before)
    
    # Filter to only mesh objects
    mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
    
    if len(mesh_objects) == 0:
        print(f"[WARNING] No mesh objects found after importing {path}")
        return None
    elif len(mesh_objects) == 1:
        print(f"  Imported 1 mesh object: {mesh_objects[0].name}")
        return mesh_objects[0]
    else:
        # Multiple mesh objects - join them into one
        print(f"  Imported {len(mesh_objects)} mesh objects, joining them...")
        
        # Deselect all, then select all mesh objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        
        # Set the first mesh as active and join
        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.join()
        
        joined_obj = bpy.context.view_layer.objects.active
        print(f"  Joined into: {joined_obj.name} ({len(joined_obj.data.vertices)} vertices)")
        return joined_obj


def export_ply_filtered(obj, output_path: str, voxel_attr_name: str = None):
    """
    Export point cloud to PLY, optionally filtering by a boolean attribute.
    
    Args:
        obj: Blender mesh object
        output_path: Path to save the PLY file
        voxel_attr_name: Name of boolean attribute to filter by (e.g., 'voxel_keep').
                        If None, exports all vertices.
    """
    import struct
    
    mesh = obj.data
    world_matrix = obj.matrix_world
    
    # Get vertex positions in world space
    vertices = []
    colors = []
    
    # Find color attribute
    color_attr = None
    color_attr_name_found = find_color_attribute_name(mesh)
    if color_attr_name_found and color_attr_name_found in mesh.color_attributes:
        color_attr = mesh.color_attributes[color_attr_name_found]
    
    # Get voxel keep attribute if specified
    keep_attr = None
    if voxel_attr_name and voxel_attr_name in mesh.attributes:
        keep_attr = mesh.attributes[voxel_attr_name]
    
    for i, v in enumerate(mesh.vertices):
        # Check if this vertex should be kept
        if keep_attr is not None and not keep_attr.data[i].value:
            continue
        
        # Get world position
        world_pos = world_matrix @ v.co
        vertices.append((world_pos.x, world_pos.y, world_pos.z))
        
        # Get color (if available)
        if color_attr is not None:
            if color_attr.domain == 'POINT':
                col = color_attr.data[i].color
                colors.append((int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)))
            else:
                colors.append((128, 128, 128))  # Default gray if corner domain
        else:
            colors.append((128, 128, 128))  # Default gray
    
    # Write PLY file
    with open(output_path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))
        
        # Write vertices
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(struct.pack('<fff', x, y, z))
            f.write(struct.pack('<BBB', r, g, b))
    
    print(f"Exported {len(vertices)} vertices to {output_path}")
    return len(vertices)


def find_color_attribute_name(mesh: bpy.types.Mesh):
    """
    Try common color attribute names imported from PLY.
    Blender 4+/5 uses mesh.color_attributes.
    """
    candidates = ["Col", "col", "COLOR", "Color", "color", "rgb", "RGBA", "rgba"]
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        names = [a.name for a in mesh.color_attributes]
        for c in candidates:
            if c in names:
                return c
        return names[0]  # fallback to first available
    # older API fallback
    if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
        return mesh.vertex_colors[0].name
    return None


def bake_color_to_point_domain(obj, src_attr_name=None, dst_attr_name="pc_color"):
    """
    Ensure color attribute is in POINT domain for geometry nodes.
    If source is CORNER domain, average loop colors per vertex.
    """
    me = obj.data
    if not hasattr(me, "color_attributes") or not me.color_attributes:
        print("No color attributes found on mesh.")
        return None

    # pick source attribute if not provided
    if src_attr_name is None:
        src_attr_name = find_color_attribute_name(me)
    if src_attr_name is None:
        print("Couldn't find a color attribute name.")
        return None

    src = me.color_attributes.get(src_attr_name)
    if src is None:
        print(f"Color attribute '{src_attr_name}' not found.")
        return None

    # If already POINT domain, just ensure it has a known name
    if src.domain == 'POINT':
        if src.name != dst_attr_name:
            dst = me.color_attributes.get(dst_attr_name)
            if dst is None:
                dst = me.color_attributes.new(name=dst_attr_name, type='FLOAT_COLOR', domain='POINT')
            for i in range(len(dst.data)):
                dst.data[i].color = src.data[i].color
            return dst_attr_name
        return src_attr_name

    # If CORNER domain: average loop colors per vertex -> POINT domain
    if src.domain == 'CORNER':
        dst = me.color_attributes.get(dst_attr_name)
        if dst is None:
            dst = me.color_attributes.new(name=dst_attr_name, type='FLOAT_COLOR', domain='POINT')

        vcount = len(me.vertices)
        sums = [[0.0, 0.0, 0.0, 0.0] for _ in range(vcount)]
        cnts = [0] * vcount

        for li, loop in enumerate(me.loops):
            vi = loop.vertex_index
            c = src.data[li].color
            sums[vi][0] += c[0]
            sums[vi][1] += c[1]
            sums[vi][2] += c[2]
            sums[vi][3] += c[3]
            cnts[vi] += 1

        for vi in range(vcount):
            if cnts[vi] > 0:
                inv = 1.0 / cnts[vi]
                dst.data[vi].color = (sums[vi][0]*inv, sums[vi][1]*inv, sums[vi][2]*inv, sums[vi][3]*inv)
            else:
                dst.data[vi].color = (1.0, 1.0, 1.0, 1.0)

        return dst_attr_name

    print(f"Unhandled color domain: {src.domain}")
    return None


# -------------------------
# Camera Helpers
# -------------------------
def ensure_camera(name="Cam") -> bpy.types.Object:
    """Create or get a camera object."""
    cam = bpy.data.objects.get(name)
    if cam and cam.type == "CAMERA":
        return cam
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)
    return cam


def set_world_black():
    """Set world background to black."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    
    # Blender 5.0+: use_nodes is deprecated, world always has nodes
    # Just ensure node tree exists and configure it
    if world.node_tree is None:
        # For older Blender versions, enable nodes
        if hasattr(world, 'use_nodes'):
            world.use_nodes = True
    
    nt = world.node_tree
    if nt is None:
        print("[WARN] Could not get world node tree")
        return
    
    bg = nt.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0, 0, 0, 1)
        bg.inputs[1].default_value = 1.0


def bbox_center_and_extent_world(obj: bpy.types.Object):
    """Get the center and extent of an object's bounding box in world space."""
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_world, Vector()) / 8.0
    min_v = Vector((min(v.x for v in bbox_world), min(v.y for v in bbox_world), min(v.z for v in bbox_world)))
    max_v = Vector((max(v.x for v in bbox_world), max(v.y for v in bbox_world), max(v.z for v in bbox_world)))
    extent = max_v - min_v
    return center, extent


def set_camera_like_open3d_bckp(
    cam_obj: bpy.types.Object,
    cam_pos,
    elev_deg: float,
    azim_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    near: float,
    far: float,
):
    """
    Set camera position and orientation using Open3D-style parameters.
    
    cam position = cam_pos (world)
    azimuth around +Z, elevation above XY plane
    forward direction = (cos(el)*cos(az), cos(el)*sin(az), sin(el))
    Blender camera local -Z points forward
    fov_deg treated as horizontal FOV
    """
    # Position
    cam_obj.location = (float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))

    # Orientation from elev/azim
    az = math.radians(float(azim_deg))
    el = math.radians(float(elev_deg))
    forward = Vector((
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    )).normalized()

    cam_obj.rotation_euler = forward.to_track_quat('-Z', 'Y').to_euler()

    # Render resolution
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)

    # Camera intrinsics
    cam_data = cam_obj.data
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'FOV'
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle = math.radians(float(fov_deg))

    # Clipping planes
    cam_data.clip_start = max(1e-6, float(near))
    cam_data.clip_end = max(cam_data.clip_start * 10.0, float(far))

def set_camera_like_open3d(
    cam_obj: bpy.types.Object,
    cam_pos,
    elev_deg: float,
    azim_deg: float,
    fov_deg: float,
    width: int,
    height: int,
    near: float,
    far: float,
    world_transform: np.ndarray = None,
    transform_space: str = "pre",  # "pre" or "post"
):
    """
    Set camera position and orientation using Open3D-style parameters.

    Base convention (before optional transform):
      - cam position = cam_pos (world)
      - azimuth around +Z, elevation above XY plane
      - forward direction = (cos(el)*cos(az), cos(el)*sin(az), sin(el))
      - Blender camera local -Z points forward
      - fov_deg treated as horizontal FOV

    Optional:
      world_transform: (4,4) matrix [[R, t],[0,0,0,1]] used to change coordinate system.
        If provided, it modifies the computed camera pose.

      transform_space:
        - "pre":  apply as  X_new = world_transform @ X_old
                 (i.e., you computed the pose in an "old" world, then map it into Blender world)
                 cam_world_new = world_transform @ cam_world_old
        - "post": apply as  X_new = X_old @ world_transform
                 (rare; useful if you want to apply an extra transform in the camera frame)
    """
    # -------------------------
    # 1) Base camera pose (in the "old" world)
    # -------------------------
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)

    az = math.radians(float(azim_deg))
    el = math.radians(float(elev_deg))
    forward = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ], dtype=np.float64)
    forward /= (np.linalg.norm(forward) + 1e-12)

    # Blender orientation: -Z is forward, +Y is up
    rot_euler = Vector(tuple(forward)).to_track_quat('-Z', 'Y').to_euler()
    R_bl = np.array(Euler(rot_euler).to_matrix(), dtype=np.float64)
    t_bl = cam_pos

    # Build 4x4 camera-to-world (C2W) matrix for Blender
    C2W = np.eye(4, dtype=np.float64)
    C2W[:3, :3] = R_bl
    C2W[:3, 3] = t_bl

    # -------------------------
    # 2) Apply optional world transform
    # -------------------------
    if world_transform is not None:
        M = np.asarray(world_transform, dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"world_transform must be shape (4,4), got {M.shape}")
        if transform_space == "pre":
            # Map pose from old world -> new world
            C2W = M @ C2W
        elif transform_space == "post":
            # Apply extra transform in camera/world composition order (less common)
            C2W = C2W @ M
        else:
            raise ValueError("transform_space must be 'pre' or 'post'")

    # -------------------------
    # 3) Write pose to Blender camera
    # -------------------------
    # Blender wants a world matrix (camera-to-world) for object placement
    cam_obj.matrix_world = Matrix(C2W.tolist())

    # -------------------------
    # 4) Render resolution
    # -------------------------
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)

    # -------------------------
    # 5) Camera intrinsics
    # -------------------------
    cam_data = cam_obj.data
    cam_data.type = 'PERSP'
    cam_data.lens_unit = 'FOV'
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.angle = math.radians(float(fov_deg))

    # -------------------------
    # 6) Clipping planes
    # -------------------------
    cam_data.clip_start = max(1e-6, float(near))
    cam_data.clip_end = max(cam_data.clip_start * 10.0, float(far))

# -------------------------
# Material Helpers
# -------------------------
def make_unlit_vertexcolor_material(obj, name="PointUnlit", color_attr="pc_color", fallback=(1, 1, 1, 1)):
    """
    Create an unlit (Emission) material that reads vertex colors.
    Double-sided to work when viewing from inside geometry.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    
    # CRITICAL: Disable backface culling so geometry is visible from both sides
    mat.use_backface_culling = False
    
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    emit.inputs["Color"].default_value = fallback

    # Check if the mesh has the color attribute
    has_attr = False
    if obj and obj.type == "MESH" and hasattr(obj.data, "color_attributes"):
        has_attr = (obj.data.color_attributes.get(color_attr) is not None)

    if has_attr:
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = color_attr
        nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    else:
        print(f"[WARN] Color attribute '{color_attr}' not found; using fallback color {fallback}")

    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def add_debug_test_sphere(location, radius=0.1):
    """
    Add a visible test sphere to verify rendering pipeline works.
    """
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    sphere = bpy.context.active_object
    sphere.name = "DebugTestSphere"
    
    # Create a bright red emission material
    mat = bpy.data.materials.new("DebugRed")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    emit.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # Bright red
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    
    sphere.data.materials.append(mat)
    print(f"Added debug test sphere at {location}")
    return sphere


def make_simple_emission_material(color=(1.0, 1.0, 1.0, 1.0), name="PointMat"):
    """Create a simple emission material with a fixed color."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    nt.nodes.clear()
    
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    emit.inputs["Color"].default_value = color
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


# -------------------------
# Point Cloud Helpers
# -------------------------
def compute_voxel_downsampling(obj, voxel_size: float = None, target_ratio: float = 0.1):
    """
    Compute voxel-based downsampling for a point cloud.
    
    This keeps one point per voxel, which naturally:
    - Preserves points in sparse regions (fewer points per voxel)
    - Reduces points in dense regions (many points per voxel -> keep only one)
    
    Args:
        obj: Blender mesh object
        voxel_size: Size of voxel grid cells. If None, auto-computed from target_ratio.
        target_ratio: Target fraction of points to keep (used to estimate voxel_size if not provided)
    
    Returns:
        Name of the boolean attribute created ("voxel_keep")
    """
    mesh = obj.data
    vertices = mesh.vertices
    num_verts = len(vertices)
    world_matrix = obj.matrix_world
    
    print(f"Computing voxel-based downsampling for {num_verts} vertices...")
    
    # Get all vertex positions in world space
    positions = []
    for v in vertices:
        world_pos = world_matrix @ v.co
        positions.append((world_pos.x, world_pos.y, world_pos.z))
    
    # Compute bounding box
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    min_z = min(p[2] for p in positions)
    max_z = max(p[2] for p in positions)
    
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent_z = max_z - min_z
    
    # Auto-compute voxel size if not provided
    if voxel_size is None:
        # Estimate: if we want target_ratio of points, and assuming uniform distribution,
        # we need approximately (1/target_ratio) points per voxel on average
        # Volume = extent_x * extent_y * extent_z
        # Num voxels needed ≈ num_verts * target_ratio
        # voxel_volume = total_volume / num_voxels
        # voxel_size = voxel_volume^(1/3)
        total_volume = extent_x * extent_y * extent_z
        if total_volume > 0 and num_verts > 0:
            target_num_voxels = max(1, num_verts * target_ratio)
            voxel_volume = total_volume / target_num_voxels
            voxel_size = voxel_volume ** (1.0 / 3.0)
        else:
            voxel_size = 0.01  # fallback
    
    print(f"  Voxel size: {voxel_size:.6f}")
    print(f"  Bounding box: ({extent_x:.4f}, {extent_y:.4f}, {extent_z:.4f})")
    
    # Compute voxel indices for each vertex
    # voxel_key = (ix, iy, iz) where ix = floor((x - min_x) / voxel_size)
    inv_voxel = 1.0 / voxel_size
    
    # Dictionary to track first vertex in each voxel
    voxel_to_vertex = {}
    keep_flags = [False] * num_verts
    
    for i, pos in enumerate(positions):
        ix = int((pos[0] - min_x) * inv_voxel)
        iy = int((pos[1] - min_y) * inv_voxel)
        iz = int((pos[2] - min_z) * inv_voxel)
        voxel_key = (ix, iy, iz)
        
        if voxel_key not in voxel_to_vertex:
            # First vertex in this voxel - keep it
            voxel_to_vertex[voxel_key] = i
            keep_flags[i] = True
    
    kept_count = sum(keep_flags)
    actual_ratio = kept_count / num_verts if num_verts > 0 else 0
    print(f"  Voxels occupied: {len(voxel_to_vertex)}")
    print(f"  Points kept: {kept_count} / {num_verts} ({actual_ratio*100:.1f}%)")
    
    # Create a boolean attribute on the mesh to store keep flags
    attr_name = "voxel_keep"
    
    # Remove existing attribute if present
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    
    # Create new boolean attribute on POINT domain
    attr = mesh.attributes.new(name=attr_name, type='BOOLEAN', domain='POINT')
    
    # Set the values
    for i, keep in enumerate(keep_flags):
        attr.data[i].value = keep
    
    print(f"  Created attribute '{attr_name}' for voxel-based selection")
    return attr_name, kept_count


def make_point_cloud_geometry_nodes(
    obj, 
    point_radius: float, 
    keep_ratio: float = 1.0, 
    color_attr_name: str = "Col", 
    use_voxel_attr: str = None, 
    use_fast_points: bool = True,
    use_distance_based_radius: bool = False,
    use_distance_culling: bool = False,
    use_backface_culling: bool = False,
    camera_position: tuple = None,
    base_distance: float = 1.0,
    culling_start_distance: float = 1.0,
    culling_end_distance: float = 5.0,
    max_render_distance: float = None,
    wonderjourney_override = False
):
    """
    Use geometry nodes to render point cloud with vertex colors.
    
    Args:
        obj: The mesh object
        point_radius: Base radius of points/instances
        keep_ratio: Random keep ratio (only used if use_voxel_attr is None)
        color_attr_name: Name of the color attribute on the mesh
        use_voxel_attr: If provided, use this boolean attribute for selection
        use_fast_points: If True, use fast native points (flat discs). If False, use icosphere instances (3D but slower)
        use_distance_based_radius: If True, scale point radius based on distance from camera
        use_distance_culling: If True, randomly delete more points as distance from camera increases
        use_backface_culling: If True, delete points whose normals face away from camera
        camera_position: (x, y, z) tuple for camera position (required for distance-based features)
        base_distance: Reference distance at which point_radius is the actual size
        culling_start_distance: Distance at which random culling starts (keep 100% before this)
        culling_end_distance: Distance at which culling is most aggressive (keep ~10% after this)
        max_render_distance: Hard cutoff - delete ALL points beyond this distance (None to disable)
    """
    # Remove existing PointCloudGN modifier if it exists (to avoid stacking)
    existing_mod = obj.modifiers.get("PointCloudGN")
    if existing_mod:
        # Also remove the old node group to avoid memory leaks
        old_ng = existing_mod.node_group
        obj.modifiers.remove(existing_mod)
        if old_ng and old_ng.users == 0:
            bpy.data.node_groups.remove(old_ng)
    
    # Create geometry nodes modifier
    mod = obj.modifiers.new(name="PointCloudGN", type='NODES')
    ng = bpy.data.node_groups.new("PointCloudGN", "GeometryNodeTree")
    mod.node_group = ng
    
    nodes = ng.nodes
    links = ng.links
    nodes.clear()
    
    # Create interface
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    inp = nodes.new("NodeGroupInput")
    out_node = nodes.new("NodeGroupOutput")
    
    # ===== Read the color attribute from input mesh =====
    named_attr = nodes.new("GeometryNodeInputNamedAttribute")
    named_attr.data_type = 'FLOAT_COLOR'
    named_attr.inputs["Name"].default_value = color_attr_name
    
    # ===== Create material that reads color =====
    mat = bpy.data.materials.new("PointCloudMat")
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    nt.nodes.clear()
    
    mat_out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    
    # Assign material to object
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    
    if use_fast_points:
        # ===== FAST PATH: Native point cloud rendering =====
        # MeshToPoints preserves POINT domain attributes with original names
        print("  Using FAST native point rendering (no icospheres)")
        
        # DEBUG: Use distance-based coloring to visualize where the "curtain" is
        # Red = close, Green = medium, Blue = far
        DEBUG_USE_DISTANCE_COLOR = False
        DEBUG_MAX_DISTANCE = 3.0  # Points at this distance will be blue
        
        if DEBUG_USE_DISTANCE_COLOR and camera_position is not None:
            print(f"  [DEBUG] Using distance-based coloring (red=near, blue=far, max={DEBUG_MAX_DISTANCE})")
            
            # Get position
            pos_node = nt.nodes.new("ShaderNodeNewGeometry")
            
            # Camera position
            cam_node = nt.nodes.new("ShaderNodeCombineXYZ")
            cam_node.inputs["X"].default_value = camera_position[0]
            cam_node.inputs["Y"].default_value = camera_position[1]
            cam_node.inputs["Z"].default_value = camera_position[2]
            
            # distance = length(position - camera)
            sub_node = nt.nodes.new("ShaderNodeVectorMath")
            sub_node.operation = 'SUBTRACT'
            nt.links.new(pos_node.outputs["Position"], sub_node.inputs[0])
            nt.links.new(cam_node.outputs["Vector"], sub_node.inputs[1])
            
            len_node = nt.nodes.new("ShaderNodeVectorMath")
            len_node.operation = 'LENGTH'
            nt.links.new(sub_node.outputs["Vector"], len_node.inputs[0])
            
            # Normalize to 0-1 range
            div_node = nt.nodes.new("ShaderNodeMath")
            div_node.operation = 'DIVIDE'
            div_node.inputs[1].default_value = DEBUG_MAX_DISTANCE
            nt.links.new(len_node.outputs["Value"], div_node.inputs[0])
            
            # Use color ramp: red (0) -> green (0.5) -> blue (1)
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.0
            ramp.color_ramp.elements[0].color = (1, 0, 0, 1)  # Red = close
            ramp.color_ramp.elements[1].position = 1.0
            ramp.color_ramp.elements[1].color = (0, 0, 1, 1)  # Blue = far
            # Add middle element for green
            mid_elem = ramp.color_ramp.elements.new(0.5)
            mid_elem.color = (0, 1, 0, 1)  # Green = medium
            
            nt.links.new(div_node.outputs["Value"], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], emit.inputs["Color"])
        else:
            # Read original color attribute in material (MeshToPoints preserves it)
            attr_node = nt.nodes.new("ShaderNodeAttribute")
            attr_node.attribute_type = 'GEOMETRY'
            attr_node.attribute_name = color_attr_name  # Use original name - it's preserved
            nt.links.new(attr_node.outputs["Color"], emit.inputs["Color"])
        
        nt.links.new(emit.outputs["Emission"], mat_out.inputs["Surface"])
        
        # Convert mesh to points (preserves POINT domain attributes like colors)
        mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
        mesh_to_points.mode = 'VERTICES'
        
        # Set point radius - either fixed or distance-based
        set_radius = nodes.new("GeometryNodeSetPointRadius")
        
        if use_distance_based_radius and camera_position is not None:
            # Distance-based radius: radius = point_radius * (distance / base_distance)
            # This helps maintain visual consistency across different depths
            print(f"  Using distance-based radius (base_dist={base_distance:.4f})")
            
            # Get point position
            position = nodes.new("GeometryNodeInputPosition")
            
            # Camera position as vector
            cam_vec = nodes.new("FunctionNodeInputVector")
            cam_vec.vector = camera_position
            
            # Compute distance: length(position - camera)
            subtract = nodes.new("ShaderNodeVectorMath")
            subtract.operation = 'SUBTRACT'
            links.new(position.outputs["Position"], subtract.inputs[0])
            links.new(cam_vec.outputs["Vector"], subtract.inputs[1])
            
            vec_length = nodes.new("ShaderNodeVectorMath")
            vec_length.operation = 'LENGTH'
            links.new(subtract.outputs["Vector"], vec_length.inputs[0])
            
            # Divide by base_distance
            divide = nodes.new("ShaderNodeMath")
            divide.operation = 'DIVIDE'
            print("BASE_DISTANCE:", base_distance)
            divide.inputs[1].default_value = max(0.001, base_distance)
            links.new(vec_length.outputs["Value"], divide.inputs[0])
            
            # Multiply by point_radius
            multiply = nodes.new("ShaderNodeMath")
            multiply.operation = 'MULTIPLY'
            multiply.inputs[1].default_value = float(point_radius)
            links.new(divide.outputs["Value"], multiply.inputs[0])
            
            # Clamp to reasonable range (0.5x to 3x base radius)
            clamp = nodes.new("ShaderNodeClamp")
            if wonderjourney_override:
                clamp.inputs["Min"].default_value = float(point_radius) * 0.01
            else:
                clamp.inputs["Min"].default_value = float(point_radius) * 0.5
            if wonderjourney_override:
                clamp.inputs["Max"].default_value = float(point_radius) * 200.0
            else:
                clamp.inputs["Max"].default_value = float(point_radius) * 3.0

            links.new(multiply.outputs["Value"], clamp.inputs["Value"])
            
            # Connect to set_radius
            links.new(clamp.outputs["Result"], set_radius.inputs["Radius"])
            # links.new(vec_length.outputs["Value"], set_radius.inputs["Radius"])
        else:
            # Fixed radius
            set_radius.inputs["Radius"].default_value = float(point_radius)
        
        # Set material
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.inputs["Material"].default_value = mat
        
        # ===== Build node chain =====
        # Input -> [Backface Cull] -> MeshToPoints -> [Voxel Delete] -> [Distance Cull] -> SetRadius -> SetMaterial -> Output
        
        # Start with input geometry (mesh)
        current_mesh_output = inp.outputs["Geometry"]
        
        # Apply backface culling BEFORE converting to points (we need mesh normals)
        if use_backface_culling and camera_position is not None:
            print(f"  Using backface culling (removing points facing away from camera)")
            
            delete_backface = nodes.new("GeometryNodeDeleteGeometry")
            delete_backface.domain = 'POINT'
            
            # Get vertex normal
            normal_node = nodes.new("GeometryNodeInputNormal")
            
            # Get vertex position
            pos_bf = nodes.new("GeometryNodeInputPosition")
            
            # Camera position vector
            cam_vec_bf = nodes.new("FunctionNodeInputVector")
            cam_vec_bf.vector = camera_position
            
            # View direction = normalize(camera - position)
            subtract_bf = nodes.new("ShaderNodeVectorMath")
            subtract_bf.operation = 'SUBTRACT'
            links.new(cam_vec_bf.outputs["Vector"], subtract_bf.inputs[0])
            links.new(pos_bf.outputs["Position"], subtract_bf.inputs[1])
            
            normalize_bf = nodes.new("ShaderNodeVectorMath")
            normalize_bf.operation = 'NORMALIZE'
            links.new(subtract_bf.outputs["Vector"], normalize_bf.inputs[0])
            
            # Dot product of normal and view direction
            # Positive = facing camera, Negative = facing away
            dot_bf = nodes.new("ShaderNodeVectorMath")
            dot_bf.operation = 'DOT_PRODUCT'
            links.new(normal_node.outputs["Normal"], dot_bf.inputs[0])
            links.new(normalize_bf.outputs["Vector"], dot_bf.inputs[1])
            
            # Delete if dot < 0 (facing away from camera)
            compare_bf = nodes.new("FunctionNodeCompare")
            compare_bf.data_type = 'FLOAT'
            compare_bf.operation = 'LESS_THAN'
            compare_bf.inputs["B"].default_value = 0.0
            links.new(dot_bf.outputs["Value"], compare_bf.inputs["A"])
            
            links.new(compare_bf.outputs["Result"], delete_backface.inputs["Selection"])
            links.new(current_mesh_output, delete_backface.inputs["Geometry"])
            current_mesh_output = delete_backface.outputs["Geometry"]
        
        # Convert mesh to points
        links.new(current_mesh_output, mesh_to_points.inputs["Mesh"])
        current_output = mesh_to_points.outputs["Points"]
        
        # Apply voxel-based selection if provided
        if use_voxel_attr:
            voxel_attr = nodes.new("GeometryNodeInputNamedAttribute")
            voxel_attr.data_type = 'BOOLEAN'
            voxel_attr.inputs["Name"].default_value = use_voxel_attr
            
            delete_voxel = nodes.new("GeometryNodeDeleteGeometry")
            delete_voxel.domain = 'POINT'
            
            bool_not = nodes.new("FunctionNodeBooleanMath")
            bool_not.operation = 'NOT'
            
            links.new(voxel_attr.outputs["Attribute"], bool_not.inputs[0])
            links.new(bool_not.outputs["Boolean"], delete_voxel.inputs["Selection"])
            links.new(current_output, delete_voxel.inputs["Geometry"])
            current_output = delete_voxel.outputs["Geometry"]
            print(f"  Using voxel-based selection: {use_voxel_attr}")
            
        elif keep_ratio < 1.0:
            # Random downsampling (only if no voxel attr)
            delete_random = nodes.new("GeometryNodeDeleteGeometry")
            delete_random.domain = 'POINT'
            
            random_val = nodes.new("FunctionNodeRandomValue")
            random_val.data_type = 'FLOAT'
            random_val.inputs["Min"].default_value = 0.0
            random_val.inputs["Max"].default_value = 1.0
            
            compare = nodes.new("FunctionNodeCompare")
            compare.data_type = 'FLOAT'
            compare.operation = 'GREATER_EQUAL'
            compare.inputs["B"].default_value = float(keep_ratio)
            
            links.new(random_val.outputs["Value"], compare.inputs["A"])
            links.new(compare.outputs["Result"], delete_random.inputs["Selection"])
            links.new(current_output, delete_random.inputs["Geometry"])
            current_output = delete_random.outputs["Geometry"]
            print(f"  Using random selection with keep_ratio: {keep_ratio}")
        
        # Apply distance-based culling if enabled
        if use_distance_culling and camera_position is not None:
            print(f"  Using distance culling (start={culling_start_distance:.2f}, end={culling_end_distance:.2f})")
            
            delete_dist = nodes.new("GeometryNodeDeleteGeometry")
            delete_dist.domain = 'POINT'
            
            # Get point position
            position = nodes.new("GeometryNodeInputPosition")
            
            # Camera position vector
            cam_vec = nodes.new("FunctionNodeInputVector")
            cam_vec.vector = camera_position
            
            # Distance = length(position - camera)
            subtract = nodes.new("ShaderNodeVectorMath")
            subtract.operation = 'SUBTRACT'
            links.new(position.outputs["Position"], subtract.inputs[0])
            links.new(cam_vec.outputs["Vector"], subtract.inputs[1])
            
            vec_length = nodes.new("ShaderNodeVectorMath")
            vec_length.operation = 'LENGTH'
            links.new(subtract.outputs["Vector"], vec_length.inputs[0])
            
            # Compute keep probability: 1.0 at start_distance, 0.1 at end_distance
            # keep_prob = 1.0 - 0.9 * clamp((distance - start) / (end - start), 0, 1)
            # We delete if random > keep_prob
            
            # (distance - start)
            sub_start = nodes.new("ShaderNodeMath")
            sub_start.operation = 'SUBTRACT'
            sub_start.inputs[1].default_value = float(culling_start_distance)
            links.new(vec_length.outputs["Value"], sub_start.inputs[0])
            
            # / (end - start)
            div_range = nodes.new("ShaderNodeMath")
            div_range.operation = 'DIVIDE'
            div_range.inputs[1].default_value = max(0.001, float(culling_end_distance - culling_start_distance))
            links.new(sub_start.outputs["Value"], div_range.inputs[0])
            
            # clamp(0, 1)
            clamp_t = nodes.new("ShaderNodeClamp")
            clamp_t.inputs["Min"].default_value = 0.0
            clamp_t.inputs["Max"].default_value = 1.0
            links.new(div_range.outputs["Value"], clamp_t.inputs["Value"])
            
            # * 0.9 (so we keep 10% at max distance)
            mult_09 = nodes.new("ShaderNodeMath")
            mult_09.operation = 'MULTIPLY'
            mult_09.inputs[1].default_value = 0.9
            links.new(clamp_t.outputs["Result"], mult_09.inputs[0])
            
            # 1.0 - result = keep_probability
            sub_one = nodes.new("ShaderNodeMath")
            sub_one.operation = 'SUBTRACT'
            sub_one.inputs[0].default_value = 1.0
            links.new(mult_09.outputs["Value"], sub_one.inputs[1])
            
            # Random value per point
            random_cull = nodes.new("FunctionNodeRandomValue")
            random_cull.data_type = 'FLOAT'
            random_cull.inputs["Min"].default_value = 0.0
            random_cull.inputs["Max"].default_value = 1.0
            random_cull.inputs["Seed"].default_value = 42  # Different seed from voxel
            
            # Delete if random > keep_probability
            compare_cull = nodes.new("FunctionNodeCompare")
            compare_cull.data_type = 'FLOAT'
            compare_cull.operation = 'GREATER_THAN'
            links.new(random_cull.outputs["Value"], compare_cull.inputs["A"])
            links.new(sub_one.outputs["Value"], compare_cull.inputs["B"])
            
            links.new(compare_cull.outputs["Result"], delete_dist.inputs["Selection"])
            links.new(current_output, delete_dist.inputs["Geometry"])
            current_output = delete_dist.outputs["Geometry"]
        
        # Apply hard distance cutoff if enabled (delete ALL points beyond max_render_distance)
        if max_render_distance is not None and camera_position is not None:
            print(f"  Using hard distance cutoff: max_render_distance={max_render_distance:.2f}")
            
            delete_far = nodes.new("GeometryNodeDeleteGeometry")
            delete_far.domain = 'POINT'
            
            # Get point position
            position_far = nodes.new("GeometryNodeInputPosition")
            
            # Camera position vector
            cam_vec_far = nodes.new("FunctionNodeInputVector")
            cam_vec_far.vector = camera_position
            
            # Distance = length(position - camera)
            subtract_far = nodes.new("ShaderNodeVectorMath")
            subtract_far.operation = 'SUBTRACT'
            links.new(position_far.outputs["Position"], subtract_far.inputs[0])
            links.new(cam_vec_far.outputs["Vector"], subtract_far.inputs[1])
            
            vec_length_far = nodes.new("ShaderNodeVectorMath")
            vec_length_far.operation = 'LENGTH'
            links.new(subtract_far.outputs["Vector"], vec_length_far.inputs[0])
            
            # Delete if distance > max_render_distance
            compare_far = nodes.new("FunctionNodeCompare")
            compare_far.data_type = 'FLOAT'
            compare_far.operation = 'GREATER_THAN'
            compare_far.inputs["B"].default_value = float(max_render_distance)
            links.new(vec_length_far.outputs["Value"], compare_far.inputs["A"])
            
            links.new(compare_far.outputs["Result"], delete_far.inputs["Selection"])
            links.new(current_output, delete_far.inputs["Geometry"])
            current_output = delete_far.outputs["Geometry"]
        
        # Connect to set_radius -> set_material -> output
        links.new(current_output, set_radius.inputs["Points"])
        links.new(set_radius.outputs["Points"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], out_node.inputs["Geometry"])
        
    else:
        # ===== SLOW PATH: Icosphere instances (3D spheres) =====
        print("  Using icosphere instances (slower but 3D)")
        
        # Read stored instance color
        attr_node = nt.nodes.new("ShaderNodeAttribute")
        attr_node.attribute_type = 'GEOMETRY'
        attr_node.attribute_name = "inst_color"
        nt.links.new(attr_node.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], mat_out.inputs["Surface"])
        
        # Create icosphere - use 0 subdivisions for speed (4 verts instead of 12)
        icosphere = nodes.new("GeometryNodeMeshIcoSphere")
        icosphere.inputs["Radius"].default_value = float(point_radius)
        icosphere.inputs["Subdivisions"].default_value = 0  # Minimal geometry!
        
        # Instance on points
        instance_on_points = nodes.new("GeometryNodeInstanceOnPoints")
        
        # Selection
        if use_voxel_attr:
            voxel_attr = nodes.new("GeometryNodeInputNamedAttribute")
            voxel_attr.data_type = 'BOOLEAN'
            voxel_attr.inputs["Name"].default_value = use_voxel_attr
            links.new(voxel_attr.outputs["Attribute"], instance_on_points.inputs["Selection"])
            print(f"  Using voxel-based selection: {use_voxel_attr}")
        elif keep_ratio < 1.0:
            random_val = nodes.new("FunctionNodeRandomValue")
            random_val.data_type = 'FLOAT'
            random_val.inputs["Min"].default_value = 0.0
            random_val.inputs["Max"].default_value = 1.0
            
            compare = nodes.new("FunctionNodeCompare")
            compare.data_type = 'FLOAT'
            compare.operation = 'LESS_THAN'
            compare.inputs["B"].default_value = float(keep_ratio)
            
            links.new(random_val.outputs["Value"], compare.inputs["A"])
            links.new(compare.outputs["Result"], instance_on_points.inputs["Selection"])
        
        # Store color on instance
        store_on_inst = nodes.new("GeometryNodeStoreNamedAttribute")
        store_on_inst.data_type = 'FLOAT_COLOR'
        store_on_inst.domain = 'INSTANCE'
        store_on_inst.inputs["Name"].default_value = "inst_color"
        
        # Realize instances
        realize = nodes.new("GeometryNodeRealizeInstances")
        
        # Set material
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.inputs["Material"].default_value = mat
        
        # Connect
        links.new(inp.outputs["Geometry"], instance_on_points.inputs["Points"])
        links.new(icosphere.outputs["Mesh"], instance_on_points.inputs["Instance"])
        links.new(instance_on_points.outputs["Instances"], store_on_inst.inputs["Geometry"])
        links.new(named_attr.outputs["Attribute"], store_on_inst.inputs["Value"])
        links.new(store_on_inst.outputs["Geometry"], realize.inputs["Geometry"])
        links.new(realize.outputs["Geometry"], set_material.inputs["Geometry"])
        links.new(set_material.outputs["Geometry"], out_node.inputs["Geometry"])
    
    print(f"Created geometry nodes (radius={point_radius}, fast_points={use_fast_points})")
    return mod, mat











# -------------------------------------------- #
# -------- Config utils from my_utils -------- #
# -------------------------------------------- #

# These functions are duplicates of ones from my_utils.
# I had to import them here to avoid blender environment errors as it uses its own python interpreter.


def printc(str, color=None):
    """Print string with color"""
    if color is None:
        print(str)
    else:
        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "end": "\033[0m",
            "gray": "\033[90m",
        }
        print(f"{colors[color]}{str}{colors['end']}")

def _parse_scalar(v: str):
    """Best-effort parse: int/float/bool/None/list/dict/strings."""
    v = v.strip()
    # common bool/none
    low = v.lower()
    if low == "true": return True
    if low == "false": return False
    if low in ("none", "null"): return None
    # numbers / literals / lists / dicts / quoted strings
    try:
        return ast.literal_eval(v)
    except Exception:
        return v  # fallback: raw string

def _collect_overrides(unknown_args):
    """
    Turn ["--a.b", "5", "--x", "true"] into [("a.b", 5), ("x", True)]
    Also supports "--a.b=5".
    """
    overrides = []
    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if not token.startswith("--"):
            i += 1
            continue

        token = token[2:]
        if "=" in token:
            k, v = token.split("=", 1)
            overrides.append((k, _parse_scalar(v)))
            i += 1
        else:
            k = token
            if i + 1 >= len(unknown_args) or unknown_args[i + 1].startswith("--"):
                # flag with no value -> treat as True
                overrides.append((k, True))
                i += 1
            else:
                overrides.append((k, _parse_scalar(unknown_args[i + 1])))
                i += 2
    return overrides

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        assert key in source.keys(), f"key {key} not in source"
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]

    return source

def yaml_load(cfg_name, load_dir):
    config_path = os.path.join(load_dir, cfg_name)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_config(cfg_name, load_dir, from_default=False, default_cfg_name='_default.yaml') :
    """Load a configuration file. If from_default is True, load 
    the default config and update it with the config file"""
    
    config = yaml_load(cfg_name, load_dir)

    if from_default :
        default_config = yaml_load(default_cfg_name, load_dir)
        config = deep_update(default_config, config)

    return config

class ConfigOverrideError(KeyError):
    pass

def set_by_dotted_path_strict(cfg, path: str, value):
    """
    Strict override:
    - All keys must already exist
    - List indices must be in range
    """
    keys = path.split(".")
    cur = cfg

    for i, k in enumerate(keys[:-1]):
        where = ".".join(keys[:i+1])

        if isinstance(cur, Mapping):
            if k not in cur:
                raise ConfigOverrideError(f"Config key does not exist: '{where}'")
            cur = cur[k]

        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            if not k.isdigit():
                raise ConfigOverrideError(
                    f"Expected list index at '{where}', got '{k}'"
                )
            idx = int(k)
            if idx >= len(cur):
                raise ConfigOverrideError(
                    f"List index out of range at '{where}' (len={len(cur)})"
                )
            cur = cur[idx]

        else:
            raise ConfigOverrideError(
                f"Cannot descend into non-container at '{where}'"
            )

    # ---- set final key ----
    last = keys[-1]
    where = ".".join(keys)

    if isinstance(cur, Mapping):
        if last not in cur:
            raise ConfigOverrideError(f"Config key does not exist: '{where}'")
        cur[last] = value

    elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
        if not last.isdigit():
            raise ConfigOverrideError(
                f"Expected list index at '{where}', got '{last}'"
            )
        idx = int(last)
        if idx >= len(cur):
            raise ConfigOverrideError(
                f"List index out of range at '{where}' (len={len(cur)})"
            )
        cur[idx] = value

    else:
        raise ConfigOverrideError(
            f"Cannot set value at non-container '{where}'"
        )

def fetch_config_via_parser(debug, debug_parser_override=None, return_img_name=False):
    if debug_parser_override is None:
        debug_parser_override = []

    repo_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="_default.yaml")
    parser.add_argument('--config_dir', type=str, default=os.path.join(repo_path, "configs"))

    # TODO: remove lines below
    parser.add_argument('--img_name', type=str, default='FD0')
    print("WARNING(Antoine): added a stuppid line in utils.py to run some quick exp. To remove later.")

    # Parse known args + keep the rest as overrides
    if debug:
        debug_message = pyfiglet.figlet_format("!Debug mode!", font="slant")
        printc(debug_message, color="red")
        args, unknown = parser.parse_known_args(debug_parser_override)
    else:
        # Filter out Blender-specific arguments when running via `blender --background --python script.py`
        # Blender passes all its args to the script, so we need to find args after "--"
        argv = sys.argv
        if "--" in argv:
            # Use only arguments after "--"
            script_args = argv[argv.index("--") + 1:]
        else:
            # Fallback: filter out known Blender args
            blender_args = {'--background', '-b', '--python', '-P', '--factory-startup'}
            script_args = []
            skip_next = False
            for i, arg in enumerate(argv[1:], 1):
                if skip_next:
                    skip_next = False
                    continue
                if arg in blender_args:
                    if arg in ('--python', '-P'):
                        skip_next = True  # skip the script path that follows
                    continue
                if arg.endswith('.py'):
                    continue  # skip script filenames
                script_args.append(arg)
        args, unknown = parser.parse_known_args(script_args)

    config = Prodict.from_dict(
        load_config(args.config, args.config_dir, from_default=True, default_cfg_name="_default.yaml")
    )

    # Apply overrides
    for k, v in _collect_overrides(unknown):
        set_by_dotted_path_strict(config, k, v)

    if return_img_name:
        return config, args.img_name
    return config

def camera_translation(pose, translation):
    """
    pose: np.array of shape [4,4]
    translation: np.array of shape [3,] in world coordinates
    """
    pose2 = pose.copy()
    pose2[:3, 3] += translation
    return pose2

def setup(config):
    seeds = [config.seed + offset for offset in config.seed_offsets]
    if config.depth_model == 'egformer':
        width = 1024
        height = 512
        print("WARNING: EGFormer depth model selected: Forcing panorama resolution to 1024x512")
    else:
        width = config.width
        height = config.height
        
    save_dir_ = Path(config.save_dir) / config.expname 
    pose_init = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    translation_direction = np.array(config.translation_direction, dtype=np.float32)
    pose_end = camera_translation(pose_init, config.delta_walk * translation_direction * (config.num_dreams-1))

    # create directories for saving:
    _partition = 'dream'
    for i in range(0, config.num_dreams):
        save_dir__ = save_dir_ / f"{_partition}_{i:02d}"
        os.makedirs(save_dir__, exist_ok=True)

    _partition = 'align'
    for i in range(1, config.num_dreams):
        save_dir__ = save_dir_ / f"{_partition}_{i:02d}"
        os.makedirs(save_dir__, exist_ok=True)


    return seeds, width, height, save_dir_, pose_init, pose_end, translation_direction

# -------------------------------------------- #
# ------------- Nerfstudio utils ------------- #
# -------------------------------------------- #

def sample_cameras(min_x, max_x, min_y, max_y, min_z, max_z, nb_points, nb_samples_per_point, seed):
    # Sample points in 3D space
    rng = np.random.default_rng(seed=seed)
    points = rng.random((nb_points, 3))
    points[:, 0] = min_x + (max_x - min_x) * points[:, 0]
    points[:, 1] = min_y + (max_y - min_y) * points[:, 1]
    points[:, 2] = min_z + (max_z - min_z) * points[:, 2]

    # Sample additional cameras around each point
    all_cameras = []
    for point in points:
        for _ in range(nb_samples_per_point):
            # Add a random elevation and azimuth angle
            elev_deg = rng.uniform(-90, 90)
            azim_deg = rng.uniform(0, 360)

            all_cameras.append((point[0], point[1], point[2], elev_deg, azim_deg))

    return all_cameras


def get_nerfstudio_frame(
    cam_pos, elev_deg, azim_deg, width, height, fov_deg, file_path=""
):
    """
    Matches the Open3D camera construction in your set_camera_from_elev_azim().
    - World Z up
    - azim around Z: 0 -> +X, 90 -> +Y
    - elev above XY plane
    - fov_deg is VERTICAL FOV (because you set FovType.Vertical)
    Returns a Nerfstudio 'frame' dict (c2w transform_matrix + intrinsics).
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)

    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)

    forward = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ], dtype=np.float64)
    forward /= (np.linalg.norm(forward) + 1e-12)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Same basis as your Open3D code
    right = np.cross(forward, world_up)
    right /= (np.linalg.norm(right) + 1e-12)

    up = np.cross(right, forward)
    up /= (np.linalg.norm(up) + 1e-12)

    # Nerfstudio convention: columns are [right, up, back], where back = +Z_cam in world
    back = -forward

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = back
    c2w[:3, 3] = cam_pos

    # Intrinsics from VERTICAL FOV
    fov = np.deg2rad(fov_deg)
    fl_y = 0.5 * height / np.tan(0.5 * fov)
    fl_x = fl_y * (width / height)  # aspect correction for vertical-FOV definition

    cx = width * 0.5
    cy = height * 0.5

    return {
        "file_path": file_path,
        "transform_matrix": c2w.tolist(),
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(width),
        "h": int(height),
    }