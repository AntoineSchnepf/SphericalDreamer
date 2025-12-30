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


def make_point_cloud_geometry_nodes(obj, point_radius: float, keep_ratio: float = 1.0, color_attr_name: str = "Col", use_voxel_attr: str = None, use_fast_points: bool = True):
    """
    Use geometry nodes to render point cloud with vertex colors.
    
    Args:
        obj: The mesh object
        point_radius: Radius of points/instances
        keep_ratio: Random keep ratio (only used if use_voxel_attr is None)
        color_attr_name: Name of the color attribute on the mesh
        use_voxel_attr: If provided, use this boolean attribute for selection
        use_fast_points: If True, use fast native points (flat discs). If False, use icosphere instances (3D but slower)
    """
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
        # Mesh to Points preserves color attributes automatically
        print("  Using FAST native point rendering (no icospheres)")
        
        # Read color attribute for material
        attr_node = nt.nodes.new("ShaderNodeAttribute")
        attr_node.attribute_type = 'GEOMETRY'
        attr_node.attribute_name = color_attr_name
        nt.links.new(attr_node.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], mat_out.inputs["Surface"])
        
        # Convert mesh to points
        mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
        mesh_to_points.mode = 'VERTICES'
        
        # Set point radius
        set_radius = nodes.new("GeometryNodeSetPointRadius")
        set_radius.inputs["Radius"].default_value = float(point_radius)
        
        # Set material
        set_material = nodes.new("GeometryNodeSetMaterial")
        set_material.inputs["Material"].default_value = mat
        
        # Selection for downsampling
        if use_voxel_attr:
            voxel_attr = nodes.new("GeometryNodeInputNamedAttribute")
            voxel_attr.data_type = 'BOOLEAN'
            voxel_attr.inputs["Name"].default_value = use_voxel_attr
            
            # Delete non-selected points
            delete_geom = nodes.new("GeometryNodeDeleteGeometry")
            delete_geom.domain = 'POINT'
            
            # Invert selection (delete where voxel_keep is False)
            bool_not = nodes.new("FunctionNodeBooleanMath")
            bool_not.operation = 'NOT'
            
            links.new(voxel_attr.outputs["Attribute"], bool_not.inputs[0])
            links.new(bool_not.outputs["Boolean"], delete_geom.inputs["Selection"])
            
            # Flow: Input -> MeshToPoints -> Delete -> SetRadius -> SetMaterial -> Output
            links.new(inp.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
            links.new(mesh_to_points.outputs["Points"], delete_geom.inputs["Geometry"])
            links.new(delete_geom.outputs["Geometry"], set_radius.inputs["Points"])
            print(f"  Using voxel-based selection: {use_voxel_attr}")
        elif keep_ratio < 1.0:
            # Random downsampling
            delete_geom = nodes.new("GeometryNodeDeleteGeometry")
            delete_geom.domain = 'POINT'
            
            random_val = nodes.new("FunctionNodeRandomValue")
            random_val.data_type = 'FLOAT'
            random_val.inputs["Min"].default_value = 0.0
            random_val.inputs["Max"].default_value = 1.0
            
            compare = nodes.new("FunctionNodeCompare")
            compare.data_type = 'FLOAT'
            compare.operation = 'GREATER_EQUAL'  # Delete if random >= keep_ratio
            compare.inputs["B"].default_value = float(keep_ratio)
            
            links.new(random_val.outputs["Value"], compare.inputs["A"])
            links.new(compare.outputs["Result"], delete_geom.inputs["Selection"])
            
            links.new(inp.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
            links.new(mesh_to_points.outputs["Points"], delete_geom.inputs["Geometry"])
            links.new(delete_geom.outputs["Geometry"], set_radius.inputs["Points"])
            print(f"  Using random selection with keep_ratio: {keep_ratio}")
        else:
            links.new(inp.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
            links.new(mesh_to_points.outputs["Points"], set_radius.inputs["Points"])
        
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
