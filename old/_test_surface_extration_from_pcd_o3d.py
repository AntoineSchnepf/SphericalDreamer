import numpy as np
import open3d as o3d
import my_utils

def build_colored_pcd(points, colors=None):
    """
    points: (N, 3) float array
    colors: (N, 3) float array in [0, 1] or uint8 in [0, 255], or None

    returns: open3d.geometry.PointCloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        colors = np.asarray(colors)
        if colors.dtype != np.float64 and colors.dtype != np.float32:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    return pcd


def poisson_reconstruct_mesh_from_pcd(
    pcd,
    voxel_downsample_size=None,
    normal_radius_factor=2.0,
    normal_max_nn=30,
    poisson_depth=9,
    trim_low_density_percentile=1e-3,
):
    """
    pcd: open3d.geometry.PointCloud (with or without normals, with colors if available)

    voxel_downsample_size: float or None
        If not None, apply voxel_down_sample to speed up reconstruction.
    normal_radius_factor: float
        Radius for normal estimation = normal_radius_factor * voxel_downsample_size or
        (diagonal_bbox / 100) if voxel_downsample_size is None.
    poisson_depth: int
        Octree depth used in Poisson reconstruction. Higher = finer but slower/more memory.
    trim_low_density_percentile: float in [0,1)
        Fraction of lowest-density vertices to trim away after Poisson.
    """
    pcd_proc = pcd

    # Optional voxel downsampling for speed on huge point clouds
    if voxel_downsample_size is not None:
        pcd_proc = pcd.voxel_down_sample(voxel_size=voxel_downsample_size)

    if len(pcd_proc.points) == 0:
        raise ValueError("Point cloud is empty after optional downsampling.")

    # Estimate normals if not present
    if not pcd_proc.has_normals():
        bbox = pcd_proc.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        if voxel_downsample_size is not None:
            radius = normal_radius_factor * voxel_downsample_size
        else:
            radius = diag / 100.0

        pcd_proc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=normal_max_nn
            )
        )
        pcd_proc.orient_normals_consistent_tangent_plane(10)

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_proc, depth=poisson_depth
    )

    densities = np.asarray(densities)
    # Trim low-density vertices (likely artifacts)
    if trim_low_density_percentile is not None and 0.0 < trim_low_density_percentile < 1.0:
        thresh = np.quantile(densities, trim_low_density_percentile)
        keep_mask = densities > thresh
        mesh = mesh.select_by_index(np.where(keep_mask)[0])

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    return mesh


def color_mesh_from_point_cloud(mesh, pcd_color_source, k=1):
    """
    Assign colors to mesh vertices by nearest neighbor in pcd_color_source.

    mesh: open3d.geometry.TriangleMesh (vertices will be colored)
    pcd_color_source: open3d.geometry.PointCloud with colors
    k: number of neighbors (1 = nearest, >1 = average k nearest)

    returns: mesh (colored, in-place)
    """
    if not pcd_color_source.has_colors():
        raise ValueError("pcd_color_source must have colors.")

    # KD-tree on source point cloud
    kdtree = o3d.geometry.KDTreeFlann(pcd_color_source)
    src_colors = np.asarray(pcd_color_source.colors)
    verts = np.asarray(mesh.vertices)

    mesh_colors = np.zeros_like(verts)

    for i, v in enumerate(verts):
        _, idx, _ = kdtree.search_knn_vector_3d(v, k)
        mesh_colors[i] = src_colors[idx].mean(axis=0)

    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
    return mesh


import open3d as o3d

def visualize_pcd_and_mesh(pcd, mesh=None, point_size=2.0):
    """
    Visualize a point cloud, optionally together with a mesh, in Open3D.
    Allows control over point size and back-face rendering.
    """
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.2, 0.6, 1.0])

    # If mesh exists: ensure normals
    if mesh is not None and not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Point Cloud + Mesh",
        width=1280,
        height=720,
        left=50,
        top=50,
    )

    # vis.add_geometry(pcd)

    # Add mesh only if provided
    if mesh is not None:
        vis.add_geometry(mesh)

    # Rendering options
    opt = vis.get_render_option()
    print("default point size", opt.point_size)
    opt.point_size = point_size
    opt.mesh_show_back_face = True  # makes the sphere visible from the inside

    vis.run()
    vis.destroy_window()

import copy

def push_mesh_inside(mesh, epsilon=1e-3):
    """
    Push the mesh slightly inward along its vertex normals, so it can act
    as a 'background' under the point cloud.

    epsilon: small positive distance in world units.
    """
    # Make a copy instead of using mesh.clone()
    if hasattr(mesh, "copy"):
        mesh_bg = mesh.copy()
    else:
        mesh_bg = copy.deepcopy(mesh)

    if not mesh_bg.has_vertex_normals():
        mesh_bg.compute_vertex_normals()

    verts = np.asarray(mesh_bg.vertices)
    norms = np.asarray(mesh_bg.vertex_normals)

    # Move vertices to normals → outward
    direction = verts
    verts_new = verts + direction * epsilon
    mesh_bg.vertices = o3d.utility.Vector3dVector(verts_new)

    return mesh_bg
if __name__ == "__main__":
    # Example: load from a PLY (with RGB)
    # pcd = o3d.io.read_point_cloud("huge_colored_cloud.ply")

    # Or from NumPy arrays (points: Nx3, colors: Nx3)
    # points in float, colors in [0, 255] or [0, 1]
    # points = np.load("points.npy")
    # colors = np.load("colors.npy")

    # Demo: synthetic noisy sphere with colors
    toydset=False
    if toydset:
        N = 1000000
        phi = 2 * np.pi * np.random.rand(N)
        costheta = 2 * np.random.rand(N) - 1
        theta = np.arccos(costheta)
        r = 1.0 #+ 0.02 * np.random.randn(N)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        points = np.stack([x, y, z], axis=-1)

        # simple radial colormap
        colors = 0.5 + 0.5 * (points / np.linalg.norm(points, axis=1, keepdims=True))

        pcd = build_colored_pcd(points, colors)
    else:
        # --- parse args: which sphere to load --- #
        expname = "24_city"
        dream_iter = 2
        open_right = True
        open_left = False
        # ---------------------------------------- #


        save_dir = "OUTPUTS/SphericalDreamerRecurse"
        save_dir_ = f"{save_dir}/{expname}"

        # --- script init --- 
        width = 1440
        height = 720
        sphere_radius = 1.0
        translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
        FAR=2.0
        NEAR=0.2
        opening_kwargs = {
            'opening_mode': 'cut+cylinder',
            'delta_cut': 2*np.pi/3,
        }
        # opening_kwargs = {
        #     'opening_mode': 'straight_cut+disk_to_square_displacement',
        #     'cut_distance':0.8,
        # }
        pose1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        # --- script init end --- 
        
        sphere_correction_kwargs = {
            "correct_depth": True,
            "near": NEAR,
            "far": FAR,
            "correct_walls": True,
            "correct_floor": True,
            "depth_threshold_for_floor_correction": 0.6,
            "remove_sky": False,
            "indoor_or_outdoor": None,
            "remove_outliers": True,
            "verbose": False,
            "plot": False,
        }

        colors1, depth1 = my_utils.load_rgbd_pano(
            dream=0,
            save_dir_=save_dir_
        )
        # depth1 = np.ones_like(depth1) * 1.0  # dummy depth for testing

        pts1_carte = my_utils.depth2cam_carte(
            depth=depth1,
            sphere_radius=sphere_radius,
            height=height,
            width=width,
        ) 
        pts1_carte_corrected, colors1_corrected = my_utils.run_corrective_pipeline_on_sphere(
            pts1_carte, # in cartesian coordinates
            colors1, 
            height, width, 
            **sphere_correction_kwargs
        )


        sphere1 = my_utils.Sphere(
            pose1, pts1_carte_corrected, colors1_corrected, 
            forward_carte=translation_direction,
            opening_kwargs=opening_kwargs,
        )

        # open 3d viewer
        if open_right and not open_left:
            pcd = sphere1.right_opened.get_world_pcd().get_o3d_pointcloud()
        elif open_left and not open_right:
            pcd = sphere1.left_opened.get_world_pcd().get_o3d_pointcloud()
        elif open_right and open_left:
            pcd = sphere1.both_opened.get_world_pcd().get_o3d_pointcloud()
        else:
            pcd = sphere1.closed.get_world_pcd().get_o3d_pointcloud()

        # o3d viz
        print("Number of points in point cloud:", np.asarray(pcd.points).shape[0])
        # o3d.visualization.draw_geometries([pcd])

    # Poisson reconstruction (tune voxel_downsample_size & depth for speed/quality tradeoff)
    mesh = poisson_reconstruct_mesh_from_pcd(
        pcd,
        voxel_downsample_size=5e-5,  # increase this for faster but coarser result
        poisson_depth=9,             # 8–10 is a good range; lower = faster
        trim_low_density_percentile=1e-1,
    )
    mesh = push_mesh_inside(mesh, epsilon=2e-2)

    # Transfer colors from original dense point cloud to mesh vertices
    mesh = color_mesh_from_point_cloud(mesh, pcd, k=1)

    # Visualize both
    visualize_pcd_and_mesh(pcd, mesh, point_size=5.0)