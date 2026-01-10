from my_utils import PointCloud, Sphere
import numpy as mp
import open3d as o3d
from pathlib import Path
import pickle
import numpy as np

import numpy as np

def overlay_purple_rgb(
    colors_rgb,
    mask,
    alpha=0.35,
    purple_rgb=(180, 130, 255)
):
    """
    Apply a purple overlay to RGB colors with given alpha.

    Parameters
    ----------
    colors_rgb : (N, 3) array
        Input RGB colors in uint8 [0,255] or float [0,1].
    mask : (N,) array
        Binary mask (1 = overlay, 0 = keep original).
    alpha : float
        Overlay strength (default: 0.35).
    purple_rgb : tuple
        Purple color in RGB [0,255].

    Returns
    -------
    out : (N, 3) array
        RGB colors with overlay applied (uint8).
    """
    colors = colors_rgb.astype(np.float32)
    if colors.max() > 1.0:
        colors /= 255.0

    purple = np.array(purple_rgb, dtype=np.float32) / 255.0
    mask = mask.astype(np.float32)[:, None]  # (N,1)

    # Alpha blending
    out = colors * (1.0 - alpha * mask) + purple * (alpha * mask)
    out = np.clip(out, 0.0, 1.0)

    return out

def get_ply_path(expname, which_ply):
    ply_fname = f"{expname}__key={which_ply}.ply"
    return ply_files_save_dir / ply_fname

def pcd_zoo_save_ply(save_dir_, expname, align_iter, which_ply):
    pointcloud_zoo = pickle.load(open(save_dir_ / f"align_{align_iter:02d}" / "2c_pointclouds_zoo.pkl", "rb"))
    pcd = pointcloud_zoo[which_ply].get_o3d_pointcloud()
    ply_path = get_ply_path(expname, which_ply)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    
expname = "forest_v3"
# expname = "Caverns"

ply_files_save_dir = Path("/home/a.schnepf/phd/SphericalDreamer/viz_ply_pointclouds")

pointcloud_zoo_keys = [
    "sphere1_init",
    "sphere2_init",
    "sphere1_open",
    "sphere2_open", #only for last iter
]

basic_spheres_key = [
    "sphere1_closed",
    "sphere1_right_opened",
    "sphere1_left_opened",
    "sphere1_both_opened",
]


conbo_spheres = {
    # "hollow_capsule": ["sphere1_open", "sphere2_open"],
    # "filled_capsule": ["sphere1_open", "sphere2_open", "blended_harmonic"],
    # "filled_capsule_colored": ["sphere1_open", "sphere2_open", "blended_harmonic"]
}

which_ply = "sphere1_open"

if __name__ == "__main__":
    OUTPUTS = Path("/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse")
    save_dir_ = OUTPUTS / expname 
    align_iter = 1

    four_spheres = False
    capsules = False
    full_world = False
    partial_world = False

    # THIS saves the actual opened spheres used during hblending
    # for which_sphere in pointcloud_zoo_keys:
    #     pcd_zoo_save_ply(save_dir_, expname, align_iter, which_sphere)
    #     print(f"Saved PLY for {which_sphere}")

    # THIS saves all state for the very first sphere
    if four_spheres:
        first_sphere_pkl_path = OUTPUTS / expname  / f"align_01" / "2a" / ".cache" / "sphere1.pkl"   
        sphere1 = Sphere.instanciate_from_saved_dict(first_sphere_pkl_path)

        sphere1_closed = sphere1.closed.get_world_pcd().get_o3d_pointcloud()
        ply_path = get_ply_path(expname, "sphere1_closed")
        o3d.io.write_point_cloud(str(ply_path), sphere1_closed)
        print(f"Saved PLY for sphere1_closed")

        sphere1_ro = sphere1.right_opened.get_world_pcd().get_o3d_pointcloud()
        ply_path = get_ply_path(expname, "sphere1_right_opened")
        o3d.io.write_point_cloud(str(ply_path), sphere1_ro)
        print(f"Saved PLY for sphere1_right_opened")

        sphere1_lo = sphere1.left_opened.get_world_pcd().get_o3d_pointcloud()
        ply_path = get_ply_path(expname, "sphere1_left_opened")
        o3d.io.write_point_cloud(str(ply_path), sphere1_lo)
        print(f"Saved PLY for sphere1_left_opened")

        sphere1_both = sphere1.both_opened.get_world_pcd().get_o3d_pointcloud()
        ply_path = get_ply_path(expname, "sphere1_both_opened")
        o3d.io.write_point_cloud(str(ply_path), sphere1_both)
        print(f"Saved PLY for sphere1_both_opened")


    # THIS saves unions of spheres after / before hblending etc
    if capsules:
        pointcloud_zoo = pickle.load(open(save_dir_ / f"align_{align_iter:02d}" / "2c_pointclouds_zoo.pkl", "rb"))
        for combo_name, sphere_keys in conbo_spheres.items():
            all_points = None
            all_colors = None
            for sphere_key in sphere_keys:
                pts = pointcloud_zoo[sphere_key].pts
                colors = pointcloud_zoo[sphere_key].colors
                if combo_name.endswith("colored"):
                    if sphere_key == "blended_harmonic":
                        # make colors a bit different to see the blend area
                        colors = overlay_purple_rgb(colors, np.ones((colors.shape[0],), dtype=np.uint8), alpha=0.35)
                if all_points is None:
                    all_points = pts
                    all_colors = colors
                else:
                    all_points = np.vstack((all_points, pts))
                    all_colors = np.vstack((all_colors, colors))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)
            ply_path = get_ply_path(expname, combo_name)
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"Saved PLY for {combo_name}")


    # THIS saves the full forest pointcloud after final dream
    if full_world:
        save_dir_full_world = OUTPUTS / "Forest"
        # for align in range(1, 6):
        all_points = []
        all_colors = []

        for align in range(1, 5):
            pointcloud_zoo = pickle.load(open(save_dir_full_world / f"align_{align:02d}" / "2c_pointclouds_zoo.pkl", "rb"))
            pts = pointcloud_zoo['blended_harmonic'].pts
            colors = pointcloud_zoo['blended_harmonic'].colors
            all_points.append(pts)
            all_colors.append(colors)
            pts = pointcloud_zoo['sphere1_open'].pts
            colors = pointcloud_zoo['sphere1_open'].colors
            all_points.append(pts)
            all_colors.append(colors)

        all_points.append(pointcloud_zoo['sphere2_open'].pts)
        all_colors.append(pointcloud_zoo['sphere2_open'].colors)
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        ply_path = get_ply_path(expname, "Forest_full_pcd_main_fig")
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved PLY for Forest_full_pcd_main_fig")

    # THIS saves the PARTIAL forest pointcloud
    if partial_world:
        save_dir_full_world = OUTPUTS / "Forest"

        all_points = []
        all_colors = []

        for align in range(1, 5):
            pointcloud_zoo = pickle.load(open(save_dir_full_world / f"align_{align:02d}" / "2c_pointclouds_zoo.pkl", "rb"))
            pts = pointcloud_zoo['sphere1_open'].pts
            colors = pointcloud_zoo['sphere1_open'].colors
            all_points.append(pts)
            all_colors.append(colors)

        all_points.append(pointcloud_zoo['sphere2_open'].pts)
        all_colors.append(pointcloud_zoo['sphere2_open'].colors)
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        ply_path = get_ply_path(expname, "Forest_partial_pcd_appendix_fig")
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved PLY for Forest_partial_pcd_appendix_fig")


    # THIS show rendering with and without LDI (single sphere)
    sphere = Sphere.instanciate_from_saved_dict(OUTPUTS / expname  / f"align_01" / "2a" / ".cache" / "sphere2.pkl" )
    single_sphere_ldi = sphere.closed.get_world_pcd()

    pts_ldi = single_sphere_ldi.pts
    colors_ldi = single_sphere_ldi.colors
    ldi_mask = single_sphere_ldi.ldi_mask

    pts_no_ldi = pts_ldi[~ldi_mask]
    colors_no_ldi = colors_ldi[~ldi_mask]

    single_sphere_ldi = o3d.geometry.PointCloud()
    single_sphere_ldi.points = o3d.utility.Vector3dVector(pts_ldi)
    single_sphere_ldi.colors = o3d.utility.Vector3dVector(colors_ldi)
    ply_path = get_ply_path(expname, "single_sphere_ldi")
    o3d.io.write_point_cloud(str(ply_path), single_sphere_ldi)
    print(f"Saved PLY for single_sphere_ldi")

    single_sphere_no_ldi = o3d.geometry.PointCloud()
    single_sphere_no_ldi.points = o3d.utility.Vector3dVector(pts_no_ldi)
    single_sphere_no_ldi.colors = o3d.utility.Vector3dVector(colors_no_ldi)
    ply_path = get_ply_path(expname, "single_sphere_no_ldi")
    o3d.io.write_point_cloud(str(ply_path), single_sphere_no_ldi)
    print(f"Saved PLY for single_sphere_no_ldi")

