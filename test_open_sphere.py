from spherical_independant_dreams import Sphere, load_sphere
import numpy as np
import my_utils

if __name__ == "__main__":
    translation_direction = my_utils.get_norm_vector(np.array([1, 0, 0], dtype=np.float32))
    width = 1440
    height = 720
    save_dir_ = "/home/a.schnepf/phd/SphericalDreamer/OUTPUTS/SphericalDreamerRecurse/22_campus_depth_debug"
    pose1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    sphere1 = load_sphere(
        0, pose1, translation_direction, 'cut+cylinder', 
        sphere_radius=1.0, 
        height=height, width=width, 
        save_dir_=save_dir_
    )

    # visualize the four different spheres
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(10, 10))
    for ax, attribute, attribute_name in zip(
        axs.flatten(), 
        [sphere1.left_opened, sphere1.right_opened, sphere1.closed, sphere1.both_opened],
        ['left_opened', 'right_opened', 'closed', 'both_opened']
    ):
        pts = attribute.get_world_pcd().pts
        colors = attribute.get_world_pcd().colors
        skip= 10
        ax.scatter(pts[::skip, 0], pts[::skip, 1], pts[::skip, 2], c=colors[::skip], s=1)
        ax.set_title(attribute_name)
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    plt.show()
