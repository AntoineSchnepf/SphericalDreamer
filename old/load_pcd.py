import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
# load pcd
pcd_path = "/home/a.schnepf/phd/SphericalDreamer/outputs/pcd_rgb.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

# visualize pintoicloud via web GUI
pcd_name = os.path.basename(pcd_path)

point = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# visualize point cloud using matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=colors, s=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(f'Point Cloud: {pcd_name}')
plt.show()
