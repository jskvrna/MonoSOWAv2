import numpy as np
import open3d as o3d

# Example paths to your .npz files:
file_path_1 = "/path/to/waymo/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord/velodyne/0000000100.npz"
file_path_2 = "/path/to/output/frames_waymo_converted/lidar_raw/segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord/pcds/0000000100.npz"

# Load the compressed arrays from each file.
# By default, npz files often store arrays under keys like 'arr_0', 'arr_1', etc.
# Check the keys if you're not sure (e.g., np.load(...) returns a dictionary-like object).
data1 = np.load(file_path_1)
data2 = np.load(file_path_2)

velo_to_cam = np.array([[0.003048476217333246, -0.9997996727443174, 0.019781839477365427, 0.06922844066205938], [0.024830532000602652, -0.019700151246131364, -0.9994975481317834, 2.0759414874546396], [0.9996870267605354, 0.0035381381029361406, 0.02476550232571813, -1.596198612982561], [0., 0., 0., 1.]])

# Suppose each file only contains a single point cloud under 'arr_0'.
points1 = data1["arr_0"][:,:3]
points2 = data2["array1"]

tmp = np.ones((points1.shape[0],1))
points1 = np.concatenate((points1, tmp), axis=1)
points1 = np.matmul(velo_to_cam, points1.transpose()).transpose()[:, :3]

# Create Open3D point cloud objects.
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points1)
pcd1.paint_uniform_color([1, 0, 0])  # set color to red

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points2)
pcd2.paint_uniform_color([0, 1, 0])  # set color to green

# Visualize both point clouds together in an Open3D window.
o3d.visualization.draw_geometries([pcd1, pcd2])
