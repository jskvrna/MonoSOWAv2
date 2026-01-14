import open3d as o3d
import numpy as np
import os
import glob

velo_to_cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03], [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02], [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]])

def load_lidar_scan(file_path):
    scan = np.fromfile(file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # Each point is [x, y, z, reflectance]
    return scan[:, :3]  # We only need [x, y, z]


# Step 2: Load the calibration data (extrinsics)
def load_calibration(calibration_file):
    return velo_to_cam


# Step 3: Transform LiDAR points to camera2 frame
def transform_lidar_to_camera2(lidar_points, Tr_velo_to_cam):
    # Convert lidar points to homogeneous coordinates
    ones = np.ones((lidar_points.shape[0], 1))
    lidar_points_hom = np.hstack((lidar_points, ones))  # (N, 4)

    # Apply the transformation
    lidar_points_in_camera2 = lidar_points_hom @ Tr_velo_to_cam.T

    return lidar_points_in_camera2[:, :3]  # Return only [x, y, z] in camera2 frame

folder_path = 'examples/'

#Open all pcds in the folder path, sorted
pcds = sorted(glob.glob(os.path.join(folder_path, '*.pcd')))

index = 0
for pcd in pcds:
    # Example usage
    lidar_file_path = pcd[:-3] + 'bin'
    calibration_file_path = pcd[:-3] + 'txt'

    # Load LiDAR scan
    lidar_points = load_lidar_scan(lidar_file_path)

    # Load transformation matrix
    Tr_velo_to_cam = load_calibration(calibration_file_path)

    # Transform LiDAR points to camera2 frame
    lidar_in_camera2 = transform_lidar_to_camera2(lidar_points, Tr_velo_to_cam)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(lidar_in_camera2)
    colors = np.zeros_like(lidar_in_camera2)
    colors[:, 2] = 1
    pcd3.colors = o3d.utility.Vector3dVector(colors)

    # load the point cloud
    pcd_visu = o3d.io.read_point_cloud(pcd)
    colors = np.zeros_like(pcd_visu.points)
    colors[:, 0] = 1
    pcd_visu.colors = o3d.utility.Vector3dVector(colors)

    #show it
    o3d.visualization.draw_geometries([pcd_visu, pcd3])

