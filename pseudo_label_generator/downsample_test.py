import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import time
import point_cloud_utils as pcu
import open3d as o3d

def downsample_voxel(filtered_lidar):
    cloud = PyntCloud(points=pd.DataFrame(data=filtered_lidar, columns=['x', 'y', 'z']))
    # Create a voxel grid
    voxelgrid_id = cloud.add_structure("voxelgrid", size_x=0.15,
                                       size_y=0.15, size_z=0.15)
    # Apply the voxel grid filter to downsample the point cloud
    downsampled = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id)
    # Access the downsampled point cloud as a NumPy array
    downsampled_array = downsampled.values
    filtered_lidar = np.array(downsampled_array)[:, 0:3]
    return filtered_lidar

def downsample_voxel3(filtered_lidar):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_lidar)

    # Define your voxel size
    voxel_size = 0.15

    # Perform voxel downsampling
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    return np.asarray(downsampled_point_cloud.points)

def downsample_voxel2(filtered_lidar):
    filtered_lidar = pcu.downsample_point_cloud_on_voxel_grid(0.15, filtered_lidar)
    return filtered_lidar


def downsample_random(filtered_lidar, number=1000):
    size = filtered_lidar.shape[0]
    if size > number:
        idxs = np.random.choice(np.arange(size), number, replace=False)
        downsampled = filtered_lidar[idxs, :]
        return downsampled
    else:
        return filtered_lidar

def downsample(lidar, method="both", random_points=1000):
    # Input lidar scan Nx3
    lidar = lidar.T
    if lidar.shape[0] > 100:
        if method == "voxel":
            lidar = downsample_voxel(lidar[:, :3])

        elif method == "random":
            lidar = downsample_random(lidar[:, :3], random_points)

        elif method == "both":
            tmp1 = downsample_random(lidar[:, :3], random_points)
            tmp2 = downsample_voxel(lidar[:, :3])

            lidar = np.concatenate((tmp1, tmp2), axis=0)

    return lidar[:, :3]

def downsample2(lidar, method="both", random_points=1000):
    # Input lidar scan Nx3
    lidar = lidar.T
    if lidar.shape[0] > 100:
        if method == "voxel":
            lidar = downsample_voxel2(lidar[:, :3])

        elif method == "random":
            lidar = downsample_random(lidar[:, :3], random_points)

        elif method == "both":
            tmp1 = downsample_random(lidar[:, :3], random_points)
            tmp2 = downsample_voxel2(lidar[:, :3])

            lidar = np.concatenate((tmp1, tmp2), axis=0)

    return lidar[:, :3]

def downsample3(lidar, method="both", random_points=1000):
    # Input lidar scan Nx3
    lidar = lidar.T
    if lidar.shape[0] > 100:
        if method == "voxel":
            lidar = downsample_voxel3(lidar[:, :3])

        elif method == "random":
            lidar = downsample_random(lidar[:, :3], random_points)

        elif method == "both":
            tmp1 = downsample_random(lidar[:, :3], random_points)
            tmp2 = downsample_voxel3(lidar[:, :3])

            lidar = np.concatenate((tmp1, tmp2), axis=0)

    return lidar[:, :3]

sequence_name = "segment-10231929575853664160_1160_000_1180_000_with_camera_labels.tfrecord/"
tmp_standing = np.load("/path/to/output/frames_waymo/" + "lidar/" + sequence_name + "60" + ".npz")
tmp_standing = [tmp_standing[key] for key in tmp_standing]

start = time.time_ns()

print(len(tmp_standing))
if len(tmp_standing) > 0:
    for i in range(len(tmp_standing)):
        if tmp_standing[i].shape[1] > 1000:
            #print(tmp_standing[i].shape)
            #x = downsample(tmp_standing[i], method="both", random_points=1000)
            #print(x.shape)
            #x = downsample2(tmp_standing[i], method="both", random_points=1000)
            #print(x.shape)
            #x = downsample3(tmp_standing[i], method="both", random_points=1000)
            #print(x.shape)

print((time.time_ns() - start) / 1000000)
print((time.time_ns() - start) / (1000000 * len(tmp_standing)))