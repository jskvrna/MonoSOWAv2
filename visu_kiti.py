import numpy as np
import open3d as o3d
import os
import random

def load_velodyne_points(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    intensity = points[:, 3]
    return xyz, intensity

def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            data = line.strip().split(' ')
            obj_class = data[0]
            if obj_class == 'DontCare':
                continue  # Skip unlabeled regions
            truncated = float(data[1])
            occluded = int(data[2])
            alpha = float(data[3])
            bbox = [float(val) for val in data[4:8]]  # 2D bounding box
            dimensions = [float(val) for val in data[8:11]]  # height, width, length
            location = [float(val) for val in data[11:14]]   # x, y, z in camera coordinates
            rotation_y = float(data[14])
            label = {
                'class': obj_class,
                'dimensions': dimensions,
                'location': location,
                'rotation_y': rotation_y
            }
            labels.append(label)
    return labels

def load_calibration(file_path):
    calib = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            key, value = line.strip().split(':', 1)
            calib[key] = np.array([float(x) for x in value.strip().split()])

    # Reshape the matrices
    calib_matrices = {}
    if 'P2' in calib:
        calib_matrices['P2'] = calib['P2'].reshape(3, 4)
    if 'R0_rect' in calib:
        calib_matrices['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    if 'Tr_velo_to_cam' in calib:
        calib_matrices['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return calib_matrices

def transform_point_cloud(xyz, calib):
    # Append a column of ones to the xyz coordinates
    xyz_hom = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    # Transformation matrix from Velodyne to camera coordinates
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    R0_rect = calib['R0_rect']

    # Apply the transformations
    xyz_cam = (Tr_velo_to_cam @ xyz_hom.T).T
    xyz_rect = (R0_rect @ xyz_cam[:, :3].T).T
    return xyz_rect


def create_3d_bbox(label, color):
    h, w, l = label['dimensions']  # height, width, length
    x, y, z = label['location']
    ry = label['rotation_y']

    # Create the bounding box in the object's local coordinate system
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = np.array([x, y - h / 2, z])  # Adjust y to the center of the box
    bbox.extent = np.array([l, h, w])  # length, height, width

    # Set rotation
    # Create rotation matrix from rotation around Y-axis (KITTI frame)
    R = bbox.get_rotation_matrix_from_axis_angle([0, ry, 0])
    bbox.R = R

    # Set the color
    bbox.color = color

    return bbox

def visualize_point_cloud_with_labels(velodyne_dir, labels_dir, pseudo_labels_dir, calib_dir):
    file_names = [f for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
    random.shuffle(file_names)
    for file_name in file_names:
        print(f'Processing {file_name}...')
        velodyne_path = os.path.join(velodyne_dir, file_name)
        label_name = file_name.replace('.bin', '.txt')
        label_path = os.path.join(labels_dir, label_name)
        pseudo_label_path = os.path.join(pseudo_labels_dir, label_name)
        calib_path = os.path.join(calib_dir, label_name)

        # Load point cloud data
        xyz, intensity = load_velodyne_points(velodyne_path)

        # Load calibration data
        if not os.path.exists(calib_path):
            print(f'Calibration file {calib_path} not found.')
            continue
        calib = load_calibration(calib_path)

        # Transform point cloud to camera coordinates
        xyz_cam = transform_point_cloud(xyz, calib)

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_cam)
        pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (xyz_cam.shape[0], 1)))

        geometries = [pcd]

        # Load and visualize ground truth labels
        if os.path.exists(label_path):
            labels = load_labels(label_path)
            for label in labels:
                bbox = create_3d_bbox(label, color=[0, 1, 0])  # Green color
                geometries.append(bbox)

        # Load and visualize pseudo labels
        if os.path.exists(pseudo_label_path):
            pseudo_labels = load_labels(pseudo_label_path)
            for label in pseudo_labels:
                bbox = create_3d_bbox(label, color=[1, 0, 0])  # Blue color
                geometries.append(bbox)

        # Visualize using Open3D
        o3d.visualization.draw_geometries(geometries)

def main():
    velodyne_dir = '/path/to/MonoDETR/data/k360_gt_gt/training/velodyne/'
    labels_dir = '/path/to/MonoDETR/data/k360_gt_gt/training/label_2/'
    pseudo_labels_dir = '/path/to/MonoDETR/data/k360_gt_gt/training/label_pseudo/'
    calib_dir = '/path/to/MonoDETR/data/k360_gt_gt/training/calib/'

    visualize_point_cloud_with_labels(velodyne_dir, labels_dir, pseudo_labels_dir, calib_dir)

if __name__ == '__main__':
    main()
