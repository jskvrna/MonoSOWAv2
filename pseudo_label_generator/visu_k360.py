import os
import numpy as np
import open3d as o3d
from pathlib import Path
import random

def load_velodyne_scan(velodyne_file):
    scan = np.fromfile(velodyne_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.strip().split()])
        # Reshape matrices
        if 'P2' in data:
            data['P2'] = data['P2'].reshape(3, 4)
        if 'R0_rect' in data:
            data['R0_rect'] = data['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in data:
            data['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(3, 4)
        elif 'Tr_velo_to_cam0' in data:
            data['Tr_velo_to_cam'] = data['Tr_velo_to_cam0'].reshape(3, 4)
        return data

def transform_point_cloud(points, calib):
    # points: Nx3 numpy array
    # calib: calibration dictionary
    # We need Tr_velo_to_cam and R0_rect

    # Get transformation matrices
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))  # make it 4x4

    R0_rect = calib['R0_rect']
    R0_rect = np.hstack((R0_rect, np.zeros((3, 1))))
    R0_rect = np.vstack((R0_rect, [0, 0, 0, 1]))  # make it 4x4

    # Transform points
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # Nx4
    points_cam = (R0_rect @ Tr_velo_to_cam @ points_hom.T).T  # Nx4
    return points_cam[:, :3]

def read_label_file(label_file):
    objects = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            obj = {}
            parts = line.strip().split(' ')
            obj['type'] = parts[0]
            obj['truncated'] = float(parts[1])
            obj['occluded'] = int(parts[2])
            obj['alpha'] = float(parts[3])
            obj['bbox'] = [float(x) for x in parts[4:8]]
            obj['dimensions'] = [float(x) for x in parts[8:11]]  # h, w, l
            obj['location'] = [float(x) for x in parts[11:14]]  # x, y, z
            obj['rotation_y'] = float(parts[14])
            objects.append(obj)
    return objects

def create_3d_bbox(obj):
    # obj: dictionary with keys 'dimensions', 'location', 'rotation_y'
    h, w, l = obj['dimensions']
    x, y, z = obj['location']
    ry = obj['rotation_y']

    # Create the 8 corners of the bounding box in object coordinate system
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    # Rotate the corners
    corners = np.array([x_corners, y_corners, z_corners])
    # Rotation matrix around y-axis
    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [0, 1, 0],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners_3d = R @ corners

    # Translate the corners to the location
    corners_3d += np.array([[x], [y], [z]])

    # Return the corners
    return corners_3d.T  # 8x3

def main():
    # Define the paths
    base_dir = Path('/path/to/test_kitti/training')  # Update this path (redacted)
    velodyne_dir = base_dir / 'velodyne'
    calib_dir = base_dir / 'calib'
    label_pred_dir = base_dir / 'label_pseudo'
    label_gt_dir = base_dir / 'label_2'

    # Get list of frames
    frame_indices = [f.stem for f in velodyne_dir.glob('*.bin')]

    seed = 666
    random.seed(seed)
    random.shuffle(frame_indices)

    for idx in frame_indices:
        print(f'Processing frame {idx}')
        # Load point cloud
        velodyne_file = velodyne_dir / f'{idx}.bin'
        calib_file = calib_dir / f'{idx}.txt'
        label_pred_file = label_pred_dir / f'{idx}.txt'
        label_gt_file = label_gt_dir / f'{idx}.txt'

        # Load data
        scan = load_velodyne_scan(velodyne_file)
        calib = read_calib_file(calib_file)
        pred_objects = read_label_file(label_pred_file)
        for obj in pred_objects:
            obj['rotation_y'] = obj['rotation_y'] + np.pi/2.
        gt_objects = read_label_file(label_gt_file)

        # Transform point cloud to camera coordinates
        points = scan[:, :3]
        points_cam = transform_point_cloud(points, calib)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cam)
        pcd.paint_uniform_color([0., 0., 0.])  # Gray color
        geometries = [pcd]

        # Create and add bounding boxes for predicted labels
        for obj in pred_objects:
            corners = create_3d_bbox(obj)
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
            bbox.color = (1, 0, 0)  # Green for predicted
            geometries.append(bbox)

        # Create and add bounding boxes for ground truth labels
        for obj in gt_objects:
            corners = create_3d_bbox(obj)
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
            bbox.color = (0, 1, 0)  # Red for ground truth
            geometries.append(bbox)

        # Visualize the frame
        o3d.visualization.draw_geometries(geometries)

if __name__ == '__main__':
    main()
