import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from tqdm import tqdm
import os
import cv2
import numpy as np
from shapely.geometry import Polygon

#dataset_folder = '/path/to/waymo_data/'
dataset_folder = '/path/to/waymo_data/'
training_folder = os.path.join(dataset_folder, 'training')
validation_folder = os.path.join(dataset_folder, 'validation')

output_dir = '/path/to/waymo_to_kitti/'
#output_dir = '/path/to/waymo_to_kitti/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class Label_3D:
    def __init__(self, label):
        self.x = label.box.center_x
        self.y = label.box.center_y
        self.z = label.box.center_z

        self.width = label.box.width
        self.length = label.box.length
        self.height = label.box.height

        self.heading = label.box.heading

        self.id = label.id

        self.left_u = -1
        self.left_v = -1
        self.right_u = -1
        self.right_v = -1

    def project_to_2D(self, calib, img_size, extrinsic):
        # Define the box corners relative to the center
        # According to a common convention:
        # length is along Z, width along X, and height along Y.
        if self.x < 0:
            return
        else:
            x_corners = [self.length / 2, self.length / 2, -self.length / 2, -self.length / 2, self.length / 2, self.length / 2, -self.length / 2, -self.length / 2]
            y_corners = [self.width / 2, self.width / 2, self.width / 2, self.width / 2, -self.width / 2, -self.width / 2, -self.width / 2, -self.width / 2]
            z_corners = [self.height / 2, -self.height / 2, -self.height / 2, self.height / 2, self.height / 2, -self.height / 2, -self.height / 2, self.height / 2]

            # Create array of corners
            corners_3D = np.vstack((x_corners, y_corners, z_corners))

            # Rotation around Y-axis
            R = np.array([
                [np.cos(self.heading), np.sin(self.heading), 0.],
                [-np.sin(self.heading), np.cos(self.heading), 0.],
                [0., 0., 1.]
            ])

            # Rotate and then translate
            corners_3D = R @ corners_3D
            corners_3D[0, :] += self.x
            corners_3D[1, :] += self.y
            corners_3D[2, :] += self.z

            corners_3D_hom = np.vstack((corners_3D, np.ones((1, corners_3D.shape[1]))))
            corners_3D = (np.linalg.inv(extrinsic) @ corners_3D_hom)[:3, :]

            # Rotate to typical camera orientation
            R = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])
            corners_3D = R @ corners_3D

            # Project into image plane
            # Convert to homogeneous coordinates for projection
            corners_3D_hom = np.vstack((corners_3D, np.ones((1, corners_3D.shape[1]))))
            corners_2D = calib @ corners_3D_hom

            # Normalize by the third row
            corners_2D[0, :] /= corners_2D[2, :]
            corners_2D[1, :] /= corners_2D[2, :]

            # corners_2D now contains [u,v,1] for each corner
            u_values = corners_2D[0, :]
            v_values = corners_2D[1, :]

            # Get min/max in image coordinates
            xmin, xmax = np.min(u_values), np.max(u_values)
            ymin, ymax = np.min(v_values), np.max(v_values)

            # Check if the box is in the image
            if xmin < 0 and ymin < 0 and xmax > img_size[1] and ymax > img_size[0]:
                self.left_u = -1
                self.left_v = -1
                self.right_u = -1
                self.right_v = -1
                return
            else:
                xmin = np.clip(xmin, 0, img_size[1] - 1)
                ymin = np.clip(ymin, 0, img_size[0] - 1)
                xmax = np.clip(xmax, 0, img_size[1] - 1)
                ymax = np.clip(ymax, 0, img_size[0] - 1)

                self.left_u = xmin
                self.left_v = ymin
                self.right_u = xmax
                self.right_v = ymax
                return


class Label_2D:
    def __init__(self, label):
        self.left_u = label.box.center_x - label.box.length / 2.
        self.left_v = label.box.center_y - label.box.width / 2.

        self.right_u = label.box.center_x + label.box.length / 2.
        self.right_v = label.box.center_y + label.box.width / 2.

        self.id = label.id

        self.x = None
        self.y = None
        self.z = None

        self.width = None
        self.length = None
        self.height = None

        self.heading = None

        self.correspondence = None

def compute_iou(label_2D, label_3D):
    # Check for invalid 3D label coordinates
    if label_3D.left_u == -1 or label_3D.left_v == -1 or label_3D.right_u == -1 or label_3D.right_v == -1:
        return 0.0

    box_1 = [(label_2D.left_u, label_2D.left_v), (label_2D.right_u, label_2D.left_v), (label_2D.right_u, label_2D.right_v), (label_2D.left_u, label_2D.right_v)]
    box_2 = [(label_3D.left_u, label_3D.left_v), (label_3D.right_u, label_3D.left_v), (label_3D.right_u, label_3D.right_v), (label_3D.left_u, label_3D.right_v)]

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def visu_labels(image, labels):
    for label in labels:
        if label.left_u == -1:
            continue
        cv2.rectangle(image, (int(label.left_u), int(label.left_v)), (int(label.right_u), int(label.right_v)), (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)

#First lets do the training
all_files_in_folder = os.listdir(training_folder)
all_files_in_folder = sorted(all_files_in_folder)

loop = tqdm(all_files_in_folder)
for file_name in loop:
    loop.set_description(file_name)
    dataset = tf.data.TFRecordDataset(os.path.join(training_folder, file_name), compression_type='')
    if not os.path.exists(os.path.join(output_dir, file_name)):
        os.makedirs(os.path.join(output_dir, file_name))
        os.makedirs(os.path.join(output_dir, file_name, 'label_2'))
        os.makedirs(os.path.join(output_dir, file_name, 'image_2'))
        os.makedirs(os.path.join(output_dir, file_name, 'velodyne'))
        os.makedirs(os.path.join(output_dir, file_name, 'calib'))

    for i, data in enumerate(dataset):
        #if i != 31:
        #    continue
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        image_size = None
        # First, lets do the image
        for index, raw_image in enumerate(frame.images):
            if index == 0:
                decoded_image = tf.image.decode_jpeg(raw_image.image)
                image_size = decoded_image.shape[:2]
                path_to_save = os.path.join(output_dir, file_name, 'image_2', str(i).zfill(10) + '.png')
                cv2.imwrite(path_to_save, cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR))

        # Now lets do the calib
        calibration = [cc for cc in frame.context.camera_calibrations][0]
        extrinsic = list(calibration.extrinsic.transform)
        intrinsic = list(calibration.intrinsic)

        calib_matrix = np.eye(4)
        calib_matrix[0, 0] = intrinsic[0]
        calib_matrix[1, 1] = intrinsic[1]
        calib_matrix[0, 2] = intrinsic[2]
        calib_matrix[1, 2] = intrinsic[3]
        calib = calib_matrix.copy()
        calib_matrix = calib_matrix[:3, :4]
        calib_matrix = calib_matrix.flatten().tolist()
        P2 = [str(c) for c in calib_matrix]

        R0_rect = np.eye(3).flatten().tolist()

        R = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        Tr_velo_to_cam = np.zeros((4,4))
        Tr_velo_to_cam[0, 1] = -1.
        Tr_velo_to_cam[1, 2] = -1.
        Tr_velo_to_cam[2, 0] = 1.
        Tr_velo_to_cam = Tr_velo_to_cam[:3, :4].flatten().tolist()

        Tr_imu_to_velo = np.linalg.inv(np.array(extrinsic).reshape(4, 4))[:3, :4].flatten().tolist()

        P2 = [str(x) for x in P2]
        R0_rect = [str(x) for x in R0_rect]
        Tr_velo_to_cam = [str(x) for x in Tr_velo_to_cam]
        Tr_imu_to_velo = [str(x) for x in Tr_imu_to_velo]

        with open(os.path.join(output_dir, file_name, 'calib', str(i).zfill(10) + '.txt'), 'w') as f:
            f.write('P0: ' + ' '.join(P2) + '\n')
            f.write('P1: ' + ' '.join(P2) + '\n')
            f.write('P2: ' + ' '.join(P2) + '\n')
            f.write('P3: ' + ' '.join(P2) + '\n')
            f.write('R0_rect: ' + ' '.join(R0_rect) + '\n')
            f.write('Tr_velo_to_cam: ' + ' '.join(Tr_velo_to_cam) + '\n')
            f.write('Tr_imu_to_velo: ' + ' '.join(Tr_imu_to_velo) + '\n')

        # Now lets do labels
        all_3d_labels = []
        for label_3d in frame.laser_labels:
            # Only if it is a vehicle
            if label_3d.type == 1:
                cur_label = Label_3D(label_3d)
                cur_label.project_to_2D(calib, image_size, np.array(extrinsic).reshape(4, 4))
                #print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                all_3d_labels.append(cur_label)

        # Now lets find 2D labels
        #print("2D labels_now")
        all_2d_labels = []
        for index, image_labels in enumerate(frame.camera_labels):
            if index == 0:
                for image_label in image_labels.labels:
                    if image_label.type == 1:
                        cur_label = Label_2D(image_label)
                        #print(cur_label.left_u, cur_label.left_v, cur_label.right_u, cur_label.right_v)
                        all_2d_labels.append(cur_label)

        #visu_labels(cv2.imread(path_to_save), all_3d_labels)

        # Now lets create correspondence between 2D and 3D labels
        ious = np.zeros((len(all_2d_labels), len(all_3d_labels)))
        for index_3d, label_3d in enumerate(all_3d_labels):
            for index_2d, label_2d in enumerate(all_2d_labels):
                iou = compute_iou(label_2d, label_3d)
                ious[index_2d, index_3d] = iou
        for index_3d, label_3d in enumerate(all_3d_labels):
            best_candidate = np.argmax(ious[:, index_3d])
            best_candidate_for_best_candidate = np.argmax(ious[best_candidate, :])
            if best_candidate_for_best_candidate == index_3d:
                label_to_modify = all_2d_labels[best_candidate]
                center = np.array([[label_3d.x, label_3d.y, label_3d.z, 1.]]).transpose()
                center = np.linalg.inv(np.array(extrinsic).reshape(4, 4)) @ center
                R = np.array([
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0]
                ])
                center = R @ center[:3, :]
                label_to_modify.x = center[0]
                label_to_modify.y = center[1]
                label_to_modify.z = center[2]
                label_to_modify.width = label_3d.width
                label_to_modify.length = label_3d.length
                label_to_modify.height = label_3d.height
                label_to_modify.heading = label_3d.heading
                label_to_modify.correspondence = label_3d.id
                all_2d_labels[best_candidate] = label_to_modify

        with open(os.path.join(output_dir, file_name, 'label_2', str(i).zfill(10) + '.txt'), 'w') as f:
            for label in all_2d_labels:
                if label.correspondence is not None:
                    f.write("Car -1 -1 -10 ")
                    f.write(f"{float(label.left_u.item() if hasattr(label.left_u, 'item') else label.left_u):.2f} ")
                    f.write(f"{float(label.left_v.item() if hasattr(label.left_v, 'item') else label.left_v):.2f} ")
                    f.write(f"{float(label.right_u.item() if hasattr(label.right_u, 'item') else label.right_u):.2f} ")
                    f.write(f"{float(label.right_v.item() if hasattr(label.right_v, 'item') else label.right_v):.2f} ")
                    f.write(f"{float(label.width.item() if hasattr(label.width, 'item') else label.width):.2f} ")
                    f.write(f"{float(label.height.item() if hasattr(label.height, 'item') else label.height):.2f} ")
                    f.write(f"{float(label.length.item() if hasattr(label.length, 'item') else label.length):.2f} ")
                    f.write(f"{float(label.x.item() if hasattr(label.x, 'item') else label.x):.2f} ")
                    f.write(f"{float(label.y.item() + label.height.item() / 2. if hasattr(label.y, 'item') else label.y):.2f} ")
                    f.write(f"{float(label.z.item() if hasattr(label.z, 'item') else label.z):.2f} ")
                    f.write(f"{float(label.heading.item() if hasattr(label.heading, 'item') else label.heading):.2f}\n")

