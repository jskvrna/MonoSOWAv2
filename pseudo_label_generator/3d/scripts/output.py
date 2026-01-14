from anno_V3 import AutoLabel3D
import numpy as np
import os
import cv2

class Output(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)

    def writetxt_cars(self, cars, pedestrians=None):
        if self.args.dataset == 'waymo':
            if not os.path.exists(self.cfg.paths.labels_path + self.file_name):
                os.makedirs(self.cfg.paths.labels_path + self.file_name + "/")
            self.f_write = open(self.cfg.paths.labels_path + self.file_name + "/" + str(self.pic_index) + '.txt', 'w')
        else:
            self.f_write = open(self.cfg.paths.labels_path + self.file_name + '.txt', 'w')

        for i in range(len(cars)):
            cur_car = cars[i]
            if cur_car.optimized and cur_car.lidar is not None:
                if cur_car.mask is not None:
                    box = self.get_bounding_box(cur_car.mask)
                else:
                    if self.cfg.optimization.skip_non_visible_cars:
                        continue
                    box = np.array([0., 0., 0., 0.])
                
                if hasattr(cur_car, 'score') and cur_car.score is not None:
                    if isinstance(cur_car.score, list):
                        score = cur_car.score[0]
                    else:
                        score = cur_car.score
                else:
                    score = 0.99

                self.f_write.write('Car -1 -1 -10 ')
                for z in range(4):
                    self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
                self.f_write.write(str(f'{cur_car.height:.2f}') + " " + str(f'{cur_car.width:.2f}') + " " + str(
                    f'{cur_car.length:.2f}') + " ")

                self.f_write.write(str(f'{float(cur_car.x):3.2f}') + " ")  # X,Y,Z center
                if self.args.dataset == 'waymo':
                    self.f_write.write(str(f'{float(cur_car.y):3.2f}') + " ")  # X,Y,Z center
                else:
                    self.f_write.write(str(f'{float(cur_car.y + cur_car.height / 2.):3.2f}') + " ")  # X,Y,Z center
                self.f_write.write(str(f'{float(cur_car.z):3.2f}') + " ")  # X,Y,Z center

                yaw = cur_car.theta
                if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                    yaw -= np.pi / 2.
                if yaw > np.pi:
                    yaw -= 2 * np.pi
                elif yaw < -np.pi:
                    yaw += 2 * np.pi

                self.f_write.write(str(f'{float(yaw):3.2f}') + " ")  # yaw
                self.f_write.write(str(f'{float(score):3.2f}') + " ")
                self.f_write.write('\n')

        if pedestrians is not None:
            for i in range(len(pedestrians)):
                cur_ped = pedestrians[i]
                if cur_ped.optimized and cur_ped.lidar is not None:
                    if cur_ped.mask is not None:
                        box = self.get_bounding_box(cur_ped.mask)
                    else:
                        if self.cfg.optimization.skip_non_visible_cars:
                            continue
                        box = np.array([0., 0., 0., 0.])
                    
                    if hasattr(cur_ped, 'score') and cur_ped.score is not None:
                        if isinstance(cur_ped.score, list):
                            score = cur_ped.score[0]
                        else:
                            score = cur_ped.score
                    else:
                        score = 0.99

                    if cur_ped.cyclist:
                        self.f_write.write('Cyclist -1 -1 -10 ')
                    else:
                        self.f_write.write('Pedestrian -1 -1 -10 ')
                    for z in range(4):
                        self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
                    self.f_write.write(str(f'{cur_ped.height:.2f}') + " " + str(f'{cur_ped.width:.2f}') + " " + str(
                        f'{cur_ped.length:.2f}') + " ")

                    self.f_write.write(str(f'{float(cur_ped.x):3.2f}') + " ")  # X,Y,Z center
                    if self.args.dataset == 'waymo':
                        self.f_write.write(str(f'{float(cur_ped.y):3.2f}') + " ")  # X,Y,Z center
                    else:
                        self.f_write.write(str(f'{float(cur_ped.y + cur_ped.height / 2.):3.2f}') + " ")  # X,Y,Z center
                    self.f_write.write(str(f'{float(cur_ped.z):3.2f}') + " ")  # X,Y,Z center

                    yaw = cur_ped.theta
                    if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                        yaw -= np.pi / 2.
                    if yaw > np.pi:
                        yaw -= 2 * np.pi
                    elif yaw < -np.pi:
                        yaw += 2 * np.pi

                    self.f_write.write(str(f'{float(yaw):3.2f}') + " ")  # yaw
                    self.f_write.write(str(f'{float(score):3.2f}') + " ")
                    self.f_write.write('\n')

        # Also write additional objects if available
        if hasattr(self, 'additional_objects') and self.additional_objects is not None:
            for cls_name, objs in self.additional_objects.items():
                for obj in objs:
                    if obj is None:
                        continue
                    bbox = getattr(obj, 'bbox', None)
                    if bbox is None:
                        continue
                    # Try to compute 2D box from mask (optional visibility)
                    if getattr(obj, 'mask', None) is not None:
                        box = self.get_bounding_box(obj.mask)
                    else:
                        if self.cfg.optimization.skip_non_visible_cars:
                            continue
                        box = np.array([0., 0., 0., 0.])

                    # Extract 7-DoF
                    cx = float(bbox.get('x', 0.0))
                    cy = float(bbox.get('y', 0.0))
                    cz = float(bbox.get('z', 0.0))
                    width = float(bbox.get('width', 0.0))
                    length = float(bbox.get('length', 0.0))
                    height = float(bbox.get('height', 0.0))
                    yaw = float(bbox.get('yaw', 0.0))

                    # Dataset-specific y shift (KITTI-style expects bottom-center)
                    y_write = cy if self.args.dataset == 'waymo' else cy + height / 2.0

                    # Yaw convention alignment like cars/peds
                    if self.args.dataset in ['kitti', 'all', 'waymo_converted', 'dsec']:
                        yaw = yaw - np.pi / 2.0
                    if yaw > np.pi:
                        yaw -= 2 * np.pi
                    elif yaw < -np.pi:
                        yaw += 2 * np.pi

                    # Prepare line
                    if hasattr(obj, 'score') and obj.score is not None:
                        if isinstance(obj.score, list):
                            score = obj.score[0]
                        else:
                            score = obj.score
                    else:
                        score = 0.99
                    type_name = str(getattr(obj, 'class_name', cls_name))
                    self.f_write.write(f"{type_name} -1 -1 -10 ")
                    for z in range(4):
                        self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
                    # dims: h, w, l
                    self.f_write.write(str(f'{height:.2f}') + " " + str(f'{width:.2f}') + " " + str(f'{length:.2f}') + " ")
                    # location: x, y, z
                    self.f_write.write(str(f'{cx:3.2f}') + " ")
                    self.f_write.write(str(f'{y_write:3.2f}') + " ")
                    self.f_write.write(str(f'{cz:3.2f}') + " ")
                    # rotation_y
                    self.f_write.write(str(f'{yaw:3.2f}') + " ")
                    # score
                    self.f_write.write(str(f'{score:3.2f}') + " ")
                    self.f_write.write('\n')

        self.f_write.close()

    def transform_kitti_box_to_lidar(self, h, w, l, x, y, z, theta, T_matrix):
        # 1. Transform Position (Center)
        center = np.array([x, y, z, 1.0])
        center_new = T_matrix @ center
        x_new, y_new, z_new = center_new[:3]

        # 2. Transform Orientation
        # Construct rotation matrix for the object in Camera frame (Rotation around Y)
        # We represent the heading as a vector in the X-Z plane.
        # v = [cos(theta), 0, sin(theta)] represents the direction.
        v_heading = np.array([np.cos(theta), 0, np.sin(theta)])
        
        # Apply the rotation part of the transformation matrix
        R_trans = T_matrix[:3, :3]
        v_heading_new = R_trans @ v_heading
        
        # Calculate new yaw angle in LiDAR frame (Rotation around Z-axis)
        # LiDAR frame is usually X-forward, Y-left, Z-up.
        # We compute the angle in the X-Y plane.
        theta_new = np.arctan2(v_heading_new[1], v_heading_new[0])
        
        return h, w, l, x_new, y_new, z_new, theta_new

    def writetxt_cars_dsec(self, cars, pedestrians=None):
        self.f_write = open(self.cfg.paths.labels_path + self.file_name + '.txt', 'w')

        T_cam1_to_velo = np.array(self.cam_to_lidar['T_lidar_camRect1'])

        for i in range(len(cars)):
            cur_car = cars[i]
            if cur_car.optimized and cur_car.lidar is not None:
                if cur_car.mask is not None:
                    box = self.get_bounding_box(cur_car.mask)
                else:
                    if self.cfg.optimization.skip_non_visible_cars:
                        continue
                    box = np.array([0., 0., 0., 0.])
                score = 0.99
                self.f_write.write('Car -1 -1 -10 ')
                for z in range(4):
                    self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
                
                # Prepare inputs for transformation
                yaw = cur_car.theta
                yaw -= np.pi / 2.
                if yaw > np.pi:
                    yaw -= 2 * np.pi
                elif yaw < -np.pi:
                    yaw += 2 * np.pi
                
                h, w, l, x_new, y_new, z_new, theta_new = self.transform_kitti_box_to_lidar(
                    cur_car.height, cur_car.width, cur_car.length, 
                    cur_car.x, cur_car.y, cur_car.z, yaw, T_cam1_to_velo)

                self.f_write.write(str(f'{h:.2f}') + " " + str(f'{w:.2f}') + " " + str(
                    f'{l:.2f}') + " ")

                self.f_write.write(str(f'{float(x_new):3.2f}') + " ")  # X,Y,Z center
                self.f_write.write(str(f'{float(y_new):3.2f}') + " ")  # X,Y,Z center
                self.f_write.write(str(f'{float(z_new):3.2f}') + " ")  # X,Y,Z center

                self.f_write.write(str(f'{float(theta_new):3.2f}') + " ")  # yaw
                self.f_write.write(str(f'{float(score):3.2f}') + " ")
                self.f_write.write('\n')

        if pedestrians is not None:
            for i in range(len(pedestrians)):
                cur_ped = pedestrians[i]
                if cur_ped.optimized and cur_ped.lidar is not None:
                    if cur_ped.mask is not None:
                        box = self.get_bounding_box(cur_ped.mask)
                    else:
                        if self.cfg.optimization.skip_non_visible_cars:
                            continue
                        box = np.array([0., 0., 0., 0.])
                    score = 0.99
                    if cur_ped.cyclist:
                        self.f_write.write('Cyclist -1 -1 -10 ')
                    else:
                        self.f_write.write('Pedestrian -1 -1 -10 ')
                    for z in range(4):
                        self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
                    
                    # Prepare inputs for transformation
                    yaw = cur_ped.theta
                    yaw -= np.pi / 2.
                    if yaw > np.pi:
                        yaw -= 2 * np.pi
                    elif yaw < -np.pi:
                        yaw += 2 * np.pi

                    h, w, l, x_new, y_new, z_new, theta_new = self.transform_kitti_box_to_lidar(
                        cur_ped.height, cur_ped.width, cur_ped.length, 
                        cur_ped.x, cur_ped.y, cur_ped.z, yaw, T_cam1_to_velo)

                    self.f_write.write(str(f'{h:.2f}') + " " + str(f'{w:.2f}') + " " + str(
                        f'{l:.2f}') + " ")

                    self.f_write.write(str(f'{float(x_new):3.2f}') + " ")  # X,Y,Z center
                    self.f_write.write(str(f'{float(y_new):3.2f}') + " ")  # X,Y,Z center
                    self.f_write.write(str(f'{float(z_new):3.2f}') + " ")  # X,Y,Z center

                    self.f_write.write(str(f'{float(theta_new):3.2f}') + " ")  # yaw
                    self.f_write.write(str(f'{float(score):3.2f}') + " ")
                    self.f_write.write('\n')

        self.f_write.close()

    def writetxt_dimensions_cars(self, cars):
        self.f_write = open(self.cfg.paths.dimensions_path + self.file_name + '.txt', 'w')

        for i in range(len(cars)):
            cur_car = cars[i]
            box = np.array([0., 0., 0., 0.])
            score = 0.99
            self.f_write.write('Car -1 -1 -10 ')
            for z in range(4):
                self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
            self.f_write.write(str(f'{cur_car.height:.2f}') + " " + str(f'{cur_car.width:.2f}') + " " + str(
                f'{cur_car.length:.2f}') + " ")

            self.f_write.write(str(f'{float(cur_car.x):3.2f}') + " ")  # X,Y,Z center
            if self.args.dataset == 'waymo':
                self.f_write.write(str(f'{float(cur_car.y):3.2f}') + " ")  # X,Y,Z center
            else:
                self.f_write.write(str(f'{float(cur_car.y + cur_car.height / 2.):3.2f}') + " ")  # X,Y,Z center
            self.f_write.write(str(f'{float(cur_car.z):3.2f}') + " ")  # X,Y,Z center

            yaw = cur_car.theta
            if self.args.dataset == 'kitti' or self.args.dataset == 'dsec':
                yaw -= np.pi / 2.
            if yaw > np.pi:
                yaw -= 2 * np.pi
            elif yaw < -np.pi:
                yaw += 2 * np.pi

            self.f_write.write(str(f'{float(yaw):3.2f}') + " ")  # yaw
            self.f_write.write(str(f'{float(score):3.2f}') + " ")
            self.f_write.write('\n')

        self.f_write.close()

    def prepare_dirs(self):
        if not os.path.exists(self.cfg.paths.merged_frames_path):
            os.makedirs(self.cfg.paths.merged_frames_path)
        if not os.path.exists(self.cfg.paths.merged_frames_path + "optimized_cars/"):
            os.makedirs(self.cfg.paths.merged_frames_path + "optimized_cars/")
        if not os.path.exists(self.cfg.paths.merged_frames_path + "transformations/"):
            os.makedirs(self.cfg.paths.merged_frames_path + "transformations/")
        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_lidar/"):
            os.makedirs(self.cfg.paths.merged_frames_path + "candidates_lidar/")
        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_masks/"):
            os.makedirs(self.cfg.paths.merged_frames_path + "candidates_masks/")
        if self.args.dataset == 'waymo' or self.args.dataset == 'waymo_converted':
            if not os.path.exists(self.cfg.paths.merged_frames_path + "lidar_raw/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "lidar_raw/")
        if self.cfg.frames_creation.extract_pedestrians:
            if not os.path.exists(self.cfg.paths.merged_frames_path + "pedestrians/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "pedestrians/")
        if self.cfg.frames_creation.tracker_for_merging == '2D':
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/"):
                    os.makedirs(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/")
            else:
                if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_2DTrack/"):
                    os.makedirs(self.cfg.paths.merged_frames_path + "cars_2DTrack/")
        else:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/"):
                    os.makedirs(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/")
            else:
                if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_3DTrack/"):
                    os.makedirs(self.cfg.paths.merged_frames_path + "cars_3DTrack/")
        if self.cfg.frames_creation.tracker_for_merging == '2D':
            if self.args.dataset == 'waymo':
                if not os.path.exists(self.cfg.paths.merged_frames_path + "homographies/"):
                    os.makedirs(self.cfg.paths.merged_frames_path + "homographies/")
            if not os.path.exists(self.cfg.paths.merged_frames_path + "detandtrackedV2/"): #TODO change to detandtracked
                os.makedirs(self.cfg.paths.merged_frames_path + "detandtrackedV2/")
        else:
            if not os.path.exists(self.cfg.paths.merged_frames_path + "masks_raw/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "masks_raw/")
        if self.cfg.frames_creation.use_gt_masks:
            if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_ids/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "candidates_ids/")

    def tensor_to_numpy(self, tensor):
        # Convert a 3xHxW PyTorch tensor to a HxWx3 numpy array
        # Rearrange from CxHxW to HxWxC
        numpy_image = tensor.permute(1, 2, 0).cpu().numpy()
        # Convert from float to uint8
        numpy_image = numpy_image.astype(np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return numpy_image

    def write_to_video(self, imgs):
        for i in range(4):
            name = 'output_video_' + str(i) + '.mp4'
            fps = 10

            # Assume all images are the same size
            height, width, _ = self.tensor_to_numpy(imgs[0][i]).shape

            # Create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            video = cv2.VideoWriter(name, fourcc, fps, (width, height))

            for z in range(len(imgs)):
                numpy_image = self.tensor_to_numpy(imgs[z][i])
                video.write(numpy_image)

            video.release()

    def get_bounding_box(self, mask):
        # Find the indices of the True values
        true_indices = np.argwhere(mask)

        # Determine the minimum and maximum indices for both dimensions
        min_row, min_col = true_indices.min(axis=0)
        max_row, max_col = true_indices.max(axis=0)

        # Define the bounding box
        bounding_box = np.array([min_row, min_col, max_row, max_col])

        return bounding_box




