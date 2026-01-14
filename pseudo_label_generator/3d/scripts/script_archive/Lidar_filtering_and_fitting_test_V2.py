import pyemd
import torch
from utils import load_velo_scan, read_car_mesh, read_calib_file, get_perfect_scale
from detectron2.utils.logger import setup_logger
import numpy as np
import cv2, random, sys
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
import glob, os
import time
from scipy.spatial.transform import Rotation as R
import scipy
from dist_compute import avg_med_distance, avg_med_distance_only_temp_to_scan, avg_trim_distance
from pyntcloud import PyntCloud

import open3d
import matplotlib.pyplot as plt
setup_logger()


class AutoLabel3D:

    def __init__(self):
        # Choose If we want to show the image and the 3D output.
        self.take_random = False #Shuffles the images, primarly debug purposes
        self.cluster = False #If cluster is true, then it changes the paths
        self.visu_img = False #If True, it shows the current image processed with 2D bboxes
        self.visu_3D = True #If True, it shows the lidar points, fitted templates and the GT bboxes and estimated bboxs
        self.visu_histo = False #If you want to visualize the histogram computed
        self.write_txt = False #If True, it writes the lables to the .txt files
        self.show_loss = False #If True, it tries to visualize the loss during the optimization on different locations
        self.iterate = False #If True, it does not compute all the fits and then show, instead it shows the 3D every time it fits something

        self.show_loaded_pclouds = False #If true it shows the histograms and also the pointclouds of the loaded templates
        self.show_whole_template = False #If True it shows in 3D visu the whole templates not the filtered one

        self.use_histogram = False #If we want to use histogram during optimization
        self.histo_nbins = 360
        self.histo_iteryaw = 360
        self.histo_random_point_scale = 100

        self.histo_use_L2 = True
        self.histo_use_L1 = False
        self.histo_use_bhat = False
        self.histo_use_emd = False

        self.location_niter = 100

        self.filter_by_mask_during_optimize = False #If true during the computation of the loss we only take points filtered by mask
        self.use_whole_template_during_optim = False #If true we optimize with the whole template, not just filtered templates
        self.use_diff_template = False #If true we take the original template in V1

        self.shift_by_mean = False #We shift the template to the mean of filtered scan points
        self.shift_by_median = True #We shift the template to the median of filtered scan points - the best
        self.shift_smart = False #We shift the template to the somewhat median of the filtered scan points

        self.loss_by_trim_mean = False #Instead of median it uses trimmed mean during optimize
        self.loss_by_median_only_to_scan = False #For each point in scan we compute NN in template and take median from this
        self.loss_by_median = True #For each point in scan we compute NN in template and take median from this and vice versa from template to scan
        self.loss_by_occlusion = False #During optimization it also takes the occlusion of points as a loss

        self.use_alternative_dimensions = False #If we want to use different template size than the kitti avg car size
        self.height_kittiavg = 1.52608343 #GOLF MK5
        self.width_kittiavg = 1.78
        self.length_kittiavg = 4.25

        self.filter_diameter = 4 #In meters, all points which are behind this diameter will be filtered
        self.trim_treshold = 0.3 #Threshold for the trimmed mean
        self.min_dist_threshold = 0 #Threshold for the distance for occlusion - if point is min_dist_threshold behind it is wrong

        self.x_range_min = -2 #Region over which we optimalize
        self.x_range_max = 2
        self.z_range_min = -2
        self.z_range_max = 2

        self.filt_template_threshold_optim = 10 #If we filter out the threshold and it has lower number of points than this threshold we continue
        self.lidar_threshold = 10 #Minimum number of points after filtering the scan with the mask
        self.angle_per_pcd = 1. #Step in angle for the filtered templates
        self.score_detectron_thresh = 0.7 #Minimum score for the detection of detectron, if the score is lower, we ignore that mask
        self.rng = np.random.default_rng(12345) #Seed for the take_random
        self.device = 'cpu' #Specify the device on which the detectron2 should run

        if self.cluster:
            self.pics = glob.glob('/path/to/KITTI/object_detection/training/image_2/*.png')  # (redacted)
            self.pics = sorted(self.pics, key=os.path.basename)
            self.detectron_config = "/path/to/detectron2/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py"  # (redacted)
            self.lidar_path = "/path/to/KITTI/object_detection/training/velodyne/"  # (redacted)
            self.calib_path = "/path/to/KITTI/object_detection/training/calib/"  # (redacted)
            self.label_path = "/path/to/KITTI/object_detection/training/label_2/"  # (redacted)
            self.output_dir = "/path/to/output/labels/"  # (redacted)

        else:
                self.pics = glob.glob('/path/to/data/kitti/training/image_2/*.png')  # (redacted)
            self.pics = sorted(self.pics, key=os.path.basename)
            self.detectron_config = "/path/to/detectron2/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py"  # (redacted)
                self.lidar_path = "/path/to/data/kitti/training/velodyne/"  # (redacted)
                self.calib_path = "/path/to/data/kitti/training/calib/"  # (redacted)
                self.label_path = "/path/to/data/kitti/training/label_2/"  # (redacted)
                self.output_dir = "/path/to/output/labels/"  # (redacted)


        self.loss_matrix = None
        self.loss_matrix_outliers = None
        self.P2_rect = None
        self.lidar_car_template_non_filt = None
        self.img_dist = None
        self.histogram = None
        self.min_dist = None
        self.opt_values = None
        self.filter_idx = None
        self.filt_success = None
        self.z_mean_lidar = None
        self.y_mean_lidar = None
        self.x_mean_lidar = None
        self.bboxes = None
        self.lidar_visu = None
        self.color = None
        self.filtered_lidar = None
        self.mask = None
        self.lidar = None
        self.out_data = None
        self.pic_index = None
        self.img_orig = None
        self.img = None
        self.file_name = None
        self.pic = None
        self.GT_yaw = None
        self.lidar_car_template = None
        self.template_histograms = None

    def load_and_init_detectron_lazy(self):
        # Load the lazyconfig, we are using this, not the classic, because this regnety learned with different idea is used
        cfg = LazyConfig.load(self.detectron_config)
        # cfg = LazyConfig.load("C:/path/to/detectron2/model_zoo/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py")  # (redacted)
        cfg.train.init_checkpoint = 'https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ/42045954/model_final_ef3a80.pkl'  # replace with the path were you have your model

        # Init the model
        model = instantiate(cfg.model)
        if self.device == 'gpu':
            model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

        model.eval()

        self.model = model

    def shift_lidar_template(self, lidar_car_template):
        # Shift the model
        x_temp_mean = np.mean(lidar_car_template[:, 0])
        y_temp_mean = np.mean(lidar_car_template[:, 1])
        z_temp_mean = np.mean(lidar_car_template[:, 2])

        # Shift it
        lidar_car_template[:, 0] -= x_temp_mean
        lidar_car_template[:, 1] -= y_temp_mean
        lidar_car_template[:, 2] -= z_temp_mean

        return lidar_car_template

    def load_and_prepare_lidar_scan(self, filename, img):
        # First get all the lidar points
        lidar = np.array(load_velo_scan(self.lidar_path + filename + '.bin'))
        # Now we need homogenous coordinates and we do not care about the reflections
        lidar[:, 3] = 1
        # Transform to the camera coordinate
        lidar = lidar.transpose()

        calib_data = read_calib_file(self.calib_path + filename + '.txt')
        # calib_data = read_calib_file('/path/to/Data_object_det/training/calib/' + filename + '.txt')  # (redacted)

        T_velo_to_cam = np.eye(4)
        T_velo_to_cam[:3, :4] = np.reshape(calib_data["Tr_velo_to_cam"], (3, 4))
        lidar = np.matmul(T_velo_to_cam, lidar)

        # Delete all points which are not in front of the camera
        mask = lidar[2, :] > 0.
        lidar = lidar[:, mask]

        R0_rect = np.eye(4)
        R0_rect[:3, :3] = np.reshape(calib_data["R0_rect"], (3, 3))

        P2 = np.eye(4)
        P2[:3, :4] = np.reshape(calib_data["P2"], (3, 4))

        P2_rect = np.matmul(P2, R0_rect)
        self.P2_rect = P2_rect

        proj_lidar = np.matmul(P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)

        # Filter lidar data based on their projection to the camera: if they actually fit?
        mask_xmin = lidar[4, :] >= 0.
        lidar = lidar[:, mask_xmin]
        mask_xmax = lidar[4, :] <= img.shape[2]  # img width
        lidar = lidar[:, mask_xmax]
        mask_ymin = lidar[5, :] >= 0.
        lidar = lidar[:, mask_ymin]
        mask_ymax = lidar[5, :] <= img.shape[1]  # img height
        lidar = lidar[:, mask_ymax]

        # Because we are thinking about pixels, we need to floor it, to use it in np.argwhere
        lidar[4:6, :] = np.floor(lidar[4:6, :])

        img_dist = -np.ones((img.shape[2], img.shape[1]))

        for i in range(lidar.shape[1]):
            img_dist[lidar[4, i].astype(int), lidar[5, i].astype(int)] = np.maximum(
                img_dist[lidar[4, i].astype(int), lidar[5, i].astype(int)], lidar[0, i] ** 2 + lidar[2, i] ** 2)

        #We need to add dilatation, because the lidar points are too sparse on camera
        footprint = np.ones((5,5))
        img_dist = scipy.ndimage.grey_dilation(img_dist, footprint=footprint)
        self.img_dist = img_dist

        self.lidar = lidar

    def filter_by_mask_and_shift(self):
        # Now, get indexes of the points which project into the mask
        tmp1 = np.argwhere(self.mask[self.lidar[4, :].astype(int), self.lidar[5, :].astype(int)])

        # Now, filter the points based on the indexes
        filtered_lidar = np.array([self.lidar[0, tmp1], self.lidar[1, tmp1], self.lidar[2, tmp1]]).transpose()[0]
        
        # Sometimes, we just lack the number of points, so if it is small, just skip it
        if filtered_lidar.shape[0] < self.lidar_threshold:
            self.filt_success = False
            return

        if self.shift_by_mean:
            # Filtered data coordinates: [0] left/right, [1] up and down, [2] far away/close,
            # Now we are in the fitting phase, we want to compute mean of the points so we know here approximately we should optimize
            x_mean = np.mean(filtered_lidar[:, 0])
            y_mean = np.mean(filtered_lidar[:, 1])
            z_mean = np.mean(filtered_lidar[:, 2])

        elif self.shift_by_median:
            # The median "mean" works better
            x_mean = np.median(filtered_lidar[:, 0])
            y_mean = np.median(filtered_lidar[:, 1])
            z_mean = np.median(filtered_lidar[:, 2])

        elif self.shift_smart:
            # The median "mean" works better
            x_mean = np.quantile(filtered_lidar[:, 0], 0.25)
            if x_mean < 0:
                x_mean = np.quantile(filtered_lidar[:, 0], 0.75)
            # y_mean = np.median(filtered_lidar[:, 1])
            z_mean = np.quantile(filtered_lidar[:, 2], 0.25)

            # Hopefully better way. Get the highest point and divide.
            high = np.min(filtered_lidar[:, 1])
            y_mean = high + 1.52608343 / 2

        #Save all the values
        self.filtered_lidar = filtered_lidar
        self.x_mean_lidar = x_mean
        self.y_mean_lidar = y_mean
        self.z_mean_lidar = z_mean
        self.filt_success = True
        self.filter_idx = tmp1

        return

    def filter_template_by_mask(self, template):
        tmp = np.ones((template.shape[0], 4))
        tmp[:, :3] = template
        template = tmp
        # Transform to the camera coordinate
        lidar = template.T

        calib_data = read_calib_file(self.calib_path + self.file_name + '.txt')

        # Delete all points which are not in front of the camera
        mask = lidar[2, :] > 0.
        lidar = lidar[:, mask]

        R0_rect = np.eye(4)
        R0_rect[:3, :3] = np.reshape(calib_data["R0_rect"], (3, 3))

        P2 = np.eye(4)
        P2[:3, :4] = np.reshape(calib_data["P2"], (3, 4))

        P2_rect = np.matmul(P2, R0_rect)
        self.P2_rect = P2_rect

        proj_lidar = np.matmul(P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)

        # Filter lidar data based on their projection to the camera: if they actually fit?
        mask_xmin = lidar[4, :] >= 0.
        lidar = lidar[:, mask_xmin]
        mask_xmax = lidar[4, :] <= self.img.shape[2]  # img width
        lidar = lidar[:, mask_xmax]
        mask_ymin = lidar[5, :] >= 0.
        lidar = lidar[:, mask_ymin]
        mask_ymax = lidar[5, :] <= self.img.shape[1]  # img height
        lidar = lidar[:, mask_ymax]

        # Because we are thinking about pixels, we need to floor it, to use it in np.argwhere
        lidar[4:6, :] = np.floor(lidar[4:6, :])

        # Now, get indexes of the points which project into the mask
        tmp1 = np.argwhere(self.mask[lidar[4, :].astype(int), lidar[5, :].astype(int)])

        # Now, filter the points based on the indexes
        filtered_lidar = np.array([lidar[0, tmp1], lidar[1, tmp1], lidar[2, tmp1]]).transpose()[0]

        return filtered_lidar

    def visu2Dbbox(self, img, box, score):
        img_tmp = img.copy()
        # box = np.array(out_data.pred_boxes[i].tensor[0])
        cv2.rectangle(img_tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
        cv2.putText(img_tmp, f'{score:.2f}', (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 1)

        cv2.imshow("", img_tmp)
        k = cv2.waitKey(3000)  # 0==wait forever

    def compute_index(self, x, z, yaw):
        angle = np.arctan2(z, x) - np.pi/2   # Shift, because for a car in front of us we want 0 rotation
        angle = angle + yaw #Add the yaw
        if angle < 0.:
            angle += 2 * np.pi
        elif angle > 2 * np.pi:
            angle -= 2 * np.pi
        template_index = round(np.rad2deg(angle) / self.angle_per_pcd, 0)
        if template_index > (360/self.angle_per_pcd) - 1:
            template_index = 0
        return int(template_index)

    def loss_template_points_behind(self, x, z, theta):
        # Function that computes loss based on the points which are behind our template.
        # First prepare the template

        template_it = np.copy(self.lidar_car_template[
                                  self.compute_index(self.x_mean_lidar + x, self.z_mean_lidar + z, theta)])

        angle_rot = -np.arctan2(self.z_mean_lidar + z,
                               self.x_mean_lidar + x) + np.pi / 2  # Shift, because for a car in front of us we want 0 rotation

        r = R.from_euler('zyx', [0, angle_rot, 0], degrees=False)
        template_it = np.matmul(r.as_matrix(), template_it.transpose()).transpose()

        template_it[:, 0] += x + self.x_mean_lidar
        template_it[:, 1] += self.y_mean_lidar
        template_it[:, 2] += z + self.z_mean_lidar

        # Now we need to do projection
        tmp = np.ones((template_it.shape[0], 4))
        tmp[:, :3] = template_it
        template = tmp
        # Transform to the camera coordinate
        lidar = template.T

        proj_lidar = np.matmul(self.P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)

        # Filter lidar data based on their projection to the camera: if they actually fit?
        mask_xmin = lidar[4, :] >= 0.
        lidar = lidar[:, mask_xmin]
        mask_xmax = lidar[4, :] <= self.img.shape[2]  # img width
        lidar = lidar[:, mask_xmax]
        mask_ymin = lidar[5, :] >= 0.
        lidar = lidar[:, mask_ymin]
        mask_ymax = lidar[5, :] <= self.img.shape[1]  # img height
        lidar = lidar[:, mask_ymax]

        # Because we are thinking about pixels, we need to floor it, to use it in np.argwhere
        lidar[4:6, :] = np.floor(lidar[4:6, :])
        dists = lidar[0, :] ** 2 + lidar[2, :] ** 2
        tmp = self.img_dist[lidar[4, :].astype(int), lidar[5, :].astype(int)]
        wrong_points = np.argwhere(tmp > dists + self.min_dist_threshold).shape[0]

        number_of_points = lidar.shape[1] + 1
        return wrong_points/number_of_points

    def optimize(self):
        if self.show_loss: #Create variables for the loss visualization
            self.loss_matrix = np.zeros((20,20,20,4))
            x_ind = 0
            z_ind = 0
            theta_ind = 0
        min_dist = np.inf
        opt_values = np.array([0., 0., 0., 0.])
        for x in np.linspace(self.x_range_min, self.x_range_max, num=20):
            if self.show_loss:
                z_ind = 0
            for z in np.linspace(self.z_range_min, self.z_range_max, num=20):
                if self.show_loss:
                    theta_ind = 0
                for theta in np.linspace(0, 2 * np.pi - (2 * np.pi/20), num=20):
                    if self.use_whole_template_during_optim: #Take the whole template and rotate it
                        template_it = self.lidar_car_template_non_filt.copy()

                        r = R.from_euler('zyx', [0, theta, 0], degrees=False)
                        template_it = np.matmul(r.as_matrix(), template_it.transpose()).transpose()
                    else: #Take the filtered templates and rotate them
                        template_it = np.copy(self.lidar_car_template[
                                                  self.compute_index(self.x_mean_lidar + x, self.z_mean_lidar + z, theta)])

                        angle_rot = -np.arctan2(self.z_mean_lidar + z,
                                               self.x_mean_lidar + x) + np.pi/2  # Shift, because for a car in front of us we want 0 rotation

                        r = R.from_euler('zyx', [0, angle_rot, 0], degrees=False)
                        template_it = np.matmul(r.as_matrix(), template_it.transpose()).transpose()

                    template_it[:, 0] += x + self.x_mean_lidar
                    template_it[:, 1] += self.y_mean_lidar
                    template_it[:, 2] += z + self.z_mean_lidar

                    if self.filter_by_mask_during_optimize:
                        template_it = self.filter_template_by_mask(template_it)
                        if template_it.shape[0] <= self.filt_template_threshold_optim: #If the template has only few points first it is not very heplpful to optimize over it and also it creates NAN in trim mean.
                            if self.show_loss:
                                self.loss_matrix[x_ind, z_ind, theta_ind, 0] = np.nan
                                self.loss_matrix[x_ind, z_ind, theta_ind, 1] = np.nan
                                self.loss_matrix[x_ind, z_ind, theta_ind, 2] = np.nan
                                self.loss_matrix[x_ind, z_ind, theta_ind, 3] = np.nan
                                theta_ind += 1
                            continue

                    if self.loss_by_trim_mean:
                        dist = avg_trim_distance(self.filtered_lidar, template_it, self.trim_treshold)
                    elif self.loss_by_median_only_to_scan:
                        dist = avg_med_distance_only_temp_to_scan(self.filtered_lidar, template_it)
                    elif self.loss_by_median:
                        dist = avg_med_distance(self.filtered_lidar, template_it)

                    if self.loss_by_occlusion:
                        loss_outliers = self.loss_template_points_behind(x, z, theta)

                    if self.show_loss:
                        self.loss_matrix[x_ind, z_ind, theta_ind, 0] = dist + loss_outliers
                        self.loss_matrix[x_ind, z_ind, theta_ind, 1] = dist
                        self.loss_matrix[x_ind, z_ind, theta_ind, 2] = loss_outliers
                        self.loss_matrix[x_ind, z_ind, theta_ind, 3] = avg_med_distance(self.filtered_lidar, template_it)

                    if self.loss_by_occlusion:
                        dist += loss_outliers

                    if dist < min_dist:
                        min_dist = dist
                        opt_values = np.array([x + self.x_mean_lidar, self.y_mean_lidar, z + self.z_mean_lidar, theta])

                    if self.show_loss:
                        theta_ind += 1
                if self.show_loss:
                    z_ind += 1
            if self.show_loss:
                x_ind += 1

        if not self.cluster:
            print(opt_values[[0, 2]] - [self.x_mean_lidar, self.z_mean_lidar])
            #print(self.loss_template_points_behind(opt_values[0]-self.x_mean_lidar, opt_values[2]-self.z_mean_lidar, opt_values[3]))
            #print(min_dist)
        self.opt_values = opt_values
        self.min_dist = min_dist

    def optimize_w_histogram(self):
        min_dist = np.inf
        opt_values = np.array([0., 0., 0., 0.])

        #First we want to find the yaw using the histogram metrics
        scan_histogram = self.compute_histogram(self.filtered_lidar)

        self.find_closest_GT_yaw()

        bins = np.rad2deg(np.linspace(-np.pi, np.pi - (2*np.pi) / self.histo_nbins, self.histo_nbins))

        #Todo need to roll the histogram.
        angle = -np.arctan2(self.z_mean_lidar,
                           self.x_mean_lidar) + np.pi / 2  # Shift, because for a car in front of us we want 0 rotation
        shift_by = np.round(angle / ((2 * np.pi) / self.histo_nbins))

        if not self.cluster:
            print("GT yaw: ", np.rad2deg(self.GT_yaw))
            print("Max histogram: ", bins[np.argmax(scan_histogram)])
            print("Angle median", np.rad2deg(angle))
            print("Shift: ", shift_by)

        losss = []
        min_histo_dist = np.inf

        for theta in np.linspace(0, 2 * np.pi - (2 * np.pi / self.histo_iteryaw), num=self.histo_iteryaw):
            #load the proper histogram.
            template_histogram = np.copy(self.template_histograms[self.compute_index(self.x_mean_lidar, self.z_mean_lidar, theta)])

            #Now I need to roll the values, because now it is located somewhere in a space. Faster than computing the whole histogram again.

            template_histogram = np.roll(template_histogram, int(shift_by))

            if self.histo_use_L2:
                loss = np.sum((template_histogram - scan_histogram)**2)
            elif self.histo_use_L1:
                loss = np.sum(np.abs(template_histogram - scan_histogram))
            elif self.histo_use_bhat:
                loss = 1 - self.bhattacharyya(template_histogram, scan_histogram)
            elif self.histo_use_emd:
                loss = self.emd(template_histogram, scan_histogram, self.compute_distance_matrix(template_histogram, scan_histogram))


            losss.append(loss)

            if loss < min_histo_dist:
                min_histo_dist = loss
                opt_values[3] = theta


        if self.visu_histo:
            fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1)

            bins2 = np.rad2deg(np.linspace(0, 2*np.pi - (2*np.pi) / self.histo_iteryaw, self.histo_iteryaw))
            ax.plot(bins, scan_histogram, label="Original")  # arguments are passed to np.histogram
            ax.plot(bins, np.roll(self.template_histograms[self.compute_index(self.x_mean_lidar, self.z_mean_lidar, opt_values[3])], int(shift_by)), label="Template")
            ax.plot(bins, np.roll(self.template_histograms[self.compute_index(self.x_mean_lidar, self.z_mean_lidar, self.GT_yaw)], int(shift_by)), label="GT Histogram")
            ax2.plot(bins2, losss, label = 'Loss')
            ax.legend()
            ax2.legend()
            plt.show()

        for x in np.linspace(self.x_range_min, self.x_range_max, num=self.location_niter):
            for z in np.linspace(self.z_range_min, self.z_range_max, num=self.location_niter):
                if self.use_whole_template_during_optim: #Take the whole template and rotate it
                    template_it = self.lidar_car_template_non_filt.copy()

                    r = R.from_euler('zyx', [0, opt_values[3], 0], degrees=False)
                    template_it = np.matmul(r.as_matrix(), template_it.transpose()).transpose()
                else: #Take the filtered templates and rotate them
                    template_it = np.copy(self.lidar_car_template[
                                              self.compute_index(self.x_mean_lidar + x, self.z_mean_lidar + z, opt_values[3])])

                    angle_rot = -np.arctan2(self.z_mean_lidar + z,
                                           self.x_mean_lidar + x) + np.pi/2  # Shift, because for a car in front of us we want 0 rotation

                    r = R.from_euler('zyx', [0, angle_rot, 0], degrees=False)
                    template_it = np.matmul(r.as_matrix(), template_it.transpose()).transpose()

                template_it[:, 0] += x + self.x_mean_lidar
                template_it[:, 1] += self.y_mean_lidar
                template_it[:, 2] += z + self.z_mean_lidar

                if self.filter_by_mask_during_optimize:
                    template_it = self.filter_template_by_mask(template_it)
                    if template_it.shape[0] <= self.filt_template_threshold_optim: #If the template has only few points first it is not very heplpful to optimize over it and also it creates NAN in trim mean.
                        continue

                if self.loss_by_trim_mean:
                    dist = avg_trim_distance(self.filtered_lidar, template_it, self.trim_treshold)
                elif self.loss_by_median_only_to_scan:
                    dist = avg_med_distance_only_temp_to_scan(self.filtered_lidar, template_it)
                elif self.loss_by_median:
                    dist = avg_med_distance(self.filtered_lidar, template_it)

                if self.loss_by_occlusion:
                    loss_outliers = self.loss_template_points_behind(x, z, opt_values[3])

                if self.loss_by_occlusion:
                    dist += loss_outliers

                if dist < min_dist:
                    min_dist = dist
                    opt_values = np.array([x + self.x_mean_lidar, self.y_mean_lidar, z + self.z_mean_lidar, opt_values[3]])

        if not self.cluster:
            print(opt_values[[0, 2]] - [self.x_mean_lidar, self.z_mean_lidar])
            #print(self.loss_template_points_behind(opt_values[0]-self.x_mean_lidar, opt_values[2]-self.z_mean_lidar, opt_values[3]))
            #print(min_dist)
            print("Opt value on fitting", np.rad2deg(opt_values[3]))
            print("Angle after fitting", np.rad2deg(-np.arctan2(opt_values[2],
                               opt_values[0]) + np.pi / 2))  # Shift, because for a car in front of us we want 0 rotation

        self.opt_values = opt_values
        self.min_dist = min_dist

    def bhattacharyya(self, hist1, hist2):
        # Compute the histogram intersection
        intersection = np.minimum(hist1, hist2)
        # Compute the normalization factor
        normalization = np.sum(intersection)
        # Compute the Bhattacharyya coefficient
        b_coeff = normalization / np.sum(hist1)
        return b_coeff

    def compute_distance_matrix(self, hist1, hist2):
        # Compute the bin centers for the two histograms
        centers1 = np.arange(len(hist1))
        centers2 = np.arange(len(hist2))
        # Compute the Euclidean distance between the bin centers
        centers_diff = np.abs(centers1[:, None] - centers2[None, :])
        distance_matrix = centers_diff
        return distance_matrix

    def emd(self, hist1, hist2, distance_matrix):
        # Compute the EMD between the two histograms
        emd_dist = pyemd.emd(hist1, hist2, distance_matrix)
        return emd_dist

    def find_closest_GT_yaw(self):
        min_dist = np.inf
        best_yaw = 0

        f = open(self.label_path + self.file_name + '.txt', 'r')
        # f = open('/path/to/Data_object_det/training/label_2/' + file_name + '.txt', 'r')  # (redacted)
        # Read all lines from the file
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car':
                center = np.array([(float(arr[i][11])), (float(arr[i][12])), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                dist = (self.x_mean_lidar - center[0])**2 + (self.z_mean_lidar - center[2])**2

                if dist < min_dist:
                    min_dist = dist
                    best_yaw = yaw

        self.GT_yaw = best_yaw

    def filter_scan_by_circle(self):
        dist_from_mean = np.sqrt((self.x_mean_lidar - self.filtered_lidar[:,0])**2 + (self.z_mean_lidar - self.filtered_lidar[:,2])**2)
        
        indexes = np.argwhere(dist_from_mean < self.filter_diameter)

        if self.visu_3D:
            self.color[self.filter_idx[indexes], 0] = self.rng.random()
            self.color[self.filter_idx[indexes], 1] = self.rng.random()
            self.color[self.filter_idx[indexes], 2] = self.rng.random()
            
        self.filtered_lidar = np.array([self.filtered_lidar[indexes, 0], self.filtered_lidar[indexes, 1], self.filtered_lidar[indexes,2]]).T[0]
        
    def load_lidar_templates(self):
        self.lidar_car_template = []
        perfect_scale = np.ones((3,1))
        mean_x = 0
        mean_y = 0
        mean_z = 0

        if self.use_histogram:
            self.template_histograms = []

        if self.use_diff_template:
            #Reads the mesh and also scales it
            templ = read_car_mesh("../data/fiat_mod.asc")
            templ = self.shift_lidar_template(templ)
        else:
            #Here it does not scale so we need to do it by ourselfs
            cloud = PyntCloud.from_file("../data/pcloud_filtered/" + "999" + ".pcd")
            templ = np.asarray(cloud.xyz)

            mean_x = np.mean(templ[:, 0])  # Need to shift in X
            mean_y = np.mean(templ[:, 1])  # Need to shift in Y
            mean_z = np.mean(templ[:, 2])  # Need to shift in Z

            #templ[:, 0] -= mean_x
            #templ[:, 1] -= mean_y
            #templ[:, 2] -= mean_z

            if self.use_alternative_dimensions:
                perfect_scale = np.array(get_perfect_scale(templ, self.height_kittiavg, self.width_kittiavg, self.length_kittiavg))
            else:
                perfect_scale = np.array(get_perfect_scale(templ))

            templ[:, 0] *= perfect_scale[0]
            templ[:, 1] *= perfect_scale[1]
            templ[:, 2] *= perfect_scale[2]

        self.lidar_car_template_non_filt = templ
        if self.show_loaded_pclouds:
            fig, ax = plt.subplots()
        for i in np.linspace(0, 360 - int(self.angle_per_pcd), int(360/self.angle_per_pcd)):
            #inp = open3d.io.read_point_cloud("../data/pcloud_filtered/" + str(int(i)) + ".pcd")
            cloud = PyntCloud.from_file("../data/pcloud_filtered/" + str(int(i)) + ".pcd")
            templ = np.asarray(cloud.xyz)
            #Works the best without the shifted mean TODO should be checked
            #templ[:, 0] -= mean_x
            #templ[:, 1] -= mean_y
            #templ[:, 2] -= mean_z

            templ[:, 0] *= perfect_scale[0]
            templ[:, 1] *= perfect_scale[1]
            templ[:, 2] *= perfect_scale[2]

            self.lidar_car_template.append(templ)

            if self.use_histogram:
                self.template_histograms.append(self.compute_histogram(templ))

            if self.show_loaded_pclouds:
                if i % 10 == 0 and i <= 90:
                    bins = np.linspace(-np.pi, np.pi - (np.pi) / 36, 36)
                    print(i)
                    ax.plot(bins,self.compute_histogram(templ),label=str(i))

                    point_cloud = open3d.geometry.PointCloud()
                    point_cloud.points = open3d.utility.Vector3dVector(templ)
                    coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
                    size = np.array([1.62858987, 1.52608343, 3.88395449])  # Height, width, length
                    center = np.array([0, 0, 0])  # x,y,z
                    yaw = np.deg2rad(i)

                    r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                    bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)

                    #visu_things = [point_cloud, coord_frame, bbox]
                    #open3d.visualization.draw_geometries(visu_things, zoom=0.1, lookat=[0, 0, 1], front=[0, -0.2, -0.5],up=[0, -1, 0])
        if self.show_loaded_pclouds:
            plt.legend()
            plt.show()

    def prepare_pic(self):
        # First get the name of the file. Sometimes for debug we want to choose it randomly
        if self.take_random:
            self.pic = random.choice(self.pics)
        else:
            self.pic = self.pics[self.pic_index]

        temp = self.pic.split("/")
        # temp = pic.split("\\")
        self.file_name = temp[-1].split(".")[0]

        # Open the image, convert
        self.img_orig = cv2.imread(self.pic)
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

    def run_detectron(self):
        # Generate the ouptut, can be modified for multiple imgs
        with torch.inference_mode():
            outputs = self.model([{'image': torch.from_numpy(self.img)}])

        out_data = outputs[0]["instances"]
        # We do not care about detections with low score -> probably occluded
        self.out_data = out_data[out_data.scores > self.score_detectron_thresh]

    def draw_GT_bboxes(self):
        # Draw the GT bboxes
        self.color = np.zeros((self.lidar.shape[1], 3))
        self.lidar_visu = self.lidar[:3, :].copy()
        self.bboxes = []

        f = open(self.label_path + self.file_name + '.txt', 'r')
        # f = open('/path/to/Data_object_det/training/label_2/' + file_name + '.txt', 'r')  # (redacted)
        # Read all lines from the file
        lines = f.readlines()
        arr = [line.strip().split(" ") for line in lines]
        for i in range(len(arr)):
            if arr[i][0] == 'Car' or arr[i][0] == 'car':
                size = np.array(
                    [(float(arr[i][8])), (float(arr[i][9])), (float(arr[i][10]))])  # Height, width, length
                center = np.array([(float(arr[i][11])), (float(arr[i][12])), (float(arr[i][13]))])  # x,y,z
                yaw = (float(arr[i][14])) - np.pi / 2.  # For unknow reasons, have to be shifted ...

                r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
                bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
                bbox.color = np.array([0, 1, 0])
                self.bboxes.append(bbox)

    def writetxt(self, index):
        box = np.array(self.out_data.pred_boxes[index].tensor[0].cpu())

        self.f_write.write('Car -1 -1 -10 ')
        for z in range(4):
            self.f_write.write(str(f'{float(box[z]):3.2f}') + ' ')
        self.f_write.write(str(self.height_kittiavg) + " " + str(self.width_kittiavg) + " " + str(
            self.length_kittiavg) + " ")  # Height width length, avg of kitti dataset
        for z in range(3):
            self.f_write.write(str(f'{float(self.opt_values[z]):3.2f}') + " ")  # X,Y,Z center
        yaw = self.opt_values[3] - np.pi / 2.
        if yaw > np.pi:
            yaw -= 2 * np.pi
        elif yaw < -np.pi:
            yaw += 2 * np.pi
        self.f_write.write(str(f'{float(yaw):3.2f}') + " ")  # yaw
        self.f_write.write(str(f'{float(self.out_data.scores[index]):3.2f}') + " ")
        self.f_write.write('\n')

    def show3d_template(self):
        if self.show_whole_template:
            lidar_car_temp = self.lidar_car_template_non_filt.copy().transpose()

            r = R.from_euler('zyx', [0, self.opt_values[3], 0], degrees=False)
            lidar_car_temp = np.matmul(r.as_matrix(), lidar_car_temp)
        else:
            lidar_car_temp = self.lidar_car_template[
                self.compute_index(self.opt_values[0], self.opt_values[2], self.opt_values[3])].copy().transpose()

            angle_rot = -np.arctan2(self.opt_values[2],
                                   self.opt_values[
                                       0]) + np.pi / 2  # Shift, because for a car in front of us we want 0 rotation

            r = R.from_euler('zyx', [0, angle_rot, 0], degrees=False)
            lidar_car_temp = np.matmul(r.as_matrix(), lidar_car_temp)

        lidar_car_temp[0, :] += self.opt_values[0]
        lidar_car_temp[1, :] += self.opt_values[1]
        lidar_car_temp[2, :] += self.opt_values[2]

        if self.filter_by_mask_during_optimize:
            lidar_car_temp = self.filter_template_by_mask(lidar_car_temp.T).T
        # Merge together the point clouds
        self.lidar_visu = np.concatenate((self.lidar_visu, lidar_car_temp), axis=1)
        # Color to differentiate detected points and the car mesh template
        color_temp = np.zeros((lidar_car_temp.shape[1], 3))
        color_temp[:, 0] = 1
        self.color = np.concatenate((self.color, color_temp))

    def show3d_template_bbox(self):
        size = np.array([self.height_kittiavg,self.width_kittiavg, self.length_kittiavg])  # Height, width, length
        center = np.array([self.opt_values[0], self.opt_values[1], self.opt_values[2]])  # x,y,z
        yaw = self.opt_values[3]

        r = R.from_euler('zyx', [0, yaw, 0], degrees=False)
        bbox = open3d.geometry.OrientedBoundingBox(center, r.as_matrix(), size)
        bbox.color = np.array([1, 0, 0])
        self.bboxes.append(bbox)

    def draw_pcloud(self):
        point_cloud = open3d.geometry.PointCloud()
        coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
        point_cloud.points = open3d.utility.Vector3dVector(self.lidar_visu.transpose())
        point_cloud.colors = open3d.utility.Vector3dVector(self.color)

        visu_things = [point_cloud]
        for k in range(len(self.bboxes)):
            visu_things.append(self.bboxes[k])
        visu_things.append(coord_frame)
        open3d.visualization.draw_geometries(visu_things, zoom=0.1, lookat=[0, 0, 1], front=[0, -0.2, -0.5],
                                             up=[0, -1, 0])
    #TODO Would be good to change it so the optimal position or angle is taken from the GT.
    def show_loss_visu(self):
        # first show over x,y
        mins = np.nanmin(self.loss_matrix[:,:,:,0], axis=2)
        x = np.linspace(self.x_range_min, self.x_range_max, num=20)
        z = np.linspace(self.z_range_min, self.z_range_max, num=20)
        theta = np.linspace(0, 360-360/20, num=20)

        X, Y = np.meshgrid(x, z, indexing='ij')

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, mins, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set(xlabel='X', ylabel='Z',
               title='Loss')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        t = np.unravel_index(np.nanargmin(mins), mins.shape)
        print("Min: ", x[t[0]], z[t[1]], np.nanmin(self.loss_matrix[t[0],t[1],:,0]))

        fig, ax = plt.subplots()
        ax.plot(theta, self.loss_matrix[t[0],t[1],:,1], label="Dist")
        ax.plot(theta, self.loss_matrix[t[0],t[1], :, 2], label="Outliers")
        ax.plot(theta, self.loss_matrix[t[0],t[1], :, 3], label="Med distance")
        ax.legend()
        plt.grid()
        plt.show()

    def compute_histogram(self, scan):
        #Function for computing histogram of angles between points
        number_of_points = scan.shape[0]

        indexes = np.random.rand(2,number_of_points*self.histo_random_point_scale) * (number_of_points - 1)
        indexes = np.around(indexes)

        x = scan[indexes[0,:].astype(int), 0] - scan[indexes[1,:].astype(int),0]
        z = scan[indexes[0,:].astype(int), 2] - scan[indexes[1,:].astype(int),2]

        angles = np.arctan2(x, z)

        counts, bins = np.histogram(angles, bins=self.histo_nbins, range=(-np.pi, np.pi - (2*np.pi) / self.histo_nbins))

        counts = counts/np.sum(counts)

        #counts = np.roll(counts, int(-(self.histo_nbins/360)*90)) #Shift by 90 degrees.

        return counts


    def main(self, argv):
        self.load_and_init_detectron_lazy()

        pic_num_min = 0
        pic_num_max = len(self.pics)
        if len(argv) > 1:
            pic_num_min = int(argv[0])
            if int(argv[1]) < len(self.pics):
                pic_num_max = int(argv[1])

        # Load point cloud templates, they are already scaled
        self.load_lidar_templates()

        for self.pic_index in range(pic_num_min, pic_num_max):
            self.prepare_pic()

            if self.write_txt:
                if os.path.isfile(self.output_dir + self.file_name + '.txt'):
                    continue
                self.f_write = open(self.output_dir + self.file_name + '.txt', 'w')

            self.run_detectron()

            self.load_and_prepare_lidar_scan(self.file_name, self.img)

            if self.visu_3D:
                self.draw_GT_bboxes()

            start = time.time_ns()
            # Now lets look on all the predicted masks
            for i in range(len(self.out_data.pred_masks)):
                # We are only interested in cars
                if self.out_data.pred_classes[i] == 2:
                    # Take the mask and transpose it
                    self.mask = np.array(self.out_data.pred_masks[i].cpu()).transpose()
                    # mask_inst = np.array(out_data.pred_masks[i]).transpose()

                    self.filter_by_mask_and_shift()

                    # print(self.x_mean_lidar, self.y_mean_lidar, self.z_mean_lidar)

                    if not self.filt_success:
                        continue

                    if self.visu_img:
                        self.visu2Dbbox(self.img_orig, np.array(self.out_data.pred_boxes[i].tensor[0].cpu()),
                                        self.out_data.scores[i])

                    if self.visu_3D:
                        self.color[self.filter_idx, 0] = self.rng.random()
                        self.color[self.filter_idx, 1] = self.rng.random()
                        self.color[self.filter_idx, 2] = self.rng.random()

                    if self.use_histogram:
                        self.filter_scan_by_circle()
                        if self.filtered_lidar.shape[0] < 10:
                            continue

                    if self.use_histogram:
                        self.optimize_w_histogram()
                    else:
                        self.optimize()
                    if not self.cluster:
                        print(self.opt_values[0],self.opt_values[1],self.opt_values[2],np.rad2deg(self.opt_values[3]))

                    if self.write_txt:
                        self.writetxt(i)

                    if self.visu_3D:
                        self.show3d_template()
                        self.show3d_template_bbox()

                    if self.iterate:
                        self.draw_pcloud()

                    if self.show_loss:
                        self.show_loss_visu()



            print("Time:", (time.time_ns() - start) / 1000000.)

            if self.visu_3D:
                self.draw_pcloud()

            if self.write_txt:
                self.f_write.close()

            print("File name: ", self.file_name)
            print("Pics_done: ", self.pic_index)


if __name__ == '__main__':
    autolabel = AutoLabel3D()
    autolabel.main(sys.argv[1:])
