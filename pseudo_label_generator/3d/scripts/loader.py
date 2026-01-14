import copy
import yaml
import time
import pytorch3d.ops
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from utils2 import load_velo_scan, get_perfect_scale, load_pseudo_lidar, load_waymoc_scan
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from scipy import stats
import cv2, os
import glob
import pickle
import open3d
import torch
import zstd
import faiss
from anno_V3 import AutoLabel3D
from pytorch3d.io import load_objs_as_meshes
from scipy.ndimage import convolve
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import hdbscan
import open3d as o3d
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import scipy.linalg # For sqrtm
import pycocotools.mask as mask_utils
import random


class Loader(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
        self.random_indexes = []
        self.mapping_data = []

        if self.args.dataset == 'waymo' or self.args.dataset == 'waymo_converted':
            with open(self.cfg.paths.waymo_info_path + "/train.txt", 'r') as f:
                self.random_indexes = [line.strip() for line in f.readlines()]
        elif self.args.dataset == 'kitti':
            with open(self.cfg.paths.kitti_path + 'object_detection/devkit_object/mapping/train_rand.txt', 'r') as f:
                line = f.readline().strip()
                self.random_indexes = line.split(',')

            with open(self.cfg.paths.kitti_path + 'object_detection/devkit_object/mapping/train_mapping.txt', 'r') as f:
                for line in f:
                    self.mapping_data.append(line.strip().split(' '))

    class Car:
        def __init__(self):
            self.lidar = None
            self.moving_scale_lidar = None
            self.scale_lidar = None
            self.locations = None
            self.mask = None
            self.info = None
            self.moving = None
            self.img_index = None
            self.ref_idx = None

            self.x = None
            self.y = None
            self.z = None
            self.theta = None
            self.length = None
            self.width = None
            self.height = None
            self.model = None
            self.optimized = False
            self.bbox = None

            self.x_scale = None
            self.y_scale = None
            self.z_scale = None
            self.theta_scale = None
            self.score = None

    class Pedestrian:
        def __init__(self):
            self.lidar = None
            self.moving_scale_lidar = None
            self.scale_lidar = None
            self.locations = None
            self.mask = None
            self.info = None
            self.moving = None
            self.img_index = None
            self.ref_idx = None

            self.x = None
            self.y = None
            self.z = None
            self.theta = None
            self.length = None
            self.width = None
            self.height = None
            self.model = None
            self.optimized = False
            self.bbox = None
            self.score = None

            self.keypoints = None
            self.all_lidars = None
            self.cyclist = False

    class AdditionalObject:
        def __init__(self, class_name: str):
            # Raw lidar points (Nx3) for this instance in the reference frame
            self.lidar = None
            # 2D binary mask corresponding to this instance (orientation consistent with other masks)
            self.mask = None
            # Class label (e.g., "truck", "bus", "bench", etc.)
            self.class_name = class_name
            # Optional properties we may compute later
            self.center = None  # np.array([x,y,z]) median of points
            self.bbox = None    # placeholder for minimal 3D bbox (to be implemented)
            self.img_index = None
            self.score = None

    # Detectron
    def load_and_init_detectron_lazy(self):
        # Load the lazyconfig, we are using this, not the classic, because this regnety learned with different idea is used
        cfg = LazyConfig.load(self.cfg.paths.detectron_config)
        cfg = LazyConfig.apply_overrides(cfg, ['train.init_checkpoint=' + self.cfg.paths.model_path])

        # Init the model
        model = instantiate(cfg.model)
        # If we are using SAM, then locally we cannot fit it also with the detectron to memory
        if self.cfg.general.device == 'gpu':
            model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        model.eval()
        if not self.cfg.general.supress_debug_prints:
            print("Detectron2 Loaded")
        self.model = model

    def load_and_init_SAM(self):
        sam = sam_model_registry["vit_h"](checkpoint=self.cfg.paths.sam_path)
        if self.cfg.general.device == 'gpu':
            sam.to(device="cuda")
        self.sam_predictor = SamPredictor(sam)
        if not self.cfg.general.supress_debug_prints:
            print("SAM Loaded")

    def load_and_prepare_lidar_scan(self, filename, img):
        # First get all the lidar points
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar = load_pseudo_lidar(self.pseudo_lidar_folder + str(self.file_number).zfill(10) + '.npz')
        else:
            lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))
        self.prepare_scan(filename, img, lidar)
        self.prepare_img_dist(img)

    def load_and_prepare_lidar_scan_all(self, filename, img):
        # First get all the lidar points
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                lidar = np.array(load_velo_scan(self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + str(int(self.number)).zfill(10) + '.bin'))
            else:
                lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))
        self.prepare_scan_all(filename, img, lidar)
        self.prepare_img_dist(img)

    def load_and_prepare_lidar_scan_waymoc(self, filename, img):
        # First get all the lidar points
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar = np.array(load_pseudo_lidar(
                self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(
                    self.number) + '.npz'))
        else:
            lidar = np.array(load_waymoc_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))
        self.prepare_scan_waymoc(filename, img, lidar)
        self.prepare_img_dist(img)

    def load_and_prepare_lidar_scan_from_multiple_pykittiV2(self, filename, img, save=False):
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar_orig = np.array(load_pseudo_lidar(self.pseudo_lidar_folder + str(self.file_number).zfill(10) + '.npz'))
        else:
            lidar_orig = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))

        if self.cfg.frames_creation.use_icp:
            transformations = self.calculate_transformationsV2(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)
        else:
            transformations = self.calculate_transformations(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Get standing car candidates")

        if self.cfg.frames_creation.use_SAM3:
            cars = self.get_SAM3_detections(transformations, "car")
            pedestrians = self.get_SAM3_detections(transformations, "pedestrian")
            cyclists = self.get_SAM3_detections(transformations, "cyclist")

            cars = self.decide_if_standing_or_moving_both5(cars, waymo=False)
            for pedestrian in pedestrians:
                pedestrian.moving = True
            for cyclist in cyclists:
                cyclist.moving = True

            cars = self.filter_moving_and_not_visible(cars, waymo=False)
            cars = self.extract_scale_lidar(cars, transformations, waymo=False)
            cars = self.choose_proper_mask(cars, waymo=False)
            pedestrians = self.filter_moving_and_not_visible(pedestrians, waymo=False)
            pedestrians = self.extract_scale_lidar(pedestrians, transformations, waymo=False)
            pedestrians = self.choose_proper_mask(pedestrians, waymo=False)
            cyclists = self.filter_moving_and_not_visible(cyclists, waymo=False)
            cyclists = self.extract_scale_lidar(cyclists, transformations, waymo=False)
            cyclists = self.choose_proper_mask(cyclists, waymo=False)

            if self.cfg.frames_creation.use_clever_aggregation:
                cars = self.standing_concatenate_lidar_clever(cars, transformations)
                pedestrians = self.standing_concatenate_lidar_clever(pedestrians, transformations)
                cyclists = self.standing_concatenate_lidar_clever(cyclists, transformations)
            else:
                cars = self.standing_concatenate_lidar(cars)
                pedestrians = self.standing_concatenate_lidar(pedestrians)
                cyclists = self.standing_concatenate_lidar(cyclists)

            cars = self.moving_lidar_keep_ref(cars, waymo=False)
            pedestrians = self.moving_lidar_keep_ref(pedestrians, waymo=False)
            cyclists = self.moving_lidar_keep_ref(cyclists, waymo=False)

            cars = self.filter_distant_cars_pseudo_lidar(cars, waymo=False)
            pedestrians = self.filter_distant_cars_pseudo_lidar(pedestrians, waymo=False)
            cyclists = self.filter_distant_cars_pseudo_lidar(cyclists, waymo=False)

            for car in cars:
                if car.lidar is not None:
                    # To reduce the size of the lidar in storage
                    if len(car.lidar) > 10000:
                        car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                    car.lidar = car.lidar.astype(np.float32)
            for pedestrian in pedestrians:
                if pedestrian.lidar is not None:
                    # To reduce the size of the lidar in storage
                    if len(pedestrian.lidar) > 10000:
                        pedestrian.lidar = self.downsample_random(pedestrian.lidar[:, :3], 10000)
                    pedestrian.lidar = pedestrian.lidar.astype(np.float32)
            for cyclist in cyclists:
                if cyclist.lidar is not None:
                    # To reduce the size of the lidar in storage
                    if len(cyclist.lidar) > 10000:
                        cyclist.lidar = self.downsample_random(cyclist.lidar[:, :3], 10000)
                    cyclist.lidar = cyclist.lidar.astype(np.float32)

            self.lidar = self.prepare_scan(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

            for pedestrian in pedestrians:
                padding = np.ones((pedestrian.lidar.shape[0], 3))

                pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T

            for cyclist in cyclists:
                padding = np.ones((cyclist.lidar.shape[0], 3))

                cyclist.lidar = np.concatenate((cyclist.lidar, padding), axis=1).T

            self.pedestrians = sorted(pedestrians + cyclists, key=lambda x: x.lidar.shape[1], reverse=True)

            return

        car_locations, car_locations_lidar, car_locations_masks, car_locations_scores = self.get_standing_car_candidates(transformations)

        if self.cfg.frames_creation.extract_pedestrians:
            #pedestrians_locations, pedestrians_lidar, pedestrians_masks = self.get_pedestrians_candidates(transformations)
            pedestrians = self.get_pedestrians_candidates_v2(transformations)
        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        moving_cars, moving_cars_lidar, moving_cars_masks, moving_cars_scores = self.perform_3D_tracking_kitti(car_locations, car_locations_lidar, car_locations_masks, car_locations_scores)

        if self.cfg.frames_creation.extract_pedestrians:
            moving_pedestrians = self.perform_3D_tracking_kitti_pedestrian(pedestrians)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, None, moving_cars_masks, moving_cars_scores)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.create_cars_from_extracted_feats_3DTrack_pedestrians(moving_pedestrians)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        #cars = self.decide_if_standing_or_moving_both(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_both2(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_bothv3(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_bothv3(cars, waymo=False)
        cars = self.decide_if_standing_or_moving_both5(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_trajectory_fit(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_sign_test(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_lidar_consistency(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_both6(cars, waymo=False)
        #cars = self.decide_if_standing_or_moving_both7(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            #pedestrians = self.decide_if_standing_or_moving_both5(pedestrians, waymo=False)
            #Assume that all pedestrians are moving
            for pedestrian in pedestrians:
                pedestrian.moving = True

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        cars = self.filter_moving_and_not_visible(cars, waymo=False)
        cars = self.extract_scale_lidar(cars, transformations, waymo=False)
        cars = self.choose_proper_mask(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.filter_moving_and_not_visible(pedestrians, waymo=False)
            pedestrians = self.extract_scale_lidar(pedestrians, transformations, waymo=False)
            pedestrians = self.choose_proper_mask(pedestrians, waymo=False)

        if not self.cfg.general.supress_debug_prints:
            print("Final Modifications")
        if self.cfg.frames_creation.use_clever_aggregation:
            cars = self.standing_concatenate_lidar_clever(cars, transformations)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.standing_concatenate_lidar_clever(pedestrians, transformations)
        else:
            cars = self.standing_concatenate_lidar(cars)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.standing_concatenate_lidar(pedestrians)

        cars = self.filter_hidden_standing_cars_tracked(cars, waymo=False)
        cars = self.moving_lidar_keep_ref(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.filter_hidden_standing_cars_tracked(pedestrians, waymo=False)
            pedestrians = self.moving_lidar_keep_ref(pedestrians, waymo=False)

        if self.cfg.frames_creation.use_pseudo_lidar:
            cars = self.filter_distant_cars_pseudo_lidar(cars, waymo=False)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.filter_distant_cars_pseudo_lidar(pedestrians, waymo=False)

        for car in cars:
            if car.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if self.cfg.frames_creation.extract_pedestrians:
            for pedestrian in pedestrians:
                if pedestrian.lidar is not None:
                    # To reduce the size of the lidar in storage
                    if len(pedestrian.lidar) > 10000:
                        pedestrian.lidar = self.downsample_random(pedestrian.lidar[:, :3], 10000)
                    pedestrian.lidar = pedestrian.lidar.astype(np.float32)

        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

            if self.cfg.frames_creation.extract_pedestrians:
                compressed_arr = zstd.compress(pickle.dumps(pedestrians, pickle.HIGHEST_PROTOCOL))
                with open(self.cfg.paths.merged_frames_path + "pedestrians/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.prepare_scan(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

            if self.cfg.frames_creation.extract_pedestrians:
                for pedestrian in pedestrians:
                    padding = np.ones((pedestrian.lidar.shape[0], 3))

                    pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T

                self.pedestrians = sorted(pedestrians, key=lambda x: x.lidar.shape[1], reverse=True)

        # Load additional classes (no temporal aggregation). We only use current frame masks.
        try:
            self._load_additional_objects_current_frame(lidar_orig, img)
        except Exception as _e:
            # Keep non-fatal; proceed even if extras fail
            if not self.cfg.general.supress_debug_prints:
                print("Additional objects loading failed:", _e)

    def get_pedestrian_poseidon(self):
        peds_to_output = []
        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            full_path_to_folder = self.path_to_folder.split("/")
            folder = full_path_to_folder[-3]
            subfolder = full_path_to_folder[-2]

            path_to_pedestrian = os.path.join(self.cfg.paths.merged_frames_path, "masks_raw_pedestrians/", folder, subfolder, f'{self.file_number + i :0>10}' + '.zstd')

            if not os.path.exists(path_to_pedestrian):
                peds_to_output.append([])
                continue

            with open(path_to_pedestrian, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            loaded_data = pickle.loads(decompressed_data)

            current_frame_peds = []
            if isinstance(loaded_data, dict):
                masks = loaded_data.get('masks', [])
                flags = loaded_data.get('flags', [])
                scores = loaded_data.get('scores', [])
                
                if masks is None or len(masks) == 0:
                    peds_to_output.append([])
                    continue

                for j in range(len(masks)):
                    ped = {}
                    ped['mask'] = masks[j].T
                    ped['cyclist_flag'] = flags[j] if len(flags) > j else 0
                    ped['score'] = scores[j] if len(scores) > j else 0.4
                    ped['keypoints'] = np.zeros((17, 3))
                    current_frame_peds.append(ped)
            else:
                current_frame_peds = loaded_data

            peds_to_output.append(current_frame_peds)

        return peds_to_output

    def get_pedestrians_candidates_v2(self, transformations):
        #pedestrians = self.get_pedestrian_smplestx()
        pedestrians = self.get_pedestrian_poseidon()

        pedestrians_out = []

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
            else:
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

            if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_velo):
                pedestrians_out.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            # Load the velo scan
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            else:
                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            lidar_cur = self.prepare_scan(self.file_name, self.img, lidar_cur, save=False)

            cur_pedestrians = pedestrians[i + self.cfg.frames_creation.nscans_before]
            pedestrian_masks = []
            pedestrian_poses = []
            pedestrian_cyc_flags = []
            pedestrian_scores = []

            for ped in cur_pedestrians:
                cur_mask = ped['mask']
                cur_pose = ped['keypoints']
                pedestrian_masks.append(cur_mask.T)
                pedestrian_cyc_flags.append(ped['cyclist_flag'])
                if 'score' in ped:
                    pedestrian_scores.append(ped['score'])
                else:
                    pedestrian_scores.append(0.4)
                
                # Swap u, v coordinates of keypoints to match the mask transpose
                transposed_pose = cur_pose.copy()
                transposed_pose[:, 0] = cur_pose[:, 0]
                transposed_pose[:, 1] = cur_pose[:, 1]

                pedestrian_poses.append(transposed_pose) 

            pedestrians_for_output = []
            ped_loc, lidar_points, masks, keypoints_lifted, scores_out = self.get_pedestrian_locations_from_img(lidar_cur, T_cur_to_ref, pedestrian_masks, pedestrian_poses, scores=pedestrian_scores) #TODO Be aware this can be problematic as it filters out predictions with less than X points.

            if False:
                for z in range(len(ped_loc)):
                    if lidar_points[z] is not None and keypoints_lifted[z] is not None:
                        # Create PointCloud for LiDAR points
                        pcd_lidar = o3d.geometry.PointCloud()
                        pcd_lidar.points = o3d.utility.Vector3dVector(lidar_cur[:3, :].T)
                        pcd_lidar.paint_uniform_color([0.8, 0.8, 0.8])  # Gray color for LiDAR

                        # Create small spheres for each lifted keypoint to make them visible
                        keypoint_geometries = []
                        # Extract only the 3D coordinates (x, y, z) from the lifted keypoints
                        keypoints_3d = keypoints_lifted[z][:, 2:5]
                        
                        for kp_3d in keypoints_3d:
                            # Check for invalid keypoints (e.g., [0,0,0] if not found)
                            if not np.all(kp_3d == 0):
                                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                                sphere.translate(kp_3d)
                                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for keypoints
                                keypoint_geometries.append(sphere)

                        # Visualize the LiDAR scan and the keypoint spheres together
                        o3d.visualization.draw_geometries([pcd_lidar] + keypoint_geometries,
                                                            window_name=f"Pedestrian {z} LiDAR and Lifted Keypoints")

            for z in range(len(ped_loc)):
                if np.abs(ped_loc[z][0]) > 0.001 and np.abs(ped_loc[z][1]) > 0.001 and np.abs(ped_loc[z][2]) > 0.001:
                    cur_pedestrians[z]['lidar'] = lidar_points[z]
                    cur_pedestrians[z]['mask'] = masks[z]
                    cur_pedestrians[z]['location'] = ped_loc[z]
                    cur_pedestrians[z]['keypoints'] = keypoints_lifted[z]
                    cur_pedestrians[z]['cyclist_flag'] = pedestrian_cyc_flags[z]
                    cur_pedestrians[z]['score'] = scores_out[z]

                    pedestrians_for_output.append(cur_pedestrians[z])

            pedestrians_out.append(pedestrians_for_output)

        return pedestrians_out


    def load_and_prepare_lidar_scan_from_multiple_all(self, filename, img, save=False):
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar_orig = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                lidar_orig = np.array(load_velo_scan(self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + str(int(self.number)).zfill(10) + '.bin'))
            else:
                lidar_orig = np.array(load_velo_scan(self.cfg.paths.all_dataset_path + 'object_detection/training/velodyne/' + filename + '.bin'))

        transformations = self.calculate_transformations_all(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Get standing car candidates")
        car_locations, car_locations_lidar, car_locations_masks, car_locations_ids, car_locations_scores = self.get_standing_car_candidates_all(transformations)
        if car_locations is None:
            return

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.get_pedestrians_candidates_all(transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        if self.cfg.frames_creation.use_gt_masks:
            moving_cars, moving_cars_lidar, moving_cars_masks = self.perform_gt_tracking(car_locations, car_locations_lidar, car_locations_masks, car_locations_ids)
        else:
            moving_cars, moving_cars_lidar, moving_cars_masks, moving_cars_scores = self.perform_3D_tracking_kitti(car_locations, car_locations_lidar, car_locations_masks, car_locations_scores)

        if self.cfg.frames_creation.extract_pedestrians:
            moving_pedestrians = self.perform_3D_tracking_kitti_pedestrian(pedestrians)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, None, moving_cars_masks, moving_cars_scores)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.create_cars_from_extracted_feats_3DTrack_pedestrians(moving_pedestrians)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        cars = self.decide_if_standing_or_moving_both5(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            #Assume that all pedestrians are moving
            for pedestrian in pedestrians:
                pedestrian.moving = True

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        if not self.cfg.frames_creation.use_gt_masks:
            cars = self.filter_moving_and_not_visible(cars, waymo=False)
        cars = self.extract_scale_lidar(cars, transformations, waymo=False)
        cars = self.choose_proper_mask(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.filter_moving_and_not_visible(pedestrians, waymo=False)
            pedestrians = self.extract_scale_lidar(pedestrians, transformations, waymo=False)
            pedestrians = self.choose_proper_mask(pedestrians, waymo=False)

        if self.cfg.frames_creation.use_clever_aggregation:
            cars = self.standing_concatenate_lidar_clever(cars, transformations)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.standing_concatenate_lidar_clever(pedestrians, transformations)
        else:
            cars = self.standing_concatenate_lidar(cars)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.standing_concatenate_lidar(pedestrians)

        if not self.cfg.frames_creation.use_gt_masks:
            cars = self.filter_hidden_standing_cars_tracked(cars, waymo=False)
        cars = self.moving_lidar_keep_ref(cars, waymo=False)

        if self.cfg.frames_creation.extract_pedestrians:
            pedestrians = self.filter_hidden_standing_cars_tracked(pedestrians, waymo=False)
            pedestrians = self.moving_lidar_keep_ref(pedestrians, waymo=False)

        if self.cfg.frames_creation.use_pseudo_lidar:
            if not self.cfg.frames_creation.use_gt_masks:
                cars = self.filter_distant_cars_pseudo_lidar(cars, waymo=False)
            if self.cfg.frames_creation.extract_pedestrians:
                pedestrians = self.filter_distant_cars_pseudo_lidar(pedestrians, waymo=False)

        for car in cars:
            if car.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if self.cfg.frames_creation.extract_pedestrians:
            for pedestrian in pedestrians:
                if pedestrian.lidar is not None:
                    # To reduce the size of the lidar in storage
                    if len(pedestrian.lidar) > 10000:
                        pedestrian.lidar = self.downsample_random(pedestrian.lidar[:, :3], 10000)
                    pedestrian.lidar = pedestrian.lidar.astype(np.float32)

        if np.all(transformations[self.cfg.frames_creation.nscans_before] == 0.):
            return False

        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

            if self.cfg.frames_creation.extract_pedestrians:
                compressed_arr = zstd.compress(pickle.dumps(pedestrians, pickle.HIGHEST_PROTOCOL))
                with open(self.cfg.paths.merged_frames_path + "pedestrians/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.prepare_scan_all(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

            if self.cfg.frames_creation.extract_pedestrians:
                for pedestrian in pedestrians:
                    padding = np.ones((pedestrian.lidar.shape[0], 3))

                    pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T

                self.pedestrians = sorted(pedestrians, key=lambda x: x.lidar.shape[1], reverse=True)

            return True

    def load_and_prepare_lidar_scan_from_multiple_waymoc(self, filename, img, save=False):
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar_orig = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            lidar_orig = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))

        transformations = self.calculate_transformations_waymoc(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Get standing car candidates")
        car_locations, car_locations_lidar, car_locations_masks, car_locations_ids, car_locations_scores = self.get_standing_car_candidates_all(transformations)
        if car_locations is None:
            return

        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        if self.cfg.frames_creation.use_gt_masks:
            moving_cars, moving_cars_lidar, moving_cars_masks = self.perform_gt_tracking(car_locations, car_locations_lidar, car_locations_masks, car_locations_ids)
        else:
            moving_cars, moving_cars_lidar, moving_cars_masks, moving_cars_scores = self.perform_3D_tracking_kitti(car_locations, car_locations_lidar, car_locations_masks, car_locations_scores)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, None, moving_cars_masks, moving_cars_scores)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        cars = self.decide_if_standing_or_moving_both5(cars, waymo=False)

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        if not self.cfg.frames_creation.use_gt_masks:
            cars = self.filter_moving_and_not_visible(cars, waymo=False)
        cars = self.extract_scale_lidar(cars, transformations, waymo=False)
        cars = self.choose_proper_mask(cars, waymo=False)

        if not self.cfg.general.supress_debug_prints:
            print("Final Modifications")
        if self.cfg.frames_creation.use_clever_aggregation:
            cars = self.standing_concatenate_lidar_clever(cars, transformations)
        else:
            cars = self.standing_concatenate_lidar(cars)
        if not self.cfg.frames_creation.use_gt_masks:
            cars = self.filter_hidden_standing_cars_tracked(cars, waymo=False)
        cars = self.moving_lidar_keep_ref(cars, waymo=False)

        if self.cfg.frames_creation.use_pseudo_lidar:
            if not self.cfg.frames_creation.use_gt_masks:
                cars = self.filter_distant_cars_pseudo_lidar(cars, waymo=False)

        for car in cars:
            if car.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if save:
            if np.all(transformations[self.cfg.frames_creation.nscans_before] == 0.):
                return
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.prepare_scan_waymoc(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def _preload_lidar_cache_dsec(self):
        cur_id = int(self.number)
        lidar_cache = {}
        for frame_idx in range(-self.cfg.frames_creation.nscans_before + cur_id, self.cfg.frames_creation.nscans_after + 1 + cur_id):
            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(frame_idx).zfill(6) + '.npz'
            else:
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{frame_idx :0>10}' + '.bin'

            if frame_idx < 0 or not os.path.exists(path_to_cur_velo):
                continue

            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            else:
                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))

            lidar_cur = self.prepare_scan_dsec(self.file_name, self.img, lidar_cur, save=False)
            lidar_cache[frame_idx] = lidar_cur
        return lidar_cache

    def load_and_prepare_lidar_scan_from_multiple_dsec(self, filename, img, save=False):
        if self.cfg.frames_creation.use_pseudo_lidar:
            lidar_orig = np.array(load_pseudo_lidar(self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(self.number) + '.npz'))
        else:
            lidar_orig = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + filename + '.bin'))

        transformations = self.calculate_transformations_dsec(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Get standing car candidates")

        lidar_cache = self._preload_lidar_cache_dsec()

        cars = self.get_SAM3_detections_dsec(transformations, lidar_cache, "car")
        pedestrians = self.get_SAM3_detections_dsec(transformations, lidar_cache, "pedestrian")
        cyclists = self.get_SAM3_detections_dsec(transformations, lidar_cache, "cyclist")

        cars = self.decide_if_standing_or_moving_both5(cars, waymo=False)
        for pedestrian in pedestrians:
            pedestrian.moving = True
        for cyclist in cyclists:
            cyclist.moving = True
        for car in cars:
            car.moving = True

        cars = self.filter_moving_and_not_visible(cars, waymo=False)
        cars = self.extract_scale_lidar(cars, transformations, waymo=False)
        cars = self.choose_proper_mask(cars, waymo=False)
        pedestrians = self.filter_moving_and_not_visible(pedestrians, waymo=False)
        pedestrians = self.extract_scale_lidar(pedestrians, transformations, waymo=False)
        pedestrians = self.choose_proper_mask(pedestrians, waymo=False)
        cyclists = self.filter_moving_and_not_visible(cyclists, waymo=False)
        cyclists = self.extract_scale_lidar(cyclists, transformations, waymo=False)
        cyclists = self.choose_proper_mask(cyclists, waymo=False)

        if False:
            visu_list = []
            visu_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(lidar_orig[:, :3])
            pcd.paint_uniform_color([0.5, 0.5, 0.5])
            visu_list.append(pcd)

            for car in cars:
                if car.lidar is not None:
                    rnd_color = [random.random(), random.random(), random.random()]
                    for lidar in car.lidar:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
                        pcd.paint_uniform_color(rnd_color)
                        visu_list.append(pcd)

            # Save the current image for debugging
            debug_img_path = os.path.join(os.getcwd(), f"debug_img_{self.file_name}.png")
            cv2.imwrite(debug_img_path, self.img_orig)
            print(f"Debug image saved to: {debug_img_path}")
            
            o3d.visualization.draw_geometries(visu_list)

        if self.cfg.frames_creation.use_clever_aggregation:
            cars = self.standing_concatenate_lidar_clever(cars, transformations)
            pedestrians = self.standing_concatenate_lidar_clever(pedestrians, transformations)
            cyclists = self.standing_concatenate_lidar_clever(cyclists, transformations)
        else:
            cars = self.standing_concatenate_lidar(cars)
            pedestrians = self.standing_concatenate_lidar(pedestrians)
            cyclists = self.standing_concatenate_lidar(cyclists)

        cars = self.moving_lidar_keep_ref(cars, waymo=False)
        pedestrians = self.moving_lidar_keep_ref(pedestrians, waymo=False)
        cyclists = self.moving_lidar_keep_ref(cyclists, waymo=False)

        cars = self.filter_distant_cars_pseudo_lidar(cars, waymo=False)
        pedestrians = self.filter_distant_cars_pseudo_lidar(pedestrians, waymo=False)
        cyclists = self.filter_distant_cars_pseudo_lidar(cyclists, waymo=False)

        for car in cars:
            if car.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)
        for pedestrian in pedestrians:
            if pedestrian.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(pedestrian.lidar) > 10000:
                    pedestrian.lidar = self.downsample_random(pedestrian.lidar[:, :3], 10000)
                pedestrian.lidar = pedestrian.lidar.astype(np.float32)
        for cyclist in cyclists:
            if cyclist.lidar is not None:
                # To reduce the size of the lidar in storage
                if len(cyclist.lidar) > 10000:
                    cyclist.lidar = self.downsample_random(cyclist.lidar[:, :3], 10000)
                cyclist.lidar = cyclist.lidar.astype(np.float32)

        self.lidar = self.prepare_scan_dsec(filename, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
        
        idx = 0
        for car in cars:
            idx += 1
            padding = np.ones((car.lidar.shape[0], 3))

            car.lidar = np.concatenate((car.lidar, padding), axis=1).T

        self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

        for pedestrian in pedestrians:
            padding = np.ones((pedestrian.lidar.shape[0], 3))

            pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T

        for cyclist in cyclists:
            padding = np.ones((cyclist.lidar.shape[0], 3))

            cyclist.lidar = np.concatenate((cyclist.lidar, padding), axis=1).T

        self.pedestrians = sorted(pedestrians + cyclists, key=lambda x: x.lidar.shape[1], reverse=True)

        return True

    def load_merged_frames_from_files_KITTI(self):
        self.load_and_prepare_lidar_scan(self.file_name, self.img)

        if self.cfg.frames_creation.use_growing_for_point_extraction:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        else:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())

        cars = pickle.loads(decompressed_data)
        self.cars = cars
        self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)

        cur_lidar = np.ascontiguousarray(self.lidar.T[:, :3]).astype('float32')
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10

        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                if len(distances) < 1:
                    continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars

        if self.cfg.frames_creation.extract_pedestrians:
            with open(self.cfg.paths.merged_frames_path + "pedestrians/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            pedestrians = pickle.loads(decompressed_data)
            self.pedestrians = pedestrians
            self.pedestrians = sorted(self.pedestrians, key=lambda x: x.lidar.shape[0], reverse=True)

            new_pedestrians = []

            for pedestrian in self.pedestrians:
                if pedestrian.lidar is not None:
                    center = np.zeros((1, 3))
                    center[0, 0] = np.median(pedestrian.lidar[:, 0])
                    center[0, 1] = np.median(pedestrian.lidar[:, 1])
                    center[0, 2] = np.median(pedestrian.lidar[:, 2])

                    idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                    if len(distances) < 1:
                        continue

                    padding = np.ones((pedestrian.lidar.shape[0], 3))

                    pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T
                    new_pedestrians.append(pedestrian)
            self.pedestrians = new_pedestrians

    def load_merged_frames_from_files_KITTI_all(self):
        self.load_and_prepare_lidar_scan_all(self.file_name, self.img)

        if self.cfg.frames_creation.use_growing_for_point_extraction:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        else:
            if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd"):
                return False
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        cars = pickle.loads(decompressed_data)
        self.cars = cars
        self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)
        cur_lidar = np.ascontiguousarray(self.lidar.T[:, :3].astype('float32'))
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10

        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                if not self.cfg.frames_creation.use_gt_masks:
                    if len(distances) < 1:
                        continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars

        if self.cfg.frames_creation.extract_pedestrians:
            self.pedestrians = []
            if os.path.exists(self.cfg.paths.merged_frames_path + "pedestrians/" + self.file_name + ".zstd"):
                with open(self.cfg.paths.merged_frames_path + "pedestrians/" + self.file_name + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                pedestrians = pickle.loads(decompressed_data)
                self.pedestrians = pedestrians
                self.pedestrians = sorted(self.pedestrians, key=lambda x: x.lidar.shape[0], reverse=True)

                new_pedestrians = []

                for pedestrian in self.pedestrians:
                    if pedestrian.lidar is not None:
                        center = np.zeros((1, 3))
                        center[0, 0] = np.median(pedestrian.lidar[:, 0])
                        center[0, 1] = np.median(pedestrian.lidar[:, 1])
                        center[0, 2] = np.median(pedestrian.lidar[:, 2])

                        idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                        if len(distances) < 1:
                            continue

                        padding = np.ones((pedestrian.lidar.shape[0], 3))

                        pedestrian.lidar = np.concatenate((pedestrian.lidar, padding), axis=1).T
                        new_pedestrians.append(pedestrian)
                self.pedestrians = new_pedestrians

        return True

    def load_merged_frames_from_files_KITTI_waymoc(self):
        self.load_and_prepare_lidar_scan_waymoc(self.file_name, self.img)

        if self.cfg.frames_creation.use_growing_for_point_extraction:
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        else:
            if not os.path.exists(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd"):
                return False
            with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
        cars = pickle.loads(decompressed_data)
        self.cars = cars
        self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)

        cur_lidar = np.ascontiguousarray(self.lidar.T[:, :3]).astype('float32')
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10

        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                if not self.cfg.frames_creation.use_gt_masks:
                    if len(distances) < 1:
                        continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars
        return True

    def load_and_prepare_lidar_scan_from_multiple_waymo(self, save=False):

        transformations = self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Convert to current frame")
        car_locations, car_locations_lidar, car_locations_info, car_locations_masks, detectron_output_arr = self.convert_to_current_frame(transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Perform 3D tracking")
        moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks = self.perform_3D_tracking(car_locations, car_locations_lidar, car_locations_info, car_locations_masks)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from features")
        cars = self.create_cars_from_extracted_feats_3DTrack(moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks)

        # Delete all moving cars which cannot be seen in the reference frames, thus probably not labeled.
        if not self.cfg.general.supress_debug_prints:
            print("Decide if moving or standing car")
        cars = self.decide_if_standing_or_moving(cars)
        #cars = self.decide_if_moving(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Filter cars")
        # Remove all moving outside of the ref frame
        cars = self.filter_moving_and_not_visible(cars)
        cars = self.choose_proper_mask(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Final Modifications")
        cars = self.standing_concatenate_lidar(cars)
        cars = self.filter_hidden_standing_cars_tracked(cars)
        cars = self.moving_lidar_keep_ref(cars)

        for car in cars:
            if car.lidar is not None:
                #To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd",'wb') as f:
                    f.write(compressed_arr)

        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.waymo_lidar[self.pic_index].T
            idx = 0
            for car in cars:
                idx += 1
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def load_and_prepare_lidar_scan_from_multiple_waymo_tracker(self, save=False):
        # The reference frame - from IMU to World
        if not self.cfg.general.supress_debug_prints:
            print("Calculating transformations")
        transformations = self.calculate_transformations_waymo(self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after)

        if not self.cfg.general.supress_debug_prints:
            print("Create cars from extracted features")
        cars = self.create_cars_from_extracted_feats()

        if not self.cfg.general.supress_debug_prints:
            print("Transform points into reference frame")
        #Transform all points into this reference frame
        cars = self.transform_lidar_points_to_reference_frame(cars, transformations)
        cars = self.transform_lidar_positions_to_reference_frame(cars, transformations)

        if not self.cfg.general.supress_debug_prints:
            print("Decide if standing or moving")
        #Now we need to decide if they are moving or not.
        cars = self.decide_if_standing_or_moving(cars)
        #cars = self.decide_if_moving(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Remove moving cars which are not visible")
        #Remove all moving outside of the ref frame
        cars = self.filter_moving_and_not_visible(cars)

        if not self.cfg.general.supress_debug_prints:
            print("Final modifications")
        #Now we have all we need. For standing concat the lidar, for moving, keep only the ref one
        cars = self.standing_concatenate_lidar(cars)
        cars = self.filter_hidden_standing_cars_tracked(cars)
        cars = self.moving_lidar_keep_ref(cars)

        for car in cars:
            if car.lidar is not None:
                #To reduce the size of the lidar in storage
                if len(car.lidar) > 10000:
                    car.lidar = self.downsample_random(car.lidar[:, :3], 10000)
                car.lidar = car.lidar.astype(np.float32)

        # We want to save the results
        if save:
            compressed_arr = zstd.compress(pickle.dumps(cars, pickle.HIGHEST_PROTOCOL))

            if self.cfg.frames_creation.use_growing_for_point_extraction:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name)

                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
            else:
                if not os.path.isdir(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name):
                    os.mkdir(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name)
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'wb') as f:
                    f.write(compressed_arr)
        else:
            # If we do not save, which means we continue in optimizing, I want to remember the original scan for finding locations of the cars.
            self.lidar = self.waymo_lidar[self.pic_index].T
            for car in cars:
                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T

            self.cars = sorted(cars, key=lambda x: x.lidar.shape[1], reverse=True)

    def choose_proper_mask(self, cars, waymo=True):
        for car in cars:
            hidden = True
            car.all_masks = copy.deepcopy(car.mask)
            for z in range(len(car.locations)):
                if car.locations[z] is not None:
                    if waymo:
                        frame_idx = car.info[z][1]
                        if self.pic_index == frame_idx:
                            car.mask = car.mask[z]
                            car.img_index = car.info[z][2]
                            hidden = False
                            break
                    else:
                        frame_idx = car.locations[z][3]
                        if frame_idx == 0:
                            car.mask = car.mask[z]
                            car.ref_idx = z
                            hidden = False
                            break
            if hidden:
                car.mask = None
        return cars

    def choose_proper_mesh(self, pedestrians, waymo=True):
        for pedestrian in pedestrians:
            hidden = True
            all_meshes = copy.deepcopy(pedestrian.mesh) # Keep a copy of all meshes
            pedestrian.mesh = None # Default to None if reference frame mesh isn't found

            for z in range(len(pedestrian.locations)):
                if pedestrian.locations[z] is not None:
                    if waymo:
                        # Assuming Waymo might use an 'info' attribute like in choose_proper_mask
                        # This part needs verification based on actual Waymo data structure for pedestrians
                        if hasattr(pedestrian, 'info') and pedestrian.info[z] is not None:
                            frame_idx = pedestrian.info[z][1]
                            if self.pic_index == frame_idx:
                                pedestrian.mesh = all_meshes[z] # Assign the correct mesh
                                hidden = False
                                break
                        else:
                                # Fallback or error handling if Waymo structure is different
                                pass
                    else: # Assuming KITTI logic
                        # Ensure location has enough elements before accessing index 3
                        if len(pedestrian.locations[z]) > 3:
                            frame_idx = pedestrian.locations[z][3]
                            if frame_idx == 0: # Reference frame for KITTI
                                pedestrian.mesh = all_meshes[z] # Assign the correct mesh
                                hidden = False
                                break
            # If hidden remains True, pedestrian.mesh is already None (set at the beginning)
        return pedestrians

    def create_cars_from_extracted_feats(self):
        cars = []
        for i in range(len(self.compressed_detandtracked)):
            car = self.Car()
            decompressed_data = zstd.decompress(self.compressed_detandtracked[i])
            to_load = pickle.loads(decompressed_data)

            car.lidar = to_load.lidar_points
            car.locations = to_load.lidar_locations
            car.info = to_load.lidar_info

            for z in range(len(car.info)):
                if car.info[z] is not None:
                    if car.info[z][1] == self.pic_index:
                        tmp_mask = to_load.masks[z]
                        tmp_img_index = car.info[z][2]
                        if tmp_mask is not None:
                            # Now convert the stitched img back to "normal" one
                            # TODO, take both masks and utilize them...
                            car.mask, car.img_index = self.convert_stitched_img_to_normal(tmp_mask, tmp_img_index)
                        break

            cars.append(car)

        return cars

    def create_cars_from_extracted_feats_3DTrack(self, moving_cars, moving_cars_lidar, moving_cars_info, moving_cars_masks, moving_cars_scores=None):
        cars = []
        for i in range(len(moving_cars)):
            car = self.Car()
            car.lidar = moving_cars_lidar[i]
            car.locations = moving_cars[i]
            if moving_cars_info is not None:
                car.info = moving_cars_info[i]
            car.mask = moving_cars_masks[i]
            if moving_cars_scores is not None:
                car.score = moving_cars_scores[i]

            cars.append(car)

        return cars

    def create_cars_from_extracted_feats_3DTrack_pedestrians(self, moving_pedestrians):
        peds = []
        for i in range(len(moving_pedestrians)):
            cur_ped = moving_pedestrians[i]
            ped = self.Pedestrian()

            ped.lidar = []
            ped.mask = []
            ped.locations = []
            ped.bbox = []
            ped.keypoints = []
            cyclist_flag = []

            for j in range(len(cur_ped)):
                ped.lidar.append(cur_ped[j]['lidar'])
                ped.mask.append(cur_ped[j]['mask'])
                ped.locations.append(cur_ped[j]['location'])
                if 'bbox' in cur_ped[j]:
                    ped.bbox.append(cur_ped[j]['bbox'])
                else:
                    mask_np = cur_ped[j]['mask']
                    if isinstance(mask_np, torch.Tensor):
                        mask_np = mask_np.cpu().numpy()
                    pos = np.where(mask_np)
                    if len(pos[0]) > 0:
                        xmin = np.min(pos[1])
                        xmax = np.max(pos[1])
                        ymin = np.min(pos[0])
                        ymax = np.max(pos[0])
                        ped.bbox.append([xmin, ymin, xmax, ymax])
                    else:
                        ped.bbox.append([0, 0, 0, 0])

                if 'keypoints' in cur_ped[j]:
                    ped.keypoints.append(cur_ped[j]['keypoints'])
                else:
                    ped.keypoints.append([])
                cyclist_flag.append(cur_ped[j]['cyclist_flag'])
            
            if 'score' in cur_ped[0]:
                ped.score = cur_ped[0]['score']
            else:
                ped.score = 0.4

            # Print the mean of cyclist flag with additional text
            print(f"The mean of the cyclist flag is: {np.mean(cyclist_flag):.2f}")
            if np.mean(cyclist_flag) > 0.5:
                ped.cyclist = True
            else:
                ped.cyclist = False

            peds.append(ped)
        return peds

    def convert_stitched_img_to_normal(self, mask, img_index):
        # Inverse of the mask
        if img_index == 0:
            if self.cfg.general.device == 'cpu':
                img0, img1 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img0, img1 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img0 = img0.numpy()[-886:, :]
            img1 = img1.numpy()

            if np.sum(img0) > np.sum(img1):
                return img0, 4
            else:
                return img1, 2

        elif img_index == 1:
            if self.cfg.general.device == 'cpu':
                img1, img2 = self.inverse_of_mask_img01(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img1, img2 = self.inverse_of_mask_img01(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img1 = img1.numpy()
            img2 = img2.numpy()

            if np.sum(img1) > np.sum(img2):
                return img1, 2
            else:
                return img2, 1


        elif img_index == 2:
            if self.cfg.general.device == 'cpu':
                img3, img2 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img3, img2 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img2 = img2.numpy()
            img3 = img3.numpy()

            if np.sum(img2) > np.sum(img3):
                return img2, 1
            else:
                return img3, 3

        else:
            if self.cfg.general.device == 'cpu':
                img4, img3 = self.inverse_of_mask_img23(mask.to(torch.float32), self.homos_all[img_index])
            else:
                img4, img3 = self.inverse_of_mask_img23(mask.cuda().to(torch.float32), self.homos_all[img_index])
            img3 = img3.numpy()
            img4 = img4.numpy()[-886:, :]

            if np.sum(img3) > np.sum(img4):
                return img3, 3
            else:
                return img4, 5

    def moving_lidar_keep_ref(self, cars, waymo=True):
        for car in cars:
            if car.moving:
                for z in range(len(car.locations)):
                    if car.locations[z] is not None:
                        if waymo:
                            frame_idx = car.info[z][1]
                            if self.pic_index == frame_idx:
                                car.lidar = car.lidar[z]
                                break
                        else:
                            frame_idx = car.locations[z][3]
                            if frame_idx == 0:
                                car.lidar = car.lidar[z]
                                break
        return cars

    def standing_concatenate_lidar(self, cars):
        for car in cars:
            if len(car.lidar) > 0 and not car.moving:
                tmp_lidar = [arr for arr in car.lidar if arr is not None]
                if len(tmp_lidar) > 0:
                    car.lidar = np.concatenate(tmp_lidar, axis=0)
                else:
                    car.lidar = None
        return cars

    def standing_concatenate_lidar_clever(self, cars, transformations):
        for car in cars:
            if len(car.lidar) > 0 and not car.moving:
                locations = [arr for arr in car.locations if arr is not None]
                dists = []
                for i in range(len(locations)):
                    loc = locations[i]
                    frame_idx = loc[3]
                    transf_idx = self.cfg.frames_creation.nscans_before + frame_idx
                    cur_transf = transformations[int(transf_idx)]
                    cur_transf = np.linalg.inv(cur_transf)
                    cur_mask = car.all_masks[i]
                    truncated = np.any(cur_mask[:10, :]) | np.any(cur_mask[-10:, :])
                    loc = np.matmul(cur_transf[0:3, 0:3], loc[:3].T).T + cur_transf[0:3, 3]
                    dist = np.sqrt(loc[0] ** 2 + loc[2] ** 2)
                    if truncated:
                        dist += 5.
                    dists.append(dist)

                best_idxs = np.argsort(np.array(dists)).tolist()
                tmp_lidar = [arr for arr in car.lidar if arr is not None]
                tmp_lidar = [tmp_lidar[i] for i in best_idxs][:10]
                if len(tmp_lidar) > 0:
                    car.lidar = np.concatenate(tmp_lidar, axis=0)
                else:
                    car.lidar = None
                car.all_masks = None
        return cars

    def decide_if_moving(self, cars):
        for car in cars:
            dists = []
            for z in range(len(car.locations) - 1):
                if car.locations[z] is not None and car.locations[z + 1] is not None:
                    dists.append(np.linalg.norm(car.locations[z][:2] - car.locations[z + 1][:2]))
            median = np.median(dists)
            if np.abs(median) > 2. / 10.:
                car.moving = True
            else:
                car.moving = False

        return cars

    def filter_moving_and_not_visible(self, cars, waymo=True):
        out_cars = []

        for car in cars:
            moving_and_hidden = True
            if car.moving:
                for z in range(len(car.locations)):
                    if car.locations[z] is not None:
                        if waymo:
                            frame_idx = car.info[z][1]
                            if self.pic_index == frame_idx:
                                moving_and_hidden = False
                                break
                        else:
                            frame_idx = car.locations[z][3]
                            if frame_idx == 0:
                                moving_and_hidden = False
                                break
                if not moving_and_hidden:
                    out_cars.append(car)
            else:
                out_cars.append(car)

        return out_cars

    def filter_hidden_standing_cars_tracked(self, cars, waymo=True):
        out_cars = []

        if self.cfg.frames_creation.use_pseudo_lidar:
            for car in cars:
                if car.mask is not None:
                    out_cars.append(car)
        else:
            # First build the faiss index of lidar
            if waymo:
                cur_lidar = self.waymo_lidar[self.pic_index][:, :3]
            else:
                if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                     path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + str(int(self.number)).zfill(10) + '.bin'
                     cur_lidar = np.array(load_velo_scan(path_to_cur_velo))
                     
                     calib_path_velo = self.cfg.paths.all_dataset_path + 'calibration/calib_cam_to_velo.txt'
                     with open(calib_path_velo, 'r') as f:
                         line = f.readline()
                         values = [float(x) for x in line.split()]
                         if len(values) == 12:
                             T_cam_to_velo = np.array(values).reshape(3, 4)
                             T_cam_to_velo = np.vstack((T_cam_to_velo, [0, 0, 0, 1]))
                         elif len(values) == 16:
                             T_cam_to_velo = np.array(values).reshape(4, 4)
                         else:
                             T_cam_to_velo = np.eye(4)
                     
                     T_velo_to_cam = np.linalg.inv(T_cam_to_velo)
                     lidar_h = np.hstack((cur_lidar[:, :3], np.ones((cur_lidar.shape[0], 1))))
                     cur_lidar = np.matmul(T_velo_to_cam, lidar_h.T).T[:, :3]
                else:
                    cur_lidar = np.array(load_velo_scan(self.cfg.paths.kitti_path + 'object_detection/training/velodyne/' + self.file_name + '.bin'))[:, :3]
            index = self.create_faiss_tree(cur_lidar)

            for car in cars:
                if not car.moving:
                    if car.lidar is not None:
                        idx, distances, indexes = index.range_search(np.ascontiguousarray(car.lidar).astype('float32'), 0.1 ** 2)
                        if len(idx) > 0:
                            out_cars.append(car)
                else:
                    out_cars.append(car)

        return out_cars

    def transform_lidar_points_to_reference_frame(self, cars, transformations):
        for car in cars:
            for z in range(len(car.lidar)):
                if car.info[z] is not None and car.lidar[z] is not None:
                    frame_idx = car.info[z][1]
                    if np.abs(self.pic_index - frame_idx) > self.cfg.frames_creation.nscans_before:
                        car.lidar[z] = None
                    elif self.pic_index - frame_idx < 0 or self.pic_index - frame_idx > 0:
                        car.lidar[z] = np.matmul(transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 0:3], car.lidar[z].T).T
                        car.lidar[z] += transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 3]
        return cars

    def transform_lidar_positions_to_reference_frame(self, cars, transformations):
        for car in cars:
            for z in range(len(car.locations)):
                if car.info[z] is not None and car.locations[z] is not None:
                    frame_idx = car.info[z][1]
                    if np.abs(self.pic_index - frame_idx) > self.cfg.frames_creation.nscans_before:
                        car.locations[z] = None
                    elif self.pic_index - frame_idx < 0 or self.pic_index - frame_idx > 0:
                        car.locations[z] = np.matmul(transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 0:3], car.locations[z].T).T
                        car.locations[z] += transformations[self.cfg.frames_creation.nscans_before - (self.pic_index - frame_idx)][0:3, 3]
        return cars

    def calculate_transformations(self, nscans_before, nscans_after, save=False):
        frames_range = self.cfg.frames_creation.nscans_transformation_range

        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            transformations = transformations[frames_range - nscans_before: frames_range + nscans_after + 1]
            
            return transformations

        else:
            T_w_imu_ref = self.kitti_data.oxts[self.file_number].T_w_imu
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))

            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number :0>10}' + '.bin'
            if self.cfg.frames_creation.use_icp:
                lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                lidar_ref = lidar_ref[:3, :].T

            for i in range(-nscans_before, nscans_after + 1):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                #Check that we have everything we need
                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                # Load cur frame - IMU to world
                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                transformations[i + nscans_before, :, :] = T_cur_to_ref

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    def transform_dsec_velo_to_cam(self, cloud):
        if self.cam_to_lidar is None:
            raise ValueError("cam_to_lidar.yaml not loaded properly.")

        self.camrect1_to_lidar = np.array(self.cam_to_lidar['T_lidar_camRect1'])
        self.lidar_to_camrect1 = np.linalg.inv(self.camrect1_to_lidar)

        cloud_homo = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
        cloud_in_cam = np.matmul(self.lidar_to_camrect1, cloud_homo.T).T[:, :3]

        return cloud_in_cam

    def read_dsec_point_clouds(self, indices):
        clouds = {}
        for idx in indices:
            velo_path = os.path.join(self.cfg.paths.dsec_path, 'lidar_sync', self.folder, 'lidar', f'{idx:06d}.bin')
            if os.path.exists(velo_path):
                cloud = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
                cloud = cloud[:, :3]  # Discard intensity
                cloud = self.transform_dsec_velo_to_cam(cloud)
                clouds[idx] = cloud
        return clouds

    def calculate_transformations_dsec(self, nscans_before, nscans_after, save=False):
        frames_range = self.cfg.frames_creation.nscans_transformation_range

        with open(os.path.join(self.cfg.paths.dsec_path, 'calibration', self.folder, 'calibration', 'cam_to_cam.yaml'), 'r') as f:
            cam_to_cam = yaml.safe_load(f)
        with open(os.path.join(self.cfg.paths.dsec_path, 'calibration', self.folder, 'calibration', 'cam_to_lidar.yaml'), 'r') as f:
            cam_to_lidar = yaml.safe_load(f)
        
        self.cam_to_cam = cam_to_cam
        self.cam_to_lidar = cam_to_lidar
        self.cam0_to_imu = np.array([[-1., 0., 0., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])

        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            transformations = transformations[frames_range - nscans_before: frames_range + nscans_after + 1]
            return transformations

        else:
            num_of_transformations = nscans_before + nscans_after + 1
            transformations = np.zeros((num_of_transformations, 4, 4))
            for i in range(num_of_transformations):
                transformations[i] = np.eye(4)
            
            indices_to_read = []
            for i in range(-nscans_before, nscans_after + 1):
                idx = int(self.number) + i
                if idx >= 0:
                    indices_to_read.append(idx)
            
            if not indices_to_read:
                return transformations

            clouds = self.read_dsec_point_clouds(indices_to_read)

            if not clouds:
                return transformations

            valid_indices = sorted(clouds.keys())
            ref_idx = int(self.number)

            def get_trans_idx(frame_idx):
                return frame_idx - ref_idx + nscans_before

            def interpolate_transformation(T_rel, fraction):
                R_rel = T_rel[:3, :3]
                t_rel = T_rel[:3, 3]
                
                t_interp = t_rel * fraction
                
                r_rel = R.from_matrix(R_rel)
                rot_vec = r_rel.as_rotvec()
                r_interp = R.from_rotvec(rot_vec * fraction)
                R_interp = r_interp.as_matrix()
                
                T_interp = np.eye(4)
                T_interp[:3, :3] = R_interp
                T_interp[:3, 3] = t_interp
                
                return T_interp

            # Determine anchor strategy
            if ref_idx in clouds:
                # Reference frame exists
                anchor_idx = ref_idx
                T_anchor_to_ref = np.eye(4)
                
                if 0 <= get_trans_idx(anchor_idx) < len(transformations):
                    transformations[get_trans_idx(anchor_idx)] = T_anchor_to_ref
                
                forward_start_idx = anchor_idx
                backward_start_idx = anchor_idx
                T_forward_start_to_ref = T_anchor_to_ref
                T_backward_start_to_ref = T_anchor_to_ref

            else:
                # Reference frame missing, look for neighbors
                prev_valid_idx = max((v for v in valid_indices if v < ref_idx), default=None)
                next_valid_idx = min((v for v in valid_indices if v > ref_idx), default=None)

                if prev_valid_idx is not None and next_valid_idx is not None:
                    # Interpolation case
                    source = clouds[next_valid_idx]
                    target = clouds[prev_valid_idx]
                    T_next_to_prev = self.icp_point_to_plane_open3d(source, target)
                    
                    total_steps = next_valid_idx - prev_valid_idx
                    steps_to_ref = ref_idx - prev_valid_idx
                    fraction_ref = steps_to_ref / total_steps
                    
                    T_ref_to_prev = interpolate_transformation(T_next_to_prev, fraction_ref)
                    T_prev_to_ref = np.linalg.inv(T_ref_to_prev)
                    
                    # Fill in everything between prev and next (inclusive)
                    for k in range(prev_valid_idx, next_valid_idx + 1):
                        fraction_k = (k - prev_valid_idx) / total_steps
                        T_k_to_prev = interpolate_transformation(T_next_to_prev, fraction_k)
                        T_k_to_ref = np.matmul(T_prev_to_ref, T_k_to_prev)
                        
                        if 0 <= get_trans_idx(k) < len(transformations):
                            transformations[get_trans_idx(k)] = T_k_to_ref
                            
                    forward_start_idx = next_valid_idx
                    backward_start_idx = prev_valid_idx
                    
                    # T_next_to_ref = T_prev_to_ref @ T_next_to_prev
                    T_forward_start_to_ref = np.matmul(T_prev_to_ref, T_next_to_prev)
                    T_backward_start_to_ref = T_prev_to_ref

                else:
                    # Fallback to closest valid (Identity assumption)
                    anchor_idx = min(valid_indices, key=lambda x: abs(x - ref_idx))
                    T_anchor_to_ref = np.eye(4)
                    
                    if 0 <= get_trans_idx(anchor_idx) < len(transformations):
                        transformations[get_trans_idx(anchor_idx)] = T_anchor_to_ref
                        
                    forward_start_idx = anchor_idx
                    backward_start_idx = anchor_idx
                    T_forward_start_to_ref = T_anchor_to_ref
                    T_backward_start_to_ref = T_anchor_to_ref
            
            # Forward pass
            last_valid_idx = forward_start_idx
            T_last_to_ref = T_forward_start_to_ref
            end_idx = ref_idx + nscans_after

            for cur_idx in range(forward_start_idx + 1, end_idx + 1):
                if cur_idx in clouds:
                    source = clouds[cur_idx]
                    target = clouds[last_valid_idx]
                    T_cur_to_last = self.icp_point_to_plane_open3d(source, target)
                    T_cur_to_ref = np.matmul(T_last_to_ref, T_cur_to_last)
                    
                    if 0 <= get_trans_idx(cur_idx) < len(transformations):
                        transformations[get_trans_idx(cur_idx)] = T_cur_to_ref

                    # Interpolate
                    steps = cur_idx - last_valid_idx
                    if steps > 1:
                        for k in range(1, steps):
                            interp_idx = last_valid_idx + k
                            fraction = k / steps
                            T_interp_rel = interpolate_transformation(T_cur_to_last, fraction)
                            T_interp_to_ref = np.matmul(T_last_to_ref, T_interp_rel)
                            
                            if 0 <= get_trans_idx(interp_idx) < len(transformations):
                                transformations[get_trans_idx(interp_idx)] = T_interp_to_ref

                    T_last_to_ref = T_cur_to_ref
                    last_valid_idx = cur_idx

            # Backward pass
            last_valid_idx = backward_start_idx
            T_last_to_ref = T_backward_start_to_ref
            start_idx = ref_idx - nscans_before

            for cur_idx in range(backward_start_idx - 1, start_idx - 1, -1):
                if cur_idx in clouds:
                    source = clouds[cur_idx]
                    target = clouds[last_valid_idx]
                    T_cur_to_last = self.icp_point_to_plane_open3d(source, target)
                    T_cur_to_ref = np.matmul(T_last_to_ref, T_cur_to_last)
                    
                    if 0 <= get_trans_idx(cur_idx) < len(transformations):
                        transformations[get_trans_idx(cur_idx)] = T_cur_to_ref

                    # Interpolate
                    steps = last_valid_idx - cur_idx
                    if steps > 1:
                        for k in range(1, steps):
                            interp_idx = last_valid_idx - k
                            fraction = k / steps
                            T_interp_rel = interpolate_transformation(T_cur_to_last, fraction)
                            T_interp_to_ref = np.matmul(T_last_to_ref, T_interp_rel)
                            
                            if 0 <= get_trans_idx(interp_idx) < len(transformations):
                                transformations[get_trans_idx(interp_idx)] = T_interp_to_ref

                    T_last_to_ref = T_cur_to_ref
                    last_valid_idx = cur_idx

        # Save Open3D transformations to file instead of printing
        if False:
            try:
                out_dir = getattr(self.cfg.paths, 'merged_frames_path', None) or os.getcwd()
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "transformations_open3d.txt")
                with open(out_path, "w") as f:
                    f.write("Open3D transformations\n")
                    for i, t in enumerate(transformations[:, :3, 3]):
                        frame_idx = i - nscans_before
                        f.write("{} {:.6f} {:.6f} {:.6f}\n".format(frame_idx, float(t[0]), float(t[1]), float(t[2])))
                if not self.cfg.general.supress_debug_prints:
                    print(f"Open3d transformations saved to: {out_path}")
            except Exception as e:
                if not self.cfg.general.supress_debug_prints:
                    print("Failed to save transformations file:", e)

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    def calculate_transformations_dsec_kiss(self, nscans_before, nscans_after, save=False):
        frames_range = self.cfg.frames_creation.nscans_transformation_range

        with open(os.path.join(self.cfg.paths.dsec_path, 'calibration', self.folder, 'calibration', 'cam_to_cam.yaml'), 'r') as f:
            cam_to_cam = yaml.safe_load(f)
        with open(os.path.join(self.cfg.paths.dsec_path, 'calibration', self.folder, 'calibration', 'cam_to_lidar.yaml'), 'r') as f:
            cam_to_lidar = yaml.safe_load(f)
        
        self.cam_to_cam = cam_to_cam
        self.cam_to_lidar = cam_to_lidar
        self.cam0_to_imu = np.array([[-1., 0., 0., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])

        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            transformations = transformations[frames_range - nscans_before: frames_range + nscans_after + 1]

            return transformations

        else:
            try:
                from kiss_icp.config import KISSConfig
                from kiss_icp.kiss_icp import KissICP
            except ImportError:
                print("KISS-ICP is not installed. Please install it via 'pip install kiss-icp'. Falling back to standard method.")
                return self.calculate_transformations_dsec(nscans_before, nscans_after, save)

            num_of_transformations = nscans_before + nscans_after + 1
            transformations = np.zeros((num_of_transformations, 4, 4))
            for i in range(num_of_transformations):
                transformations[i] = np.eye(4)
            
            indices_to_read = []
            for i in range(-nscans_before, nscans_after + 1):
                idx = int(self.number) + i
                if idx >= 0:
                    indices_to_read.append(idx)
            
            if not indices_to_read:
                return transformations

            clouds = self.read_dsec_point_clouds(indices_to_read)

            if not clouds:
                return transformations

            valid_indices = sorted(clouds.keys())
            ref_idx = int(self.number)

            # Initialize KISS-ICP
            config = KISSConfig()
            config.mapping.voxel_size = 0.25
            odometry = KissICP(config)
            
            # Store poses: idx -> 4x4 matrix
            poses = {}

            # Load timestamps
            timestamps_path = os.path.join(self.cfg.paths.dsec_path, 'images', self.folder, 'images', 'timestamps.txt')
            all_timestamps = []
            if os.path.exists(timestamps_path):
                with open(timestamps_path, 'r') as f:
                    for line in f:
                        try:
                            all_timestamps.append(float(line.strip()))
                        except ValueError:
                            pass
            else:
                if not self.cfg.general.supress_debug_prints:
                    print(f"Warning: timestamps.txt not found at {timestamps_path}. Using synthetic 10Hz timestamps.")
            
            # Determine start time for normalization to avoid large numbers
            start_ts = 0.0
            if all_timestamps and valid_indices:
                first_idx = valid_indices[0]
                if first_idx < len(all_timestamps):
                    start_ts = all_timestamps[first_idx]
                    if start_ts > 1e16: start_ts /= 1e9
                    elif start_ts > 1e10: start_ts /= 1e6

            # Run KISS-ICP on all available scans in order
            registered_indices = []
            for i, idx in enumerate(valid_indices):
                frame = clouds[idx]
                
                # Remove NaNs and Infs
                if np.any(~np.isfinite(frame)):
                    frame = frame[np.isfinite(frame).all(axis=1)]

                if frame.shape[0] < 100:
                    if not self.cfg.general.supress_debug_prints:
                        print(f"Warning: Frame {idx} has too few points ({frame.shape[0]}). Skipping.")
                    continue

                # KISS-ICP expects float64 usually
                
                if all_timestamps and idx < len(all_timestamps):
                     ts = all_timestamps[idx]
                     # Convert to seconds if necessary (heuristic)
                     if ts > 1e16: # nanoseconds
                         ts /= 1e9
                     elif ts > 1e10: # microseconds
                         ts /= 1e6
                     
                     # Normalize timestamp (start from 10.0 to avoid 0.0 issues)
                     ts = (ts - start_ts) + 10.0
                else:
                     # Synthetic 10Hz (0.1s per frame)
                     ts = 10.0 + (i * 0.1)
                
                timestamps = np.full(frame.shape[0], ts, dtype=np.float64)

                odometry.register_frame(frame.astype(np.float64), timestamps)
                poses[idx] = odometry.poses[-1]
                registered_indices.append(idx)
            
            valid_indices = registered_indices
            if not valid_indices:
                return transformations

            def get_trans_idx(frame_idx):
                return frame_idx - ref_idx + nscans_before

            def interpolate_transformation(T_rel, fraction):
                R_rel = T_rel[:3, :3]
                t_rel = T_rel[:3, 3]
                
                t_interp = t_rel * fraction
                
                r_rel = R.from_matrix(R_rel)
                rot_vec = r_rel.as_rotvec()
                r_interp = R.from_rotvec(rot_vec * fraction)
                R_interp = r_interp.as_matrix()
                
                T_interp = np.eye(4)
                T_interp[:3, :3] = R_interp
                T_interp[:3, 3] = t_interp
                
                return T_interp

            full_poses = {}
            
            # 1. Interpolate between valid frames
            for i in range(len(valid_indices) - 1):
                idx1 = valid_indices[i]
                idx2 = valid_indices[i+1]
                
                T1 = poses[idx1]
                T2 = poses[idx2]
                
                # T2 = T1 @ T_rel -> T_rel = inv(T1) @ T2
                T_rel = np.linalg.inv(T1) @ T2
                
                full_poses[idx1] = T1
                
                steps = idx2 - idx1
                for k in range(1, steps):
                    fraction = k / steps
                    T_interp_rel = interpolate_transformation(T_rel, fraction)
                    T_interp = T1 @ T_interp_rel
                    full_poses[idx1 + k] = T_interp
            
            full_poses[valid_indices[-1]] = poses[valid_indices[-1]]
            
            # 2. Extrapolate edges (nearest neighbor / constant pose)
            min_valid = valid_indices[0]
            max_valid = valid_indices[-1]
            
            start_idx = ref_idx - nscans_before
            end_idx = ref_idx + nscans_after
            
            for idx in range(start_idx, min_valid):
                full_poses[idx] = poses[min_valid]
                
            for idx in range(max_valid + 1, end_idx + 1):
                full_poses[idx] = poses[max_valid]
                
            # 3. Compute transformations relative to reference frame
            if ref_idx in full_poses:
                T_ref_world = full_poses[ref_idx]
            else:
                # Should not happen given the logic above
                T_ref_world = np.eye(4)
                
            T_ref_world_inv = np.linalg.inv(T_ref_world)
            
            for idx in range(start_idx, end_idx + 1):
                if idx in full_poses:
                    T_cur_world = full_poses[idx]
                    # We want T_cur_to_ref
                    # P_ref = T_cur_to_ref @ P_cur
                    # We have P_world = T_cur_world @ P_cur
                    # And P_ref = T_ref_world_inv @ P_world
                    # So P_ref = T_ref_world_inv @ T_cur_world @ P_cur
                    # Thus T_cur_to_ref = T_ref_world_inv @ T_cur_world
                    
                    T_cur_to_ref = T_ref_world_inv @ T_cur_world
                    
                    t_idx = get_trans_idx(idx)
                    if 0 <= t_idx < len(transformations):
                        transformations[t_idx] = T_cur_to_ref

        # Save KISS transformations to file instead of printing
        try:
            out_dir = getattr(self.cfg.paths, 'merged_frames_path', None) or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "transformations_kiss.txt")
            with open(out_path, "w") as f:
                f.write("KISS transformations\n")
                for i, t in enumerate(transformations[:, :3, 3]):
                    frame_idx = i - nscans_before
                    f.write("{} {:.6f} {:.6f} {:.6f}\n".format(frame_idx, float(t[0]), float(t[1]), float(t[2])))
                
                if not self.cfg.general.supress_debug_prints:
                    print(f"KISS transformations saved to: {out_path}")

        except Exception as e:
            if not self.cfg.general.supress_debug_prints:
                print("Failed to save transformations file:", e)
        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    def calculate_transformations_all(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            # We know that the transformation were generated for -50 and 120 frames.
            transformations_out = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            with open(self.cfg.paths.all_dataset_path + '/data_poses/' + self.folder + '/cam0_to_world.txt', 'r') as f:
                lines = f.readlines()
                max_frame = lines[-1].split(' ')[0]
                transformations = np.zeros((int(max_frame) + 1, 4, 4))
                for line in lines:
                    frame, T = line.split(' ', 1)
                    T = np.array([float(x) for x in T.split()]).reshape((4, 4))
                    transformations[int(frame), :, :] = T

                if int(self.number) > int(max_frame):
                    transformations = np.zeros((nscans_before + nscans_after + 1, 4, 4))
                    if save:
                        np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
                    return transformations

            T_ref = transformations[int(self.number)]
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations_out = np.zeros((num_of_transformations, 4, 4))
            if not np.all(T_ref == 0):
                for i in range(-nscans_before, nscans_after + 1):
                    # Load cur frame - IMU to world
                    if int(self.number) + i < 0 or int(self.number) + i >= len(transformations):
                        continue
                    T_cur = transformations[int(self.number) + i]
                    if np.all(T_cur == 0):
                        continue

                    T_cur_to_ref = np.matmul(np.linalg.inv(T_ref), T_cur)

                    transformations_out[i + nscans_before, :, :] = T_cur_to_ref

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations_out)
        else:
            return transformations_out

    def calculate_transformations_waymoc(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            # We know that the transformation were generated for -50 and 120 frames.
            transformations_out = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            transformations_out = np.zeros((nscans_before + nscans_after + 1, 4, 4), dtype=np.float32)
            ref_path = os.path.join(self.cfg.paths.all_dataset_path, 'training', self.folder, 'calib', f'{self.number :0>10}' + '.txt')
            ref_calib = self.load_full_calib(ref_path)
            ref_pose = ref_calib['Cur_pose'].reshape((4, 4))
            inv_ref_pose = np.linalg.inv(ref_pose)
            for i in range(int(self.number) - nscans_before, int(self.number) + nscans_after + 1):
                path_to_calib = os.path.join(self.cfg.paths.all_dataset_path, 'training', self.folder, 'calib', f'{i:0>10}.txt')
                if os.path.exists(path_to_calib):
                    calib = self.load_full_calib(path_to_calib)
                    cur_pose = calib['Cur_pose'].reshape((4, 4))
                    velo_to_cam = calib['Tr_velo_to_cam'].reshape((3, 4))
                    velo_to_cam = np.vstack((velo_to_cam, np.array([0, 0, 0, 1])))
                    cam_to_velo = np.linalg.inv(velo_to_cam)

                    T_cur_to_ref = np.matmul(cur_pose, cam_to_velo)
                    T_cur_to_ref = np.matmul(np.linalg.inv(ref_pose), T_cur_to_ref)
                    T_cur_to_ref = np.matmul(velo_to_cam, T_cur_to_ref)
                    #print(i, T_cur_to_ref)
                    transformations_out[i - int(self.number) + nscans_before, :, :] = T_cur_to_ref

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations_out)
        else:
            return transformations_out

    #Always works with ICP
    def calculate_transformationsV2(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + '.npy')
            transformations = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))
            transformations[nscans_before, :, :] = np.eye(4)
            tmp_transformations = np.zeros((num_of_transformations, 4, 4))

            jump_step = 5

            for i in range(tmp_transformations.shape[0]):
                tmp_transformations[i, :, :] = np.eye(4)
                transformations[i, :, :] = np.eye(4)

            if not self.cfg.general.supress_debug_prints:
                print("Calculating transformations with step")
            for i in range(-nscans_before, nscans_after + 1, jump_step):
                if i == 0:
                    continue

                if i < 0:
                    if self.cfg.frames_creation.use_pseudo_lidar:
                        path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                        path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i + jump_step:0>10}' + '.npz'
                    else:
                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step:0>10}' + '.bin'
                else:
                    if self.cfg.frames_creation.use_pseudo_lidar:
                        path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i - jump_step:0>10}' + '.npz'
                        path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                    else:
                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step:0>10}' + '.bin'
                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                    if i < 0 and i + jump_step * 2 <= 0:
                        if self.cfg.frames_creation.use_pseudo_lidar:
                            path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                            path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i + jump_step * 2:0>10}' + '.npz'
                        else:
                            path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 2:0>10}' + '.bin'
                    elif i >= 0 and i - jump_step * 2 >= 0:
                        if self.cfg.frames_creation.use_pseudo_lidar:
                            path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i - jump_step * 2:0>10}' + '.npz'
                            path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                        else:
                            path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 2 :0>10}' + '.bin'
                            path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                    else:
                        continue

                    if self.file_number + i < 0 or self.file_number + i >= len(
                            self.kitti_data.oxts) or not os.path.exists(
                            path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                        if i < 0 and i + jump_step * 3 <= 0:
                            if self.cfg.frames_creation.use_pseudo_lidar:
                                path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                                path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i + jump_step * 3:0>10}' + '.npz'
                            else:
                                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 3:0>10}' + '.bin'
                        elif i >= 0 and i - jump_step * 3 >= 0:
                            if self.cfg.frames_creation.use_pseudo_lidar:
                                path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i - jump_step * 3:0>10}' + '.npz'
                                path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                            else:
                                path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 3:0>10}' + '.bin'
                                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                        else:
                            continue

                        if self.file_number + i < 0 or self.file_number + i >= len(
                                self.kitti_data.oxts) or not os.path.exists(
                            path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                            if i < 0 and i + jump_step * 4 <= 0:
                                if self.cfg.frames_creation.use_pseudo_lidar:
                                    path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                                    path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i + jump_step * 4:0>10}' + '.npz'
                                else:
                                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 4:0>10}' + '.bin'
                            elif i >= 0 and i - jump_step * 4 >= 0:
                                if self.cfg.frames_creation.use_pseudo_lidar:
                                    path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i - jump_step * 4:0>10}' + '.npz'
                                    path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                                else:
                                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 4:0>10}' + '.bin'
                                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                            else:
                                continue

                            if self.file_number + i < 0 or self.file_number + i >= len(
                                    self.kitti_data.oxts) or not os.path.exists(
                                path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                                if i < 0 and i + jump_step * 5 <= 0:
                                    if self.cfg.frames_creation.use_pseudo_lidar:
                                        path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                                        path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i + jump_step * 5:0>10}' + '.npz'
                                    else:
                                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i + jump_step * 5:0>10}' + '.bin'
                                elif i >= 0 and i - jump_step * 5 >= 0:
                                    if self.cfg.frames_creation.use_pseudo_lidar:
                                        path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + i - jump_step * 5:0>10}' + '.npz'
                                        path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                                    else:
                                        path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i - jump_step * 5:0>10}' + '.bin'
                                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                                else:
                                    continue

                                if self.file_number + i < 0 or self.file_number + i >= len(
                                        self.kitti_data.oxts) or not os.path.exists(
                                    path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                                    continue
                                else:
                                    if i < 0:
                                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 5].T_w_imu
                                    else:
                                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 5].T_w_imu
                                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                            else:
                                if i < 0:
                                    T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                    T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 4].T_w_imu
                                else:
                                    T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 4].T_w_imu
                                    T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                        else:
                            if i < 0:
                                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                                T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 3].T_w_imu
                            else:
                                T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 3].T_w_imu
                                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                    else:
                        if i < 0:
                            T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                            T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step * 2].T_w_imu
                        else:
                            T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step * 2].T_w_imu
                            T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                else:
                    if i < 0:
                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i + jump_step].T_w_imu
                    else:
                        T_w_imu_ref = self.kitti_data.oxts[self.file_number + i - jump_step].T_w_imu
                        T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu

                if self.cfg.frames_creation.use_pseudo_lidar:
                    lidar_ref = load_pseudo_lidar(path_to_ref_velo)[:, :3]
                else:
                    lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                    lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                    lidar_ref = lidar_ref[:3, :].T

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                if self.cfg.frames_creation.use_pseudo_lidar:
                    lidar_cur = load_pseudo_lidar(path_to_cur_velo)[:, :3].T
                else:
                    lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                    lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                    lidar_cur = lidar_cur[:3, :]
                # Transform the points between frames.
                lidar_cur = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur).T
                lidar_cur += T_cur_to_ref[0:3, 3]

                new_transformation = self.icp_point_to_plane_open3d(lidar_cur, lidar_ref)
                new_transformation = copy.deepcopy(new_transformation)
                rot = R.from_matrix(new_transformation[:3, :3])
                T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

                tmp_transformations[i + nscans_before, :, :] = T_cur_to_ref

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations with step")
            for i in range(-nscans_before, 0, jump_step):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, jump_step):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            for i in range(nscans_after, 0, -jump_step):
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i:0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, -jump_step):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations for all frames")
            for i in range(-nscans_before, nscans_after + 1):
                if i % jump_step == 0:
                    continue
                if i < 0:
                    ref_index = int(np.rint(np.ceil(i/float(jump_step))*jump_step))
                else:
                    ref_index = int(np.rint(np.floor(i/float(jump_step))*jump_step))

                if np.array_equal(transformations[nscans_before + ref_index], np.eye(4)) and ref_index != 0:
                    if i < 0:
                        ref_index = int(np.rint(np.ceil(i / float(jump_step * 2)) * jump_step * 2))
                    else:
                        ref_index = int(np.rint(np.floor(i / float(jump_step * 2)) * jump_step * 2))

                if np.array_equal(transformations[nscans_before + ref_index], np.eye(4)) and ref_index != 0:
                    if i < 0:
                        ref_index = int(np.rint(np.ceil(i / float(jump_step * 3)) * jump_step * 3))
                    else:
                        ref_index = int(np.rint(np.floor(i / float(jump_step * 3)) * jump_step * 3))

                if self.cfg.frames_creation.use_pseudo_lidar:
                    path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
                    path_to_ref_velo = self.pseudo_lidar_folder + f'{self.file_number + ref_index :0>10}' + '.npz'
                else:
                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                    path_to_ref_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + ref_index :0>10}' + '.bin'

                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_velo) or not os.path.exists(path_to_ref_velo):
                    continue

                T_w_imu_cur = self.kitti_data.oxts[self.file_number + i].T_w_imu
                T_w_imu_ref = self.kitti_data.oxts[self.file_number + ref_index].T_w_imu

                if self.cfg.frames_creation.use_pseudo_lidar:
                    lidar_ref = load_pseudo_lidar(path_to_ref_velo)[:, :3]
                else:
                    lidar_ref = np.array(load_velo_scan(path_to_ref_velo))
                    lidar_ref = self.transform_velo_to_cam(self.file_name, lidar_ref, filter_points=False)
                    lidar_ref = lidar_ref[:3, :].T

                # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
                T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)
                # Now we need to go from IMU to CAM2
                T_imu_to_cam = self.kitti_data.calib.T_cam2_imu

                T_cur_to_ref = np.matmul(T_cur_to_ref, np.linalg.inv(T_imu_to_cam))
                T_cur_to_ref = np.matmul(T_imu_to_cam, T_cur_to_ref)

                if self.cfg.frames_creation.use_pseudo_lidar:
                    lidar_cur = load_pseudo_lidar(path_to_cur_velo)[:, :3].T
                else:
                    lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                    lidar_cur = self.transform_velo_to_cam(self.file_name, lidar_cur, filter_points=False)
                    lidar_cur = lidar_cur[:3, :]
                # Transform the points between frames.
                lidar_cur = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur).T
                lidar_cur += T_cur_to_ref[0:3, 3]

                new_transformation = self.icp_point_to_plane_open3d(lidar_cur, lidar_ref)
                new_transformation = copy.deepcopy(new_transformation)
                rot = R.from_matrix(new_transformation[:3, :3])
                T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

                transformations[i + nscans_before, :, :] = np.matmul(transformations[nscans_before + ref_index], T_cur_to_ref)

        if save:
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name, transformations)
        else:
            return transformations

    def calculate_transformations_waymo(self, nscans_before, nscans_after, save=False):
        if self.load_merged_frames or self.load_transformations:
            transformations = np.load(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + "/" + str(self.pic_index) + '.npy')
            # We know that the transformation were generated for -50 and 120 frames.
            transformations = transformations[self.cfg.frames_creation.nscans_transformation_range - nscans_before: self.cfg.frames_creation.nscans_transformation_range + 1 + nscans_after]
        else:
            if os.path.isfile(self.cfg.paths.merged_frames_path + "/transformations/" + self.file_name + "/" + str(self.pic_index) + '.npy'):
                return
            num_of_transformations = nscans_before + nscans_after + 1  # +1 because we actually want 1 more
            transformations = np.zeros((num_of_transformations, 4, 4))
            transformations[nscans_before, :, :] = np.eye(4)
            tmp_transformations = np.zeros((num_of_transformations, 4, 4))

            for i in range(tmp_transformations.shape[0]):
                tmp_transformations[i, :, :] = np.eye(4)
                transformations[i, :, :] = np.eye(4)

            if not self.cfg.general.supress_debug_prints:
                print("Calculating transformations with step")
            for i in range(-nscans_before, nscans_after + 1):
                if i == 0:
                    continue

                transformation = self.get_transformation_icp(i)
                if transformation is not None:
                    tmp_transformations[i + nscans_before, :, :] = transformation
                    transformations[i + nscans_before, :, :] = transformation

            if not self.cfg.general.supress_debug_prints:
                print("Calculating absolute transformations with step")
            for i in range(-nscans_before, 0):
                if i + self.pic_index < 0:
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix

            for i in range(nscans_after, 0, -1):
                if i + self.pic_index >= len(self.waymo_data):
                    continue

                t_matrix = np.eye(4)
                for z in range(i, 0, -1):
                    t_matrix = np.matmul(tmp_transformations[z + nscans_before, :, :], t_matrix)
                transformations[i + nscans_before, :, :] = t_matrix
        if save:
            if not os.path.isdir(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name):
                os.mkdir(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name)
            np.save(self.cfg.paths.merged_frames_path + "transformations/" + self.file_name + "/" + str(self.pic_index), transformations)
        else:
            return transformations

    def get_transformation_icp(self, i):
        if i < 0:
            if i + self.pic_index < 0:
                return None
            frame_cur = self.waymo_frame[self.pic_index + i]
            frame_ref = self.waymo_frame[self.pic_index + i + 1]

            T_w_imu_cur = np.array(frame_cur.pose.transform).reshape((4, 4))
            T_w_imu_ref = np.array(frame_ref.pose.transform).reshape((4, 4))

            lidar_cur = self.waymo_lidar[self.pic_index + i][:, :3]
            lidar_ref = self.waymo_lidar[self.pic_index + i + 1][:, :3]

        else:
            if i + self.pic_index >= len(self.waymo_data):
                return None
            frame_cur = self.waymo_frame[self.pic_index + i]
            frame_ref = self.waymo_frame[self.pic_index + i - 1]

            T_w_imu_cur = np.array(frame_cur.pose.transform).reshape((4, 4))
            T_w_imu_ref = np.array(frame_ref.pose.transform).reshape((4, 4))

            lidar_cur = self.waymo_lidar[self.pic_index + i][:, :3]
            lidar_ref = self.waymo_lidar[self.pic_index + i - 1][:, :3]

        # Compute the transformation between frames - IMU cur to world then to IMU again but to the ref frame
        T_cur_to_ref = np.matmul(np.linalg.inv(T_w_imu_ref), T_w_imu_cur)

        # Transform the points between frames.
        lidar_cur_tmp = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_cur.T)
        lidar_cur_tmp += T_cur_to_ref[0:3, 3].reshape((3, 1))

        new_transformation = self.icp_point_to_plane_open3d(lidar_cur_tmp.T, lidar_ref)
        new_transformation = copy.deepcopy(new_transformation)
        T_cur_to_ref = np.matmul(new_transformation, T_cur_to_ref)

        return T_cur_to_ref

    def get_standing_car_candidates(self, transformations):
        car_locations = []
        car_locations_lidar = []
        car_locations_masks = []
        car_locations_scores = []

        if self.cfg.frames_creation.use_codetr:
            precomputed_masks = self.precompute_detectron_kitti_v2()
        else:
            precomputed_masks = self.precompute_detectron_kitti()

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            # Ignore the reference scan and also do not search for not existing datas

            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
            else:
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

            if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_velo):
                car_locations.append([])
                car_locations_lidar.append([])
                car_locations_masks.append([])
                car_locations_scores.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            # Load the velo scan
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            else:
                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            lidar_cur = self.prepare_scan(self.file_name, self.img, lidar_cur, save=False)

            masks = precomputed_masks[i + self.cfg.frames_creation.nscans_before]
            scores = None
            if isinstance(masks, dict):
                scores = masks['scores']
                masks = masks['masks']

            car_loc, lidar_points, masks, scores = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, masks, scores)
            car_locations.append(car_loc)
            car_locations_lidar.append(lidar_points)
            car_locations_masks.append(masks)
            car_locations_scores.append(scores)

        return car_locations, car_locations_lidar, car_locations_masks, car_locations_scores

    def get_pedestrians_candidates(self, transformations):
        pedestrian_locations = []
        pedestrian_lidar = []
        pedestrian_masks = []

        if self.cfg.frames_creation.use_codetr:
            precomputed_masks = self.precompute_detectron_kitti_v2(pedestrian=True)
        else:
            precomputed_masks = self.precompute_detectron_kitti() #TODO Update this for pedestrians

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.pseudo_lidar_folder + f'{self.file_number + i :0>10}' + '.npz'
            else:
                path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

            if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_velo):
                pedestrian_locations.append([])
                pedestrian_lidar.append([])
                pedestrian_masks.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            # Load the velo scan
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            else:
                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            lidar_cur = self.prepare_scan(self.file_name, self.img, lidar_cur, save=False)

            masks = precomputed_masks[i + self.cfg.frames_creation.nscans_before]
            ped_loc, lidar_points, masks = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, masks) #TODO Be aware this can be problematic as it filters out predictions with less than X points.
            pedestrian_locations.append(ped_loc)
            pedestrian_lidar.append(lidar_points)
            pedestrian_masks.append(masks)

        return pedestrian_locations, pedestrian_lidar, pedestrian_masks


    def get_precomputed_candidates_pedestrians(self, T_cur_to_ref, offset=0):
        filename = self.folder + '_' + str(int(self.number) + offset).zfill(10)
        
        path_lidar = self.cfg.paths.merged_frames_path + "candidates_lidar_pedestrians/" + filename + ".zstd"
        if not os.path.exists(path_lidar):
             return None, None, None, None, None

        with open(path_lidar, 'rb') as f:
            decompressed_data = zstd.decompress(f.read())
        lidar_points = pickle.loads(decompressed_data)
        
        with open(self.cfg.paths.merged_frames_path + "candidates_masks_pedestrians/" + filename + ".zstd", 'rb') as f:
            decompressed_data = zstd.decompress(f.read())
        out_masks = pickle.loads(decompressed_data)
        
        path_flags = self.cfg.paths.merged_frames_path + "candidates_cyclist_flags_pedestrians/" + filename + ".zstd"
        if os.path.exists(path_flags):
            with open(path_flags, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            cyclist_flags = pickle.loads(decompressed_data)
        else:
            cyclist_flags = [False] * len(lidar_points)

        path_scores = self.cfg.paths.merged_frames_path + "candidates_scores_pedestrians/" + filename + ".zstd"
        if os.path.exists(path_scores):
            with open(path_scores, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_scores = pickle.loads(decompressed_data)
        else:
            out_scores = [0.4] * len(lidar_points)

        # Apply transformation T_cur_to_ref
        means = []
        lidars = []
        masks = []
        flags = []
        scores = []

        for i in range(len(lidar_points)):
            if lidar_points[i] is None or len(lidar_points[i]) == 0:
                 continue
            
            # Transform lidar points
            pts = lidar_points[i]
            pts = np.matmul(T_cur_to_ref[0:3, 0:3], pts.T).T
            pts += T_cur_to_ref[0:3, 3]
            
            x_mean, y_mean, z_mean = np.median(pts, axis=0)
            
            means.append(np.array([x_mean, y_mean, z_mean]))
            lidars.append(pts)
            masks.append(out_masks[i])
            flags.append(cyclist_flags[i])
            scores.append(out_scores[i])

        return means, lidars, masks, flags, scores

    def get_pedestrians_candidates_all(self, transformations):
        pedestrians_out = []

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]
            
            # Try precomputed
            ped_loc, lidar_points, masks, cyclist_flags, scores = self.get_precomputed_candidates_pedestrians(T_cur_to_ref, offset=i)
            
            if ped_loc is not None:
                 # Use precomputed
                 pedestrians_for_output = []
                 for z in range(len(ped_loc)):
                    if np.abs(ped_loc[z][0]) > 0.001 and np.abs(ped_loc[z][1]) > 0.001 and np.abs(ped_loc[z][2]) > 0.001:
                        cur_ped = {}
                        cur_ped['lidar'] = lidar_points[z]
                        cur_ped['mask'] = masks[z]
                        cur_ped['location'] = ped_loc[z]
                        cur_ped['cyclist_flag'] = cyclist_flags[z]
                        cur_ped['score'] = scores[z]

                        mask = masks[z]
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        if not np.any(rows) or not np.any(cols):
                            cur_ped['bbox'] = [0, 0, 0, 0]
                        else:
                            x_min, x_max = np.where(rows)[0][[0, -1]]
                            y_min, y_max = np.where(cols)[0][[0, -1]]
                            cur_ped['bbox'] = [x_min, y_min, x_max, y_max]
                        cur_ped['keypoints'] = []

                        pedestrians_for_output.append(cur_ped)
                 pedestrians_out.append(pedestrians_for_output)
                 continue

            # If not precomputed, compute it now
            pedestrian_masks, cyclist_flags, scores = self.precompute_detectron_pedestrians_all(offset=i)

            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(int(self.number) + i).zfill(10) + '.npz'
            else:
                if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                    path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + f'{int(self.number) + i :0>10}' + '.bin'
                else:
                    filename = self.folder + '_' + str(int(self.number) + i).zfill(10)
                    path_to_cur_velo = self.cfg.paths.all_dataset_path + 'object_detection/training/velodyne/' + filename + '.bin'

            if int(self.number) + i < 0 or not os.path.exists(path_to_cur_velo):
                pedestrians_out.append([])
                continue

            # Load the velo scan
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            else:
                lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            lidar_cur = self.prepare_scan_all(self.file_name, self.img, lidar_cur, save=False)

            pedestrians_for_output = []
            ped_loc, lidar_points, masks, out_flags, out_scores = self.get_pedestrian_locations_from_img(lidar_cur, T_cur_to_ref, pedestrian_masks, cyclist_flags, scores=scores)

            for z in range(len(ped_loc)):
                if np.abs(ped_loc[z][0]) > 0.001 and np.abs(ped_loc[z][1]) > 0.001 and np.abs(ped_loc[z][2]) > 0.001:
                    cur_ped = {}
                    cur_ped['lidar'] = lidar_points[z]
                    cur_ped['mask'] = masks[z]
                    cur_ped['location'] = ped_loc[z]
                    cur_ped['cyclist_flag'] = out_flags[z]
                    cur_ped['score'] = out_scores[z]

                    mask = masks[z]
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        cur_ped['bbox'] = [0, 0, 0, 0]
                    else:
                        x_min, x_max = np.where(rows)[0][[0, -1]]
                        y_min, y_max = np.where(cols)[0][[0, -1]]
                        cur_ped['bbox'] = [x_min, y_min, x_max, y_max]
                    cur_ped['keypoints'] = []

                    pedestrians_for_output.append(cur_ped)
            
            pedestrians_out.append(pedestrians_for_output)

        return pedestrians_out

    def get_standing_car_candidates_all(self, transformations):
        car_locations = []
        car_locations_lidar = []
        car_locations_masks = []
        car_locations_ids = []
        car_locations_scores = []

        if self.cfg.frames_creation.use_gt_masks:
            path_to_ref_masks = self.cfg.paths.all_dataset_path + str(self.folder) + '/instance/' + str(int(self.number)).zfill(10) + '.png'
            #If label is not present
            if not os.path.exists(path_to_ref_masks):
                return None, None, None, None, None

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            if self.cfg.frames_creation.use_pseudo_lidar:
                path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(int(self.number) + i).zfill(10) + '.npz'
            else:
                if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                    path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + f'{int(self.number) + i :0>10}' + '.bin'
                else:
                    path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            if int(self.number) + i < 0 or not os.path.exists(path_to_cur_velo) or np.all(T_cur_to_ref == 0):
                car_locations.append([])
                car_locations_lidar.append([])
                car_locations_masks.append([])
                car_locations_scores.append([])
                if self.cfg.frames_creation.use_gt_masks:
                    car_locations_ids.append([])
                continue

            if self.cfg.frames_creation.use_gt_masks:
                path_to_cur_masks = self.cfg.paths.all_dataset_path + str(self.folder) + '/instance/' + str(int(self.number) + i).zfill(10) + '.png'
                if not os.path.exists(path_to_cur_masks):
                    car_locations.append([])
                    car_locations_lidar.append([])
                    car_locations_masks.append([])
                    car_locations_ids.append([])
                    car_locations_scores.append([])
                    continue

            # Load the velo scan
            #if self.cfg.frames_creation.use_pseudo_lidar:
                #lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
            #else:
                #lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
            #lidar_cur = self.prepare_scan_all(self.file_name, self.img, lidar_cur, save=False)

            #masks = self.precompute_detectron_all(offset=i)
            #car_loc, lidar_points, masks = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, masks, img_idx=i)
            if i ==0:
                x= 0
            if self.cfg.frames_creation.use_gt_masks:
                car_loc, lidar_points, masks, ids = self.get_precomputed_candidates_gt(T_cur_to_ref, offset=i)
                car_locations_ids.append(ids)
                scores = [1.0] * len(car_loc)
            else:
                car_loc, lidar_points, masks, scores = self.get_precomputed_candidates(T_cur_to_ref, offset=i)
            car_locations.append(car_loc)
            car_locations_lidar.append(lidar_points)
            car_locations_masks.append(masks)
            car_locations_scores.append(scores)

        return car_locations, car_locations_lidar, car_locations_masks, car_locations_ids, car_locations_scores

    def get_SAM3_detections(self, transformations, class_name="car"):
        all_objects = []
        
        pkl_path = os.path.join(self.cfg.paths.sam3_detections_path, class_name, self.map_data_cur[0] + '_' + self.map_data_cur[1], str.zfill(str(self.file_number), 6) + '.pkl')

        # Load the tracklet data
        with open(pkl_path, 'rb') as f:
            tracklet_data = pickle.load(f)

        cur_id = self.file_number

        for object_id in tracklet_data.keys():

            cur_object = tracklet_data[object_id]

            if not self.cfg.frames_creation.extract_standing_vehicles_not_visible and cur_id not in cur_object['masks']:
                continue

            if class_name == "car":
                object_class = self.Car()
            elif class_name == "pedestrian":
                object_class = self.Pedestrian()
            elif class_name == "cyclist":
                object_class = self.Pedestrian()
                object_class.cyclist = True
            else:
                raise ValueError("Unsupported class name for SAM3 detections.")

            object_class.locations = []
            object_class.lidar = []
            object_class.masks = []

            correctly_visible_in_ref_frame = True

            for frame_idx in range(-self.cfg.frames_creation.nscans_before + cur_id, self.cfg.frames_creation.nscans_after + 1 + cur_id):
                if self.cfg.frames_creation.use_pseudo_lidar:
                    path_to_cur_velo = self.pseudo_lidar_folder + f'{frame_idx :0>10}' + '.npz'
                else:
                    if self.args.dataset == 'kitti360':
                        path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                    else:
                        path_to_cur_velo = self.path_to_folder + 'velodyne_points/data/' + f'{self.file_number + i :0>10}' + '.bin'
                        #object_class.locations.append([])
                        #object_class.lidar.append([])
                        #object_class.masks.append([])
                        continue

                if frame_idx in cur_object['masks']:
                    cur_mask_rle = cur_object['masks'][frame_idx]
                    cur_mask = mask_utils.decode(cur_mask_rle).T
                    cur_mask = np.array(cur_mask > 0).astype(bool)

                    T_cur_to_ref = transformations[frame_idx - cur_id + self.cfg.frames_creation.nscans_before, :, :]

                    if self.cfg.frames_creation.use_pseudo_lidar:
                        lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
                    else:
                        lidar_cur = np.array(load_velo_scan(path_to_cur_velo))
                    lidar_cur = self.prepare_scan(self.file_name, self.img, lidar_cur, save=False)

                    location, lidar_points, masks = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, [cur_mask])

                    if len(lidar_points) == 0:
                        if frame_idx == cur_id:
                            if not self.cfg.frames_creation.extract_standing_vehicles_not_visible:
                                correctly_visible_in_ref_frame = False
                        continue

                    location = np.append(location[0], [frame_idx - cur_id])
                    
                    object_class.locations.append(location)
                    object_class.lidar.append(lidar_points[0])
                    object_class.masks.append(masks[0])

                    if frame_idx == cur_id:
                        object_class.bbox = cur_object['boxes'][frame_idx]
                        object_class.ref_idx = cur_id
                        
                else:
                    #object_class.locations.append([])
                    #object_class.lidar.append([])
                    #object_class.masks.append([])
                    continue
            
            object_class.mask = object_class.masks

            if len(object_class.locations) == 0:
                continue
            if correctly_visible_in_ref_frame:
                all_objects.append(object_class)

        return all_objects

    def get_SAM3_detections_dsec(self, transformations, lidar_cache, class_name="car"):
        all_objects = []
        
        pkl_path = os.path.join(self.cfg.paths.sam3_detections_path, class_name, self.folder, str.zfill(str(self.number), 6) + '.pkl')

        # Load the tracklet data
        with open(pkl_path, 'rb') as f:
            tracklet_data = pickle.load(f)

        cur_id = int(self.number)

        for object_id in tracklet_data.keys():

            cur_object = tracklet_data[object_id]

            if not self.cfg.frames_creation.extract_standing_vehicles_not_visible and cur_id not in cur_object['masks']:
                continue

            if class_name == "car":
                object_class = self.Car()
            elif class_name == "pedestrian":
                object_class = self.Pedestrian()
            elif class_name == "cyclist":
                object_class = self.Pedestrian()
                object_class.cyclist = True
            else:
                raise ValueError("Unsupported class name for SAM3 detections.")

            object_class.locations = []
            object_class.lidar = []
            object_class.masks = []

            correctly_visible_in_ref_frame = True

            for frame_idx in range(-self.cfg.frames_creation.nscans_before + cur_id, self.cfg.frames_creation.nscans_after + 1 + cur_id):
                if frame_idx not in lidar_cache:
                    #object_class.locations.append([])
                    #object_class.lidar.append([])
                    #object_class.masks.append([])
                    continue

                if frame_idx in cur_object['masks']:
                    cur_mask_rle = cur_object['masks'][frame_idx]
                    cur_mask = mask_utils.decode(cur_mask_rle).T
                    cur_mask = np.array(cur_mask > 0).astype(bool)

                    T_cur_to_ref = transformations[frame_idx - cur_id + self.cfg.frames_creation.nscans_before, :, :]

                    lidar_cur = lidar_cache[frame_idx]

                    location, lidar_points, masks = self.get_car_locations_from_img(lidar_cur, T_cur_to_ref, [cur_mask])

                    if len(lidar_points) == 0:
                        if frame_idx == cur_id:
                            if not self.cfg.frames_creation.extract_standing_vehicles_not_visible:
                                correctly_visible_in_ref_frame = False
                        continue

                    location = np.append(location[0], [frame_idx - cur_id])
                    
                    object_class.locations.append(location)
                    object_class.lidar.append(lidar_points[0])
                    object_class.masks.append(masks[0])

                    if frame_idx == cur_id:
                        object_class.bbox = cur_object['boxes'][frame_idx]
                        object_class.ref_idx = cur_id
                        
                else:
                    #object_class.locations.append([])
                    #object_class.lidar.append([])
                    #object_class.masks.append([])
                    continue
            
            object_class.mask = object_class.masks

            if len(object_class.locations) == 0:
                continue
            if correctly_visible_in_ref_frame:
                all_objects.append(object_class)

        return all_objects
        
    def get_gt_masks(self, lidar, masks_path, T_cur_to_ref):
        masks_img = cv2.imread(masks_path, cv2.IMREAD_UNCHANGED)
        masks_img = np.array(masks_img)

        idx_in_img = np.unique(masks_img)
        masks_arr = []
        ids = []

        for idx in idx_in_img:
            if 26000 <= idx < 27000:
                mask = (masks_img == idx)
                masks_arr.append(mask.T)
                ids.append(idx)

        # Overlay the masks
        #result = self.overlay_masks_on_rgb(cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2BGR), masks_arr, colors, alpha=0.5)
        locs, points, masks = self.get_car_locations_from_img_gt(lidar, T_cur_to_ref, masks_arr)
        return locs, points, masks, ids

    def overlay_masks_on_rgb(self, image, masks, colors, alpha=0.5):
        """
        Overlay binary masks on an RGB image with specified colors.

        :param image: Original RGB image as a NumPy array.
        :param masks: List of binary masks as NumPy arrays.
        :param colors: List of colors for each mask (R, G, B).
        :param alpha: Transparency for the overlay (0-1).
        :return: Image with masks overlaid.
        """
        overlay = image.copy()  # Copy the image to preserve the original

        for mask, color in zip(masks, colors):
            # Create a colored mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for i in range(3):  # Apply the color for each channel
                colored_mask[:, :, i] = mask * color[i]

            # Overlay the colored mask on the original image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        return overlay

    def precompute_candidates(self):
        if self.cfg.frames_creation.use_pseudo_lidar:
            path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(int(self.number)).zfill(10) + '.npz'
            lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
        else:
            if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + str(int(self.number)).zfill(10) + '.bin'
            else:
                path_to_cur_velo = self.cfg.paths.kitti_path + '/object/training/velodyne/' + str(int(self.number)).zfill(6) + '.bin'
            
            if not os.path.exists(path_to_cur_velo):
                return

            lidar_cur = np.array(load_velo_scan(path_to_cur_velo))

        lidar_cur = self.prepare_scan_all(self.file_name, self.img, lidar_cur, save=False)
        masks, scores = self.precompute_detectron_all()

        self.get_car_locations_from_img_all(lidar_cur, masks, scores=scores)

        if self.cfg.frames_creation.extract_pedestrians:
            self.precompute_candidates_pedestrians(lidar_cur)

        return

    def precompute_candidates_pedestrians(self, lidar_cur):
        pedestrian_masks, cyclist_flags, scores = self.precompute_detectron_pedestrians_all()

        if len(pedestrian_masks) == 0:
            return

        ped_loc, lidar_points, masks, out_flags, out_scores = self.get_pedestrian_locations_from_img(lidar_cur, np.eye(4), pedestrian_masks, cyclist_flags, scores=scores)

        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_lidar_pedestrians/"):
             os.makedirs(self.cfg.paths.merged_frames_path + "candidates_lidar_pedestrians/", exist_ok=True)
        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_masks_pedestrians/"):
             os.makedirs(self.cfg.paths.merged_frames_path + "candidates_masks_pedestrians/", exist_ok=True)
        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_cyclist_flags_pedestrians/"):
             os.makedirs(self.cfg.paths.merged_frames_path + "candidates_cyclist_flags_pedestrians/", exist_ok=True)
        if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_scores_pedestrians/"):
             os.makedirs(self.cfg.paths.merged_frames_path + "candidates_scores_pedestrians/", exist_ok=True)

        compressed_arr = zstd.compress(pickle.dumps(lidar_points, pickle.HIGHEST_PROTOCOL))
        with open(self.cfg.paths.merged_frames_path + "candidates_lidar_pedestrians/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

        compressed_arr = zstd.compress(pickle.dumps(masks, pickle.HIGHEST_PROTOCOL))
        with open(self.cfg.paths.merged_frames_path + "candidates_masks_pedestrians/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)
            
        compressed_arr = zstd.compress(pickle.dumps(out_flags, pickle.HIGHEST_PROTOCOL))
        with open(self.cfg.paths.merged_frames_path + "candidates_cyclist_flags_pedestrians/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

        compressed_arr = zstd.compress(pickle.dumps(out_scores, pickle.HIGHEST_PROTOCOL))
        with open(self.cfg.paths.merged_frames_path + "candidates_scores_pedestrians/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

    def precompute_candidates_waymoc(self):
        path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(int(self.number)).zfill(10) + '.npz'

        lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
        lidar_cur = self.prepare_scan_waymoc(self.file_name, self.img, lidar_cur, save=False)
        masks = self.precompute_detectron_all()

        self.get_car_locations_from_img_all(lidar_cur, masks)

        return

    def precompute_candidates_gt(self):
        if self.cfg.frames_creation.use_pseudo_lidar:
            path_to_cur_velo = self.cfg.paths.merged_frames_path + '/lidar_raw/' + str(self.folder) + '/pcds/' + str(int(self.number)).zfill(10) + '.npz'
            lidar_cur = np.array(load_pseudo_lidar(path_to_cur_velo))
        else:
            if self.args.dataset == 'kitti360' or self.args.dataset == 'all':
                path_to_cur_velo = self.cfg.paths.all_dataset_path + 'data_3d_raw/' + self.folder + '/velodyne_points/data/' + str(int(self.number)).zfill(10) + '.bin'
            else:
                path_to_cur_velo = self.cfg.paths.kitti_path + '/object/training/velodyne/' + str(int(self.number)).zfill(6) + '.bin'
            
            if not os.path.exists(path_to_cur_velo):
                return

            lidar_cur = np.array(load_velo_scan(path_to_cur_velo))

        lidar_cur = self.prepare_scan_all(self.file_name, self.img, lidar_cur, save=False)

        masks_path = self.cfg.paths.all_dataset_path + str(self.folder) + '/instance/' + str(int(self.number)).zfill(10) + '.png'
        masks_img = cv2.imread(masks_path, cv2.IMREAD_UNCHANGED)
        masks_img = np.array(masks_img)

        idx_in_img = np.unique(masks_img)
        masks_arr = []
        ids = []

        for idx in idx_in_img:
            if idx is None:
                continue
            if 26000 <= idx < 27000:
                mask = (masks_img == idx)
                masks_arr.append(mask.T)
                ids.append(idx)

        locs, points, masks = self.get_car_locations_from_img_gt(lidar_cur, np.eye(4), masks_arr)

        compressed_arr = zstd.compress(pickle.dumps(points, pickle.HIGHEST_PROTOCOL))

        with open(self.cfg.paths.merged_frames_path + "candidates_lidar/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)
        compressed_arr = zstd.compress(pickle.dumps(masks, pickle.HIGHEST_PROTOCOL))

        with open(self.cfg.paths.merged_frames_path + "candidates_masks/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

        compressed_arr = zstd.compress(pickle.dumps(ids, pickle.HIGHEST_PROTOCOL))

        with open(self.cfg.paths.merged_frames_path + "candidates_ids/" + self.file_name + ".zstd", 'wb') as f:
            f.write(compressed_arr)

        return

    def get_precomputed_candidates_gt(self, T_cur_to_ref, offset=0):
        try:
            with open(self.cfg.paths.merged_frames_path + "candidates_lidar/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            lidar_points = pickle.loads(decompressed_data)
            with open(self.cfg.paths.merged_frames_path + "candidates_masks/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_masks = pickle.loads(decompressed_data)
            with open(self.cfg.paths.merged_frames_path + "candidates_ids/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_ids = pickle.loads(decompressed_data)
        except FileNotFoundError:
            return np.array([]), [], np.array([]), []

        means = []
        lidars = []
        masks = []
        ids = []

        for i in range(len(lidar_points)):
            lidar_points[i] = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_points[i].T).T
            lidar_points[i] += T_cur_to_ref[0:3, 3]
            x_mean, y_mean, z_mean = np.median(lidar_points[i], axis=0)
            means.append(np.array([x_mean, y_mean, z_mean]))
            lidars.append(lidar_points[i])
            masks.append(out_masks[i])
            ids.append(out_ids[i])

        means = np.array(means)
        masks = np.array(masks)
        return means, lidars, masks, ids

    def get_precomputed_candidates(self, T_cur_to_ref, offset=0):
        try:
            with open(self.cfg.paths.merged_frames_path + "candidates_lidar/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            lidar_points = pickle.loads(decompressed_data)
            with open(self.cfg.paths.merged_frames_path + "candidates_masks/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_masks = pickle.loads(decompressed_data)
        except FileNotFoundError:
            return np.array([]), [], np.array([]), []

        path_scores = self.cfg.paths.merged_frames_path + "candidates_scores/" + self.folder + '_' + str(int(self.number) + offset).zfill(10) + ".zstd"
        if os.path.exists(path_scores):
            with open(path_scores, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_scores = pickle.loads(decompressed_data)
        else:
            out_scores = [0.4] * len(out_masks)

        means = []
        lidars = []
        masks = []
        scores = []

        for i in range(len(lidar_points)):
            lidar_points[i] = np.matmul(T_cur_to_ref[0:3, 0:3], lidar_points[i].T).T
            lidar_points[i] += T_cur_to_ref[0:3, 3]
            x_mean, y_mean, z_mean = np.median(lidar_points[i], axis=0)
            if z_mean > 0:
                means.append(np.array([x_mean, y_mean, z_mean]))
                lidars.append(lidar_points[i])
                masks.append(out_masks[i])
                scores.append(out_scores[i])

        means = np.array(means)
        masks = np.array(masks)
        return means, lidars, masks, scores

    def precompute_detectron_waymo(self):
        if self.generate_raw_masks_or_tracking:
            img_arr = []
            for i in range(0, len(self.waymo_frame)):
                arr_temp = []
                images_sorted = sorted(self.waymo_frame[i].images, key=lambda z: z.name)
                for index, image in enumerate(images_sorted):
                    decoded_image = tf.image.decode_jpeg(image.image).numpy()

                    # Open the image, convert
                    img = np.array(decoded_image, dtype=np.uint8)
                    arr_temp.append(np.moveaxis(img, -1, 0))  # the model expects the image to be in channel first format

                img_arr.append(arr_temp)

            for i in range(len(img_arr)):
                if not self.cfg.general.supress_debug_prints:
                    print("Processing image: ", i, " from ", len(img_arr))
                tmp_img_arr = img_arr[i]
                for z in range(len(tmp_img_arr)):
                    if os.path.exists(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + '.npz'):
                        continue
                    out_dete = self.run_detectron(tmp_img_arr[z], save=False)
                    out_dete = out_dete[out_dete.scores > self.cfg.filtering.score_detectron_thresh]
                    masks_to_save = []
                    for k in range(len(out_dete.pred_masks)):
                        # We are only interested in cars
                        if out_dete.pred_classes[k] == 2 or out_dete.pred_classes[k] == 7:
                            # Take the mask and transpose it
                            mask = np.array(out_dete.pred_masks[k].cpu()).transpose()
                            masks_to_save.append(mask)

                    masks_to_save = np.array(masks_to_save)

                    if not os.path.isdir(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name):
                        os.mkdir(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name)
                    np.savez_compressed(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + ".npz", np.float32(masks_to_save))

            return None
        else:
            masks_arr = []
            for i in range(0, len(self.waymo_frame)):
                tmp_masks_arr = []
                for z in range(5):
                    masks_raw = np.load(
                        self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + "/" + str(i) + "_" + str(z) + '.npz')
                    masks_raw = [np.bool_(masks_raw[key]) for key in masks_raw]
                    tmp_masks_arr.append(masks_raw)
                masks_arr.append(tmp_masks_arr)

            return masks_arr

    def precompute_detectron_kitti(self):
        if self.generate_raw_masks_or_tracking:
            img_arr = []
            #img_arr_visu = []
            for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
                path_to_cur_img = self.path_to_folder + 'image_02/data/' + f'{self.file_number + i :0>10}' + '.png'
                if self.file_number + i < 0 or self.file_number + i >= len(self.kitti_data.oxts) or not os.path.exists(path_to_cur_img):
                    continue
                img_cv2 = cv2.imread(path_to_cur_img)
                img_cv2 = np.array(img_cv2, dtype=np.uint8)
                #img_arr_visu.append(img_cv2)
                img_cv2 = np.moveaxis(img_cv2, -1, 0)  # the model expects the image to be in channel first format
                img_arr.append(img_cv2)

            detectron_output_arr_tmp = []
            detectron_output_arr = []
            # Get all the outputs from detectron/SAM
            num_of_full_batches = ((len(img_arr)) // self.cfg.general.batch_size)
            for i in range(num_of_full_batches):
                tmp_img_arr = []
                for z in range(self.cfg.general.batch_size):
                    tmp_img_arr.append(img_arr[z + (i * self.cfg.general.batch_size)])
                if self.cfg.frames_creation.use_SAM:
                    out_dete = self.run_SAM_batch(tmp_img_arr)
                else:
                    out_dete = self.run_detectron_batch(tmp_img_arr)
                for z in range(len(out_dete)):
                    detectron_output_arr_tmp.append(out_dete[z])

            last_batch = ((len(img_arr)) % self.cfg.general.batch_size)
            if last_batch > 0:
                tmp_img_arr = []
                for z in range(last_batch):
                    tmp_img_arr.append(img_arr[z + (num_of_full_batches * self.cfg.general.batch_size)])
                if self.cfg.frames_creation.use_SAM:
                    out_dete = self.run_SAM_batch(tmp_img_arr)
                else:
                    out_dete = self.run_detectron_batch(tmp_img_arr)
                for z in range(len(out_dete)):
                    detectron_output_arr_tmp.append(out_dete[z])

            # Merge detector output and missing output
            idx = 0
            for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
                path_to_cur_img = self.path_to_folder + 'image_02/data/' + f'{self.file_number + i :0>10}' + '.png'
                if self.file_number + i < 0 or self.file_number + i >= len(
                        self.kitti_data.oxts) or not os.path.exists(
                        path_to_cur_img):
                    detectron_output_arr.append([])
                else:
                    detectron_output_arr.append(detectron_output_arr_tmp[idx])
                    idx += 1

            for i in range(len(detectron_output_arr)):
                if len(detectron_output_arr[i]) > 0:
                    detections = detectron_output_arr[i]
                    detections = detections[detections.scores > self.cfg.filtering.score_detectron_thresh]
                    masks_to_save = []
                    for k in range(len(detections.pred_masks)):
                        if detections.pred_classes[k] == 2: #or detections.pred_classes[k] == 7:
                            # Take the mask and transpose it
                            mask = np.array(detections.pred_masks[k].cpu()).transpose()
                            masks_to_save.append(mask)

                            #img_arr_visu[i][mask.T] = [255, 0, 0]

                    #result_image = Image.fromarray(img_arr_visu[i])
                    '''
                    # Display the result
                    plt.imshow(result_image)
                    plt.axis('off')  # Hide the axes
                    plt.show()
                    '''
                    masks_to_save = np.array(masks_to_save)
                    detectron_output_arr[i] = masks_to_save

            compressed_arr = zstd.compress(pickle.dumps(detectron_output_arr, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            return detectron_output_arr
        else:
            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            masks = pickle.loads(decompressed_data)
            return masks[100 - self.cfg.frames_creation.nscans_before: 100 + self.cfg.frames_creation.nscans_after + 1]

    def precompute_detectron_kitti_v2(self, pedestrian=False):
        if self.generate_raw_masks_or_tracking:
            #TODO Implement for mvitv2
            return None

        else:
            masks_to_output = []
            for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
                full_path_to_folder = self.path_to_folder.split("/")
                folder = full_path_to_folder[-3]
                subfolder = full_path_to_folder[-2]

                if pedestrian:
                    path_to_mask = os.path.join(self.cfg.paths.merged_frames_path, "masks_raw_pedestrians/", folder, subfolder,f'{self.file_number + i :0>10}' + '.zstd')
                else:
                    path_to_mask = os.path.join(self.cfg.paths.merged_frames_path, "masks_raw/", folder, subfolder, f'{self.file_number + i :0>10}' + '.zstd')
                if not os.path.exists(path_to_mask):
                    masks_to_output.append([])
                    continue

                #TODO parse the target folder and load all masks before and after, then return them :)
                with open(path_to_mask, 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                loaded_data = pickle.loads(decompressed_data)

                if isinstance(loaded_data, dict):
                    masks = loaded_data['masks']
                    scores = loaded_data.get('scores', None)
                else:
                    masks = loaded_data
                    scores = None

                if scores is not None:
                    masks_to_output.append({'masks': masks, 'scores': scores})
                else:
                    masks_to_output.append(masks)

            return masks_to_output

    def get_pedestrian_smplestx(self):
        peds_to_output = []
        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            full_path_to_folder = self.path_to_folder.split("/")
            folder = full_path_to_folder[-3]
            subfolder = full_path_to_folder[-2]

            path_to_pedestrian = os.path.join(self.cfg.paths.merged_frames_path, "pose_pedestrians/", folder, subfolder, f'{self.file_number + i :0>10}' + '.zstd')

            if not os.path.exists(path_to_pedestrian):
                peds_to_output.append([])
                continue

            with open(path_to_pedestrian, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            pedestrian = pickle.loads(decompressed_data)
            peds_to_output.append(pedestrian)

        return peds_to_output
    
    def _class_threshold_str(self, thr_val: float) -> str:
        try:
            return f"{int(float(thr_val) * 100):03d}"
        except Exception:
            return "050"

    def _load_additional_objects_current_frame(self, lidar_orig, img):
        """
        Load additional object classes for the current frame only (no temporal aggregation),
        using precomputed Co-DETR mask outputs under masks_raw_<class>_<THR>/folder/subfolder/<frame>.zstd.

        Populates self.additional_objects as a dict: {class_name: [AdditionalObject, ...]}.
        """
        # Config guards and defaults
        frames_cfg = getattr(self.cfg, 'frames_creation', None)
        paths_cfg = getattr(self.cfg, 'paths', None)
        if frames_cfg is None or paths_cfg is None:
            return
        additional_enabled = bool(getattr(frames_cfg, 'extract_additional_objects', False))
        if not additional_enabled:
            return

        additional_classes = list(getattr(frames_cfg, 'additional_classes', []) or [])
        if len(additional_classes) == 0:
            return

        add_thr = float(getattr(frames_cfg, 'additional_classes_thr', 0.5))
        t_str = self._class_threshold_str(add_thr)

        base_output = getattr(paths_cfg, 'merged_frames_path', None)
        if base_output is None:
            return

        # Prepare the scan for the current frame (reference frame)
        try:
            scan = self.prepare_scan(self.file_name, img, lidar_orig, save=False, crop=not self.cfg.visualization.show_pcdet and not self.cfg.visualization.visu_whole_lidar)
        except Exception:
            # Fallback without cropping flags if not present
            scan = self.prepare_scan(self.file_name, img, lidar_orig, save=False)

        # Build folder/subfolder and frame id
        full_path_to_folder = self.path_to_folder.split("/")
        if len(full_path_to_folder) >= 3:
            folder = full_path_to_folder[-3]
            subfolder = full_path_to_folder[-2]
        else:
            # If path structure is unexpected, we'll rely on a slower glob fallback
            folder, subfolder = None, None

        frame_str = f"{self.file_number:0>10}"

        additional_objects = {}

        for cls_name in additional_classes:
            masks_file = None
            if folder is not None and subfolder is not None:
                # Try COCO-style folder first
                candidate1 = os.path.join(base_output, f"masks_raw_{cls_name}_{t_str}", folder, subfolder, frame_str + '.zstd')
                # Try LVIS-style folder naming
                candidate2 = os.path.join(base_output, f"masks_raw_lvis_{cls_name}_{t_str}", folder, subfolder, frame_str + '.zstd')
                if os.path.exists(candidate1):
                    masks_file = candidate1
                elif os.path.exists(candidate2):
                    masks_file = candidate2
            if masks_file is None:
                # Fallback: recursive search (slower but robust if layout differs)
                pattern1 = os.path.join(base_output, f"masks_raw_{cls_name}_{t_str}", "**", frame_str + ".zstd")
                pattern2 = os.path.join(base_output, f"masks_raw_lvis_{cls_name}_{t_str}", "**", frame_str + ".zstd")
                matches = glob.glob(pattern1, recursive=True)
                if len(matches) == 0:
                    matches = glob.glob(pattern2, recursive=True)
                if len(matches) > 0:
                    masks_file = matches[0]

            if masks_file is None or not os.path.exists(masks_file):
                continue

            try:
                with open(masks_file, 'rb') as f:
                    decompressed = zstd.decompress(f.read())
                masks_arr = pickle.loads(decompressed)
            except Exception:
                # Skip class if we cannot read
                continue

            # Expect masks_arr to be an iterable of binary masks; reuse existing extraction util
            if masks_arr is None or len(masks_arr) == 0:
                continue

            T_identity = np.eye(4, dtype=float)
            try:
                means, lidars, out_masks = self.get_car_locations_from_img(scan, T_identity, masks_arr)
            except Exception:
                # As a fallback, try GT variant which is slightly more permissive
                try:
                    means, lidars, out_masks = self.get_car_locations_from_img_gt(scan, T_identity, masks_arr)
                except Exception:
                    continue

            per_class_objects = []
            for i in range(len(lidars)):
                if lidars[i] is None or (hasattr(lidars[i], 'shape') and lidars[i].shape[0] == 0):
                    continue
                obj = self.AdditionalObject(cls_name)
                obj.lidar = lidars[i].astype(np.float32)
                obj.mask = out_masks[i]
                try:
                    obj.center = means[i]
                except Exception:
                    # Compute median if means index fails
                    obj.center = np.median(obj.lidar, axis=0) if obj.lidar is not None and len(obj.lidar) > 0 else None
                per_class_objects.append(obj)

            if len(per_class_objects) > 0:
                additional_objects[cls_name] = per_class_objects

        # Attach to instance for downstream use
        self.additional_objects = additional_objects

    def precompute_detectron_all(self, offset=0):
        if self.generate_raw_masks_or_tracking:
            tmp_img_arr = [self.img]
            out_dete = self.run_detectron_batch(tmp_img_arr)[0]

            masks_to_save = []
            scores_to_save = []
            if len(out_dete) > 0:
                detections = out_dete
                detections = detections[detections.scores > self.cfg.filtering.score_detectron_thresh]
                for k in range(len(detections.pred_masks)):
                    if detections.pred_classes[k] == 2: #or detections.pred_classes[k] == 7:
                        # Take the mask and transpose it
                        mask = np.array(detections.pred_masks[k].cpu()).transpose()
                        masks_to_save.append(mask)
                        scores_to_save.append(float(detections.scores[k]))

                masks_to_save = np.array(masks_to_save)
                scores_to_save = np.array(scores_to_save)

            payload = {
                'masks': masks_to_save,
                'scores': scores_to_save
            }
            compressed_arr = zstd.compress(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            return masks_to_save, scores_to_save

        elif self.cfg.frames_creation.use_codetr:
            file_name = self.folder + '/' + str(int(self.number) + offset).zfill(10)
            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            loaded_data = pickle.loads(decompressed_data)
            
            if isinstance(loaded_data, dict):
                masks = loaded_data['masks']
                scores = loaded_data.get('scores', [0.4] * len(masks))
            else:
                masks = loaded_data
                scores = [0.4] * len(masks)
            return masks, scores
        
        else:
            file_name = self.folder + '_' + str(int(self.number) + offset).zfill(10)
            with open(self.cfg.paths.merged_frames_path + "masks_raw/" + file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            loaded_data = pickle.loads(decompressed_data)
            
            if isinstance(loaded_data, dict):
                masks = loaded_data['masks']
                scores = loaded_data.get('scores', [0.4] * len(masks))
            else:
                masks = loaded_data
                scores = [0.4] * len(masks)
            return masks, scores

    def get_precomputed_detectron_waymo(self, idx_start, idx_end):
        if idx_start < 0:
            idx_start = 0
        if idx_end >= len(self.waymo_frame):
            idx_end = len(self.waymo_frame)

        return self.prec_detectron_output[idx_start: idx_end]

    def precompute_standing_car_candidates_waymo(self):
        car_locations = []
        car_locations_lidar = []
        car_locations_info = []
        car_locations_masks = []
        detectron_output_arr = self.get_precomputed_detectron_waymo(0, len(self.waymo_lidar))

        for i in range(0, len(self.waymo_lidar)):
            # Load the velo scan
            lidar_cur = self.waymo_lidar[i]

            out_det = detectron_output_arr[i]

            car_loc_tmp = None
            car_locations_lidar_tmp = []
            car_locations_info_tmp = []
            car_locations_masks_tmp = []
            for z in range(len(out_det)):
                out_det_tmp = out_det[z]
                if self.cfg.frames_creation.use_growing_for_point_extraction:
                    car_loc, lidar_points, info, masks = self.get_car_locations_from_img_waymo_growing(z + 1, lidar_cur, i, out_det_tmp)
                else:
                    car_loc, lidar_points, info, masks = self.get_car_locations_from_img_waymo(z + 1, lidar_cur, i, out_det_tmp)
                if len(car_loc) > 0:
                    if car_loc_tmp is None:
                        car_loc_tmp = car_loc
                    else:
                        car_loc_tmp = np.concatenate((car_loc_tmp, car_loc), axis=0)
                for k in range(len(lidar_points)):
                    car_locations_lidar_tmp.append(lidar_points[k])
                    car_locations_info_tmp.append(info[k])
                    car_locations_masks_tmp.append(masks[k])
            if car_loc_tmp is None:
                car_loc_tmp = []
            car_locations.append(car_loc_tmp)
            car_locations_lidar.append(car_locations_lidar_tmp)
            car_locations_info.append(car_locations_info_tmp)
            car_locations_masks.append(car_locations_masks_tmp)
        if not self.cfg.general.supress_debug_prints:
            print("New Car Locations")
        self.car_locations = car_locations
        self.car_locations_lidar = car_locations_lidar
        self.car_locations_info = car_locations_info
        self.car_locations_masks = car_locations_masks

        return None

    def convert_to_current_frame(self, transformations):
        car_locations = []
        car_locations_lidar = []
        car_locations_info = []
        car_locations_masks = []
        detectron_output_arr_tmp = self.get_precomputed_detectron_waymo(self.pic_index - self.cfg.frames_creation.nscans_before,
                                                                        self.pic_index + self.cfg.frames_creation.nscans_after + 1)
        detectron_output_arr = []

        # Merge detector output and missing output
        idx = 0
        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            if self.pic_index + i < 0 or self.pic_index + i >= len(self.waymo_frame) or len(self.car_locations[self.pic_index + i]) == 0:
                detectron_output_arr.append([])
            else:
                detectron_output_arr.append(detectron_output_arr_tmp[idx])
                idx += 1

        for i in range(-self.cfg.frames_creation.nscans_before, self.cfg.frames_creation.nscans_after + 1):
            # Ignore the reference scan and also do not search for not existing data

            if self.pic_index + i < 0 or self.pic_index + i >= len(self.waymo_frame) or len(self.car_locations[self.pic_index + i]) == 0:
                car_locations.append([])
                car_locations_lidar.append([])
                car_locations_info.append([])
                car_locations_masks.append([])
                continue

            T_cur_to_ref = transformations[i + self.cfg.frames_creation.nscans_before, :, :]

            tmp_arr = np.matmul(T_cur_to_ref[:3, :3], self.car_locations[self.pic_index + i].T).T
            tmp_arr += T_cur_to_ref[0:3, 3]
            car_locations.append(tmp_arr)

            tmp_arr = []
            for z in range(len(self.car_locations_lidar[self.pic_index + i])):
                tmp_arr.append(np.matmul(T_cur_to_ref[:3, :3], self.car_locations_lidar[self.pic_index + i][z].T).T)
                tmp_arr[z] += T_cur_to_ref[0:3, 3]
            car_locations_lidar.append(tmp_arr)
            car_locations_info.append(self.car_locations_info[self.pic_index + i])
            car_locations_masks.append(self.car_locations_masks[self.pic_index + i])

        return car_locations, car_locations_lidar, car_locations_info, car_locations_masks, detectron_output_arr

    def perform_3D_tracking(self, standing_cars_candidates, car_locations_lidar, car_locations_info, car_locations_masks):
        ref_cars = standing_cars_candidates[0]  # Reference locations of the cars, including the standing cars
        ref_cars_lidar = car_locations_lidar[0]
        ref_cars_info = car_locations_info[0]
        ref_cars_mask = car_locations_masks[0]

        final_moving_cars = []
        final_moving_cars_lidar = []
        final_moving_cars_info = []
        final_moving_cars_masks = []

        if len(ref_cars) > 0:
            tmp_arr = []
            tmp_arr_lidar = []
            tmp_arr_info = []
            tmp_arr_mask = []
            for z in range(len(ref_cars)):
                tmp_arr.append([np.append(ref_cars[z], -self.cfg.frames_creation.nscans_before)])
                tmp_arr_lidar.append([ref_cars_lidar[z]])
                tmp_arr_info.append([ref_cars_info[z]])
                tmp_arr_mask.append([ref_cars_mask[z]])
            moving_cars = tmp_arr
            moving_cars_lidar = tmp_arr_lidar
            moving_cars_info = tmp_arr_info
            moving_cars_masks = tmp_arr_mask

        else:
            moving_cars = []
            moving_cars_lidar = []
            moving_cars_info = []
            moving_cars_masks = []

        #Now lets find moving cars
        for i in range(-self.cfg.frames_creation.nscans_before + 1, self.cfg.frames_creation.nscans_after + 1):
            #Take the cars in the current frame
            cur_cars = standing_cars_candidates[i + self.cfg.frames_creation.nscans_before]
            cur_cars_lidar = car_locations_lidar[i + self.cfg.frames_creation.nscans_before]
            cur_cars_info = car_locations_info[i + self.cfg.frames_creation.nscans_before]
            cur_cars_masks = car_locations_masks[i + self.cfg.frames_creation.nscans_before]
            # Create a mask which looks if the all cars have been found in this frame, otherwise we will discard them and save them
            mask = np.zeros(len(moving_cars), dtype=np.bool_)
            if cur_cars is not None and len(cur_cars) > 0:
                cur_detected_cars = cur_cars
                cur_detected_cars_lidar = cur_cars_lidar
                cur_detected_cars_info = cur_cars_info
                cur_detected_cars_masks = cur_cars_masks

                moving_cars_estimate_locations = []
                for z in range(len(moving_cars)):
                    moving_car = moving_cars[z]
                    # In this case, this is the first time we have seen this car so we cannot predict velocity
                    if len(moving_car) == 1:
                        moving_cars_estimate_locations.append(moving_car[0][:3])
                    else:
                        # Last location + velocity from the last frame
                        est1 = np.array(moving_car[-1][:3] - np.array(moving_car[-2][:3]))
                        if len(moving_car) > 2:
                            est2 = np.array(moving_car[-2][:3] - np.array(moving_car[-3][:3]))
                            if len(moving_car) > 3:
                                est3 = np.array(moving_car[-3][:3] - np.array(moving_car[-4][:3]))
                                if len(moving_car) > 4:
                                    est4 = np.array(moving_car[-4][:3] - np.array(moving_car[-5][:3]))
                                    est = (est1 + est2 + est3 + est4) / 4
                                else:
                                    est = (est1 + est2 + est3) / 3
                            else:
                                est = (est1 + est2) / 2
                        else:
                            est = est1

                        est += np.array(moving_car[-1][:3])

                        moving_cars_estimate_locations.append(est.tolist())
                # Now we got the distance matrix and now we want to do the matching
                new_moving_cars = []
                new_moving_cars_lidar = []
                new_moving_cars_info = []
                new_moving_cars_masks = []

                if len(moving_cars) > 0 and len(cur_detected_cars) > 0:
                    dists = cdist(cur_detected_cars, moving_cars_estimate_locations)
                    mins_cur_to_mov = np.argmin(dists, axis=1)
                    mins_mov_to_cur = np.argmin(dists, axis=0)

                    for z in range(len(cur_detected_cars)):
                        #Check if the closest cur is also closest to the moving car
                        if mins_mov_to_cur[mins_cur_to_mov[z]] == z or True:
                            dist = np.linalg.norm(cur_detected_cars[z][:2] - moving_cars_estimate_locations[mins_cur_to_mov[z]][:2])
                            if dist < self.cfg.frames_creation.dist_treshold_tracking:
                                mask[mins_cur_to_mov[z]] = True
                                moving_cars[mins_cur_to_mov[z]].append(np.append(cur_detected_cars[z], i))
                                moving_cars_lidar[mins_cur_to_mov[z]].append(cur_detected_cars_lidar[z])
                                moving_cars_info[mins_cur_to_mov[z]].append(cur_detected_cars_info[z])
                                moving_cars_masks[mins_cur_to_mov[z]].append(cur_detected_cars_masks[z])
                            else:
                                new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                                new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                                new_moving_cars_info.append([cur_detected_cars_info[z]])
                                new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                        # We didnt find a match so it is probably a new moving car
                        else:
                            new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                            new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                            new_moving_cars_info.append([cur_detected_cars_info[z]])
                            new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                else:
                    for z in range(len(cur_detected_cars)):
                        new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                        new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                        new_moving_cars_info.append([cur_detected_cars_info[z]])
                        new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                index_to_keep = []
                for z in range(len(moving_cars)):
                    if not mask[z] and False:
                        # We did not find a corresponding car so we think we cannot track it anymore
                        final_moving_cars.append(moving_cars[z])
                        final_moving_cars_lidar.append(moving_cars_lidar[z])
                        final_moving_cars_info.append(moving_cars_info[z])
                        final_moving_cars_masks.append(moving_cars_masks[z])
                    else:
                        index_to_keep.append(z)
                tmp_moving_cars = []
                tmp_moving_cars_lidar = []
                tmp_moving_cars_info = []
                tmp_moving_cars_masks = []
                for z in index_to_keep:
                    tmp_moving_cars.append(moving_cars[z])
                    tmp_moving_cars_lidar.append(moving_cars_lidar[z])
                    tmp_moving_cars_info.append(moving_cars_info[z])
                    tmp_moving_cars_masks.append(moving_cars_masks[z])
                moving_cars = tmp_moving_cars
                moving_cars_lidar = tmp_moving_cars_lidar
                moving_cars_info = tmp_moving_cars_info
                moving_cars_masks = tmp_moving_cars_masks

                moving_cars = moving_cars + new_moving_cars
                moving_cars_lidar = moving_cars_lidar + new_moving_cars_lidar
                moving_cars_info = moving_cars_info + new_moving_cars_info
                moving_cars_masks = moving_cars_masks + new_moving_cars_masks

        for z in range(len(moving_cars)):
            final_moving_cars.append(moving_cars[z])
            final_moving_cars_lidar.append(moving_cars_lidar[z])
            final_moving_cars_info.append(moving_cars_info[z])
            final_moving_cars_masks.append(moving_cars_masks[z])

        return final_moving_cars, final_moving_cars_lidar, final_moving_cars_info, final_moving_cars_masks

    def perform_gt_tracking(self, car_locations, car_locations_lidar, car_locations_masks, car_locations_idxs):
        ref_idx = car_locations_idxs[self.cfg.frames_creation.nscans_before]

        final_moving_cars = []
        final_moving_cars_lidar = []
        final_moving_cars_masks = []

        for car_idx in ref_idx:
            cur_locs = []
            cur_lidars = []
            cur_masks = []
            for i in range(0, self.cfg.frames_creation.nscans_after + 1 + self.cfg.frames_creation.nscans_before):
                for z in range(len(car_locations_idxs[i])):
                    if car_locations_idxs[i][z] == car_idx:
                        cur_locs.append(np.append(car_locations[i][z], i - self.cfg.frames_creation.nscans_before))
                        cur_lidars.append(car_locations_lidar[i][z])
                        cur_masks.append(car_locations_masks[i][z])
            final_moving_cars.append(cur_locs)
            final_moving_cars_lidar.append(cur_lidars)
            final_moving_cars_masks.append(cur_masks)

        return final_moving_cars, final_moving_cars_lidar, final_moving_cars_masks


    def perform_3D_tracking_kitti(self, standing_cars_candidates, car_locations_lidar, car_locations_masks, car_locations_scores=None):
        ref_cars = standing_cars_candidates[0]  # Reference locations of the cars, including the standing cars
        ref_cars_lidar = car_locations_lidar[0]
        ref_cars_mask = car_locations_masks[0]
        
        if car_locations_scores is not None:
            ref_cars_scores = car_locations_scores[0]
        else:
            ref_cars_scores = None

        final_moving_cars = []
        final_moving_cars_lidar = []
        final_moving_cars_masks = []
        final_moving_cars_scores = []

        if len(ref_cars) > 0:
            tmp_arr = []
            tmp_arr_lidar = []
            tmp_arr_mask = []
            tmp_arr_score = []
            for z in range(len(ref_cars)):
                tmp_arr.append([np.append(ref_cars[z], -self.cfg.frames_creation.nscans_before)])
                tmp_arr_lidar.append([ref_cars_lidar[z]])
                tmp_arr_mask.append([ref_cars_mask[z]])
                if ref_cars_scores is not None:
                    tmp_arr_score.append([ref_cars_scores[z]])
                else:
                    tmp_arr_score.append([0.4])
            moving_cars = tmp_arr
            moving_cars_lidar = tmp_arr_lidar
            moving_cars_masks = tmp_arr_mask
            moving_cars_scores = tmp_arr_score

        else:
            moving_cars = []
            moving_cars_lidar = []
            moving_cars_masks = []
            moving_cars_scores = []
        
        # Initialize moving_cars_info to avoid crash if it was used (though it seems undefined in original)
        moving_cars_info = [[] for _ in range(len(moving_cars))]

        #Now lets find moving cars
        for i in range(-self.cfg.frames_creation.nscans_before + 1, self.cfg.frames_creation.nscans_after + 1):
            #Take the cars in the current frame
            cur_cars = standing_cars_candidates[i + self.cfg.frames_creation.nscans_before]
            cur_cars_lidar = car_locations_lidar[i + self.cfg.frames_creation.nscans_before]
            cur_cars_masks = car_locations_masks[i + self.cfg.frames_creation.nscans_before]
            if car_locations_scores is not None:
                cur_cars_scores = car_locations_scores[i + self.cfg.frames_creation.nscans_before]
            else:
                cur_cars_scores = None

            # Create a mask which looks if the all cars have been found in this frame, otherwise we will discard them and save them
            mask = np.zeros(len(moving_cars), dtype=np.bool_)
            if cur_cars is not None and len(cur_cars) > 0:
                cur_detected_cars = cur_cars
                cur_detected_cars_lidar = cur_cars_lidar
                cur_detected_cars_masks = cur_cars_masks
                cur_detected_cars_scores = cur_cars_scores

                moving_cars_estimate_locations = []
                for z in range(len(moving_cars)):
                    moving_car = moving_cars[z]
                    # In this case, this is the first time we have seen this car so we cannot predict velocity
                    if len(moving_car) == 1:
                        moving_cars_estimate_locations.append(moving_car[0][:3])
                    else:
                        # Last location + velocity from the last frame
                        est1 = np.array(moving_car[-1][:3] - np.array(moving_car[-2][:3]))
                        if len(moving_car) > 2:
                            est2 = np.array(moving_car[-2][:3] - np.array(moving_car[-3][:3]))
                            if len(moving_car) > 3:
                                est3 = np.array(moving_car[-3][:3] - np.array(moving_car[-4][:3]))
                                if len(moving_car) > 4:
                                    est4 = np.array(moving_car[-4][:3] - np.array(moving_car[-5][:3]))
                                    est = (est1 + est2 + est3 + est4) / 4
                                else:
                                    est = (est1 + est2 + est3) / 3
                            else:
                                est = (est1 + est2) / 2
                        else:
                            est = est1

                        est += np.array(moving_car[-1][:3])

                        moving_cars_estimate_locations.append(est.tolist())
                # Now we got the distance matrix and now we want to do the matching
                new_moving_cars = []
                new_moving_cars_lidar = []
                new_moving_cars_masks = []
                new_moving_cars_scores = []

                if len(moving_cars) > 0 and len(cur_detected_cars) > 0:
                    dists = cdist(cur_detected_cars, moving_cars_estimate_locations)
                    mins_cur_to_mov = np.argmin(dists, axis=1)
                    mins_mov_to_cur = np.argmin(dists, axis=0)

                    for z in range(len(cur_detected_cars)):
                        #Check if the closest cur is also closest to the moving car
                        if mins_mov_to_cur[mins_cur_to_mov[z]] == z:
                            dist = np.linalg.norm(cur_detected_cars[z][:3] - moving_cars_estimate_locations[mins_cur_to_mov[z]][:3])
                            if dist < self.cfg.frames_creation.dist_treshold_tracking:
                                mask[mins_cur_to_mov[z]] = True
                                moving_cars[mins_cur_to_mov[z]].append(np.append(cur_detected_cars[z], i))
                                moving_cars_lidar[mins_cur_to_mov[z]].append(cur_detected_cars_lidar[z])
                                moving_cars_masks[mins_cur_to_mov[z]].append(cur_detected_cars_masks[z])
                                if cur_detected_cars_scores is not None:
                                    moving_cars_scores[mins_cur_to_mov[z]].append(cur_detected_cars_scores[z])
                                else:
                                    moving_cars_scores[mins_cur_to_mov[z]].append(0.4)
                            else:
                                new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                                new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                                new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                                if cur_detected_cars_scores is not None:
                                    new_moving_cars_scores.append([cur_detected_cars_scores[z]])
                                else:
                                    new_moving_cars_scores.append([0.4])
                        # We didnt find a match so it is probably a new moving car
                        else:
                            new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                            new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                            new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                            if cur_detected_cars_scores is not None:
                                new_moving_cars_scores.append([cur_detected_cars_scores[z]])
                            else:
                                new_moving_cars_scores.append([0.4])
                else:
                    for z in range(len(cur_detected_cars)):
                        new_moving_cars.append([np.append(cur_detected_cars[z], i)])
                        new_moving_cars_lidar.append([cur_detected_cars_lidar[z]])
                        new_moving_cars_masks.append([cur_detected_cars_masks[z]])
                        if cur_detected_cars_scores is not None:
                            new_moving_cars_scores.append([cur_detected_cars_scores[z]])
                        else:
                            new_moving_cars_scores.append([0.4])
                index_to_keep = []
                for z in range(len(moving_cars)):
                    if not mask[z] and False:
                        # We did not find a corresponding car so we think we cannot track it anymore
                        final_moving_cars.append(moving_cars[z])
                        final_moving_cars_lidar.append(moving_cars_lidar[z])
                        final_moving_cars_masks.append(moving_cars_masks[z])
                        final_moving_cars_scores.append(moving_cars_scores[z])
                    else:
                        index_to_keep.append(z)
                tmp_moving_cars = []
                tmp_moving_cars_lidar = []
                tmp_moving_cars_masks = []
                tmp_moving_cars_scores = []
                for z in index_to_keep:
                    tmp_moving_cars.append(moving_cars[z])
                    tmp_moving_cars_lidar.append(moving_cars_lidar[z])
                    tmp_moving_cars_masks.append(moving_cars_masks[z])
                    tmp_moving_cars_scores.append(moving_cars_scores[z])
                moving_cars = tmp_moving_cars
                moving_cars_lidar = tmp_moving_cars_lidar
                moving_cars_masks = tmp_moving_cars_masks
                moving_cars_scores = tmp_moving_cars_scores

                moving_cars = moving_cars + new_moving_cars
                moving_cars_lidar = moving_cars_lidar + new_moving_cars_lidar
                moving_cars_masks = moving_cars_masks + new_moving_cars_masks
                moving_cars_scores = moving_cars_scores + new_moving_cars_scores

        for z in range(len(moving_cars)):
            final_moving_cars.append(moving_cars[z])
            final_moving_cars_lidar.append(moving_cars_lidar[z])
            final_moving_cars_masks.append(moving_cars_masks[z])
            final_moving_cars_scores.append(moving_cars_scores[z])

        return final_moving_cars, final_moving_cars_lidar, final_moving_cars_masks, final_moving_cars_scores

    def perform_3D_tracking_kitti_pedestrian(self, pedestrians):
        ref_pedestrians = pedestrians[0]

        final_moving_pedestrians = []

        if len(ref_pedestrians) > 0:
            moving_peds = []
            for i in range(len(ref_pedestrians)):
                ref_pedestrians[i]['location'] = np.append(ref_pedestrians[i]['location'], -self.cfg.frames_creation.nscans_before)
                moving_peds.append([ref_pedestrians[i]])
        else:
            moving_peds = []

        #Now lets find moving cars
        for i in range(-self.cfg.frames_creation.nscans_before + 1, self.cfg.frames_creation.nscans_after + 1):
            #Take the cars in the current frame
            cur_peds = pedestrians[i + self.cfg.frames_creation.nscans_before]

            # Create a mask which looks if the all cars have been found in this frame, otherwise we will discard them and save them
            mask = np.zeros(len(moving_peds), dtype=np.bool_)
            if cur_peds is not None and len(cur_peds) > 0:
                moving_peds_estimate_locations = []
                for z in range(len(moving_peds)):
                    moving_ped = moving_peds[z]
                    # In this case, this is the first time we have seen this car so we cannot predict velocity
                    if len(moving_ped) == 1:
                        moving_peds_estimate_locations.append(moving_ped[0]['location'][:3])
                    else:
                        # Last location + velocity from the last frame
                        est1 = np.array(moving_ped[-1]['location'][:3] - np.array(moving_ped[-2]['location'][:3]))
                        if len(moving_ped) > 2:
                            est2 = np.array(moving_ped[-2]['location'][:3] - np.array(moving_ped[-3]['location'][:3]))
                            if len(moving_ped) > 3:
                                est3 = np.array(moving_ped[-3]['location'][:3] - np.array(moving_ped[-4]['location'][:3]))
                                if len(moving_ped) > 4:
                                    est4 = np.array(moving_ped[-4]['location'][:3] - np.array(moving_ped[-5]['location'][:3]))
                                    est = (est1 + est2 + est3 + est4) / 4
                                else:
                                    est = (est1 + est2 + est3) / 3
                            else:
                                est = (est1 + est2) / 2
                        else:
                            est = est1

                        est += np.array(moving_ped[-1]['location'][:3])

                        moving_peds_estimate_locations.append(est.tolist())

                # Now we got the distance matrix and now we want to do the matching
                new_moving_peds = []

                if len(moving_peds) > 0 and len(cur_peds) > 0:
                    cur_peds_location = []
                    for cur_ped in cur_peds:
                        cur_peds_location.append(cur_ped['location'][:3])
                    dists = cdist(cur_peds_location, moving_peds_estimate_locations)
                    mins_cur_to_mov = np.argmin(dists, axis=1)
                    mins_mov_to_cur = np.argmin(dists, axis=0)

                    for z in range(len(cur_peds)):
                        cur_ped = cur_peds[z]
                        cur_ped['location'] = np.append(cur_ped["location"], i)
                        #Check if the closest cur is also closest to the moving car
                        if mins_mov_to_cur[mins_cur_to_mov[z]] == z:
                            dist = np.linalg.norm(cur_peds[z]['location'][:3] - moving_peds_estimate_locations[mins_cur_to_mov[z]][:3])
                            if dist < self.cfg.frames_creation.dist_treshold_tracking:
                                mask[mins_cur_to_mov[z]] = True
                                moving_peds[mins_cur_to_mov[z]].append(cur_ped)
                            else:
                                new_moving_peds.append([cur_ped])
                        # We didnt find a match so it is probably a new moving car
                        else:
                            new_moving_peds.append([cur_ped])
                else:
                    for z in range(len(cur_peds)):
                        cur_ped = cur_peds[z]
                        cur_ped['location'] = np.append(cur_ped['location'], i)
                        new_moving_peds.append([cur_ped])
                index_to_keep = []
                for z in range(len(moving_peds)):
                    if not mask[z] and False:
                        # We did not find a corresponding car so we think we cannot track it anymore
                        final_moving_pedestrians.append(moving_peds[z])
                    else:
                        index_to_keep.append(z)
                tmp_moving_peds = []
                for z in index_to_keep:
                    tmp_moving_peds.append(moving_peds[z])
                moving_peds = tmp_moving_peds

                moving_peds = moving_peds + new_moving_peds

        for z in range(len(moving_peds)):
            final_moving_pedestrians.append(moving_peds[z])

        return final_moving_pedestrians

    def decide_if_standing_or_moving(self, cars, waymo=True):
        for i in range(len(cars)):
            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                if waymo:
                    dist_traveled = (np.power(start[0] - end[0], 2) +
                                     np.power(start[1] - end[1], 2))
                else:
                    dist_traveled = (np.power(start[0] - end[0], 2) +
                                     np.power(start[2] - end[2], 2))

                if np.sqrt(dist_traveled) > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_both(self, cars, waymo=True):
        for i in range(len(cars)):
            old_loc = None
            diffs_arr = []
            for loc in cars[i].locations:
                if loc is not None:
                    if old_loc is None:
                        old_loc = loc
                        continue
                    diff = np.array(loc[:3]) - np.array(old_loc[:3])
                    diffs_arr.append(diff)
                    old_loc = loc
                else:
                    old_loc = None

            if len(diffs_arr) <= 1:
                cars[i].moving = False
                continue
            diffs_arr = np.array(diffs_arr)
            means = np.mean(diffs_arr, axis=0)
            std_deltas = np.std(diffs_arr, axis=0)
            sigma = std_deltas / np.sqrt(2)

            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                net_displacement_vector = end[:3] - start[:3]
                net_displacement = np.linalg.norm(net_displacement_vector)

                N = len(diffs_arr)
                sigma_net = np.linalg.norm(sigma)
                std_net_displacement = sigma_net * np.sqrt(N)

                z_score = net_displacement / std_net_displacement
                p_value = 1 - stats.norm.cdf(z_score)

                alpha = self.cfg.frames_creation.alpha_value

                if not self.cfg.general.supress_debug_prints:
                    print(f"Estimated noise standard deviation (σ) in each axis:")
                    print(f"σ_x: {sigma[0]:.4f}, σ_y: {sigma[1]:.4f}, σ_z: {sigma[2]:.4f}")
                    print(f"Net displacement over {N} frames: {net_displacement:.4f} units")
                    print(f"Expected net displacement due to noise: {std_net_displacement:.4f} units")
                    print(f"Z-score: {z_score:.4f}")
                    print(f"P-value: {p_value:.4f}")

                if p_value < alpha and net_displacement > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_both2(self, cars, waymo=True, suppress=True):
        for i in range(len(cars)):
            old_loc = None
            diffs_arr = []
            for loc in cars[i].locations:
                if loc is not None:
                    if old_loc is None:
                        old_loc = loc
                        continue
                    diff = np.array(loc[:3]) - np.array(old_loc[:3])
                    diffs_arr.append(diff)
                    old_loc = loc
                else:
                    old_loc = None

            if len(diffs_arr) <= 1:
                cars[i].moving = True  # Assume moving by default
                continue
            diffs_arr = np.array(diffs_arr)
            std_deltas = np.std(diffs_arr, axis=0)
            sigma = std_deltas / np.sqrt(2)

            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = True  # Assume moving by default
            else:
                median_diffs = np.mean(diffs_arr, axis=0)
                if not suppress:
                    print("Pre: ", median_diffs, sigma)

                median_diffs = np.sqrt(median_diffs[0] ** 2 + median_diffs[2] ** 2)
                sigma = np.sqrt(sigma[0] ** 2 + sigma[2] ** 2)
                whole_diff = end - start
                whole_dist = np.sqrt(whole_diff[0] ** 2 + whole_diff[2] ** 2)
                if not suppress:
                    print("Post: ", median_diffs, sigma)
                    print("Number: ", len(diffs_arr))
                if whole_dist > self.cfg.frames_creation.dist_moving or median_diffs > self.cfg.frames_creation.speed_moving or (median_diffs / sigma) >= self.cfg.frames_creation.ratio_moving:
                    cars[i].moving = True  # Car is standing
                else:
                    cars[i].moving = False  # Car is moving

                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_bothv3(self, cars, waymo=False):
        for i in range(len(cars)):
            #Classify based on LiDAR
            concatenated_lidar = np.concatenate(cars[i].lidar, axis=0)

            center_of_lidar = np.mean(concatenated_lidar, axis=0)

            dists = np.linalg.norm(concatenated_lidar - center_of_lidar, axis=1)

            close_points = dists < 5.

            #Classify based on locations
            locations = cars[i].locations
            locs = np.zeros((len(locations), 3))
            for z in range(len(locations)):
                loc = locations[z]
                if loc is not None:
                    locs[z, :] = loc[:3]

            center_of_locations = np.mean(locs, axis=0)

            dists = np.linalg.norm(locs - center_of_locations, axis=1)

            close_locations = dists < 5.
            print("Points: ", np.sum(close_points), concatenated_lidar.shape[0])
            print("Locations: ", np.sum(close_locations), len(locations))
            if np.sum(close_points) < 0.95 * concatenated_lidar.shape[0]:
                cars[i].moving = True
            else:
                cars[i].moving = False

            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_both4(self, cars, waymo=True):
        for i in range(len(cars)):
            old_loc = None
            diffs_arr = []
            for loc in cars[i].locations:
                if loc is not None:
                    if old_loc is None:
                        old_loc = loc
                        continue
                    diff = np.array(loc[:3]) - np.array(old_loc[:3])
                    diffs_arr.append(diff)
                    old_loc = loc
                else:
                    old_loc = None

            if len(diffs_arr) <= 1:
                cars[i].moving = False
                continue
            diffs_arr = np.array(diffs_arr)
            means = np.mean(diffs_arr, axis=0)
            std_deltas = np.std(diffs_arr, axis=0)
            sigma = std_deltas / np.sqrt(2)

            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                net_displacement_vector = end[:3] - start[:3]
                net_displacement = np.linalg.norm(net_displacement_vector)

                N = len(diffs_arr)
                sigma_net = np.linalg.norm(sigma)
                std_net_displacement = sigma_net * np.sqrt(N)

                z_score = net_displacement / std_net_displacement
                #z_score = np.linalg.norm(means) / sigma_net

                if not self.cfg.general.supress_debug_prints:
                    print("means: ", means, np.linalg.norm(means))
                    print("z score: ", z_score)
                    print("sigmas: ", sigma, np.linalg.norm(sigma))

                if z_score > 1.0 and net_displacement > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_both5(self, cars, waymo=True):
        for i in range(len(cars)):
            old_loc = None
            diffs_arr = []
            for loc in cars[i].locations:
                if loc is not None:
                    if old_loc is None:
                        old_loc = loc
                        continue
                    diff = np.array(loc[:3]) - np.array(old_loc[:3])
                    diffs_arr.append(diff)
                    old_loc = loc
                else:
                    old_loc = None

            if len(diffs_arr) <= 1:
                cars[i].moving = False
                continue
            diffs_arr = np.array(diffs_arr)
            means = np.mean(diffs_arr, axis=0)
            std_deltas = np.std(diffs_arr, axis=0)
            sigma = std_deltas / np.sqrt(2)

            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                net_displacement_vector = end[:3] - start[:3]
                net_displacement = np.linalg.norm(net_displacement_vector)

                N = len(diffs_arr)
                sigma_net = np.linalg.norm(sigma)
                std_net_displacement = sigma_net * np.sqrt(N)

                #z_score = net_displacement / std_net_displacement
                z_score = np.linalg.norm(means) / sigma_net

                if not self.cfg.general.supress_debug_prints:
                    print("means: ", means, np.linalg.norm(means))
                    print("z score: ", z_score)
                    print("sigmas: ", sigma, np.linalg.norm(sigma))

                if z_score > 0.2 and net_displacement > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_both6(self, cars, waymo=True):
        for i in range(len(cars)):
            old_loc = None
            diffs_arr = []
            for loc in cars[i].locations:
                if loc is not None:
                    if old_loc is None:
                        old_loc = loc
                        continue
                    diff = np.array(loc[:3]) - np.array(old_loc[:3])
                    diffs_arr.append(diff)
                    old_loc = loc
                else:
                    old_loc = None

            if len(diffs_arr) <= 1:
                cars[i].moving = False
                continue
            diffs_arr = np.array(diffs_arr)
            means = np.mean(diffs_arr, axis=0)
            std_deltas = np.std(diffs_arr, axis=0)
            sigma = std_deltas / np.sqrt(2)

            start = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[z] is not None:
                    start = cars[i].locations[z]
                    break

            end = None
            for z in range(len(cars[i].locations)):
                if cars[i].locations[-(z + 1)] is not None:
                    end = cars[i].locations[-(z + 1)]
                    break

            if start is None or end is None:
                cars[i].moving = False
            else:
                net_displacement_vector = end[:3] - start[:3]
                net_displacement = np.linalg.norm(net_displacement_vector)

                N = len(diffs_arr)
                sigma_net = np.linalg.norm(sigma)

                #z_score = net_displacement / std_net_displacement
                z_score = np.linalg.norm(means) - sigma_net

                if not self.cfg.general.supress_debug_prints:
                    print("means: ", means, np.linalg.norm(means))
                    print("z score: ", z_score)
                    print("sigmas: ", sigma, np.linalg.norm(sigma))

                if z_score > 2. and net_displacement > self.cfg.frames_creation.dist_treshold_moving:
                    cars[i].moving = True
                else:
                    cars[i].moving = False

                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_trajectory_fit(self, cars, waymo=True):
        """
        Legacy method - kept for compatibility. Use decide_if_standing_or_moving_robust instead.
        """
        return self.decide_if_standing_or_moving_robust(cars, waymo)

    def decide_if_standing_or_moving_robust(self, cars, waymo=True):
        """
        Robust multi-feature motion classification.
        
        Key insight: For a STANDING vehicle, frame-to-frame displacements are pure noise,
        so velocity vectors point in RANDOM directions (expected cosine similarity ≈ 0).
        For a MOVING vehicle, velocity vectors are roughly ALIGNED (cosine similarity → 1).
        
        This method combines multiple orthogonal features:
        1. Velocity Direction Consistency - Are velocities pointing the same way?
        2. Path Efficiency - Is the path direct (net/total ≈ 1) or wandering (≈ 0)?
        3. Trajectory Fit Quality - Does position follow a smooth polynomial curve?
        4. Signal-to-Noise Ratio - Is mean velocity magnitude >> noise level?
        
        A vehicle is classified as MOVING if it passes multiple criteria, providing
        robustness against edge cases that fool any single method.
        """
        for i in range(len(cars)):
            # Extract valid locations and times
            locs = []
            times = []
            for t, loc in enumerate(cars[i].locations):
                if loc is not None:
                    locs.append(loc[:3])
                    times.append(t)
            
            # Insufficient data - default to standing (safer for pseudo-label aggregation)
            if len(locs) < 3:
                cars[i].moving = False
                continue
            
            locs = np.array(locs)
            times = np.array(times)
            N = len(locs)
            
            # ===== FEATURE 1: Net Displacement =====
            net_displacement = np.linalg.norm(locs[-1] - locs[0])
            
            # Quick rejection: if displacement is tiny, definitely standing
            if net_displacement < self.cfg.frames_creation.dist_treshold_moving * 0.5:
                cars[i].moving = False
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue
            
            # ===== FEATURE 2: Velocity Direction Consistency =====
            # Compute frame-to-frame velocity vectors
            velocities = np.diff(locs, axis=0)  # Shape: (N-1, 3)
            vel_magnitudes = np.linalg.norm(velocities, axis=1)
            
            # Compute cosine similarity between consecutive velocity vectors
            direction_scores = []
            for j in range(len(velocities) - 1):
                v1, v2 = velocities[j], velocities[j + 1]
                mag1, mag2 = vel_magnitudes[j], vel_magnitudes[j + 1]
                # Only compare if both velocities are non-negligible
                if mag1 > 1e-6 and mag2 > 1e-6:
                    cos_sim = np.dot(v1, v2) / (mag1 * mag2)
                    direction_scores.append(cos_sim)
            
            # Mean direction consistency: ~1 for moving, ~0 for standing
            if len(direction_scores) >= 2:
                mean_direction_consistency = np.mean(direction_scores)
            else:
                mean_direction_consistency = 0.0
            
            # ===== FEATURE 3: Path Efficiency =====
            # Ratio of net displacement to total path length
            # Moving straight: efficiency ≈ 1, Random walk: efficiency << 1
            total_path_length = np.sum(vel_magnitudes)
            path_efficiency = net_displacement / (total_path_length + 1e-9)
            
            # ===== FEATURE 4: Signal-to-Noise Ratio (using MAD for robustness) =====
            # Mean velocity magnitude vs. spread (noise estimate)
            mean_vel_mag = np.mean(vel_magnitudes)
            # Use Median Absolute Deviation - robust to outliers
            mad = np.median(np.abs(vel_magnitudes - np.median(vel_magnitudes)))
            # Convert MAD to standard deviation estimate (for Gaussian: σ ≈ 1.4826 * MAD)
            noise_estimate = 1.4826 * mad + 1e-9
            snr = mean_vel_mag / noise_estimate
            
            # ===== FEATURE 5: Trajectory Fit Quality (Adaptive Polynomial) =====
            # Try both linear (constant velocity) and quadratic (acceleration)
            best_r2 = 0.0
            for degree in [1, 2]:
                if N < degree + 2:  # Need enough points for meaningful fit
                    continue
                r2_per_dim = []
                for dim in range(3):
                    y = locs[:, dim]
                    var_y = np.var(y)
                    if var_y < 1e-9:  # No variation in this dimension
                        continue
                    
                    coeffs = np.polyfit(times, y, degree)
                    y_pred = np.polyval(coeffs, times)
                    
                    rss = np.sum((y - y_pred) ** 2)
                    tss = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - rss / (tss + 1e-9)
                    r2_per_dim.append(r2)
                
                if r2_per_dim:
                    # Use max R² across dimensions (motion may be primarily in one axis)
                    best_r2 = max(best_r2, np.max(r2_per_dim))
            
            # ===== DECISION: Combine Features =====
            # Count how many indicators suggest motion
            motion_votes = 0
            
            # Vote 1: Sufficient net displacement
            if net_displacement > self.cfg.frames_creation.dist_treshold_moving:
                motion_votes += 1
            
            # Vote 2: Consistent velocity direction (strong indicator)
            if mean_direction_consistency > 0.3:
                motion_votes += 1
            if mean_direction_consistency > 0.6:
                motion_votes += 1  # Extra vote for strong consistency
            
            # Vote 3: Efficient path (not wandering randomly)
            if path_efficiency > 0.5:
                motion_votes += 1
            
            # Vote 4: Good signal-to-noise ratio
            if snr > 1.5:
                motion_votes += 1
            
            # Vote 5: Trajectory fits a smooth curve
            if best_r2 > 0.5:
                motion_votes += 1
            if best_r2 > 0.8:
                motion_votes += 1  # Extra vote for excellent fit
            
            # Final decision: Need at least 3 votes (out of max 7)
            # This ensures robustness - no single noisy feature can dominate
            cars[i].moving = (motion_votes >= 3) and (net_displacement > self.cfg.frames_creation.dist_treshold_moving)
            
            if not self.cfg.general.supress_debug_prints:
                print(f"Car {i}: disp={net_displacement:.3f}, dir_cons={mean_direction_consistency:.3f}, "
                      f"path_eff={path_efficiency:.3f}, snr={snr:.3f}, r2={best_r2:.3f}, "
                      f"votes={motion_votes} -> {'MOVING' if cars[i].moving else 'STANDING'}")
            
            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_sign_test(self, cars, waymo=True):
        """
        Simple and elegant motion classification using the Sign Test.
        
        Core idea: For a STANDING vehicle, frame-to-frame displacements are random,
        so the SIGN of each displacement is like a coin flip (50% positive, 50% negative).
        For a MOVING vehicle, displacements consistently point in the travel direction,
        so signs are predominantly the same.
        
        Statistical foundation: Under the null hypothesis (standing), the number of 
        positive signs follows Binomial(n, 0.5). We reject if too many signs agree.
        
        This is non-parametric - no assumptions about noise distribution!
        """
        for i in range(len(cars)):
            # Extract valid locations
            locs = []
            for loc in cars[i].locations:
                if loc is not None:
                    locs.append(loc[:3])
            
            if len(locs) < 4:  # Need at least 3 displacements
                cars[i].moving = False
                continue
            
            locs = np.array(locs)
            
            # Net displacement check
            net_displacement = np.linalg.norm(locs[-1] - locs[0])
            if net_displacement < self.cfg.frames_creation.dist_treshold_moving:
                cars[i].moving = False
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue
            
            # Compute displacements
            displacements = np.diff(locs, axis=0)  # (N-1, 3)
            n_displacements = len(displacements)
            
            # For each axis, compute sign consistency
            # Use the axis with largest total displacement (most informative)
            net_per_axis = np.abs(locs[-1] - locs[0])
            primary_axis = np.argmax(net_per_axis)
            
            # Get displacements along primary axis
            d = displacements[:, primary_axis]
            net_sign = np.sign(locs[-1, primary_axis] - locs[0, primary_axis])
            
            # Count how many displacements agree with net direction
            if net_sign == 0:
                cars[i].moving = False
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue
            
            n_agree = np.sum(np.sign(d) == net_sign)
            agreement_ratio = n_agree / n_displacements
            
            # Under null (standing): expected ratio = 0.5
            # Binomial test: P(X >= n_agree) where X ~ Binomial(n, 0.5)
            # For simplicity, use threshold: if >70% agree, it's moving
            # (This corresponds to ~95% confidence for n>=10)
            
            # Adaptive threshold based on sample size
            # Smaller samples need higher agreement to be confident
            if n_displacements >= 10:
                threshold = 0.65
            elif n_displacements >= 6:
                threshold = 0.70
            else:
                threshold = 0.75
            
            cars[i].moving = agreement_ratio > threshold
            
            if not self.cfg.general.supress_debug_prints:
                print(f"Car {i}: net_disp={net_displacement:.3f}, axis={primary_axis}, "
                      f"agree={n_agree}/{n_displacements} ({agreement_ratio:.2f}) "
                      f"-> {'MOVING' if cars[i].moving else 'STANDING'}")
            
            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_runs_test(self, cars, waymo=True):
        """\
        Motion classification using a Wald–Wolfowitz runs test on displacement signs.

        Good idea, kept simple:
        - Standing: displacement sign sequence is random -> many runs (frequent sign changes)
        - Moving (even accelerating): signs are mostly consistent -> few runs

        This is non-parametric (doesn't assume Gaussian noise) and uses only sign patterns,
        so it is robust to scale/units and outliers.
        """
        for i in range(len(cars)):
            locs = []
            for loc in cars[i].locations:
                if loc is not None:
                    locs.append(loc[:3])

            if len(locs) < 5:  # need at least 4 displacements for a meaningful runs statistic
                cars[i].moving = False
                continue

            locs = np.array(locs)

            net_displacement = np.linalg.norm(locs[-1] - locs[0])
            if net_displacement < self.cfg.frames_creation.dist_treshold_moving:
                cars[i].moving = False
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue

            displacements = np.diff(locs, axis=0)
            net_per_axis = np.abs(locs[-1] - locs[0])
            primary_axis = int(np.argmax(net_per_axis))
            d = displacements[:, primary_axis]

            # Convert to signs, drop zeros (no information)
            signs = np.sign(d)
            signs = signs[signs != 0]
            if len(signs) < 4:
                cars[i].moving = False
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue

            n_pos = int(np.sum(signs > 0))
            n_neg = int(np.sum(signs < 0))
            if n_pos == 0 or n_neg == 0:
                # All signs identical -> extremely consistent direction
                cars[i].moving = True
                if not self.cfg.general.supress_debug_prints:
                    print(
                        f"Car {i}: net_disp={net_displacement:.3f}, axis={primary_axis}, all_signs_same -> MOVING"
                    )
                if waymo:
                    for z in range(len(cars[i].locations)):
                        if cars[i].locations[z] is not None:
                            cars[i].locations[z] = cars[i].locations[z][:3]
                continue

            # Count runs in the sign sequence
            runs = 1
            for j in range(1, len(signs)):
                if signs[j] != signs[j - 1]:
                    runs += 1

            # Runs test approximation (normal) under H0 of randomness
            n = n_pos + n_neg
            mu = (2.0 * n_pos * n_neg) / n + 1.0
            var = (
                (2.0 * n_pos * n_neg) * (2.0 * n_pos * n_neg - n)
            ) / (n * n * (n - 1.0) + 1e-9)
            sigma = np.sqrt(max(var, 1e-12))

            # Continuity correction: because runs is discrete
            z_score = (runs - mu) / sigma
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

            # Decision: moving if the sequence has *too few* runs (too structured)
            # Use a simple significance threshold.
            alpha = 0.05
            cars[i].moving = (z_score < -1.96) and (p_value < alpha)

            if not self.cfg.general.supress_debug_prints:
                print(
                    f"Car {i}: net_disp={net_displacement:.3f}, axis={primary_axis}, "
                    f"runs={runs}, n_pos={n_pos}, n_neg={n_neg}, z={z_score:.2f}, p={p_value:.3f} -> "
                    f"{'MOVING' if cars[i].moving else 'STANDING'}"
                )

            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def decide_if_standing_or_moving_lidar_consistency(self, cars, waymo=True):
        """\
        Motion classification using *per-frame* instance LiDAR point clouds.

        Why this can beat location-only classification:
        - Location tracks are often dominated by detector jitter.
        - Instance LiDAR (already cropped by the mask) carries geometric evidence.
        - In the reference frame, a standing object produces overlapping per-frame clouds.
        - A moving object produces non-overlapping clouds (temporal smear).

        Kept simple (no ICP):
        1) Robust per-frame centers (median) and within-frame spread.
        2) Between-frame center dispersion ratio (scale-free).
        3) Overlap ratio: median 1-NN distance to a reference frame cloud (scale-free).

        Decision: moving if either ratio is sufficiently large.
        """

        def _downsample(points_xyz: np.ndarray, max_points: int) -> np.ndarray:
            if points_xyz.shape[0] <= max_points:
                return points_xyz
            idx = np.random.choice(points_xyz.shape[0], size=max_points, replace=False)
            return points_xyz[idx]

        min_points_per_frame = 20
        max_ref_points = 4000
        max_query_points = 1500

        for i in range(len(cars)):
            if cars[i].lidar is None or len(cars[i].lidar) == 0:
                cars[i].moving = False
                continue

            frames_xyz = []
            for pts in cars[i].lidar:
                if pts is None:
                    continue
                pts_xyz = np.asarray(pts)[:, :3]
                if pts_xyz.shape[0] < min_points_per_frame:
                    continue
                frames_xyz.append(pts_xyz)

            if len(frames_xyz) < 2:
                cars[i].moving = False
                continue

            centers = []
            spreads = []
            for pts_xyz in frames_xyz:
                center = np.median(pts_xyz, axis=0)
                centers.append(center)

                d = np.linalg.norm(pts_xyz - center[None, :], axis=1)
                spreads.append(float(np.percentile(d, 70)))

            centers = np.asarray(centers)
            within = float(np.median(spreads)) + 1e-6

            center_med = np.median(centers, axis=0)
            between = float(np.median(np.linalg.norm(centers - center_med[None, :], axis=1)))
            center_ratio = between / within

            # Overlap score using nearest-neighbor distance to the densest frame
            ref_idx = int(np.argmax([p.shape[0] for p in frames_xyz]))
            ref = _downsample(frames_xyz[ref_idx].astype(np.float32), max_ref_points)
            ref = np.ascontiguousarray(ref)

            overlap_dists = []
            try:
                index = faiss.IndexFlatL2(3)
                index.add(ref)
                for j, pts_xyz in enumerate(frames_xyz):
                    if j == ref_idx:
                        continue
                    qry = _downsample(pts_xyz.astype(np.float32), max_query_points)
                    qry = np.ascontiguousarray(qry)
                    D, _I = index.search(qry, 1)  # squared distances
                    overlap_dists.append(float(np.median(np.sqrt(D[:, 0] + 1e-12))))
            except Exception:
                overlap_dists = []

            overlap = float(np.median(overlap_dists)) if len(overlap_dists) > 0 else 0.0
            overlap_ratio = overlap / within

            # Simple, scale-free thresholds
            cars[i].moving = (center_ratio > 0.45) or (overlap_ratio > 0.55)

            if not self.cfg.general.supress_debug_prints:
                print(
                    f"Car {i}: within={within:.3f}, between={between:.3f}, center_ratio={center_ratio:.2f}, "
                    f"overlap={overlap:.3f}, overlap_ratio={overlap_ratio:.2f} -> "
                    f"{'MOVING' if cars[i].moving else 'STANDING'}"
                )

            if waymo:
                for z in range(len(cars[i].locations)):
                    if cars[i].locations[z] is not None:
                        cars[i].locations[z] = cars[i].locations[z][:3]

        return cars

    def load_merged_frames_from_files_waymo_tracker(self, track2D=False, merge_two_trackers=False):
        self.lidar = self.waymo_lidar[self.pic_index].T

        if merge_two_trackers:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars2D = pickle.loads(decompressed_data)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars3D = pickle.loads(decompressed_data)
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars2D = pickle.loads(decompressed_data)
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                cars3D = pickle.loads(decompressed_data)
            self.cars = cars2D + cars3D
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[1], reverse=True)
            self.cars3D_start = len(cars2D)

            #Make the optimization area significantly smaller, as we dont need so much precision for the merging :)
            self.opt_param1_iters = 20  # X
            self.opt_param2_iters = 20  # Y or Z, depending on dataset
            self.opt_param3_iters = 20  # Theta

        elif track2D:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_2DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            cars = pickle.loads(decompressed_data)
            self.cars = cars
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)
        else:
            if self.cfg.frames_creation.use_growing_for_point_extraction:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack_growing/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            else:
                with open(self.cfg.paths.merged_frames_path + "cars_3DTrack/" + self.file_name + "/" + str(self.pic_index) + ".zstd", 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
            cars = pickle.loads(decompressed_data)
            self.cars = cars
            self.cars = sorted(self.cars, key=lambda x: x.lidar.shape[0], reverse=True)

        cur_lidar = np.ascontiguousarray(self.lidar.T[:,:3]).astype('float32')
        quantizer = faiss.IndexFlatL2(cur_lidar.shape[1])
        index_faiss = faiss.IndexIVFFlat(quantizer, cur_lidar.shape[1], int(np.floor(np.sqrt(cur_lidar.shape[0]))))
        index_faiss.train(cur_lidar)
        index_faiss.add(cur_lidar)
        index_faiss.nprobe = 10
        
        new_cars = []

        for car in self.cars:
            if car.lidar is not None:
                center = np.zeros((1, 3))
                center[0, 0] = np.median(car.lidar[:, 0])
                center[0, 1] = np.median(car.lidar[:, 1])
                center[0, 2] = np.median(car.lidar[:, 2])

                idx, distances, indexes = index_faiss.range_search(np.ascontiguousarray(center).astype('float32'), 2. ** 2)

                if len(distances) < 1:
                    continue

                padding = np.ones((car.lidar.shape[0], 3))

                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
                new_cars.append(car)
        self.cars = new_cars

        '''
        for car in self.cars:
            if car.lidar is not None:
                padding = np.ones((car.lidar.shape[0], 3))
                car.lidar = np.concatenate((car.lidar, padding), axis=1).T
        '''

    def non_maxima_surpression(self, cars):
        num_of_cars = len(cars)

        indx = 0
        to_be_optimized = []

        while indx < num_of_cars:
            num_of_cars = len(cars)
            if cars[indx].lidar is None or cars[indx].optimized is False:
                indx += 1
                continue
            else:
                bbox_ref_center = [cars[indx].x, cars[indx].y, cars[indx].z]
                if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                    bbox_ref_size = [cars[indx].width, cars[indx].height, cars[indx].length]
                elif self.args.dataset == 'waymo':
                    bbox_ref_size = [cars[indx].length, cars[indx].width, cars[indx].height]
                else:
                    raise ValueError('Dataset not supported')
                bbox_ref_theta = cars[indx].theta

                scaled_cube = np.diag(bbox_ref_size).dot(self.unit_cube.T).T
                if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                    rotation = R.from_euler('y', bbox_ref_theta, degrees=False)
                elif self.args.dataset == 'waymo':
                    rotation = R.from_euler('z', bbox_ref_theta, degrees=False)
                else:
                    raise ValueError('Dataset not supported')

                rotated_cube = rotation.apply(scaled_cube)
                bbox_ref_points = rotated_cube + bbox_ref_center

                for i in range(indx + 1, num_of_cars):
                    if cars[i].lidar is None or not cars[i].optimized:
                        continue
                    else:
                        bbox_cur_center = [cars[i].x, cars[i].y, cars[i].z]
                        if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                            bbox_cur_size = [cars[i].width, cars[i].height, cars[i].length]
                        elif self.args.dataset == 'waymo':
                            bbox_cur_size = [cars[i].length, cars[i].width, cars[i].height]
                        else:
                            raise ValueError('Dataset not supported')
                        bbox_cur_theta = cars[i].theta

                        scaled_cube = np.diag(bbox_cur_size).dot(self.unit_cube.T).T
                        if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == 'waymo_converted' or self.args.dataset == 'dsec':
                            rotation = R.from_euler('y', bbox_cur_theta, degrees=False)
                        elif self.args.dataset == 'waymo':
                            rotation = R.from_euler('z', bbox_cur_theta, degrees=False)
                        else:
                            raise ValueError('Dataset not supported')
                        rotated_cube = rotation.apply(scaled_cube)
                        bbox_cur_points = rotated_cube + bbox_cur_center
                        #print(bbox_ref_points, bbox_cur_points)
                        vol, iou = pytorch3d.ops.box3d_overlap(torch.tensor(bbox_ref_points, dtype=torch.float32).unsqueeze(0), torch.tensor(bbox_cur_points, dtype=torch.float32).unsqueeze(0))
                        if iou[0].item() > self.cfg.optimization.nms_threshold:
                            if self.cfg.optimization.nms_merge_and_reopt:
                                cars[indx].lidar = np.concatenate((cars[indx].lidar, cars[i].lidar), axis=1)
                                to_be_optimized.append(indx)
                            cars[i].lidar = None
                            cars[i].optimized = False
                indx += 1

        if self.cfg.optimization.merge_two_trackers:
            to_be_optimized = np.array(list(range(len(cars))))
            self.opt_param1_iters = 40  # X
            self.opt_param2_iters = 40  # Y or Z, depending on dataset
            self.opt_param3_iters = 40  # Theta
        else:
            to_be_optimized = np.array(to_be_optimized)
        unique = np.unique(to_be_optimized)

        return cars, list(unique)

    # Function which returns the location of the cars in the picture. The locations are a median of points detected
    def get_car_locations_from_img(self, scan, T_cur_to_ref, masks, scores=None, img_idx=None):
        transformed_means = []
        lidar_points = []
        out_masks = []
        out_scores = []

        #depth_map = self.create_depth_map(scan)

        for z in range(len(masks)):
            mask = masks[z]
            score = scores[z] if scores is not None else 0.4
            mask_old = copy.deepcopy(mask)
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)

            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        continue

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # look for the mean on the filtered data by circle, which will hopefully get better results
            if filtered_lidar.shape[0] > 0:
                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            if self.cfg.frames_creation.use_pseudo_lidar:
                dist = np.sqrt(x_mean ** 2 + y_mean ** 2 + z_mean ** 2)
                if dist > self.cfg.frames_creation.max_distance_pseudo_lidar:
                    continue

            # Transform the points between frames.
            mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            mean_transformed += T_cur_to_ref[0:3, 3]
            # Check if the car is atleast infront of us
            if mean_transformed[2] > 0.:
                # Lets save the lidar points for the moving cars detection.
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]
                #filtered_depths = depth_map[mask_old]

                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                # Filter by circle
                dist_from_mean = np.sqrt(
                    (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

                indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                filtered_lidar = \
                    np.array(
                        [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]
                #filtered_depths = filtered_depths[indexes]

                #mean_md = np.mean(filtered_depths)
                #std_md = np.std(filtered_depths)

                #lower_threshold = mean_md - std_md
                #upper_threshold = mean_md + std_md

                #inlier_mask_car = (filtered_depths >= lower_threshold) | (filtered_depths <= upper_threshold)
                #inlier_mask_car = inlier_mask_car.flatten()

                #filtered_lidar = filtered_lidar[inlier_mask_car, :]
                # Filter points with hdbscan
                if self.cfg.frames_creation.use_hdbscan:
                   filtered_lidar = self.ensamble_clustering(filtered_lidar)

                # Transform the points between frames.
                filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
                filtered_lidar += T_cur_to_ref[0:3, 3]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    continue

                transformed_means.append(mean_transformed)
                lidar_points.append(filtered_lidar)
                out_masks.append(mask_old)
                out_scores.append(score)

        return np.array(transformed_means), lidar_points, np.array(out_masks), np.array(out_scores)

    def get_car_locations_from_img_gt(self, scan, T_cur_to_ref, masks, img_idx=None):
        transformed_means = []
        lidar_points = []
        out_masks = []

        for z in range(len(masks)):
            mask = masks[z]
            mask_old = copy.deepcopy(mask)
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)

            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Transform the points between frames.
            mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            mean_transformed += T_cur_to_ref[0:3, 3]
            # Check if the car is atleast infront of us
            # Lets save the lidar points for the moving cars detection.
            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]
            #filtered_depths = depth_map[mask_old]

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]
            #filtered_depths = filtered_depths[indexes]

            #mean_md = np.mean(filtered_depths)
            #std_md = np.std(filtered_depths)

            #lower_threshold = mean_md - std_md
            #upper_threshold = mean_md + std_md

            #inlier_mask_car = (filtered_depths >= lower_threshold) | (filtered_depths <= upper_threshold)
            #inlier_mask_car = inlier_mask_car.flatten()

            #filtered_lidar = filtered_lidar[inlier_mask_car, :]
            # Filter points with hdbscan
            if self.cfg.frames_creation.use_hdbscan:
               filtered_lidar = self.ensamble_clustering(filtered_lidar)

            # Transform the points between frames.
            filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
            filtered_lidar += T_cur_to_ref[0:3, 3]

            transformed_means.append(mean_transformed)
            lidar_points.append(filtered_lidar)
            out_masks.append(mask_old)

        return np.array(transformed_means), lidar_points, np.array(out_masks)

    def get_car_locations_from_img_all(self, scan, masks, img_idx=None, scores=None):
        if self.generate_candidates:
            lidar_points = []
            out_masks = []
            out_scores = []

            if scores is None:
                scores = [0.4] * len(masks)

            for z in range(len(masks)):
                mask = masks[z]
                score = scores[z]
                mask_old = copy.deepcopy(mask)
                # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
                struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)

                mask = np.invert(mask)
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)

                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

                # Sometimes, we just lack the number of points, so if it is small, just skip it
                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    struct_size = 1
                    mask = np.invert(copy.deepcopy(mask_old))
                    mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                    mask = np.invert(mask)
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        # Now, get indexes of the points which project into the mask
                        tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                        # Now, filter the points based on the indexes
                        filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                            0]

                        if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                            continue

                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                # Filter by circle
                dist_from_mean = np.sqrt(
                    (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

                indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                filtered_lidar = \
                    np.array(
                        [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

                # look for the mean on the filtered data by circle, which will hopefully get better results
                if filtered_lidar.shape[0] > 0:
                    x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                if self.cfg.frames_creation.use_pseudo_lidar:
                    dist = np.sqrt(x_mean ** 2 + y_mean ** 2 + z_mean ** 2)
                    if dist > self.cfg.frames_creation.max_distance_pseudo_lidar:
                        continue

                # Lets save the lidar points for the moving cars detection.
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]
                #filtered_depths = depth_map[mask_old]

                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                # Filter by circle
                dist_from_mean = np.sqrt(
                    (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

                indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                filtered_lidar = \
                    np.array(
                        [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]
                #filtered_depths = filtered_depths[indexes]

                #mean_md = np.mean(filtered_depths)
                #std_md = np.std(filtered_depths)

                #lower_threshold = mean_md - std_md
                #upper_threshold = mean_md + std_md

                #inlier_mask_car = (filtered_depths >= lower_threshold) | (filtered_depths <= upper_threshold)
                #inlier_mask_car = inlier_mask_car.flatten()

                #filtered_lidar = filtered_lidar[inlier_mask_car, :]
                # Filter points with hdbscan
                if self.cfg.frames_creation.use_hdbscan:
                    print(filtered_lidar.shape)
                    filtered_lidar = self.ensamble_clustering(filtered_lidar)

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    continue

                lidar_points.append(filtered_lidar)
                out_masks.append(mask_old)
                out_scores.append(score)

            out_masks = np.array(out_masks)
            out_scores = np.array(out_scores)
            compressed_arr = zstd.compress(pickle.dumps(lidar_points, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.merged_frames_path + "candidates_lidar/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)
            compressed_arr = zstd.compress(pickle.dumps(out_masks, pickle.HIGHEST_PROTOCOL))

            with open(self.cfg.paths.merged_frames_path + "candidates_masks/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)

            if not os.path.exists(self.cfg.paths.merged_frames_path + "candidates_scores/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "candidates_scores/", exist_ok=True)

            compressed_arr = zstd.compress(pickle.dumps(out_scores, pickle.HIGHEST_PROTOCOL))
            with open(self.cfg.paths.merged_frames_path + "candidates_scores/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)
        else:
            with open(self.cfg.paths.merged_frames_path + "candidates_lidar/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            lidar_points = pickle.loads(decompressed_data)
            with open(self.cfg.paths.merged_frames_path + "candidates_masks/" + self.file_name + ".zstd", 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            out_masks = pickle.loads(decompressed_data)
            
            path_scores = self.cfg.paths.merged_frames_path + "candidates_scores/" + self.file_name + ".zstd"
            if os.path.exists(path_scores):
                with open(path_scores, 'rb') as f:
                    decompressed_data = zstd.decompress(f.read())
                out_scores = pickle.loads(decompressed_data)
            else:
                out_scores = [0.4] * len(out_masks)

        return lidar_points, out_masks, out_scores

    def get_car_locations_from_img_waymo(self, img_index, scan, frame_index, out_det):
        transformed_means = []
        lidar_points = []
        info = []
        masks = []

        scan = scan[scan[:, 3] == img_index]
        scan = scan.T

        out_det = out_det[0]

        for z in range(len(out_det)):
            # Take the mask and transpose it
            mask = np.copy(out_det[z])

            mask_old = copy.deepcopy(mask)
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            #print("old: ", np.count_nonzero(mask))
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)
            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)
            # print("new: ", np.count_nonzero(mask))

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        continue

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (y_mean - filtered_lidar[:, 1]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # look for the mean on the filtered data by circle, which will hopefully get better results
            if filtered_lidar.shape[0] > 0:
                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Transform the points between frames.
            #mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            #mean_transformed += T_cur_to_ref[0:3, 3]
            mean_transformed = np.array([x_mean, y_mean, z_mean])

            # Lets save the lidar points for the moving cars detection.
            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                0]

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (y_mean - filtered_lidar[:, 1]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # Transform the points between frames.
            #filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
            #filtered_lidar += T_cur_to_ref[0:3, 3]

            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                continue

            transformed_means.append(mean_transformed)
            lidar_points.append(filtered_lidar)

            info_item = np.array([0, frame_index, img_index, 0])
            info.append(info_item)
            masks.append(out_det[z])

        return np.array(transformed_means), lidar_points, info, np.array(masks)

    def get_car_locations_from_img_waymo_growing(self, img_index, scan, frame_index, out_det):
        transformed_means = []
        lidar_points = []
        info = []
        masks = []

        scan_orig = np.copy(scan)
        out_det = out_det[0]

        for z in range(len(out_det)):
            # Take the mask and transpose it
            mask = np.copy(out_det[z])

            start = time.time_ns()
            filtered_lidar = self.perform_growing(mask, img_index, scan_orig)
            if not self.cfg.general.supress_debug_prints:
                print("Time to perform growing: ", (time.time_ns() - start) / 1000000)

            if filtered_lidar is not None:
                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    continue
                else:
                    x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                    mean_transformed = np.array([x_mean, y_mean, z_mean])

                    transformed_means.append(mean_transformed)
                    lidar_points.append(filtered_lidar)

                    info_item = np.array([0, frame_index, img_index, 0])
                    info.append(info_item)
                    masks.append(out_det[z])

            else:
                continue

        return np.array(transformed_means), lidar_points, info, np.array(masks)

    def precompute_detectron_pedestrians_all(self, offset=0):
        if self.generate_raw_masks_or_tracking:
            tmp_img_arr = [self.img]
            out_dete = self.run_detectron_batch(tmp_img_arr)[0]

            masks_to_save = []
            cyclist_flags_to_save = []
            scores_to_save = []

            if len(out_dete) > 0:
                detections = out_dete
                # Filter by score
                valid_indices = detections.scores > self.cfg.filtering.score_detectron_thresh
                detections = detections[valid_indices]
                
                # Separate persons and bicycles
                person_indices = (detections.pred_classes == 0).nonzero(as_tuple=True)[0]
                bicycle_indices = (detections.pred_classes == 1).nonzero(as_tuple=True)[0]
                
                persons = detections[person_indices]
                bicycles = detections[bicycle_indices]
                
                # Helper for IoU
                def compute_iou(box_a, box_b):
                    # box is tensor [x1, y1, x2, y2]
                    x1 = torch.max(box_a[0], box_b[0])
                    y1 = torch.max(box_a[1], box_b[1])
                    x2 = torch.min(box_a[2], box_b[2])
                    y2 = torch.min(box_a[3], box_b[3])
                    
                    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
                    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    
                    return inter_area / (box_a_area + box_b_area - inter_area + 1e-6)

                for k in range(len(persons)):
                    # Check cyclist flag
                    is_cyclist = False
                    p_box = persons.pred_boxes[k].tensor[0]
                    for b_idx in range(len(bicycles)):
                        b_box = bicycles.pred_boxes[b_idx].tensor[0]
                        if compute_iou(p_box, b_box) > 0.3: 
                            is_cyclist = True
                            break
                    
                    mask = np.array(persons.pred_masks[k].cpu()).transpose()
                    masks_to_save.append(mask)
                    cyclist_flags_to_save.append(is_cyclist)
                    scores_to_save.append(float(persons.scores[k]))

            masks_to_save = np.array(masks_to_save)
            cyclist_flags_to_save = np.array(cyclist_flags_to_save)
            scores_to_save = np.array(scores_to_save)

            payload = {
                'masks': masks_to_save,
                'flags': cyclist_flags_to_save.astype(np.uint8),
                'scores': scores_to_save
            }

            compressed_arr = zstd.compress(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))

            if not os.path.exists(self.cfg.paths.merged_frames_path + "masks_raw_pedestrians/"):
                os.makedirs(self.cfg.paths.merged_frames_path + "masks_raw_pedestrians/", exist_ok=True)

            with open(self.cfg.paths.merged_frames_path + "masks_raw_pedestrians/" + self.file_name + ".zstd", 'wb') as f:
                f.write(compressed_arr)

            return masks_to_save, cyclist_flags_to_save, scores_to_save
        
        else:
            file_name = self.folder + '/' + str(int(self.number) + offset).zfill(10)
            path = self.cfg.paths.merged_frames_path + "masks_raw_pedestrians/" + file_name + ".zstd"
            
            if not os.path.exists(path):
                return [], [], []
            with open(path, 'rb') as f:
                decompressed_data = zstd.decompress(f.read())
            loaded_data = pickle.loads(decompressed_data)

            if isinstance(loaded_data, dict):
                masks = loaded_data['masks']
                flags = loaded_data['flags']
                scores = loaded_data.get('scores', [0.4] * len(masks))
            else:
                masks = loaded_data
                flags = [False] * len(masks)
                scores = [0.4] * len(masks)

            return masks, flags, scores

    def get_pedestrian_locations_from_img(self, scan, T_cur_to_ref, masks, cyclist_flags, img_idx=None, scores=None):
        transformed_means = []
        lidar_points = []
        out_masks = []
        out_flags = []
        out_scores = []

        if scores is None:
            scores = [0.4] * len(masks)

        for z in range(len(masks)):
            mask = masks[z]
            score = scores[z]
            mask_old = copy.deepcopy(mask)
            
            # Shrink the mask to approx half of the area to avoid detecting outliers as standing cars
            struct_size = int(2 + np.sqrt(np.count_nonzero(mask)) // 10)

            mask = np.invert(mask)
            mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
            mask = np.invert(mask)

            # Now, get indexes of the points which project into the mask
            tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

            # Now, filter the points based on the indexes
            filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

            # Sometimes, we just lack the number of points, so if it is small, just skip it
            if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                struct_size = 1
                mask = np.invert(copy.deepcopy(mask_old))
                mask = scipy.ndimage.binary_dilation(mask, iterations=struct_size)
                mask = np.invert(mask)
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                    0]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    # Now, get indexes of the points which project into the mask
                    tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                    # Now, filter the points based on the indexes
                    filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[
                        0]

                    if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                        transformed_means.append(np.array([0.0, 0.0, 0.0]))
                        lidar_points.append(None)
                        out_masks.append(mask_old)
                        out_flags.append(cyclist_flags[z])
                        out_scores.append(score)
                        continue

            x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            # Filter by circle
            dist_from_mean = np.sqrt(
                (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

            indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

            filtered_lidar = \
                np.array(
                    [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

            # look for the mean on the filtered data by circle, which will hopefully get better results
            if filtered_lidar.shape[0] > 0:
                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

            if self.cfg.frames_creation.use_pseudo_lidar:
                dist = np.sqrt(x_mean ** 2 + y_mean ** 2 + z_mean ** 2)
                if dist > self.cfg.frames_creation.max_distance_pseudo_lidar:
                    transformed_means.append(np.array([0.0, 0.0, 0.0]))
                    lidar_points.append(None)
                    out_masks.append(mask_old)
                    out_flags.append(cyclist_flags[z])
                    out_scores.append(score)
                    continue

            # Transform the points between frames.
            mean_transformed = np.matmul(T_cur_to_ref[0:3, 0:3], np.array([x_mean, y_mean, z_mean]).T).transpose()
            mean_transformed += T_cur_to_ref[0:3, 3]
            # Check if the car is atleast infront of us
            if mean_transformed[2] > 0.:
                # Lets save the lidar points for the moving cars detection.
                # Now, get indexes of the points which project into the mask
                tmp1 = np.argwhere(mask_old[scan[4, :].astype(int), scan[5, :].astype(int)])

                # Now, filter the points based on the indexes
                filtered_lidar = np.array([scan[0, tmp1], scan[1, tmp1], scan[2, tmp1]]).transpose()[0]

                x_mean, y_mean, z_mean = self.compute_mean(filtered_lidar)

                # Filter by circle
                dist_from_mean = np.sqrt(
                    (x_mean - filtered_lidar[:, 0]) ** 2 + (z_mean - filtered_lidar[:, 2]) ** 2)

                indexes = np.argwhere(dist_from_mean < self.cfg.filtering.filter_diameter)

                filtered_lidar = \
                    np.array(
                        [filtered_lidar[indexes, 0], filtered_lidar[indexes, 1], filtered_lidar[indexes, 2]]).T[0]

                if self.cfg.frames_creation.use_hdbscan:
                   filtered_lidar = self.ensamble_clustering(filtered_lidar)

                # Transform the points between frames.
                filtered_lidar = np.matmul(T_cur_to_ref[0:3, 0:3], filtered_lidar.T).T
                filtered_lidar += T_cur_to_ref[0:3, 3]

                if filtered_lidar.shape[0] < self.cfg.filtering.moving_detection_threshold:
                    transformed_means.append(np.array([0.0, 0.0, 0.0]))
                    lidar_points.append(None)
                    out_masks.append(mask_old)
                    out_flags.append(cyclist_flags[z])
                    out_scores.append(score)
                    continue

                transformed_means.append(mean_transformed)
                lidar_points.append(filtered_lidar)
                out_masks.append(mask_old)
                out_flags.append(cyclist_flags[z])
                out_scores.append(score)

        return np.array(transformed_means), lidar_points, np.array(out_masks), out_flags, out_scores


    def prepare_scan(self, filename, img, lidar, save=True, crop=True, visu=False):
        self.P2_rect = self.kitti_data.calib.P_rect_00
        if visu:
            if self.cfg.visualization.show_real_lidar:
                lidar = self.transform_velo_to_cam(filename, lidar)
            elif self.cfg.frames_creation.use_pseudo_lidar:
                lidar = lidar.T
            else:
                lidar = self.transform_velo_to_cam(filename, lidar)
        else:
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar = lidar.T
            else:
                lidar = self.transform_velo_to_cam(filename, lidar)
        return self.project_lidar_points(lidar, img, save, crop)

    def prepare_scan_all(self, filename, img, lidar, save=True, crop=True, visu=False, lidar_is_raw=None):
        calib_path = self.cfg.paths.all_dataset_path + 'calibration/perspective.txt'
        self.P2_rect = self.load_calibration_all(calib_path)
        self.P2_rect = np.concatenate((self.P2_rect, np.array([[0, 0, 0, 1]])), axis=0)
        
        # Determine if lidar is raw (needs Velo->Cam transform) or pseudo (already in cam coords)
        if lidar_is_raw is None:
            lidar_is_raw = not self.cfg.frames_creation.use_pseudo_lidar
        
        if lidar_is_raw:
             calib_path_velo = self.cfg.paths.all_dataset_path + 'calibration/calib_cam_to_velo.txt'
             with open(calib_path_velo, 'r') as f:
                 line = f.readline()
                 values = [float(x) for x in line.split()]
                 if len(values) == 12:
                     T_cam_to_velo = np.array(values).reshape(3, 4)
                     T_cam_to_velo = np.vstack((T_cam_to_velo, [0, 0, 0, 1]))
                 elif len(values) == 16:
                     T_cam_to_velo = np.array(values).reshape(4, 4)
                 else:
                     T_cam_to_velo = np.eye(4)
             
             T_velo_to_cam = np.linalg.inv(T_cam_to_velo)
             
             lidar_h = np.hstack((lidar[:, :3], np.ones((lidar.shape[0], 1))))
             lidar = np.matmul(T_velo_to_cam, lidar_h.T)
             
             # Filter out points behind the camera (Z < 0)
             lidar = lidar[:, lidar[2, :] >= 0]
        else:
             lidar = lidar.T

        return self.project_lidar_points_all(lidar, img, save, crop)

    def prepare_scan_dsec(self, filename, img, lidar, save=True, crop=True, visu=False):
        x = self.cam_to_cam['intrinsics']['camRect1']['camera_matrix']
        P2_rect = np.array([[x[0], 0, x[2], 0],
                           [0, x[1], x[3], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.P2_rect = P2_rect

        if visu:
            if self.cfg.visualization.show_real_lidar:
                lidar = self.transform_velo_to_cam_dsec(lidar)
            elif self.cfg.frames_creation.use_pseudo_lidar:
                lidar = lidar.T
            else:
                lidar = self.transform_velo_to_cam_dsec(lidar)
        else:
            if self.cfg.frames_creation.use_pseudo_lidar:
                lidar = lidar.T
            else:
                lidar = self.transform_velo_to_cam_dsec(lidar)

        return self.project_lidar_points_all(lidar, img, save, crop)

    def prepare_scan_waymoc(self, filename, img, lidar, save=True, crop=True, visu=False):
        calib_path = os.path.join(self.cfg.paths.all_dataset_path, 'training', self.folder, 'calib', self.number + '.txt')
        self.P2_rect = self.load_calibration(calib_path)
        self.P2_rect = np.concatenate((self.P2_rect, np.array([[0, 0, 0, 1]])), axis=0)
        lidar = lidar.T
        self.calib = self.load_full_calib(calib_path)
        return self.project_lidar_points_all(lidar, img, save, crop)

    def transform_velo_to_cam(self, filename, lidar, filter_points=True):
        # Now we need homogenous coordinates and we do not care about the reflections
        lidar[:, 3] = 1
        # Transform to the camera coordinate
        lidar = lidar.transpose()
        # This should be rectified already
        T_velo_to_cam = self.kitti_data.calib.T_cam2_velo
        lidar = np.matmul(T_velo_to_cam, lidar)

        if filter_points:
            # Delete all points which are not in front of the camera
            mask = lidar[2, :] > 0.
            lidar = lidar[:, mask]

        self.velo_to_cam = T_velo_to_cam
        return lidar

    def transform_velo_to_cam_dsec(self, lidar):
        T_cam1_to_velo = self.cam_to_lidar['T_lidar_camRect1']
        T_velo_to_cam1 = np.linalg.inv(T_cam1_to_velo)

        lidar[:, 3] = 1
        lidar = np.matmul(T_velo_to_cam1, lidar.T)
        return lidar

    def project_lidar_points(self, lidar, img, save=True, crop=True):
        proj_lidar = np.matmul(self.P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)
        lidar[4:6, :] = np.rint(lidar[4:6, :])

        # Filter lidar data based on their projection to the camera: if they actually fit?
        if crop:
            mask_xmin = lidar[4, :] >= 0.
            lidar = lidar[:, mask_xmin]
            mask_xmax = lidar[4, :] < img.shape[2]  # img width
            lidar = lidar[:, mask_xmax]
            mask_ymin = lidar[5, :] >= 0.
            lidar = lidar[:, mask_ymin]
            mask_ymax = lidar[5, :] < img.shape[1]  # img height
            lidar = lidar[:, mask_ymax]

        if save:
            self.lidar = lidar
        else:
            return lidar

    def project_lidar_points_all(self, lidar, img, save=True, crop=True):
        proj_lidar = np.matmul(self.P2_rect, lidar)
        proj_lidar = proj_lidar[0:2, :] / proj_lidar[2, :]

        # Add projected data to the lidar data
        lidar = np.concatenate((lidar, proj_lidar), axis=0)
        lidar[4:6, :] = np.rint(lidar[4:6, :])

        # Filter lidar data based on their projection to the camera: if they actually fit?
        if crop:
            mask_xmin = lidar[4, :] >= 0.
            lidar = lidar[:, mask_xmin]
            mask_xmax = lidar[4, :] < img.shape[2]  # img width
            lidar = lidar[:, mask_xmax]
            mask_ymin = lidar[5, :] >= 0.
            lidar = lidar[:, mask_ymin]
            mask_ymax = lidar[5, :] < img.shape[1]  # img height
            lidar = lidar[:, mask_ymax]
        if save:
            self.lidar = lidar
        else:
            return lidar

    def prepare_img_dist(self, img):
        img_dist = -np.ones((img.shape[2], img.shape[1]))

        for i in range(self.lidar.shape[1]):
            img_dist[self.lidar[4, i].astype(int), self.lidar[5, i].astype(int)] = np.maximum(
                img_dist[self.lidar[4, i].astype(int), self.lidar[5, i].astype(int)],
                self.lidar[0, i] ** 2 + self.lidar[2, i] ** 2)

        # We need to add dilatation, because the lidar points are too sparse on camera
        footprint = np.ones((5, 5))
        img_dist = scipy.ndimage.grey_dilation(img_dist, footprint=footprint)
        self.img_dist = img_dist

    def load_current_segment(self):
        file_name = self.cfg.paths.waymo_path + self.random_indexes[self.segment_index]
        self.file_name = self.random_indexes[self.segment_index]
        dataset = tf.data.TFRecordDataset(file_name, compression_type='')
        if not self.cfg.general.supress_debug_prints:
            print("Segment: ", file_name)

        self.waymo_data = []
        self.waymo_frame = []
        self.waymo_lidar = []

        for i, data in enumerate(dataset):
            if i >10: break
            self.waymo_data.append(data)
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            self.waymo_frame.append(frame)

            if self.generate_raw_lidar:
                self.generate_raw_lidar_frame(frame, i)
            else:
                lidar_raw = np.load(
                    self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name + "/" + str(i) + '.npz')
                lidar_raw = [lidar_raw[key] for key in lidar_raw][0]

                self.waymo_lidar.append(lidar_raw)

        if not self.cfg.general.supress_debug_prints:
            print("Segment loaded")

    def generate_raw_lidar_frame(self, frame, i):
        (range_images, camera_projections, _,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images,
                                                                           camera_projections,
                                                                           range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        cp_points_all_concat = np.concatenate([points_all, cp_points_all[..., 0:3]], axis=-1)

        self.waymo_lidar.append(cp_points_all_concat)

        if not os.path.isdir(self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name):
            os.mkdir(self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name)
        np.savez_compressed(
            self.cfg.paths.merged_frames_path + "lidar_raw/" + self.file_name + "/" + str(i) + ".npz",
            np.float32(cp_points_all_concat))

    def load_lidar_templatesv2(self):
        pcd1, mesh1, mesh_p3d_1 = self.load_and_sample_fiat()
        #pcd1, _ = self.load_and_sample_cube()
        pcd2, mesh2, mesh_p3d_2 = self.load_and_sample_passat()
        pcd3, mesh3, mesh_p3d_3 = self.load_and_sample_suv()
        pcd4, mesh4, mesh_p3d_4 = self.load_and_sample_mpv()
        #pcd1_scale, _ = self.load_and_sample_fiat_scale()
        #pcd2_scale, _ = self.load_and_sample_passat_scale()
        #pcd1_scale, _ = self.load_and_sample_cube()
        #pcd2_scale, _ = self.load_and_sample_cube_scale()

        #coord_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.)  # Only for visu purpose
        #open3d.visualization.draw_geometries([pcd4, coord_frame])

        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)
        pcd3 = np.asarray(pcd3.points)
        pcd4 = np.asarray(pcd4.points)
        #pcd1_scale = np.asarray(pcd1_scale.points)
        #pcd2_scale = np.asarray(pcd2_scale.points)

        if self.args.dataset == 'waymo':
            pcd1[:, 2] += self.cfg.templates.offset_fiat
            pcd2[:, 2] += self.cfg.templates.offset_passat
            pcd3[:, 2] += self.cfg.templates.offset_suv
            pcd4[:, 2] += self.cfg.templates.offset_mpv
        else:
            pcd1[:, 1] -= self.cfg.templates.offset_fiat
            pcd2[:, 1] -= self.cfg.templates.offset_passat
            pcd3[:, 1] -= self.cfg.templates.offset_suv
            pcd4[:, 1] -= self.cfg.templates.offset_mpv

        self.lidar_car_template_non_filt = [pcd1, pcd2, pcd3, pcd4]
        self.lidar_car_template_scale = [pcd1, pcd2, pcd3, pcd4]
        self.mesh_templates = [mesh1, mesh2, mesh3, mesh4]
        self.mesh_templates_p3d = [mesh_p3d_1, mesh_p3d_2, mesh_p3d_3, mesh_p3d_4]

    def load_and_sample_fiat(self):
        mesh = open3d.io.read_triangle_mesh("../data/fiat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi/2, np.pi,0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' and torch.cuda.is_available() else "cpu")
        mesh_p3d = load_objs_as_meshes(["../data/fiat_deformed2.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_fiat_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/fiat_scale.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi / 2, 0, -np.pi / 2))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_passat(self):
        mesh = open3d.io.read_triangle_mesh("../data/passat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' else "cpu")
        mesh_p3d = load_objs_as_meshes(["../data/passat_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_passat_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/passat_scale.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_cube(self):
        mesh = open3d.io.read_triangle_mesh("../data/cube.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((0, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_cube_scale(self):
        mesh = open3d.io.read_triangle_mesh("../data/cube_top.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((0, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((-np.pi / 2, -np.pi / 2, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        return pcd, mesh

    def load_and_sample_suv(self):
        mesh = open3d.io.read_triangle_mesh("../data/suv.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi/2, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, 0))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' else "cpu")
        mesh_p3d = load_objs_as_meshes(["../data/suv_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_mpv(self):
        mesh = open3d.io.read_triangle_mesh("../data/minivan.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        if self.args.dataset == 'waymo':
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
            mesh.transform(T)
        else:
            T = np.eye(4)
            T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi/2))  # Y rotation -||-
            mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        if self.args.dataset == 'waymo':
            scale_ax0 = self.cfg.templates.template_length / ax0_size
            scale_ax1 = self.cfg.templates.template_width / ax1_size
            scale_ax2 = self.cfg.templates.template_height / ax2_size
        else:
            scale_ax0 = self.cfg.templates.template_width / ax0_size
            scale_ax1 = self.cfg.templates.template_height / ax1_size
            scale_ax2 = self.cfg.templates.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' else "cpu")
        mesh_p3d = load_objs_as_meshes(["../data/mpv_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def prepare_pic(self):
        self.pic = self.pics[self.pic_index]

        temp = self.pic.split("/")
        self.file_name = temp[-1].split(".")[0]

        self.img_orig = cv2.imread(self.pic)
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

        if not self.cfg.general.supress_debug_prints:
            print(self.mapping_data[int(self.random_indexes[int(self.pic_index)]) - 1])

    def prepare_pic_all(self, idx):
        pic = self.index_of_all_imgs[idx]
        img_path = self.cfg.paths.all_dataset_path + pic[0] + "/image_00/data_rect/" + pic[1] + '.png'
        self.pic = img_path

        self.file_name = pic[0] + '_' + pic[1]
        self.folder = pic[0]
        self.number = pic[1]

        self.img_orig = cv2.imread(self.pic)
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

    def prepare_pic_waymoc(self, idx):
        pic = self.index_of_all_imgs[idx]
        img_path = self.cfg.paths.all_dataset_path + 'training/' + pic[0] + "/image_2/" + pic[1] + '.png'
        self.pic = img_path

        self.file_name = pic[0] + '_' + pic[1]
        self.folder = pic[0]
        self.number = pic[1]

        self.img_orig = cv2.imread(self.pic)
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

    def prepare_pic_dsec(self, idx):
        pic = self.index_of_all_imgs[idx]
        # pic is (folder, file_name, path_to_imgs)
        folder = pic[0]
        file_name = pic[1]
        path_to_imgs = pic[2]
        
        img_path = os.path.join(path_to_imgs, file_name + '.png')
        self.pic = img_path

        self.file_name = folder + '_' + file_name
        self.folder = folder
        self.number = file_name

        self.img_orig = cv2.imread(self.pic)
        if self.img_orig is None:
             raise ValueError(f"Could not read image {self.pic}")
        img = np.array(self.img_orig, dtype=np.uint8)
        self.img = np.moveaxis(img, -1, 0)  # the model expects the image to be in channel first format

    def prepare_pic_waymo(self, data):
        # First get the name of the file. Sometimes for debug we want to choose it randomly
        if self.cfg.visualization.show_3D_scan:
            self.bboxes = []

        self.img = []
        images_sorted = sorted(data.images, key=lambda i: i.name)
        for index, image in enumerate(images_sorted):
            decoded_image = tf.image.decode_jpeg(image.image).numpy()

            # Open the image, convert
            img = np.array(decoded_image, dtype=np.uint8)
            self.img.append(np.moveaxis(img, -1, 0))  # the model expects the image to be in channel first format

    def compute_mean(self, lidar):
        x_mean = np.median(lidar[:, 0])
        y_mean = np.median(lidar[:, 1])
        z_mean = np.median(lidar[:, 2])

        return x_mean, y_mean, z_mean

    def icp_point_to_plane_open3d(self, source_points, target_points, max_iterations=50, tolerance=1e-6):
        # Convert numpy arrays to Open3D point clouds
        source_cloud = open3d.geometry.PointCloud()
        source_cloud.points = open3d.utility.Vector3dVector(source_points)
        source_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=.5, max_nn=30))
        #source_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        target_cloud = open3d.geometry.PointCloud()
        target_cloud.points = open3d.utility.Vector3dVector(target_points)
        target_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=.5, max_nn=30))
        #target_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))

        # Perform Point-to-Plane ICP registration
        icp_result = open3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, 0.1, np.eye(4),
            open3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Get the transformation matrix from the ICP result
        transformation = icp_result.transformation

        return transformation

    def icp_point_to_point_open3d(self, source_points, target_points, max_iterations=50, tolerance=1e-6):
        # Convert numpy arrays to Open3D point clouds
        source_cloud = open3d.geometry.PointCloud()
        source_cloud.points = open3d.utility.Vector3dVector(source_points)
        #source_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        #source_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2., max_nn=30))
        target_cloud = open3d.geometry.PointCloud()
        target_cloud.points = open3d.utility.Vector3dVector(target_points)
        #target_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1., max_nn=30))
        #target_cloud.estimate_covariances(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2., max_nn=30))

        # Perform Point-to-Plane ICP registration
        icp_result = open3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, 0.1, np.eye(4),
            open3d.pipelines.registration.TransformationEstimationPointToPoint())

        # Get the transformation matrix from the ICP result
        transformation = icp_result.transformation

        return transformation

    def filter_distant_cars_pseudo_lidar(self, cars, waymo=False):
        new_cars = []
        for car in cars:
            if car.lidar is not None:
                location = np.median(car.lidar, axis=0)
                distance = np.sqrt(location[0] ** 2 + location[1] ** 2 + location[2] ** 2)

                if distance < self.cfg.frames_creation.max_distance_pseudo_lidar:
                    new_cars.append(car)

        return new_cars

    def extract_scale_lidar(self, cars, transformations, waymo=False):
        for car in cars:
            car.moving_scale_lidar = []
            distances = []
            for z in range(len(car.locations)):
                if car.locations[z] is not None and car.mask[z] is not None:
                    if waymo:
                        frame_idx = car.info[z][1]
                        if self.pic_index == frame_idx:
                            car.lidar = car.lidar[z]
                            break
                    else:
                        frame_idx = car.locations[z][3]
                        T_cur_to_ref = transformations[int(frame_idx) + self.cfg.frames_creation.nscans_before, :, :]
                        T_ref_to_cur = np.linalg.inv(T_cur_to_ref)

                        cur_lidar = copy.deepcopy(car.lidar[z])
                        cur_lidar = np.matmul(T_ref_to_cur[0:3, 0:3], cur_lidar.T).T
                        cur_lidar += T_ref_to_cur[0:3, 3]

                        cur_loc = np.median(cur_lidar, axis=0)
                        distance = np.sqrt(cur_loc[0] ** 2 + cur_loc[1] ** 2 + cur_loc[2] ** 2)

                        distances.append(distance)
                else:
                    distances.append(np.inf)

            idxs = np.argsort(distances)
            taken = 0
            iter_idx = 0
            while taken < self.cfg.frames_creation.k_to_scale_estimation_save and iter_idx < len(idxs):
                cur_idx = idxs[iter_idx]
                cur_mask = car.mask[cur_idx]
                if not np.any(cur_mask[0:10, :]) and not np.any(cur_mask[-10:, :]) and not np.any(
                        cur_mask[:, 0:10]) and not np.any(cur_mask[:, -10:]):
                    car.moving_scale_lidar.append(car.lidar[cur_idx])
                    taken += 1
                iter_idx += 1

        return cars

    def create_depth_map(self, scan):
        x_coords = scan[0]
        y_coords = scan[1]
        z_coords = scan[2]

        depth_values = np.sqrt(x_coords ** 2 + y_coords ** 2 + z_coords ** 2)

        u_coords = scan[4].astype(int)
        v_coords = scan[5].astype(int)

        max_u = np.max(u_coords)
        max_v = np.max(v_coords)

        depth_image = np.full((max_v + 1, max_u + 1), np.inf)

        flat_indices = np.ravel_multi_index((v_coords, u_coords), depth_image.shape)

        np.minimum.at(depth_image.ravel(), flat_indices, depth_values)

        depth_image[depth_image == np.inf] = 0

        kernel = np.array([
            [0.125, 0.125, 0.125],
            [0.125, -1., 0.125],
            [0.125, 0.125, 0.125]
        ])

        convolved_depth = convolve(depth_image, kernel, mode='nearest')

        return convolved_depth.transpose()

    def hdbscan_clustering(self, pcloud):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
        cluster_labels = clusterer.fit_predict(pcloud)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        mask = unique_labels != -1
        unique_labels = unique_labels[mask]
        counts = counts[mask]

        if len(unique_labels) > 0:
            # Identify the label of the biggest cluster
            biggest_cluster_label = unique_labels[np.argmax(counts)]

            # Extract data points belonging to the biggest cluster
            biggest_cluster = pcloud[cluster_labels == biggest_cluster_label]

            # Visualize the biggest cluster together with input
            #pcd = o3d.geometry.PointCloud()
            #cd.points = o3d.utility.Vector3dVector(pcloud)
            #pcd_input = o3d.geometry.PointCloud()
            #pcd_input.points = o3d.utility.Vector3dVector(biggest_cluster)
            #pcd.paint_uniform_color([0.8, 0.8, 0.8])
            #pcd_input.paint_uniform_color([1.0, 0, 0])
            #o3d.visualization.draw_geometries([pcd, pcd_input], window_name='Inliers and Outliers', width=800, height=600)

            return biggest_cluster
        else:
            return pcloud

    def isolation_forest_clustering(self, pcloud):
        # Initialize the Isolation Forest
        iso_forest = IsolationForest(contamination=0.01, random_state=42)

        # Fit the model
        iso_forest.fit(pcloud)

        # Predict anomalies (-1 for outliers, 1 for inliers)
        predictions = iso_forest.predict(pcloud)

        # Separate inliers and outliers
        inliers = pcloud[predictions == 1]
        outliers = pcloud[predictions == -1]

        return inliers

    def ensamble_clustering(self, pcloud):
        #overall_start = time.time_ns()
        if pcloud.shape[0] <= 3:
            return pcloud
        # Initialize models
        hbos = HBOS()
        #print("Models initialized.")

        # Scale data
        #start = time.time_ns()
        scaler = StandardScaler()
        pcloud_scaled = scaler.fit_transform(pcloud)
        #print("Data scaling: ", (time.time_ns() - start) / 1e6, "ms")

        # Z-Score method
        #start = time.time_ns()
        z_scores = np.abs((pcloud_scaled - np.mean(pcloud_scaled, axis=0)) / np.std(pcloud_scaled, axis=0))
        threshold = 3  # Number of standard deviations from the mean
        z_score_pred = np.where(np.max(z_scores, axis=1) > threshold, 1, 0)  # 1 for outliers, 0 for inliers
        #print("Z-Score: ", (time.time_ns() - start) / 1e6, "ms")

        # HBOS
        #start = time.time_ns()
        hbos.fit(pcloud_scaled)
        hbos_pred = hbos.predict(pcloud_scaled)  # 0 for inliers, 1 for outliers
        #print("HBOS: ", (time.time_ns() - start) / 1e6, "ms")

        # Statistical Outlier Removal (SOR)
        #start = time.time_ns()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcloud)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=1.0)
        sor_pred = np.ones(len(pcloud), dtype=int)
        sor_pred[ind] = 0
        #print("SOR: ", (time.time_ns() - start) / 1e6, "ms")

        # HDBSCAN
        #start = time.time_ns()
        hdbscan_clusterer = hdbscan.HDBSCAN()
        hdbscan_labels = hdbscan_clusterer.fit_predict(pcloud_scaled)
        hdbscan_pred = np.where(hdbscan_labels == -1, 1, 0)
        #print("HDBSCAN: ", (time.time_ns() - start) / 1e6, "ms")

        # DBSCAN
        #start = time.time_ns()
        dbscan_clusterer = DBSCAN(eps=0.2, min_samples=10)  # Adjust parameters as needed
        dbscan_labels = dbscan_clusterer.fit_predict(pcloud_scaled)
        dbscan_pred = np.where(dbscan_labels == -1, 1, 0)
        #print("DBSCAN: ", (time.time_ns() - start) / 1e6, "ms")

        # Combine predictions
        predictions = np.stack((z_score_pred, hbos_pred, sor_pred, hdbscan_pred, dbscan_pred), axis=1)

        # Apply majority voting (threshold of 3 for 5 methods)
        final_pred = (np.sum(predictions, axis=1) >= self.cfg.frames_creation.ensamble_threshold).astype(int)

        # Extract inliers
        inliers = pcloud[final_pred == 0]
        #total_time = (time.time_ns() - overall_start) / 1e6
        #print("Total time taken: ", total_time, "ms")
        return inliers

    def load_full_calib(self, path):
        out_dict = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Split at the first colon to separate the key from the data
                key, values_str = line.split(':', 1)
                key = key.strip()  # e.g., "P0", "P1", etc.

                # Split the values into a list of float strings
                values_list = values_str.strip().split()

                # Convert each value to float and store as a NumPy array
                values_arr = np.array([float(x) for x in values_list], dtype=np.float64)

                # Store in dictionary
                out_dict[key] = values_arr

        return out_dict
    
    def calculate_compensation(self, pedestrians, transformations):
        for pedestrian in pedestrians:
            pedestrian.yaw_angles = []
            for location in pedestrian.locations:
                cur_loc = location[0:3]
                cur_idx = location[3]

                ref_yaw = np.arctan2(cur_loc[2], cur_loc[0])

                T_cur_to_ref = transformations[int(cur_idx) + self.cfg.frames_creation.nscans_before, :, :]

                T_ref_to_cur = np.linalg.inv(T_cur_to_ref)
                cur_loc = np.matmul(T_ref_to_cur[0:3, 0:3], cur_loc.T).T
                cur_loc += T_ref_to_cur[0:3, 3]

                yaw_angle = np.arctan2(cur_loc[2], cur_loc[0])

                compensation_yaw_angle = ref_yaw - yaw_angle
                pedestrian.yaw_angles.append(compensation_yaw_angle)
        
        return pedestrians

    def perform_compensation(self, pedestrians):
        for pedestrian in pedestrians:
            if pedestrian.yaw_angles == None:
                continue
            for i in range(len(pedestrian.full_pose)):
                global_pose_axis_angle = pedestrian.full_pose[i][0:3]
                compensation_yaw_angle = pedestrian.yaw_angles[i]

                # Convert original global pose from axis-angle to Rotation object
                original_rotation = R.from_rotvec(global_pose_axis_angle)

                # Create rotation object for the compensation yaw angle around the Y-axis
                # Assuming Y is the up-axis for yaw in the coordinate system
                compensation_rotation = R.from_euler('y', compensation_yaw_angle)

                # Combine the rotations: apply compensation to the original rotation
                # R_comp * R_orig transforms points from local -> original -> compensated frame
                combined_rotation = compensation_rotation * original_rotation

                # Convert the combined rotation back to axis-angle format
                compensated_pose_axis_angle = combined_rotation.as_rotvec()
                pedestrian.full_pose[i][0:3] = compensated_pose_axis_angle

        return pedestrians
    
    def choose_closest_pedestrian(self, pedestrians, transformations):
        for pedestrian in pedestrians:
            dists = []
            for i in range(len(pedestrian.full_pose)):
                cur_loc = pedestrian.locations[i][0:3]
                cur_idx = pedestrian.locations[i][3]
                T_cur_to_ref = transformations[int(cur_idx) + self.cfg.frames_creation.nscans_before, :, :]
                T_ref_to_cur = np.linalg.inv(T_cur_to_ref)

                cur_loc = np.matmul(T_ref_to_cur[0:3, 0:3], cur_loc.T).T
                cur_loc += T_ref_to_cur[0:3, 3]

                dist = np.sqrt(cur_loc[0] ** 2 + cur_loc[1] ** 2 + cur_loc[2] ** 2)
                dists.append(dist)
            
            idxs = np.argsort(dists)

            pedestrian.sorted_idxs = idxs

            idxs = idxs[:self.cfg.pedestrian.body_betas_number_of_frames]
            pedestrian.all_lidars = []
            for idx in idxs:
                pedestrian.all_lidars.append(pedestrian.lidar[idx])
        
        return pedestrians

    def kalman_filter_denoise_3d_correlated_noise(self,
        noisy_positions,      # Shape (N, 3): N timesteps, [x, y, z]
        initial_velocity,     # Shape (3,): [vx0, vy0, vz0]
        dt,                   # Scalar: time step between measurements
        sigma_w_noise,        # Scalar or array (3,): std dev of white noise driving the Markov noise
        a_noise,              # Scalar or array (3,): correlation coeff for Markov noise (0 < a < 1)
        sigma_a_motion,       # Scalar or array (3,): std dev of unmodeled acceleration (process noise for car motion)
        R_val=1e-5            # Scalar: diagonal value for measurement noise covariance (small, for stability)
    ):
        """
        Denoises 3D car positions corrupted by first-order Gaussian Markov noise
        using a Kalman Filter with state augmentation and a constant velocity motion model.

        State vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T (9x1)
            p: position
            v: velocity
            n: correlated noise component

        Args:
            noisy_positions (np.ndarray): Array of noisy (x,y,z) measurements, shape (N, 3).
            initial_velocity (np.ndarray): Initial velocity [vx, vy, vz], shape (3,).
            dt (float): Time step between measurements.
            sigma_w_noise (float or np.ndarray): Standard deviation of the white noise
                                                driving the Markov noise process.
                                                If scalar, assumed same for x, y, z.
            a_noise (float or np.ndarray): Correlation coefficient for the Markov noise (0 < a < 1).
                                        If scalar, assumed same for x, y, z.
            sigma_a_motion (float or np.ndarray): Standard deviation of the process noise
                                                (unmodeled acceleration) for the car's motion.
                                                If scalar, assumed same for x, y, z.
            R_val (float): Small diagonal value for the measurement noise covariance matrix R.
                        This accounts for any tiny, unmodeled white noise in measurements
                        beyond the correlated noise.

        Returns:
            np.ndarray: Denoised positions, shape (N, 3).
            np.ndarray: Estimated velocities, shape (N, 3).
            np.ndarray: Estimated noise components, shape (N, 3).
        """
        N = noisy_positions.shape[0]
        if N == 0:
            return np.array([]), np.array([]), np.array([])

        # Ensure noise parameters are arrays
        if np.isscalar(sigma_w_noise):
            sigma_w_noise = np.full(3, sigma_w_noise)
        if np.isscalar(a_noise):
            a_noise = np.full(3, a_noise)
        if np.isscalar(sigma_a_motion):
            sigma_a_motion = np.full(3, sigma_a_motion)

        # --- State Vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T ---
        # Initial state estimate (x_hat)
        x_hat = np.zeros(9)
        x_hat[0:3] = noisy_positions[0]  # Initial position from first measurement
        x_hat[3:6] = initial_velocity    # Given initial velocity
        x_hat[6:9] = 0.0                 # Initial noise components assumed to be zero

        # Initial state covariance (P)
        # Reflects uncertainty in the initial state
        P = np.eye(9) * 1.0  # Start with moderate uncertainty
        # For position, uncertainty is related to noise variance
        # Steady-state variance of Markov noise: sigma_w^2 / (1 - a^2)
        for i in range(3):
            if 1 - a_noise[i]**2 > 1e-6: # Avoid division by zero if a is close to 1
                P[i, i] = (sigma_w_noise[i]**2) / (1 - a_noise[i]**2) if (1 - a_noise[i]**2) > 0 else sigma_w_noise[i]**2
                P[i+6, i+6] = (sigma_w_noise[i]**2) / (1 - a_noise[i]**2) if (1 - a_noise[i]**2) > 0 else sigma_w_noise[i]**2
            else: # if a is very close to 1, use a large variance for initial noise estimate
                P[i, i] = sigma_w_noise[i]**2 * 100 # Heuristic large variance
                P[i+6, i+6] = sigma_w_noise[i]**2 * 100
        P[3,3] = (0.1 * abs(initial_velocity[0]) if initial_velocity[0] != 0 else 0.1)**2 + 0.01 # Velocity uncertainty
        P[4,4] = (0.1 * abs(initial_velocity[1]) if initial_velocity[1] != 0 else 0.1)**2 + 0.01
        P[5,5] = (0.1 * abs(initial_velocity[2]) if initial_velocity[2] != 0 else 0.1)**2 + 0.01


        # --- State Transition Matrix (F) ---
        # Describes how the state evolves: x_k = F * x_{k-1}
        F = np.eye(9)
        # Position update: p_k = p_{k-1} + v_{k-1}*dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        # Noise update: n_k = a * n_{k-1}
        F[6, 6] = a_noise[0]
        F[7, 7] = a_noise[1]
        F[8, 8] = a_noise[2]

        # --- Process Noise Covariance (Q) ---
        # Describes uncertainty introduced by the model (e.g., unmodeled accelerations)
        # Q is block diagonal: Q_motion for (p,v) and Q_noise_driver for (n)
        Q = np.zeros((9, 9))
        # For position-velocity part (constant velocity model with discrete white noise acceleration)
        # Assumes acceleration is process noise on velocity
        # sigma_a_motion^2 is variance of acceleration
        for i in range(3):
            sigma_a_sq = sigma_a_motion[i]**2
            Q[i, i]         = (dt**4 / 4) * sigma_a_sq  # process noise for position
            Q[i, i+3]       = (dt**3 / 2) * sigma_a_sq
            Q[i+3, i]       = (dt**3 / 2) * sigma_a_sq
            Q[i+3, i+3]     = (dt**2) * sigma_a_sq      # process noise for velocity
            Q[i+6, i+6]     = sigma_w_noise[i]**2       # driving noise for the Markov process n

        # --- Measurement Matrix (H) ---
        # Relates state to measurement: z_k = H * x_k
        # We measure position (p) + noise (n)
        H = np.zeros((3, 9))
        H[0, 0] = 1  # z_x = p_x + n_x
        H[0, 6] = 1
        H[1, 1] = 1  # z_y = p_y + n_y
        H[1, 7] = 1
        H[2, 2] = 1  # z_z = p_z + n_z
        H[2, 8] = 1

        # --- Measurement Noise Covariance (R) ---
        # Uncertainty of the measurement sensor (after accounting for correlated noise in state)
        # This is typically small if the dominant noise is modeled in the state.
        R = np.eye(3) * R_val

        # Storage for results
        denoised_positions_history = np.zeros((N, 3))
        estimated_velocities_history = np.zeros((N, 3))
        estimated_noise_history = np.zeros((N, 3))

        # Kalman Filter Loop
        for k in range(N):
            # --- Prediction Step ---
            x_hat_pred = F @ x_hat
            P_pred = F @ P @ F.T + Q

            # --- Update Step ---
            z_k = noisy_positions[k, :]  # Current measurement
            y_k = z_k - H @ x_hat_pred   # Innovation (measurement residual)
            S_k = H @ P_pred @ H.T + R   # Innovation covariance
            
            try:
                K_k = P_pred @ H.T @ np.linalg.inv(S_k) # Kalman Gain
            except np.linalg.LinAlgError:
                # If S_k is singular, R might be too small or model issues
                # Use pseudo-inverse or add a small identity matrix to S_k
                print(f"Warning: S_k singular at step {k}. Using pseudo-inverse.")
                K_k = P_pred @ H.T @ np.linalg.pinv(S_k)


            x_hat = x_hat_pred + K_k @ y_k
            P = (np.eye(9) - K_k @ H) @ P_pred
            # Ensure P remains symmetric (important for numerical stability)
            P = (P + P.T) / 2

            # Store results
            denoised_positions_history[k, :] = x_hat[0:3]
            estimated_velocities_history[k, :] = x_hat[3:6]
            estimated_noise_history[k, :] = x_hat[6:9]

        return denoised_positions_history, estimated_velocities_history, estimated_noise_history

    def kalman_filter_denoise_3d_correlated_noise_with_gt_velocity(self,
        noisy_positions,      # Shape (N, 3): N timesteps, [x, y, z]
        GT_velocities,        # Shape (N, 3): Ground truth velocities [vx, vy, vz]
        dt,                   # Scalar: time step between measurements
        sigma_w_noise,        # Scalar or array (3,): std dev of white noise driving the Markov noise
        a_noise,              # Scalar or array (3,): correlation coeff for Markov noise (0 < a < 1)
        # sigma_a_motion is now effectively ignored as velocities are known
        sigma_a_motion_ignored, # Scalar or array (3,): std dev of unmodeled acceleration (process noise for car motion)
        R_val=1e-5,           # Scalar: diagonal value for measurement noise covariance (small, for stability)
        tiny_P_vel_diag=1e-12 # Scalar: small diagonal value for velocity covariance when GT is used
    ):
        """
        Denoises 3D car positions corrupted by first-order Gaussian Markov noise
        using a Kalman Filter with state augmentation and a constant velocity motion model,
        where true velocities are provided as Ground Truth (GT).

        State vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T (9x1)
            p: position
            v: velocity (taken from GT_velocities)
            n: correlated noise component

        Args:
            noisy_positions (np.ndarray): Array of noisy (x,y,z) measurements, shape (N, 3).
            GT_velocities (np.ndarray): Ground Truth velocities [vx, vy, vz] for each timestep, shape (N, 3).
            dt (float): Time step between measurements.
            sigma_w_noise (float or np.ndarray): Standard deviation of the white noise
                                                driving the Markov noise process.
                                                If scalar, assumed same for x, y, z.
            a_noise (float or np.ndarray): Correlation coefficient for the Markov noise (0 < a < 1).
                                        If scalar, assumed same for x, y, z.
            sigma_a_motion_ignored (float or np.ndarray): Standard deviation of the process noise
                                                (unmodeled acceleration) for the car's motion.
                                                THIS PARAMETER IS IGNORED because GT velocities are used.
            R_val (float): Small diagonal value for the measurement noise covariance matrix R.
                        This accounts for any tiny, unmodeled white noise in measurements
                        beyond the correlated noise.
            tiny_P_vel_diag (float): A very small value for the diagonal elements of the covariance
                                     matrix P corresponding to the velocity states, reflecting
                                     high certainty due to GT.

        Returns:
            np.ndarray: Denoised positions, shape (N, 3).
            np.ndarray: Estimated velocities (will be identical to GT_velocities), shape (N, 3).
            np.ndarray: Estimated noise components, shape (N, 3).
        """
        N = noisy_positions.shape[0]
        if N == 0:
            return np.array([]), np.array([]), np.array([])
        if GT_velocities.shape[0] != N:
            raise ValueError("GT_velocities must have the same number of timesteps as noisy_positions.")

        # Ensure noise parameters are arrays
        if np.isscalar(sigma_w_noise):
            sigma_w_noise = np.full(3, sigma_w_noise)
        if np.isscalar(a_noise):
            a_noise = np.full(3, a_noise)
        # sigma_a_motion is ignored, but we can create the array for consistency if needed by other parts (not here)
        # if np.isscalar(sigma_a_motion_ignored):
        #     sigma_a_motion_ignored = np.full(3, sigma_a_motion_ignored)

        # --- State Vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T ---
        vel_idx = slice(3, 6) # Indices for velocity components in the state vector

        # Initial state estimate (x_hat)
        x_hat = np.zeros(9)
        x_hat[0:3] = noisy_positions[0]  # Initial position from first measurement
        x_hat[vel_idx] = GT_velocities[0] # Use GT initial velocity
        x_hat[6:9] = 0.0                 # Initial noise components assumed to be zero

        # Initial state covariance (P)
        P = np.eye(9) * 1.0  # Start with moderate uncertainty for non-velocity states
        # For position, uncertainty is related to noise variance
        # Steady-state variance of Markov noise: sigma_w^2 / (1 - a^2)
        for i in range(3):
            var_noise_ss = (sigma_w_noise[i]**2) / (1 - a_noise[i]**2) if (1 - a_noise[i]**2) > 1e-6 else sigma_w_noise[i]**2 * 100
            P[i, i] = var_noise_ss
            P[i+6, i+6] = var_noise_ss
        
        # For velocity, uncertainty is very small because it's GT
        P[vel_idx, vel_idx] = tiny_P_vel_diag * np.eye(3)
        # Zero out cross-covariances for initial velocity
        P[vel_idx, :vel_idx.start] = 0
        P[:vel_idx.start, vel_idx] = 0
        P[vel_idx, vel_idx.stop:] = 0
        P[vel_idx.stop:, vel_idx] = 0


        # --- State Transition Matrix (F) ---
        F = np.eye(9)
        F[0, 3] = dt # px_k = px_{k-1} + vx_{k-1}*dt
        F[1, 4] = dt # py_k = py_{k-1} + vy_{k-1}*dt
        F[2, 5] = dt # pz_k = pz_{k-1} + vz_{k-1}*dt
        F[6, 6] = a_noise[0] # nx_k = a_x * nx_{k-1}
        F[7, 7] = a_noise[1] # ny_k = a_y * ny_{k-1}
        F[8, 8] = a_noise[2] # nz_k = a_z * nz_{k-1}
        # Velocity transition v_k = v_{k-1} is implicit in eye(9) for F[3:6, 3:6]
        # but will be overridden by GT.

        # --- Process Noise Covariance (Q) ---
        # Since velocities are GT, sigma_a_motion (unmodeled acceleration) is effectively zero.
        # Q only contains noise for the Markov process n.
        Q = np.zeros((9, 9))
        for i in range(3):
            # Motion model parts of Q are zero because velocity is known
            # Q[i, i]         = (dt**4 / 4) * 0 (was sigma_a_sq)
            # Q[i, i+3]       = (dt**3 / 2) * 0
            # Q[i+3, i]       = (dt**3 / 2) * 0
            # Q[i+3, i+3]     = (dt**2) * 0
            Q[i+6, i+6]     = sigma_w_noise[i]**2 # driving noise for the Markov process n

        # --- Measurement Matrix (H) ---
        # Relates state to measurement: z_k = H * x_k
        # We measure position (p) + noise (n)
        H = np.zeros((3, 9))
        H[0, 0] = 1; H[0, 6] = 1  # z_x = p_x + n_x
        H[1, 1] = 1; H[1, 7] = 1  # z_y = p_y + n_y
        H[2, 2] = 1; H[2, 8] = 1  # z_z = p_z + n_z

        # --- Measurement Noise Covariance (R) ---
        R = np.eye(3) * R_val

        # Storage for results
        denoised_positions_history = np.zeros((N, 3))
        estimated_velocities_history = np.zeros((N, 3)) # Will store GT velocities
        estimated_noise_history = np.zeros((N, 3))

        # Store initial state
        denoised_positions_history[0, :] = x_hat[0:3]
        estimated_velocities_history[0, :] = x_hat[vel_idx]
        estimated_noise_history[0, :] = x_hat[6:9]
        
        # Kalman Filter Loop (starts from k=1 if first point is handled by initialization,
        # or k=0 if we re-process the first point. Let's re-process for consistency)
        for k in range(N):
            # --- Prediction Step (if k > 0, else use initial x_hat, P) ---
            if k == 0: # Use initial x_hat and P for the first measurement
                x_hat_pred = x_hat 
                P_pred = P
            else: # Standard prediction for k > 0
                # x_hat from previous step (k-1) already has GT_velocities[k-1]
                x_hat_pred = F @ x_hat 
                P_pred = F @ P @ F.T + Q
            
            # --- Incorporate GT velocity for current step k into prediction ---
            x_hat_pred[vel_idx] = GT_velocities[k]
            
            # Adjust P_pred to reflect known velocity (high certainty)
            P_pred[vel_idx, vel_idx] = tiny_P_vel_diag * np.eye(3)
            P_pred[vel_idx, :vel_idx.start] = 0
            P_pred[:vel_idx.start, vel_idx] = 0
            P_pred[vel_idx, vel_idx.stop:] = 0
            P_pred[vel_idx.stop:, vel_idx] = 0
            P_pred = (P_pred + P_pred.T) / 2 # Ensure symmetry

            # --- Update Step ---
            z_k = noisy_positions[k, :]    # Current measurement
            y_k = z_k - H @ x_hat_pred     # Innovation (measurement residual)
            S_k = H @ P_pred @ H.T + R     # Innovation covariance
            
            try:
                K_k = P_pred @ H.T @ np.linalg.inv(S_k) # Kalman Gain
            except np.linalg.LinAlgError:
                print(f"Warning: S_k singular at step {k}. Using pseudo-inverse.")
                K_k = P_pred @ H.T @ np.linalg.pinv(S_k)

            x_hat_updated = x_hat_pred + K_k @ y_k
            P_updated = (np.eye(9) - K_k @ H) @ P_pred
            
            # Final state and covariance for step k
            x_hat = x_hat_updated
            P = P_updated

            # Enforce GT velocity and its certainty on the final state x_hat and covariance P
            x_hat[vel_idx] = GT_velocities[k] # Re-affirm GT velocity
            
            P[vel_idx, vel_idx] = tiny_P_vel_diag * np.eye(3)
            P[vel_idx, :vel_idx.start] = 0
            P[:vel_idx.start, vel_idx] = 0
            P[vel_idx, vel_idx.stop:] = 0
            P[vel_idx.stop:, vel_idx] = 0
            P = (P + P.T) / 2 # Ensure P remains symmetric

            # Store results
            denoised_positions_history[k, :] = x_hat[0:3]
            estimated_velocities_history[k, :] = x_hat[vel_idx] # This will be GT_velocities[k]
            estimated_noise_history[k, :] = x_hat[6:9]

        return denoised_positions_history, estimated_velocities_history, estimated_noise_history

    def kalman_filter_denoise_3d_partial_gt_velocity(self,
            noisy_positions,      # Shape (N, 3): N timesteps, [x, y, z]
            GT_velocities,        # Shape (N, 3): Ground truth velocities [vx_gt, vy_for_init_or_compare, vz_gt]
            dt,                   # Scalar: time step between measurements
            sigma_w_noise,        # Scalar or array (3,): std dev of white noise driving the Markov noise
            a_noise,              # Scalar or array (3,): correlation coeff for Markov noise (0 < a < 1)
            sigma_a_motion,       # Scalar or array (3,): std dev of unmodeled acceleration.
                                  # [sigma_ax (ignored), sigma_ay (used for vy), sigma_az (ignored)]
            R_val=1e-5,           # Scalar: diagonal value for measurement noise covariance (small, for stability)
            tiny_P_vel_diag=1e-12 # Scalar: small diagonal value for GT velocity covariance
        ):
            """
            Denoises 3D car positions corrupted by first-order Gaussian Markov noise
            using a Kalman Filter with state augmentation and a constant velocity motion model.
            - X and Z velocities are taken from GT_velocities.
            - Y velocity is estimated by the Kalman Filter.

            State vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T (9x1)
                p: position
                v: velocity (vx, vz from GT; vy estimated)
                n: correlated noise component

            Args:
                noisy_positions (np.ndarray): Array of noisy (x,y,z) measurements, shape (N, 3).
                GT_velocities (np.ndarray): Ground Truth velocities [vx, vy, vz] for each timestep, shape (N, 3).
                                            GT_velocities[:,0] (vx) and GT_velocities[:,2] (vz) are used as true GT.
                                            GT_velocities[:,1] (vy) can be used for initializing vy_hat[0] or for comparison.
                dt (float): Time step between measurements.
                sigma_w_noise (float or np.ndarray): Standard deviation of the white noise
                                                    driving the Markov noise process.
                a_noise (float or np.ndarray): Correlation coefficient for the Markov noise (0 < a < 1).
                sigma_a_motion (float or np.ndarray): Standard deviation of the process noise
                                                    (unmodeled acceleration).
                                                    The Y-component is used for estimating vy.
                                                    X and Z components are effectively ignored due to GT vx, vz.
                R_val (float): Small diagonal value for the measurement noise covariance matrix R.
                tiny_P_vel_diag (float): A very small value for the diagonal elements of P
                                         for GT velocity states (vx, vz).

            Returns:
                np.ndarray: Denoised positions, shape (N, 3).
                np.ndarray: Estimated velocities [vx_GT, vy_estimated, vz_GT], shape (N, 3).
                np.ndarray: Estimated noise components, shape (N, 3).
            """
            N = noisy_positions.shape[0]
            if N == 0:
                return np.array([]), np.array([]), np.array([])
            if GT_velocities.shape[0] != N:
                raise ValueError("GT_velocities must have the same number of timesteps as noisy_positions.")

            # Ensure noise parameters are arrays
            if np.isscalar(sigma_w_noise):
                sigma_w_noise = np.full(3, sigma_w_noise)
            if np.isscalar(a_noise):
                a_noise = np.full(3, a_noise)
            if np.isscalar(sigma_a_motion):
                sigma_a_motion = np.full(3, sigma_a_motion)

            # --- State Vector X = [px, py, pz, vx, vy, vz, nx, ny, nz]^T ---
            # Indices for convenience
            px_idx, py_idx, pz_idx = 0, 1, 2
            vx_idx, vy_idx, vz_idx = 3, 4, 5
            nx_idx, ny_idx, nz_idx = 6, 7, 8

            # Initial state estimate (x_hat)
            x_hat = np.zeros(9)
            x_hat[0:3] = noisy_positions[0]  # Initial position from first measurement
            x_hat[vx_idx] = GT_velocities[0, 0] # Use GT initial vx
            x_hat[vy_idx] = GT_velocities[0, 1] # Use provided vy for initial estimate (could also be 0)
            x_hat[vz_idx] = GT_velocities[0, 2] # Use GT initial vz
            x_hat[6:9] = 0.0                 # Initial noise components assumed to be zero

            # Initial state covariance (P)
            P = np.eye(9) * 1.0
            for i in range(3): # Position and noise components
                var_noise_ss = (sigma_w_noise[i]**2) / (1 - a_noise[i]**2) if (1 - a_noise[i]**2) > 1e-9 else sigma_w_noise[i]**2 * 1e4
                P[i, i] = var_noise_ss
                P[i+6, i+6] = var_noise_ss

            # Velocity covariances
            P[vx_idx, vx_idx] = tiny_P_vel_diag  # vx is GT
            P[vz_idx, vz_idx] = tiny_P_vel_diag  # vz is GT
            # vy is estimated: initial uncertainty can be related to sigma_a_motion_y
            # A simple initialization: variance of velocity after one step of random acceleration
            P[vy_idx, vy_idx] = (dt * sigma_a_motion[1])**2 if (dt * sigma_a_motion[1])**2 > 1e-9 else 1.0
            if P[vy_idx, vy_idx] < tiny_P_vel_diag * 10: # Ensure it's not too small
                 P[vy_idx, vy_idx] = 0.1

            # Zero out cross-covariances for initial GT velocities (vx, vz)
            for gt_v_idx in [vx_idx, vz_idx]:
                P[gt_v_idx, :gt_v_idx] = 0
                P[:gt_v_idx, gt_v_idx] = 0
                P[gt_v_idx, gt_v_idx+1:] = 0
                P[gt_v_idx+1:, gt_v_idx] = 0
            P = (P + P.T) / 2 # Ensure symmetry

            # --- State Transition Matrix (F) ---
            F = np.eye(9)
            F[px_idx, vx_idx] = dt
            F[py_idx, vy_idx] = dt
            F[pz_idx, vz_idx] = dt
            F[nx_idx, nx_idx] = a_noise[0]
            F[ny_idx, ny_idx] = a_noise[1]
            F[nz_idx, nz_idx] = a_noise[2]

            # --- Process Noise Covariance (Q) ---
            Q = np.zeros((9, 9))
            # Noise for Markov process n
            for i in range(3):
                Q[i+6, i+6] = sigma_w_noise[i]**2

            # Process noise for py and vy (Y-axis motion)
            sigma_ay_sq = sigma_a_motion[1]**2
            Q[py_idx, py_idx]     = (dt**4 / 4) * sigma_ay_sq
            Q[py_idx, vy_idx]     = (dt**3 / 2) * sigma_ay_sq
            Q[vy_idx, py_idx]     = (dt**3 / 2) * sigma_ay_sq
            Q[vy_idx, vy_idx]     = (dt**2)     * sigma_ay_sq
            # Process noise for X and Z motion components is zero because vx, vz are GT.

            # --- Measurement Matrix (H) ---
            H = np.zeros((3, 9))
            H[0, px_idx] = 1; H[0, nx_idx] = 1  # z_x = p_x + n_x
            H[1, py_idx] = 1; H[1, ny_idx] = 1  # z_y = p_y + n_y
            H[2, pz_idx] = 1; H[2, nz_idx] = 1  # z_z = p_z + n_z

            # --- Measurement Noise Covariance (R) ---
            R = np.eye(3) * R_val

            denoised_positions_history = np.zeros((N, 3))
            estimated_velocities_history = np.zeros((N, 3))
            estimated_noise_history = np.zeros((N, 3))

            # Store initial state (after potential GT override)
            denoised_positions_history[0, :] = x_hat[0:3]
            estimated_velocities_history[0, :] = x_hat[3:6]
            estimated_noise_history[0, :] = x_hat[6:9]

            for k in range(N):
                # --- Prediction Step ---
                if k == 0: # Use initial x_hat and P for the first measurement update
                    x_hat_pred = x_hat
                    P_pred = P
                else: # Standard prediction for k > 0
                    x_hat_pred = F @ x_hat
                    P_pred = F @ P @ F.T + Q

                # --- Incorporate GT for vx, vz into prediction ---
                x_hat_pred[vx_idx] = GT_velocities[k, 0]
                x_hat_pred[vz_idx] = GT_velocities[k, 2]

                # Adjust P_pred to reflect known vx, vz
                for gt_v_idx in [vx_idx, vz_idx]:
                    P_pred[gt_v_idx, :] = 0  # Zero out row
                    P_pred[:, gt_v_idx] = 0  # Zero out col
                    P_pred[gt_v_idx, gt_v_idx] = tiny_P_vel_diag
                P_pred = (P_pred + P_pred.T) / 2 # Ensure symmetry

                # --- Update Step ---
                z_k = noisy_positions[k, :]
                y_k = z_k - H @ x_hat_pred
                S_k = H @ P_pred @ H.T + R

                try:
                    K_k = P_pred @ H.T @ np.linalg.inv(S_k)
                except np.linalg.LinAlgError:
                    # print(f"Warning: S_k singular at step {k}. Using pseudo-inverse.")
                    K_k = P_pred @ H.T @ np.linalg.pinv(S_k)

                x_hat_updated = x_hat_pred + K_k @ y_k
                P_updated = (np.eye(9) - K_k @ H) @ P_pred
                P_updated = (P_updated + P_updated.T) / 2 # Ensure symmetry

                # Final state and covariance for step k
                x_hat = x_hat_updated
                P = P_updated

                # --- Enforce GT velocity (vx, vz) and its certainty on the final state x_hat and P ---
                x_hat[vx_idx] = GT_velocities[k, 0] # Re-affirm GT vx
                x_hat[vz_idx] = GT_velocities[k, 2] # Re-affirm GT vz

                for gt_v_idx in [vx_idx, vz_idx]:
                    P[gt_v_idx, :] = 0    # Zero out row
                    P[:, gt_v_idx] = 0    # Zero out col
                    P[gt_v_idx, gt_v_idx] = tiny_P_vel_diag
                P = (P + P.T) / 2 # Ensure P remains symmetric

                # Store results
                denoised_positions_history[k, :] = x_hat[0:3]
                estimated_velocities_history[k, :] = x_hat[3:6] # vx_gt, vy_est, vz_gt
                estimated_noise_history[k, :] = x_hat[6:9]

            return denoised_positions_history, estimated_velocities_history, estimated_noise_history

    def generate_sigma_points(self, x_hat, P, alpha_ukf_param, beta_ukf_param, kappa_ukf_param):
        """Generates sigma points and weights for UKF."""
        L = x_hat.shape[0]  # Dimension of state
        lambda_ = alpha_ukf_param**2 * (L + kappa_ukf_param) - L
        gamma = np.sqrt(L + lambda_)

        sigma_points = np.zeros((L, 2 * L + 1))
        try:
            # P_sqrt = scipy.linalg.sqrtm(P) # Can be slow or fail for non-PSD
            P_sqrt = np.linalg.cholesky(P) # Numerically more stable and faster for PSD
        except np.linalg.LinAlgError:
            print("Warning: P not positive semi-definite in sigma point generation. Adding small identity to P for Cholesky.")
            # Ensure P is at least positive semi-definite by adding a small regularization
            P_reg = P + np.eye(L) * 1e-9
            P_sqrt = np.linalg.cholesky(P_reg)


        sigma_points[:, 0] = x_hat
        for i in range(L):
            sigma_points[:, i + 1]       = x_hat + gamma * P_sqrt[:, i]
            sigma_points[:, i + L + 1] = x_hat - gamma * P_sqrt[:, i]

        Wm = np.zeros(2 * L + 1)
        Wc = np.zeros(2 * L + 1)
        Wm[0] = lambda_ / (L + lambda_)
        Wc[0] = lambda_ / (L + lambda_) + (1 - alpha_ukf_param**2 + beta_ukf_param)
        for i in range(1, 2 * L + 1):
            Wm[i] = 1 / (2 * (L + lambda_))
            Wc[i] = 1 / (2 * (L + lambda_))

        return sigma_points, Wm, Wc

    # --- Define State Transition and Measurement Functions for Speed/Heading Model ---
    # State vector X = [px, py, pz, s, psi, vz, nx, ny, nz]^T (9x1)

    def f_state_transition_ukf_speed_heading(self, x_k_minus_1, dt, a_noise_param_sh):
        """
        State transition function f(x) for UKF with speed and heading model.
        x_k_minus_1: current state sigma point (9x1)
                    [px, py, pz, s, psi, vz, nx, ny, nz]^T
        dt: time step
        a_noise_param_sh: array of [a_nx, a_ny, a_nz] for correlated noise
        Returns: propagated state sigma point (9x1)
        """
        px, py, pz, s, psi, vz, nx, ny, nz = x_k_minus_1
        x_k = np.zeros_like(x_k_minus_1)

        # Motion model
        delta_px = s * np.cos(psi) * dt
        delta_py = s * np.sin(psi) * dt
        delta_pz = vz * dt

        x_k[0] = px + delta_px
        x_k[1] = py + delta_py
        x_k[2] = pz + delta_pz
        x_k[3] = s      # Speed (s_dot is process noise)
        x_k[4] = psi    # Heading (psi_dot is process noise)
        x_k[5] = vz     # Vertical velocity (vz_dot is process noise)

        # Noise model (for augmented noise states)
        x_k[6] = a_noise_param_sh[0] * nx
        x_k[7] = a_noise_param_sh[1] * ny
        x_k[8] = a_noise_param_sh[2] * nz
        
        # Normalize heading angle psi to [-pi, pi]
        x_k[4] = (x_k[4] + np.pi) % (2 * np.pi) - np.pi

        return x_k

    def h_measurement_ukf_cartesian_pos(self, x_k_pred_sh):
        """
        Measurement function h(x) for UKF, measuring Cartesian position.
        x_k_pred_sh: predicted state sigma point (9x1)
                    [px, py, pz, s, psi, vz, nx, ny, nz]^T
        Returns: predicted measurement (3x1) [mx, my, mz]
        """
        z_pred = np.zeros(3)
        px, py, pz, _, _, _, nx, ny, nz = x_k_pred_sh

        z_pred[0] = px + nx # measured_x = true_px + noise_x
        z_pred[1] = py + ny # measured_y = true_py + noise_y
        z_pred[2] = pz + nz # measured_z = true_pz + noise_z
        return z_pred


    # --- Main UKF Denoising Function for Speed/Heading Model ---
    def ukf_denoise_speed_heading(self,
        noisy_positions,      # Shape (N, 3): N timesteps, [x, y, z]
        initial_cartesian_velocity, # Shape (3,): [vx0, vy0, vz0]
        dt,                   # Scalar: time step
        sigma_w_noise_sh,     # Scalar or array (3,): std dev of white noise driving Markov noise
        a_noise_sh,           # Scalar or array (3,): correlation coeff for Markov noise
        sigma_s_dot_sh,       # Scalar: std dev of speed change (longitudinal accel noise)
        sigma_psi_dot_sh,     # Scalar: std dev of heading rate (yaw rate noise)
        sigma_vz_dot_sh,      # Scalar: std dev of vertical velocity change (vertical accel noise)
        R_val_sh=1e-5,        # Scalar: diagonal value for measurement noise covariance
        ukf_alpha=1e-3,
        ukf_beta=2.0,
        ukf_kappa=0.0
    ):
        N = noisy_positions.shape[0]
        if N == 0:
            return np.array([]), np.array([]), np.array([])

        # Ensure noise parameters are arrays if scalar
        if np.isscalar(sigma_w_noise_sh): sigma_w_noise_sh = np.full(3, sigma_w_noise_sh)
        if np.isscalar(a_noise_sh): a_noise_sh = np.full(3, a_noise_sh)

        L_state = 9 # Dimension of state vector
        L_meas = 3  # Dimension of measurement

        # --- Initial state estimate (x_hat) ---
        # X = [px, py, pz, s, psi, vz, nx, ny, nz]^T
        x_hat = np.zeros(L_state)
        x_hat[0:3] = noisy_positions[0] # Initial px, py, pz

        vx0, vy0, vz0 = initial_cartesian_velocity
        s0 = np.sqrt(vx0**2 + vy0**2)
        psi0 = np.arctan2(vy0, vx0)
        x_hat[3] = s0    # Initial speed
        x_hat[4] = psi0  # Initial heading
        x_hat[5] = vz0   # Initial vertical velocity

        x_hat[6:9] = 0.0 # Initial noise components assumed zero

        # --- Initial state covariance (P) ---
        P = np.eye(L_state) * 1.0 # General moderate uncertainty
        # Position uncertainty (related to noise)
        for i in range(3):
            var_n_steady = (sigma_w_noise_sh[i]**2) / (1 - a_noise_sh[i]**2) if (1 - a_noise_sh[i]**2) > 1e-6 else sigma_w_noise_sh[i]**2 * 100
            P[i, i] = var_n_steady
            P[i+6, i+6] = var_n_steady # Noise component uncertainty
        # Speed uncertainty (e.g., 10% of initial speed + small base)
        P[3,3] = (0.1 * abs(s0) if s0 != 0 else 0.1)**2 + 0.01
        # Heading uncertainty (can be higher if speed is low, atan2 is sensitive)
        # For simplicity, a fixed moderate uncertainty. Could be refined based on s0.
        P[4,4] = (0.1)**2 # Radians^2, e.g., ~5.7 degrees std dev
        # Vertical velocity uncertainty
        P[5,5] = (0.1 * abs(vz0) if vz0 != 0 else 0.1)**2 + 0.01


        # --- Process Noise Covariance (Q) ---
        # Models uncertainty in the state transition (unmodeled accelerations/rates)
        Q = np.zeros((L_state, L_state))
        # Q for speed (s_dot), heading (psi_dot), vertical velocity (vz_dot)
        Q[3, 3] = (sigma_s_dot_sh * dt)**2    # Variance of speed change over dt
        Q[4, 4] = (sigma_psi_dot_sh * dt)**2  # Variance of heading change over dt
        Q[5, 5] = (sigma_vz_dot_sh * dt)**2   # Variance of vertical velocity change over dt
        # Q for augmented noise states (driving noise for Markov process n)
        Q[6, 6] = sigma_w_noise_sh[0]**2
        Q[7, 7] = sigma_w_noise_sh[1]**2
        Q[8, 8] = sigma_w_noise_sh[2]**2
        # Q[0,0], Q[1,1], Q[2,2] (for px,py,pz) are often left as zero or very small,
        # as position uncertainty arises from propagation of s, psi, vz uncertainties.
        # For example, adding a tiny bit for robustness:
        # Q[0,0] = Q[1,1] = Q[2,2] = (0.001 * dt)**2


        # --- Measurement Noise Covariance (R_matrix) ---
        R_matrix = np.eye(L_meas) * R_val_sh

        # Storage for results
        denoised_positions_history = np.zeros((N, 3))
        estimated_s_psi_vz_history = np.zeros((N, 3)) # s, psi, vz
        estimated_noise_history = np.zeros((N, 3))

        # UKF Loop
        for k in range(N):
            sigma_pts, Wm, Wc = self.generate_sigma_points(x_hat, P, ukf_alpha, ukf_beta, ukf_kappa)

            sigma_pts_pred = np.zeros_like(sigma_pts)
            for i in range(2 * L_state + 1):
                sigma_pts_pred[:, i] = self.f_state_transition_ukf_speed_heading(sigma_pts[:, i], dt, a_noise_sh)

            x_hat_pred = np.sum(Wm[:, np.newaxis] * sigma_pts_pred.T, axis=0)
            # Normalize predicted heading
            x_hat_pred[4] = (x_hat_pred[4] + np.pi) % (2 * np.pi) - np.pi


            P_pred = np.zeros((L_state, L_state))
            for i in range(2 * L_state + 1):
                # Ensure heading difference is handled correctly for covariance calculation
                diff_state = sigma_pts_pred[:, i] - x_hat_pred
                diff_state[4] = (diff_state[4] + np.pi) % (2 * np.pi) - np.pi # Normalize heading difference
                diff_state = diff_state.reshape(-1,1)
                P_pred += Wc[i] * (diff_state @ diff_state.T)
            P_pred += Q

            sigma_pts_meas_pred = np.zeros((L_meas, 2 * L_state + 1))
            for i in range(2 * L_state + 1):
                sigma_pts_meas_pred[:, i] = self.h_measurement_ukf_cartesian_pos(sigma_pts_pred[:, i])

            z_hat_pred = np.sum(Wm[:, np.newaxis] * sigma_pts_meas_pred.T, axis=0)

            S_k = np.zeros((L_meas, L_meas))
            for i in range(2 * L_state + 1):
                diff_meas = (sigma_pts_meas_pred[:, i] - z_hat_pred).reshape(-1,1)
                S_k += Wc[i] * (diff_meas @ diff_meas.T)
            S_k += R_matrix

            P_xz = np.zeros((L_state, L_meas))
            for i in range(2 * L_state + 1):
                diff_state = sigma_pts_pred[:, i] - x_hat_pred
                diff_state[4] = (diff_state[4] + np.pi) % (2 * np.pi) - np.pi # Normalize heading difference
                diff_state = diff_state.reshape(-1,1)

                diff_meas = (sigma_pts_meas_pred[:, i] - z_hat_pred).reshape(-1,1)
                P_xz += Wc[i] * (diff_state @ diff_meas.T)

            try:
                K_k = P_xz @ np.linalg.inv(S_k)
            except np.linalg.LinAlgError:
                print(f"Warning: S_k singular at step {k}. Using pseudo-inverse.")
                K_k = P_xz @ np.linalg.pinv(S_k)

            z_k = noisy_positions[k, :]
            innovation = z_k - z_hat_pred
            x_hat = x_hat_pred + K_k @ innovation
            # Normalize updated heading
            x_hat[4] = (x_hat[4] + np.pi) % (2 * np.pi) - np.pi

            P = P_pred - K_k @ S_k @ K_k.T
            P = (P + P.T) / 2.0 # Ensure symmetry

            denoised_positions_history[k, :] = x_hat[0:3]
            estimated_s_psi_vz_history[k, :] = x_hat[3:6] # s, psi, vz
            estimated_noise_history[k, :] = x_hat[6:9]

        return denoised_positions_history, estimated_s_psi_vz_history, estimated_noise_history


    def simple_kalman(self, positions_noisy_gps, imu_velocities, alpha, dt, sigma_values):
        n_states = 6
        n_measurements = 3

        # --- Kalman Filter Matrices ---
        # F - State Transition Matrix
        F = np.eye(n_states)
        F[0, 0] = 1; F[1, 1] = 1; F[2, 2] = 1 # Position is previous position + velocity*dt (handled by B)
        F[3, 3] = alpha; F[4, 4] = alpha; F[5, 5] = alpha # Noise is alpha * previous_noise + process_noise

        # B - Control Input Matrix
        B = np.zeros((n_states, 3))
        B[0, 0] = dt
        B[1, 1] = dt
        B[2, 2] = dt

        # H - Measurement Matrix
        H = np.zeros((n_measurements, n_states))
        H[0, 0] = 1; H[0, 3] = 1  # gps_x = px + nx
        H[1, 1] = 1; H[1, 4] = 1  # gps_y = py + ny
        H[2, 2] = 1; H[2, 5] = 1  # gps_z = pz + nz

        # Q - Process Noise Covariance
        # Uncertainty in IMU velocity affecting position prediction
        # Use the same sigma_imu_velocity_noise as used for IMU data generation for consistency,
        # or this can be a tuning parameter.
        # This is the std dev of the error in the IMU-based velocity *used by the filter model*.
        # It's not necessarily the same as sigma_imu_velocity_noise used to *generate* the IMU data.
        # It's a tuning parameter for the filter.
        kf_sigma_v_imu = np.array([0.05, 0.05, 0.03]) # m/s - expected IMU velocity error std dev
        q_pos = (kf_sigma_v_imu * dt)**2

        # Process noise for the G-M noise states (this should match the generation)
        # beta_kf is calculated using the filter's assumed tau (which should match the true tau)
        beta_kf = sigma_values * np.sqrt(1 - alpha**2) # Same as beta_gen if tau and sigma_values are known
        q_noise = beta_kf**2

        Q = np.diag(np.concatenate([q_pos, q_noise]))


        # R - Measurement Noise Covariance
        # This is for additional white noise on the GPS sensor itself, beyond the correlated noise.
        sigma_gps_white = np.array([0.1, 0.1, 0.1]) # meters - small additional white noise std dev
        R = np.diag(sigma_gps_white**2)


        # --- Initial Kalman Filter State ---
        x_est = np.zeros(n_states)
        # Initial position: first noisy GPS reading
        x_est[0:3] = positions_noisy_gps[0, :]
        # Initial noise: assume zero
        x_est[3:6] = 0.0

        # Initial State Covariance P
        # High uncertainty for position (sigma_values of G-M noise)
        # High uncertainty for noise state (sigma_values of G-M noise)
        P_est = np.diag(np.concatenate([sigma_values**2, sigma_values**2])) # Or larger if less certain


        # --- Kalman Filter Loop ---
        num_steps = positions_noisy_gps.shape[0]
        estimated_positions = np.zeros((num_steps, 3))
        estimated_noise = np.zeros((num_steps, 3))

        for k in range(num_steps):
            # --- Prediction ---
            u_k = imu_velocities[k, :] # Control input (IMU velocity)
            x_pred = F @ x_est + B @ u_k
            P_pred = F @ P_est @ F.T + Q

            # --- Update ---
            z_k = positions_noisy_gps[k, :] # Current noisy GPS measurement
            
            y_tilde = z_k - H @ x_pred       # Innovation
            S = H @ P_pred @ H.T + R         # Innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S) # Kalman Gain

            x_est = x_pred + K @ y_tilde
            P_est = (np.eye(n_states) - K @ H) @ P_pred
            
            # Store results
            estimated_positions[k, :] = x_est[0:3] # Estimated true position
            estimated_noise[k, :] = x_est[3:6]     # Estimated noise component

        return estimated_positions, estimated_noise