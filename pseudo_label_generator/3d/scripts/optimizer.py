import torch
import re
from scipy.spatial.transform import Rotation as R
import numpy as np
from anno_V3 import AutoLabel3D
from pytorch3d.transforms import euler_angles_to_matrix
import open3d as o3d
from pytorch3d.ops.knn import knn_points
import copy

class Optimizer(AutoLabel3D):
    def __init__(self, args):
        super().__init__(args)
    
    def _get_expected_dims(self, class_name):
        """
        Helper to safely retrieve expected dimensions for a class.
        Returns (width, length, height) tuple or None if not found.
        """
        if not class_name:
            return None
        
        # Hardcoded fallback dimensions (can be overridden by config)
        default_dims = {
            'armchair': [0.8, 0.8, 1.0],
            'trash_can': [0.6, 0.6, 1.0],
            'ball': [0.3, 0.3, 0.3],
            'beer_bottle': [0.08, 0.08, 0.30],
            'beer_can': [0.08, 0.08, 0.15],
            'bench': [0.6, 1.8, 0.9],
            'billboard': [0.5, 4.0, 3.0],
            'blinker': [0.2, 0.3, 0.3],
            'bottle': [0.10, 0.10, 0.35],
            'box': [0.8, 0.8, 0.8],
            'bulldozer': [3.0, 6.0, 3.5],
            'chair': [0.6, 0.6, 1.0],
            'clock': [0.2, 0.5, 0.5],
            'deck_chair': [0.7, 1.4, 1.0],
            'dining_table': [1.0, 1.8, 0.8],
            'flagpole': [0.3, 0.3, 8.0],
            'garbage': [0.7, 0.8, 1.2],
            'ladder': [0.5, 3.0, 0.3],
            'lamp': [0.6, 0.6, 2.0],
            'lamppost': [0.4, 0.4, 8.0],
            'postbox': [0.6, 0.6, 1.4],
            'street_sign': [0.6, 0.8, 3.0],
            'streetlight': [0.5, 0.5, 8.0],
        }
        
        # Normalize class name: lowercase and spaces to underscores
        key = str(class_name).lower().replace(' ', '_')
        
        # Try config first
        try:
            frames_cfg = getattr(self.cfg, 'frames_creation', None)
            if frames_cfg:
                exp_map = getattr(frames_cfg, 'additional_expected_dims', None)
                if exp_map and hasattr(exp_map, '__getitem__'):
                    # Try direct lookup
                    if key in exp_map:
                        return tuple(map(float, exp_map[key]))
                    # Try original case
                    orig_key = str(class_name).replace(' ', '_')
                    if orig_key in exp_map:
                        return tuple(map(float, exp_map[orig_key]))
        except Exception:
            pass  # Fall through to defaults
        
        # Use hardcoded defaults
        if key in default_dims:
            return tuple(default_dims[key])
        
        return None

    def optimize_car(self, car):
        car.length = self.cfg.templates.template_length
        car.width = self.cfg.templates.template_width
        car.height = self.cfg.templates.template_height
        car.model = 0
        if not car.moving:
            car = self.optimize_coarse(car)
            car = self.optimize_fine(car)
        else:
            car = self.optimize_moving(car)
        car.optimized = True
        if self.cfg.optimization.use_dimensions_estimation_during_optim:
            car = self.estimate_dimensions(car)
        return car

    def optimize_additional_object(self, obj, depth_radius: float = 5.0, min_points: int = 10, oriented: bool = True):
        """
        Compute a 7-DoF 3D bounding box for an AdditionalObject instance.

        Output format (stored in obj.bbox):
        {
          'format': '7dof',
          'x': cx, 'y': cy, 'z': cz,
          'width': w, 'length': l, 'height': h,
          'yaw': yaw
        }

        Notes:
        - yaw is defined around the vertical axis (Z for Waymo, Y for KITTI-like)
        - width and length lie in the ground plane; height is along the vertical axis
        - Uses 2D PCA on the ground plane to estimate yaw, then aligns points to compute AABB extents
        """
        if obj is None or obj.lidar is None or len(obj.lidar) == 0:
            return obj

        pts = np.asarray(obj.lidar, dtype=np.float32)
        # Ensure center guess for filtering
        center_guess = obj.center if getattr(obj, 'center', None) is not None else np.median(pts, axis=0)

        # Distance filter around center
        diffs = pts - center_guess
        dists = np.linalg.norm(diffs, axis=1)
        keep = dists <= float(depth_radius)
        pts_f = pts[keep]
        if pts_f.shape[0] < max(min_points, 3):
            pts_f = pts  # fallback to all points

        # Dataset-specific axes
        is_waymo = self.args.dataset in ['waymo', 'waymo_converted']
        if is_waymo:
            ground_idx = (0, 1)  # X, Y
            vert_idx = 2         # Z
        else:
            ground_idx = (0, 2)  # X, Z
            vert_idx = 1         # Y

        try:
            # 2D PCA on ground plane for yaw
            gp = pts_f[:, ground_idx]
            gp_center = np.median(gp, axis=0)
            gp_local = gp - gp_center
            if gp_local.shape[0] >= 3:
                cov2 = np.cov(gp_local.T)
                vals2, vecs2 = np.linalg.eigh(cov2)
                order2 = np.argsort(vals2)[::-1]
                main_vec = vecs2[:, order2[0]]  # principal direction in ground plane
                # Ensure deterministic sign (optional)
                if main_vec[0] < 0:
                    main_vec = -main_vec
                # yaw depending on dataset convention
                if is_waymo:
                    yaw = float(np.arctan2(main_vec[1], main_vec[0]))  # XY plane, rotate about Z
                    # Rotation matrix around Z by -yaw to axis-align
                    c, s = np.cos(-yaw), np.sin(-yaw)
                    R_align = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                else:
                    # ground plane XZ -> yaw about Y; compute later in the non-waymo branch as atan2(vx, vz)
                    pass
            else:
                raise ValueError("Not enough points for PCA")

            # For KITTI-like datasets, recompute yaw and rotation properly (separate branch to keep clarity)
            if not is_waymo:
                # main_vec corresponds to [vx, vz]
                vx, vz = main_vec[0], main_vec[1]
                yaw = float(np.arctan2(vx, vz))  # rotate about Y (XZ plane)
                c, s = np.cos(-yaw), np.sin(-yaw)
                R_align = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

            # Align points to compute AABB extents in the aligned frame
            pts_aligned = (R_align @ pts_f.T).T
            mins = pts_aligned.min(axis=0)
            maxs = pts_aligned.max(axis=0)
            
            # Use median of LiDAR points as the center (in aligned frame)
            center_aligned = np.median(pts_aligned, axis=0)

            # Dimensions: length along X, width along ground-other axis, height along vertical
            if is_waymo:
                length = float(maxs[0] - mins[0])
                width = float(maxs[1] - mins[1])
                height = float(maxs[2] - mins[2])
            else:
                length = float(maxs[0] - mins[0])
                width = float(maxs[2] - mins[2])
                height = float(maxs[1] - mins[1])

            # Apply dimension constraints if available
            exp_dims = self._get_expected_dims(getattr(obj, 'class_name', None))
            if exp_dims:
                exp_w, exp_l, exp_h = exp_dims
                # Clamp dimensions - center is from median, stays fixed
                width = min(width, exp_w)
                length = min(length, exp_l)
                height = min(height, exp_h)

            # Transform center back to world frame
            if is_waymo:
                R_inv = R_align.T  # rotation matrices are orthonormal
            else:
                R_inv = R_align.T
            center_world = (R_inv @ center_aligned.T).T

            cx, cy, cz = float(center_world[0]), float(center_world[1]), float(center_world[2])

            bbox_7 = {
                'format': '7dof',
                'x': cx, 'y': cy, 'z': cz,
                'width': float(width), 'length': float(length), 'height': float(height),
                'yaw': float(yaw),
            }
        except Exception:
            # Fallback: axis-aligned box in original coordinates (yaw=0)
            mins = pts_f.min(axis=0)
            maxs = pts_f.max(axis=0)
            
            # Use median of LiDAR points as the center
            center_world = np.median(pts_f, axis=0)
            
            if is_waymo:
                length = float(maxs[0] - mins[0])
                width = float(maxs[1] - mins[1])
                height = float(maxs[2] - mins[2])
            else:
                length = float(maxs[0] - mins[0])
                width = float(maxs[2] - mins[2])
                height = float(maxs[1] - mins[1])

            # Apply dimension constraints if available (fallback path)
            exp_dims = self._get_expected_dims(getattr(obj, 'class_name', None))
            if exp_dims:
                exp_w, exp_l, exp_h = exp_dims
                # Clamp dimensions - center is from median, stays fixed
                width = min(width, exp_w)
                length = min(length, exp_l)
                height = min(height, exp_h)
            
            cx, cy, cz = float(center_world[0]), float(center_world[1]), float(center_world[2])
            bbox_7 = {
                'format': '7dof',
                'x': cx, 'y': cy, 'z': cz,
                'width': float(width), 'length': float(length), 'height': float(height),
                'yaw': 0.0,
            }

        # Save back
        obj.center = np.array([bbox_7['x'], bbox_7['y'], bbox_7['z']], dtype=np.float32)
        obj.bbox = bbox_7
        return obj

    def optimize_all_additional_objects(self, depth_radius: float = None):
        if not hasattr(self, 'additional_objects') or self.additional_objects is None:
            return
        radius = depth_radius
        if radius is None:
            # Default or from config
            radius = float(getattr(self.cfg.frames_creation, 'additional_depth_radius', 5.0))
        for cls_name, objs in list(self.additional_objects.items()):
            new_list = []
            for obj in objs:
                optimized = self.optimize_additional_object(obj, depth_radius=radius, oriented=True)
                if optimized is not None:
                    new_list.append(optimized)
            self.additional_objects[cls_name] = new_list

    def optimize_car_robust(self, car):
        car.length = self.cfg.templates.template_length
        car.width = self.cfg.templates.template_width
        car.height = self.cfg.templates.template_height
        car.model = 0
        if not car.moving:
            car = self.estimate_dimensions(car)
            car = self.optimize_loc_only(car)
        else:
            car = self.optimize_moving(car)
        car.optimized = True
        if self.cfg.optimization.use_dimensions_estimation_during_optim:
            car = self.estimate_dimensions(car)
        return car
    
    def estimate_dimensions(self, car, est_theta=False):
        if car.moving_scale_lidar is None:
            return car

        if est_theta:
            car.moving_scale_lidar = [car.lidar.T[:, :3], car.lidar.T[:, :3]] + car.moving_scale_lidar

        est_theta_arr  = np.zeros((min(len(car.moving_scale_lidar), self.cfg.frames_creation.k_to_scale_estimation)))

        for i in range(min(len(car.moving_scale_lidar), self.cfg.frames_creation.k_to_scale_estimation)):
            if car.moving_scale_lidar[i] is None:
                continue
            cur_lidar = copy.deepcopy(car.moving_scale_lidar[i])

            cur_lidar, center = self.move_pcloud_to_center_numpy(cur_lidar)

            points = cur_lidar

            best_params, best_theta, best_extent = self.estimate_best_params(points)

            y_min, y_max = self.estimate_height(points)

            rectangle_2d = self.construct_rectangle(best_params)

            bottom_points = np.hstack((rectangle_2d[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_min,
                                       rectangle_2d[:, 1].reshape(-1, 1)))  # Z = 0 for bottom
            top_points = np.hstack((rectangle_2d[:, 0].reshape(-1, 1), np.ones((4, 1)) * y_max,
                                    rectangle_2d[:, 1].reshape(-1, 1)))  # Z = height for top
            all_points = np.vstack((bottom_points, top_points))  # Combine into one array

            our_bbox, angle, bbox_center, extent = self.get_3D_bbox(all_points)
            our_bbox.color = (0, 0, 1)  # Blue color for the OBB

            center = center + bbox_center

            est_theta_arr[i] = angle - np.pi / 2.
        
        car.theta = np.median(est_theta_arr)

        return car
    
    def get_3D_bbox(self, points):
        points_xz = points[:, [0, 2]]
        centroid = np.mean(points_xz, axis=0)
        centered_points = points_xz - centroid
        cov_matrix = np.cov(centered_points, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]
        rotated_points = centered_points @ eigenvectors
        min_coords = np.min(rotated_points, axis=0)
        max_coords = np.max(rotated_points, axis=0)
        extents = max_coords - min_coords
        center_rotated = (min_coords + max_coords) / 2

        obb_center = eigenvectors @ center_rotated + centroid
        angle = -np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        height = y_max - y_min
        y_center = (y_max + y_min) / 2

        obb_center = np.array([obb_center[0], y_center, obb_center[1]])
        extents = np.array([extents[0], height, extents[1]])

        rot_matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        oriented_bbox = o3d.geometry.OrientedBoundingBox(center=obb_center, R=rot_matrix, extent=extents)

        return oriented_bbox, angle, obb_center, extents
    
    def estimate_location(self, car):
        if car.lidar is not None:
            cur_lidar = car.lidar.T[:, :3]

            location = np.median(cur_lidar, axis=0)
            car.x = location[0]
            car.y = location[1]
            car.z = location[2]
            car.theta = 0.
            return car
        else:
            car.x = 0
            car.y = 0
            car.z = 0
            car.theta = 0
            return car

    def estimate_best_params(self, points):
        best_theta = None
        max_criterion = -np.inf
        best_params = None
        best_extent = None

        points_xz = points[:, [0, 2]]

        for theta in range(0, 90, 1):
            theta_rad = np.deg2rad(theta)
            e1 = np.array([np.cos(theta_rad), np.sin(theta_rad)])
            e2 = np.array([-np.sin(theta_rad), np.cos(theta_rad)])

            c1 = np.matmul(points_xz, e1)
            c2 = np.matmul(points_xz, e2)

            percentile_90_c1 = np.percentile(c1, 90)
            percentile_10_c1 = np.percentile(c1, 10)
            percentile_90_c2 = np.percentile(c2, 90)
            percentile_10_c2 = np.percentile(c2, 10)

            # Calculate the distance to the nearest edge
            d1 = np.minimum(np.abs(c1 - percentile_10_c1), np.abs(percentile_90_c1 - c1))
            d2 = np.minimum(np.abs(c2 - percentile_10_c2), np.abs(percentile_90_c2 - c2))
            #d1 = np.minimum(np.abs(c1 - np.min(c1)), np.abs(np.max(c1) - c1))
            #d2 = np.minimum(np.abs(c2 - np.min(c2)), np.abs(np.max(c2) - c2))

            d1 = 1.0 / (1.0 + np.exp(-d1 * self.cfg.optimization.steepness))
            d2 = 1.0 / (1.0 + np.exp(-d2 * self.cfg.optimization.steepness))

            # Compute the closeness score
            #closeness_score = -np.sum(np.maximum(np.minimum(d1, d2), 1.))
            closeness_score = -np.sum(np.minimum(d1, d2))

            if closeness_score > max_criterion:
                max_criterion = closeness_score
                best_theta = theta_rad
                # Determine the edge parameters
                best_extent = np.array([c1.max() - c1.min(), c2.max() - c2.min()])
                best_params = {
                    'a1': np.cos(theta_rad), 'b1': np.sin(theta_rad), 'c1': c1.min(),
                    'a2': -np.sin(theta_rad), 'b2': np.cos(theta_rad), 'c2': c2.min(),
                    'a3': np.cos(theta_rad), 'b3': np.sin(theta_rad), 'c3': c1.max(),
                    'a4': -np.sin(theta_rad), 'b4': np.cos(theta_rad), 'c4': c2.max()
                }

        return best_params, best_theta, best_extent
    
    def estimate_height(self, points):
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        return y_min, y_max

    def move_pcloud_to_center_numpy(self, pcloud):
        centroid = np.mean(pcloud, axis=0)
        pcloud -= centroid
        return pcloud, centroid

    def construct_rectangle(self, params):
        """
        Constructs a rectangle from line parameters.
        Assumes params contains keys: 'a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'a3', 'b3', 'c3', 'a4', 'b4', 'c4'
        Each ai, bi, ci corresponds to the line equation ai*x + bi*y = ci
        """
        # Retrieve the parameters for the four edges
        a1, b1, c1 = params['a1'], params['b1'], params['c1']
        a2, b2, c2 = params['a2'], params['b2'], params['c2']
        a3, b3, c3 = params['a3'], params['b3'], params['c3']
        a4, b4, c4 = params['a4'], params['b4'], params['c4']

        # Calculate the intersection points
        point1 = self.line_intersection(a1, b1, c1, a2, b2, c2)  # Intersection of edge 1 and 2
        point2 = self.line_intersection(a1, b1, c1, a4, b4, c4)  # Intersection of edge 1 and 4
        point3 = self.line_intersection(a3, b3, c3, a2, b2, c2)  # Intersection of edge 3 and 2
        point4 = self.line_intersection(a3, b3, c3, a4, b4, c4)  # Intersection of edge 3 and 4

        return np.array([point1, point2, point3, point4])

    def line_intersection(self, a1, b1, c1, a2, b2, c2):
        """
        Finds the intersection of two lines given the coefficients of the lines.
        Line 1: a1*x + b1*y = c1
        Line 2: a2*x + b2*y = c2
        """
        A = np.array([[a1, b1], [a2, b2]])
        b = np.array([c1, c2])
        x, y = np.linalg.solve(A, b)
        return x, y

    def optimize_car_scale(self, car):
        if car.optimized and car.scale_lidar is not None and car.scale_lidar.shape[1] > 0:
            car = self.optimize_scale(car)
            return car
        else:
            return car

    def optimize_pedestrian(self, pedestrian):

        if hasattr(pedestrian, 'estimated_angle'):
            estimated_angle = pedestrian.estimated_angle
        else:
            estimated_angle = self.estimate_angle_from_movement_tracked(pedestrian)
            pedestrian.estimated_angle = estimated_angle

        if estimated_angle is None:
            if self.cfg.frames_creation.use_SAM3:
                pedestrian = self.optimize_ped_bbox_sam(pedestrian)
            else:
                pedestrian = self.optimize_ped_bbox(pedestrian)
        elif self.cfg.frames_creation.use_SAM3:
            pedestrian = self.optimize_ped_bbox_traj_sam(pedestrian)
        else:
            pedestrian = self.optimize_ped_bbox_traj(pedestrian)

        pedestrian.optimized = True
        # Hardcode pedestrian dimensions to median values (requested override)
        # Source stats:
        #   Median Height: 1.77, Median Width: 0.65, Median Length: 0.88
        # Internal attribute semantics are (length, width, height).
        if pedestrian.cyclist:
            pedestrian.length = 1.74
            pedestrian.width = 0.66
            pedestrian.height = 1.74
        else:
            pedestrian.length = 0.88
            pedestrian.width = 0.65
            pedestrian.height = 1.77

        # pedestrian = self.estimate_body_betas(pedestrian)

        return pedestrian

    def optimize_coarse(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        elif self.cfg.loss_functions.loss_function == 'binary1way' or self.cfg.loss_functions.loss_function == 'binary2way':
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters):
            for opt_param2 in np.linspace(self.cfg.optimization.opt_param2_min, self.cfg.optimization.opt_param2_max, num=self.cfg.optimization.opt_param2_iters):
                for opt_param3 in np.linspace(0, 2 * np.pi - (2 * np.pi/self.cfg.optimization.opt_param3_iters), num=self.cfg.optimization.opt_param3_iters):
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
        car.theta = opt_values[2]

        return car

    def optimize_fine(self, car):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_value = 0.

        for theta in np.linspace(0, 2 * np.pi - (2 * np.pi/360), num=360):
            template_it = self.get_template(car.x, car.y, car.z, theta)

            loss = self.compute_loss(template_it)

            if loss < min_loss:
                min_loss = loss
                opt_value = theta

        car.theta = opt_value
        return car

    def optimize_loc_only(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        elif self.cfg.loss_functions.loss_function == 'binary1way' or self.cfg.loss_functions.loss_function == 'binary2way':
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters):
            for opt_param2 in np.linspace(self.cfg.optimization.opt_param2_min, self.cfg.optimization.opt_param2_max, num=self.cfg.optimization.opt_param2_iters):
                for opt_param3 in [car.theta, car.theta + np.pi]:
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
            car.theta = opt_values[2]
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
            car.theta = opt_values[2]

        return car

    def optimize_fine_dimensions_estimation(self, car):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_value = 0.

        for theta in np.linspace(0, 2 * np.pi - (2 * np.pi/360), num=360):
            scale_length = car.length / self.cfg.templates.template_length
            scale_width = car.width / self.cfg.templates.template_width
            scale_height = car.height / self.cfg.templates.template_height
            template_it = self.get_template(car.x, car.y, car.z, theta, scale_length=scale_length, scale_width=scale_width, scale_height=scale_height)

            loss = self.compute_loss(template_it)

            if loss < min_loss:
                min_loss = loss
                opt_value = theta

        car.theta = opt_value - np.pi/2
        return car

    def optimize_scale(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0., 0., 0., 0., 0.])

        if car.scale_lidar is not None:
            tmp_opt_values = [car.x, car.y, car.z, car.theta]
            cur_lidar = car.scale_lidar.T[:, :3]
        else:
            return car

        if self.args.dataset == 'waymo':
            height_of_car = np.amax(cur_lidar[:, 2]) - np.amin(cur_lidar[:, 2])
        else:
            height_of_car = np.amax(cur_lidar[:, 1]) - np.amin(cur_lidar[:, 1])
        perfect_height_scale = height_of_car / self.cfg.templates.template_height
        perfect_height_scale = np.clip(perfect_height_scale, 0.75, 1.25)

        # Now we want to compute how much we want to move in each direction. If the car is parallel to our car (0 degrees)
        # Then we want to move more in the z axis than the x axis, because the length is usually our biggest problem.
        opt_param1_max = np.abs(np.cos(tmp_opt_values[3]) + np.sin(tmp_opt_values[3]))
        opt_param2_max = np.abs(np.sin(tmp_opt_values[3]) + np.cos(tmp_opt_values[3]))
        opt_param1_range = np.linspace(-opt_param1_max, opt_param1_max, num=self.cfg.optimization.opt_param1_iters)
        opt_param2_range = np.linspace(-opt_param2_max, opt_param2_max, num=self.cfg.optimization.opt_param2_iters)
        length_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.scale_num_scale_iters)

        if self.cfg.scale_detector.use_independent_width_scaling:
            width_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.width_num_scale_iters)
        else:
            width_range = [1.]

        #We are currently scaling the moving car.
        angle_to_iterate = [tmp_opt_values[3]]

        cur_lidar = self.downsample_lidar(cur_lidar)[:, :3]

        self.index = self.create_faiss_tree(cur_lidar)

        for template_index in range(self.cfg.scale_detector.num_of_templates):
            for scale_length in length_range:
                for scale_width in width_range:
                    for opt_param1 in opt_param1_range:
                        for opt_param2 in opt_param2_range:
                            for theta in angle_to_iterate:
                                if self.args.dataset == 'waymo':
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], opt_param2 + tmp_opt_values[1], tmp_opt_values[2], theta, template_index, scale_length, scale_width, perfect_height_scale, scale=True)
                                    else:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], opt_param2 + tmp_opt_values[1], tmp_opt_values[2], theta, template_index, scale_length, scale_length, perfect_height_scale, scale=True)
                                else:
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], tmp_opt_values[1], opt_param2 + tmp_opt_values[2], theta, template_index, scale_length, scale_width, perfect_height_scale, scale=True)
                                    else:
                                        template_it = self.get_template(opt_param1 + tmp_opt_values[0], tmp_opt_values[1], opt_param2 + tmp_opt_values[2], theta, template_index, scale_length, scale_length, perfect_height_scale, scale=True)

                                loss = self.compute_loss(template_it)

                                if loss < min_loss:
                                    min_loss = loss
                                    # Compute the loss of the template, the inputs are switched so we actually get loss for template.
                                    if self.cfg.scale_detector.use_independent_width_scaling:
                                        opt_values = np.array([template_index, scale_length, scale_width, opt_param1, opt_param2, theta])
                                    else:
                                        opt_values = np.array([template_index, scale_length, scale_length, opt_param1, opt_param2, theta])

        car.template_index = opt_values[0]
        car.length = opt_values[1] * self.cfg.templates.template_length
        car.width = opt_values[2] * self.cfg.templates.template_width
        car.height = perfect_height_scale * self.cfg.templates.template_height
        car.x_scale = opt_values[3] + tmp_opt_values[0]
        if self.args.dataset == 'waymo':
            car.y_scale = opt_values[4] + tmp_opt_values[1]
            car.z_scale = tmp_opt_values[2]
        else:
            car.y_scale = tmp_opt_values[1]
            car.z_scale = opt_values[4] + tmp_opt_values[2]
        car.theta_scale = opt_values[5]

        #Now lets iterate over height and Z value
        opt_param1_range = np.linspace(-opt_param1_max, opt_param1_max, num=20)
        height_range = np.linspace(self.cfg.scale_detector.scale_min, self.cfg.scale_detector.scale_max, num=self.cfg.optimization.scale_num_scale_iters)
        min_loss = np.inf

        for scale_height in height_range:
            for opt_param1 in opt_param1_range:
                if self.args.dataset == 'waymo':
                    if self.cfg.scale_detector.use_independent_width_scaling:
                        template_it = self.get_template(car.x_scale, car.y_scale, opt_param1 + tmp_opt_values[2], car.theta, 1, car.length, car.width, scale_height, scale=True)
                    else:
                        template_it = self.get_template(car.x_scale, car.y_scale, opt_param1 + tmp_opt_values[2], car.theta, 1, car.length, car.width, scale_height, scale=True)
                else:
                    if self.cfg.scale_detector.use_independent_width_scaling:
                        template_it = self.get_template(car.x_scale, opt_param1 + tmp_opt_values[1], car.z_scale, car.theta, 1, car.length, car.width, scale_height, scale=True)
                    else:
                        template_it = self.get_template(car.x_scale, opt_param1 + tmp_opt_values[1], car.z_scale, car.theta, 1, car.length, car.width, scale_height, scale=True)

                loss = self.compute_loss(template_it)

                if loss < min_loss:
                    min_loss = loss
                    # Compute the loss of the template, the inputs are switched so we actually get loss for template.
                    opt_values = np.array([opt_param1, scale_height])

        car.height = opt_values[1] * self.cfg.templates.template_height
        if self.args.dataset == 'waymo':
            car.z_scale = opt_values[0] + tmp_opt_values[2]
        else:
            car.y_scale = opt_values[0] + tmp_opt_values[1]

        return car


    def optimize_moving(self, car=None):
        if car is None:
            raise Exception("Car is None")

        min_loss = np.inf
        opt_values = np.array([0., 0., 0.])

        if hasattr(car, 'estimated_angle'):
            estimated_angle = car.estimated_angle
        else:
            estimated_angle = self.estimate_angle_from_movement_tracked(car)

        opt_param1_range = np.linspace(self.cfg.optimization.opt_param1_min, self.cfg.optimization.opt_param1_max, num=self.cfg.optimization.opt_param1_iters)
        opt_param2_range = np.linspace(self.cfg.optimization.opt_param2_min + 1., self.cfg.optimization.opt_param2_max + 1., num=self.cfg.optimization.opt_param2_iters)
        if estimated_angle is not None:
            opt_param3_range = [estimated_angle]
        else:
            opt_param3_range = np.linspace(0, 2 * np.pi - (2 * np.pi/self.cfg.optimization.opt_param3_iters), num=self.cfg.optimization.opt_param3_iters)

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                self.filtered_lidar = torch.tensor(self.filtered_lidar).cuda()
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0]).cuda()
            else:
                self.filtered_lidar = torch.tensor(self.filtered_lidar)
                self.lidar_car_template_non_filt[0] = torch.tensor(self.lidar_car_template_non_filt[0])
        else:
            self.index = self.create_faiss_tree(self.filtered_lidar)

        for opt_param1 in opt_param1_range:
            for opt_param2 in opt_param2_range:
                for opt_param3 in opt_param3_range:
                    if self.args.dataset == 'waymo':
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, opt_param2 + self.y_mean_lidar, self.z_mean_lidar, opt_param3)
                    else:
                        template_it = self.get_template(opt_param1 + self.x_mean_lidar, self.y_mean_lidar, opt_param2 + self.z_mean_lidar, opt_param3)

                    loss = self.compute_loss(template_it)

                    if loss < min_loss:
                        min_loss = loss
                        opt_values = np.array([opt_param1, opt_param2, opt_param3])

        car.x = opt_values[0] + self.x_mean_lidar
        if self.args.dataset == 'waymo':
            car.y = opt_values[1] + self.y_mean_lidar
            car.z = self.z_mean_lidar
        else:
            car.y = self.y_mean_lidar
            car.z = opt_values[1] + self.z_mean_lidar
        car.theta = opt_values[2]

        return car

    def estimate_angle_from_movement_tracked(self, car):
        if self.args.dataset == 'waymo':
            moving_car_info = car.info
        else:
            moving_car_info = []
        moving_car_locations = car.locations

        # If the car has been seen on only few frames, then we cannot estimate anything.
        if len(moving_car_info) < 3 and len(moving_car_locations) < 3:
            car.theta = None
            return None
        else:
            # First we need to find the frame, where it was in the reference frame
            ref_idx = None
            if self.args.dataset == 'waymo':
                for i in range(len(moving_car_info)):
                    if moving_car_info[i] is not None:
                        if moving_car_info[i][1] == self.pic_index:
                            ref_idx = i
                if ref_idx is None:
                    car.theta = None
                    return None
            else:
                for i in range(len(moving_car_locations)):
                        if moving_car_locations[i][3] == 0:
                            ref_idx = i
                if ref_idx is None:
                    car.theta = None
                    return None

            estimation_arr = []

            i = ref_idx - 1
            count = 0
            while i >= 0 and count < 5:
                if moving_car_locations[i] is not None:
                    if self.args.dataset == 'waymo':
                        dist = np.sqrt((np.power(moving_car_locations[ref_idx][0] - moving_car_locations[i][0], 2) + np.power(moving_car_locations[ref_idx][1] - moving_car_locations[i][1], 2)))
                    else:
                        dist = np.sqrt((np.power(moving_car_locations[ref_idx][0] - moving_car_locations[i][0],2) + np.power(moving_car_locations[ref_idx][2] - moving_car_locations[i][2], 2)))
                    if dist > self.cfg.optimization.moving_cars_min_dist_for_angle:
                        if self.args.dataset == 'waymo':
                            angle = np.arctan2(moving_car_locations[ref_idx][1] - moving_car_locations[i][1],moving_car_locations[ref_idx][0] - moving_car_locations[i][0])
                        else:
                            angle = np.arctan2(moving_car_locations[ref_idx][2] - moving_car_locations[i][2], moving_car_locations[ref_idx][0] - moving_car_locations[i][0])
                        estimation_arr.append(angle)
                        count += 1
                i -= 1

            i = ref_idx + 1
            count = 0
            while i < len(moving_car_locations) and count < 5:
                if moving_car_locations[i] is not None:
                    if self.args.dataset == 'waymo':
                        dist = np.sqrt((np.power(moving_car_locations[i][0] - moving_car_locations[ref_idx][0], 2) + np.power(moving_car_locations[i][1] - moving_car_locations[ref_idx][1], 2)))
                    else:
                        dist = np.sqrt((np.power(moving_car_locations[i][0] - moving_car_locations[ref_idx][0], 2) + np.power(moving_car_locations[i][2] - moving_car_locations[ref_idx][2], 2)))
                    if dist > self.cfg.optimization.moving_cars_min_dist_for_angle:
                        if self.args.dataset == 'waymo':
                            angle = np.arctan2(moving_car_locations[i][1] - moving_car_locations[ref_idx][1], moving_car_locations[i][0] - moving_car_locations[ref_idx][0])
                        else:
                            angle = np.arctan2(moving_car_locations[i][2] - moving_car_locations[ref_idx][2],moving_car_locations[i][0] - moving_car_locations[ref_idx][0])
                        estimation_arr.append(angle)
                        count += 1
                i += 1

        if len(estimation_arr) < 3:
            return None
        else:
            if len(estimation_arr) % 2 == 0:
                estimation_arr.append(estimation_arr[-1])
            estimation_arr = np.array(estimation_arr)
            predicted_angle = np.median(estimation_arr)
            if predicted_angle > np.pi:
                predicted_angle -= 2 * np.pi
            if self.args.dataset == 'kitti' or self.args.dataset == 'all' or self.args.dataset == "waymo_converted" or self.args.dataset == 'dsec':
                predicted_angle = -predicted_angle + np.pi/2
            return predicted_angle

    def get_template(self, x, y, z, theta, template_index=0, scale_length=1.0, scale_width=1.0, scale_height=1.0, scale=False):
        if self.cfg.loss_functions.loss_function == 'diffbin':
            if scale:
                template_it = self.lidar_car_template_scale[template_index].clone()
            else:
                template_it = self.lidar_car_template_non_filt[template_index].clone()
        else:
            if scale:
                template_it = self.lidar_car_template_scale[template_index].copy()
            else:
                template_it = self.lidar_car_template_non_filt[template_index].copy()

        if scale_length != 1.0 or scale_width != 1.0 or scale_height != 1.0:
            if self.args.dataset == 'waymo':
                if template_index == 0:
                    template_it[:, 2] -= self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 2] -= self.cfg.templates.offset_passat
            else:
                if template_index == 0:
                    template_it[:, 1] += self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 1] += self.cfg.templates.offset_passat

            template_it[:, 0] *= scale_length
            template_it[:, 1] *= scale_width
            template_it[:, 2] *= scale_height

            if self.args.dataset == 'waymo':
                if template_index == 0:
                    template_it[:, 2] += self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 2] += self.cfg.templates.offset_passat
            else:
                if template_index == 0:
                    template_it[:, 1] -= self.cfg.templates.offset_fiat
                elif template_index == 1:
                    template_it[:, 1] -= self.cfg.templates.offset_passat

        if self.cfg.loss_functions.loss_function == 'diffbin':
            if self.cfg.general.device == 'gpu':
                if self.args.dataset == 'waymo':
                    r = euler_angles_to_matrix(torch.tensor([theta, 0., 0.]), "ZYX").cuda()
                else:
                    r = euler_angles_to_matrix(torch.tensor([0., theta, 0.]), "ZYX").cuda()
            else:
                if self.args.dataset == 'waymo':
                    r = euler_angles_to_matrix(torch.tensor([theta, 0., 0.]), "ZYX")
                else:
                    r = euler_angles_to_matrix(torch.tensor([0., theta, 0.]), "ZYX")
            template_it = torch.matmul(r, template_it.T).T
        else:
            if self.args.dataset == 'waymo':
                r = R.from_euler('zyx', [theta, 0, 0], degrees=False)
            else:
                r = R.from_euler('zyx', [0, theta, 0], degrees=False)
            template_it = np.matmul(r.as_matrix(), template_it.T).T

        template_it[:, 0] += x
        template_it[:, 1] += y
        template_it[:, 2] += z

        return template_it

    def optimize_ped_bbox(self, pedestrian):
        theta = 0.0
        
        center = np.median(pedestrian.lidar, axis=1)[:3]

        # Update pedestrian attributes
        pedestrian.theta = theta + np.pi / 2
        pedestrian.x, pedestrian.y, pedestrian.z = center[0], center[1], center[2]

        return pedestrian
    
    def optimize_ped_bbox_sam(self, pedestrian):
        theta = 0.0
        
        center = np.median(pedestrian.lidar, axis=1)[:3]

        # Update pedestrian attributes
        pedestrian.theta = theta + np.pi / 2
        pedestrian.x, pedestrian.y, pedestrian.z = center[0], center[1], center[2]

        return pedestrian
    
    def optimize_ped_bbox_traj(self, pedestrian):
        estimated_angle = pedestrian.estimated_angle
        theta = estimated_angle - np.pi / 2
        
        center = np.median(pedestrian.lidar, axis=1)[:3]

        pedestrian.theta = theta + np.pi / 2
        pedestrian.x, pedestrian.y, pedestrian.z = center[0], center[1], center[2]
        
        return pedestrian
    
    def optimize_ped_bbox_traj_sam(self, pedestrian):

        estimated_angle = pedestrian.estimated_angle

        # The orientation of the bbox (theta) is perpendicular to the shoulder line
        theta = estimated_angle - np.pi / 2

        # Create rotation matrix to align points with axes
        if self.args.dataset == 'waymo':
            r = R.from_euler('z', -theta)
        else:
            r = R.from_euler('y', -theta)
        
        center = np.median(pedestrian.lidar, axis=1)[:3]

        # Update pedestrian attributes
        pedestrian.theta = theta + np.pi / 2
        pedestrian.x, pedestrian.y, pedestrian.z = center[0], center[1], center[2]

        return pedestrian
    


