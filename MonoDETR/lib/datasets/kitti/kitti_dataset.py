import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
from lib.datasets.kitti.eval_custom_nusc import eval_nusc_like
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy
from .pd import PhotometricDistort
import dill
import zstd
import point_cloud_utils as pcu
import open3d
import torch

class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('root_dir')
        self.split = split
        
        # Get writelist first to determine number of classes dynamically
        self.writelist = cfg.get('writelist', ['Car'])
        
        # Dynamically build class mappings based on writelist
        self.class_name = self.writelist.copy()
        self.cls2id = {cls_name: idx for idx, cls_name in enumerate(self.class_name)}
        self.num_classes = len(self.class_name)
        
        self.max_objs = 50
        self.resolution = np.array([1280, 384])  # W * H #[960, 640]
        self.use_3d_center = cfg.get('use_3d_center', True)
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get('bbox2d_type', 'anno')
        assert self.bbox2d_type in ['anno', 'proj']
        self.meanshape = cfg.get('meanshape', False)
        self.class_merging = cfg.get('class_merging', False)
        self.use_dontcare = cfg.get('use_dontcare', False)
        self.use_add_data = cfg.get('use_add_data', False)
        if self.use_add_data:
            self.add_data_path = cfg.get('add_data_path', None)
        self.use_depth = cfg.get('use_depth', False)
        if self.use_depth:
            self.depth_path = cfg.get('depth_path', None)

        if self.class_merging:
            self.writelist.extend(['Van', 'Truck'])
        if self.use_dontcare:
            self.writelist.extend(['DontCare'])

        # data split loading
        assert self.split in ['train', 'val', 'trainval', 'test']
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Mean sizes for each class [height, width, length] in meters
        # Create mean size array dynamically based on class_name
        default_mean_sizes = {
            'Pedestrian': [1.76255119, 0.66068622, 0.84422524],
            'Car': [1.52563191462, 1.62856739989, 3.88311640418],
            'Cyclist': [1.73698127, 0.59706367, 1.76282397],
            # Default sizes for additional classes (will be adjusted based on actual data)
            'armchair': [1.0, 0.8, 0.8],
            'trash_can': [1.0, 0.5, 0.5],
            'ball': [0.3, 0.3, 0.3],
            'beer_bottle': [0.3, 0.1, 0.1],
            'beer_can': [0.15, 0.07, 0.07],
            'bench': [0.8, 1.5, 0.5],
            'billboard': [3.0, 4.0, 0.2],
            'blinker': [0.3, 0.2, 0.2],
            'bottle': [0.3, 0.1, 0.1],
            'box': [0.5, 0.5, 0.5],
            'bulldozer': [3.0, 2.5, 5.0],
            'chair': [1.0, 0.5, 0.5],
            'clock': [0.5, 0.5, 0.2],
            'deck_chair': [1.0, 0.7, 1.2],
            'dining_table': [0.8, 1.5, 0.8],
            'flagpole': [5.0, 0.2, 0.2],
            'garbage': [0.5, 0.5, 0.5],
            'ladder': [2.0, 0.5, 0.3],
            'lamp': [0.5, 0.3, 0.3],
            'lamppost': [4.0, 0.3, 0.3],
            'postbox': [1.5, 0.6, 0.6],
            'street_sign': [1.5, 0.8, 0.2],
            'streetlight': [4.0, 0.3, 0.3],
        }
        
        # Build cls_mean_size array in the order of class_name
        self.cls_mean_size = np.array([
            default_mean_sizes.get(cls, [1.0, 1.0, 1.0]) for cls in self.class_name
        ], dtype=np.float32)
        
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

        self.lidar_templates = []
        self.rendering_templates = []

        self.offset_fiat = cfg.get('offset_fiat', 0.0)
        self.offset_passat = cfg.get('offset_passat', 0.0)
        self.offset_suv = cfg.get('offset_suv', 0.0)
        self.offset_mpv = cfg.get('offset_mpv', 0.0)

        self.template_width = cfg.get('template_width', 1.63)
        self.template_height = cfg.get('template_height', 1.526)
        self.template_length = cfg.get('template_length', 3.88)

        self.use_canonical_module = cfg.get('use_canonical_module', False)
        self.canonical_focal_length = cfg.get('canonical_focal_length', 1000.0)

        self.output_lidar = cfg.get('output_lidar', False)

        #self.load_lidar_templates()

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_velodyne(self, idx):
        velo_file = os.path.join(self.data_dir, 'velodyne', '%06d.bin' % idx)
        assert os.path.exists(velo_file)
        return np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)

    def get_depth(self, idx):
        depth_file = os.path.join(self.depth_path, '%06d.zst' % idx)
        assert os.path.exists(depth_file)
        with open(depth_file, 'rb') as f:
            decompressed = zstd.decompress(f.read())
            rgbd = dill.loads(decompressed)
        return rgbd

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        # Ensure detections and GTs are aligned by image ids
        dt_annos = kitti.get_label_annos(results_dir, img_ids)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        # Build test_id dynamically from cls2id
        test_id = self.cls2id.copy()

        # Define supported classes for KITTI evaluation (only these are supported in eval code)
        supported_classes = ['Car', 'Pedestrian', 'Cyclist']
        
        # Filter writelist to only include supported classes for evaluation
        eval_classes = [cls for cls in self.writelist if cls in supported_classes]
        
        if not eval_classes:
            logger.warning("No supported classes found in writelist for evaluation. Supported classes are: Car, Pedestrian, Cyclist")
            return 0.0
        
        if len(eval_classes) < len(self.writelist):
            skipped = [cls for cls in self.writelist if cls not in supported_classes]
            logger.info(f"Only evaluating supported classes: {eval_classes}. Skipping: {skipped}")

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in eval_classes:
            if category not in test_id:
                logger.warning(f"Category {category} not found in class mapping, skipping evaluation")
                continue
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == 'Car':
                car_moderate = mAP3d_R40
            logger.info(results_str)

        # nuScenes-like metrics (AOE/ATE/ADE/IoU2D) during training/testing
        logger.info('==> Evaluating (nuScenes-like metrics) ...')
        # Use only supported classes for nuScenes-like evaluation
        nusc_summary = eval_nusc_like(dt_annos, gt_annos, classes=tuple(eval_classes))

        def _fmt(v):
            return 'nan' if v != v else f"{v:.4f}"

        for cls, m in nusc_summary['per_class'].items():
            logger.info(
                f"[nusc-like] {cls}: TP={m['TP']} FP={m['FP']} FN={m['FN']} | "
                f"AOE(deg)={_fmt(m['AOE(deg)'])} ATE(m)={_fmt(m['ATE(m)'])} "
                f"ADE_rel={_fmt(m['ADE_rel(mean(|d-p|/p))'])} ADE_abs(m)={_fmt(m['ADE_abs(m)'])} IoU2D={_fmt(m['IoU2D'])}"
            )
        o = nusc_summary['overall']
        logger.info(
            f"[nusc-like] OVERALL: TP={o['TP']} FP={o['FP']} FN={o['FN']} | "
            f"AOE(deg)={_fmt(o['AOE(deg)'])} ATE(m)={_fmt(o['ATE(m)'])} "
            f"ADE_rel={_fmt(o['ADE_rel(mean(|d-p|/p))'])} ADE_abs(m)={_fmt(o['ADE_abs(m)'])} IoU2D={_fmt(o['IoU2D'])}"
        )
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        calib = self.get_calib(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        if self.use_depth:
            depth = self.get_depth(index).numpy()

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False

        if self.data_augmentation:
            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if self.use_depth:
                    depth = np.fliplr(depth)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        #img.save('path_where_to_save_image.jpg', 'JPEG')

        if self.use_depth:
            depth_img = Image.fromarray(depth, mode="F")
            transformed_depth_map_img = depth_img.transform(
                tuple(self.resolution.tolist()),  # Same resolution as img
                method=Image.AFFINE,
                data=tuple(trans_inv.reshape(-1).tolist()),  # Same transformation data as img
                resample=Image.BILINEAR  # Same resampling method as img
            )
            depth = np.array(transformed_depth_map_img).astype(depth.dtype)

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        if self.use_depth:
            depth = np.clip(depth, 0, 150.)
            depth = depth / 75.
            #depth = (depth - np.mean(depth)) / np.std(depth)
            depth -= 1.
            depth = np.expand_dims(depth, axis=0)
            img = np.concatenate((img, depth), axis=0)

        if self.use_canonical_module:
            fu, fv, cu, cv, height_cropped = self.adjust_intrinsics(calib.fu, calib.fv, calib.cu, calib.cv, img_size, center, crop_scale, crop_size, random_flip_flag)
            canonical_scale = self.canonical_focal_length / fu
        else:
            fu, fv, cu, cv, height_cropped = self.adjust_intrinsics(calib.fu, calib.fv, calib.cu, calib.cv, img_size, center, crop_scale, crop_size, random_flip_flag)
            canonical_scale = 1.

        #height_cropped = 1.

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size,
                'canonical_scale': canonical_scale,
                'height_crop': height_cropped}

        if self.split == 'test':
            calib = self.get_calib(index)
            return img, calib.P2, img, info

        #  ============================   get labels   ==============================
        objects = self.get_label(index)

        # data augmentation for labels #TODO Implement this also for our additional data
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        scores = np.zeros((self.max_objs), dtype=np.float32)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        objects_out = np.zeros((self.max_objs, 7), dtype=np.float32)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.

            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                 dtype=np.float32)  # W * H
            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space

            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
            center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment
            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id
            scores[i] = objects[i].score

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            if self.use_canonical_module:
                objects[i].pos[-1] *= canonical_scale

            # encoding depth
            if self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale

            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale

            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = calib.P2

            objects_out[i] = np.array([objects[i].h, objects[i].w, objects[i].l, objects[i].pos[0], objects[i].pos[1], objects[i].pos[2], objects[i].ry], dtype=np.float32)

        # collect return data
        inputs = img
        targets = {
            'calibs': calibs,
            'indices': indices,
            'img_size': img_size,
            'labels': labels,
            'scores': scores,
            'boxes': boxes,
            'boxes_3d': boxes_3d,
            'depth': depth,
            'size_2d': size_2d,
            'size_3d': size_3d,
            'src_size_3d': src_size_3d,
            'heading_bin': heading_bin,
            'heading_res': heading_res,
            'mask_2d': mask_2d,
            'objects': objects_out}

        masks_out = np.zeros((self.max_objs, self.resolution[0], self.resolution[1]), dtype=np.bool_)
        lidar_out = np.zeros((self.max_objs, 10000, 3), dtype=np.float32)
        calib_T_cam2_velo = np.zeros((4,4), dtype=np.float32)
        calib_P_rect_00 = np.zeros((3,4), dtype=np.float32)
        moving = np.zeros((self.max_objs), dtype=np.bool_)
        angle_if_moving = np.zeros((self.max_objs), dtype=np.float32)

        if self.use_add_data:
            # Load the cars data
            with open(self.add_data_path + '/optimized_cars/' + str(index).zfill(6) + ".zstd", 'rb') as f:
                compressed_arr = f.read()
                cars = dill.loads(zstd.decompress(compressed_arr))

                for idx, car in enumerate(cars):
                    mask = Image.fromarray(car.mask.transpose().astype('uint8') * 255)
                    #if random_flip_flag:
                    #    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_transformed = mask.transform(
                        size=tuple(self.resolution.tolist()),
                        method=Image.AFFINE,
                        data=tuple(trans_inv.reshape(-1).tolist()),
                        resample=Image.NEAREST
                    )
                    masks_out[idx] = (np.array(mask_transformed) > 127).transpose()

                    lidar_out[idx] = self.downsample_lidar(car.lidar.T[:, :3])
                    if car.moving:
                        moving[idx] = True
                        angle_if_moving[idx] = car.theta

            # Load the calib info
            with open(self.add_data_path + '/optimized_cars/' + str(index).zfill(6) + "_calib" + ".zstd", 'rb') as f:
                compressed_arr = f.read()
                calibs = dill.loads(zstd.decompress(compressed_arr))
                calib_T_cam2_velo = calibs[0]
                calib_P_rect_00 = calibs[1]

        if self.output_lidar:
            lidar = self.get_velodyne(index)
            lidar[:, 3] = 1.
            velo_to_cam = calib.V2C
            lidar = np.matmul(velo_to_cam, lidar.T).T
            lidar = np.matmul(calib.R0, lidar.T).T
        else:
            lidar = np.zeros(1)

        lidar_templates = self.lidar_templates
        lidar_templates = np.array(lidar_templates, dtype=np.float32)

        templates_dimensions = np.array([self.template_height, self.template_width, self.template_length], dtype=np.float32)

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size,
                'masks': masks_out,
                'lidar': lidar_out,
                'calib_T_cam2_velo': calib_T_cam2_velo,
                'calib_P_rect_00': calib_P_rect_00,
                'moving': moving,
                'angle_if_moving': angle_if_moving,
                'affine': trans,
                'affine_inv': trans_inv,
                'scale_depth': crop_scale,
                'calib_P2': calib.P2,
                'calib_R0': calib.R0,
                'calib_V2C': calib.V2C,
                'lidar_whole': lidar,
                'lidar_bool': np.array([self.output_lidar], dtype=np.bool_),
                'resolution': self.resolution,
                'lidar_templates': lidar_templates,
                'templates_dimensions': templates_dimensions,
                'flip': random_flip_flag,
                'canonical_scale': canonical_scale,
                'height_crop': height_cropped
                }
        return inputs, calib.P2, targets, info

    def adjust_intrinsics(self, fx, fy, cx, cy, img_size, center, crop_scale, crop_size, random_flip_flag):
        # Initialize adjusted intrinsics
        fx_adj = fx
        fy_adj = fy
        cx_adj = cx
        cy_adj = cy

        # Apply horizontal flip adjustment
        if random_flip_flag:
            cx_adj = img_size[0] - 1 - cx_adj

        # Apply random scaling
        fx_adj *= crop_scale
        fy_adj *= crop_scale
        cx_adj *= crop_scale
        cy_adj *= crop_scale

        # Apply center shift (cropping)
        delta_x = center[0] - (img_size[0] / 2)
        delta_y = center[1] - (img_size[1] / 2)
        cx_adj -= delta_x
        cy_adj -= delta_y

        # Compute scaling factors for affine transformation
        scale_x = self.resolution[0] / crop_size[0]

        # Apply scaling to intrinsics
        fx_adj *= scale_x
        fy_adj *= scale_x
        cx_adj *= scale_x
        cy_adj *= scale_x

        height_cropped = cy_adj / (self.resolution[1] / 2.)

        return fx_adj, fy_adj, cx_adj, cy_adj, height_cropped

    def downsample_voxel(self, filtered_lidar):
        filtered_lidar = pcu.downsample_point_cloud_on_voxel_grid(0.15, filtered_lidar)
        return filtered_lidar

    def downsample_random(self, filtered_lidar, number=1000):
        size = filtered_lidar.shape[0]
        if size > number:
            idxs = np.random.choice(np.arange(size), number, replace=False)
            downsampled = filtered_lidar[idxs, :]
            return downsampled
        else:
            return filtered_lidar

    def downsample_lidar(self, lidar, method="both", output_points=10000):
        # Input lidar scan Nx3
        voxel = self.downsample_voxel(lidar[:, :3])

        num_random = np.min([output_points - voxel.shape[0], lidar.shape[0]])
        if num_random < 0:
            lidar = self.downsample_random(voxel, output_points)
        else:
            random_points = self.downsample_random(lidar[:, :3], num_random)

            lidar = np.concatenate((voxel, random_points), axis=0)

        if lidar.shape[0] < output_points:
            lidar = np.pad(lidar, ((0, output_points - lidar.shape[0]), (0, 0)), 'constant', constant_values=(0))

        return lidar[:, :3]

    def load_lidar_templates(self):
        pcd1, mesh1, mesh_p3d_1 = self.load_and_sample_fiat()
        pcd2, mesh2, mesh_p3d_2 = self.load_and_sample_passat()
        pcd3, mesh3, mesh_p3d_3 = self.load_and_sample_suv()
        pcd4, mesh4, mesh_p3d_4 = self.load_and_sample_mpv()

        pcd1 = np.asarray(pcd1.points)
        pcd2 = np.asarray(pcd2.points)
        pcd3 = np.asarray(pcd3.points)
        pcd4 = np.asarray(pcd4.points)

        pcd1[:, 1] -= self.offset_fiat
        pcd2[:, 1] -= self.offset_passat
        pcd3[:, 1] -= self.offset_suv
        pcd4[:, 1] -= self.offset_mpv

        self.lidar_templates = [pcd1, pcd2, pcd3, pcd4]

    def load_and_sample_fiat(self):
        mesh = open3d.io.read_triangle_mesh("../pseudo_label_generator/3d/data/fiat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        T = np.eye(4)
        T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, np.pi/2, 0))  # Y rotation -||-
        mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        scale_ax0 = self.template_width / ax0_size
        scale_ax1 = self.template_height / ax1_size
        scale_ax2 = self.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' and torch.cuda.is_available() else "cpu")
        mesh_p3d = load_objs_as_meshes(["../pseudo_label_generator/3d/data/fiat_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_passat(self):
        mesh = open3d.io.read_triangle_mesh("../pseudo_label_generator/3d/data/passat2.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        T = np.eye(4)
        T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi))  # Y rotation -||-
        mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        scale_ax0 = self.template_width / ax0_size
        scale_ax1 = self.template_height / ax1_size
        scale_ax2 = self.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda" if self.cfg.general.device == 'gpu' and torch.cuda.is_available() else "cpu")
        mesh_p3d = load_objs_as_meshes(["../pseudo_label_generator/3d/data/passat_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_suv(self):
        mesh = open3d.io.read_triangle_mesh("../pseudo_label_generator/3d/data/suv.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        T = np.eye(4)
        T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, 0))  # Y rotation -||-
        mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        scale_ax0 = self.template_width / ax0_size
        scale_ax1 = self.template_height / ax1_size
        scale_ax2 = self.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda")
        mesh_p3d = load_objs_as_meshes(["../pseudo_label_generator/3d/data/suv_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def load_and_sample_mpv(self):
        mesh = open3d.io.read_triangle_mesh("../pseudo_label_generator/3d/data/minivan.gltf")  # Read mesh of fiat uno converted via blender
        bbox = mesh.get_minimal_oriented_bounding_box()
        T = np.eye(4)
        T[:3, 3] = (-bbox.center[0], -bbox.center[1], -bbox.center[2])
        mesh.transform(T)

        T = np.eye(4)
        T[:3, :3] = open3d.geometry.get_rotation_matrix_from_zxy((np.pi, 0, np.pi/2))  # Y rotation -||-
        mesh.transform(T)

        vertices = np.asarray(mesh.vertices)
        ax0_size = np.amax(vertices[:, 0]) - np.amin(vertices[:, 0])
        ax1_size = np.amax(vertices[:, 1]) - np.amin(vertices[:, 1])
        ax2_size = np.amax(vertices[:, 2]) - np.amin(vertices[:, 2])

        scale_ax0 = self.template_width / ax0_size
        scale_ax1 = self.template_height / ax1_size
        scale_ax2 = self.template_length / ax2_size

        vertices[:, 0] *= scale_ax0
        vertices[:, 1] *= scale_ax1
        vertices[:, 2] *= scale_ax2

        mesh.vertices = open3d.utility.Vector3dVector(vertices)

        mesh.compute_vertex_normals()  # Looks better in visu with normals
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

        device = torch.device("cuda")
        mesh_p3d = load_objs_as_meshes(["../pseudo_label_generator/3d/data/mpv_deformed.obj"], device=device)

        return pcd, mesh, mesh_p3d

    def get_lidar_templates(self):
        return self.lidar_templates

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'root_dir': '../../../data/KITTI',
           'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.8, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI_Dataset('train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
