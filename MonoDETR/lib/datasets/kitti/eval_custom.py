import kitti_eval_python.kitti_common as kitti
from kitti_eval_python.eval import get_official_eval_result, get_distance_eval_result
import os
import datetime
import numpy as np

def eval(results_dir, gt_dir, val_idxs_path, logger, writelist=None, class_mapping=None):
    """
    Evaluate 3D object detection results.
    
    Args:
        results_dir: Directory containing detection results
        gt_dir: Directory containing ground truth labels
        val_idxs_path: Path to validation indices file
        logger: Logger for output
        writelist: List of classes to evaluate (default: ['Car', 'Pedestrian'])
        class_mapping: Dictionary mapping class names to indices (default: standard KITTI mapping)
    """
    print("==> Loading detections and GTs...")
    with open(val_idxs_path, 'r') as f:
        idx_list_lines = f.readlines()
        idx_list = [x.strip() for x in idx_list_lines]
    img_ids = [int(id) for id in idx_list]
    dt_annos = kitti.get_label_annos(results_dir, img_ids)
    gt_annos = kitti.get_label_annos(gt_dir, img_ids)

    # Use provided class_mapping or create default
    if class_mapping is None:
        # Standard KITTI classes
        class_mapping = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    
    if writelist is None:
        writelist = ['Car', 'Pedestrian']

    print('==> Evaluating (official) ...')
    car_moderate = 0
    for category in writelist:
        if category not in class_mapping:
            print(f"Warning: Category {category} not found in class mapping, skipping")
            continue
        results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, class_mapping[category])
        if category == 'Car':
            car_moderate = mAP3d_R40
        print(results_str)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        with open(os.path.join(results_dir, 'eval_' + timestamp + '.txt'), 'w') as f:
            f.write(results_str)

    return car_moderate

prediction_path2 = None

#KITTI
#prediction_path = "/REDACTED_PATH/outputs/data/"
#prediction_path = "/REDACTED_PATH/monosowa_kitti_labels_cars_060_v2/"
#prediction_path = "/REDACTED_PATH/labels_kitti_codetr_tfl_lidar_newclass_scores/"
#prediction_path = "/REDACTED_PATH/kitti_pseudo_nofilt/training/label_2/"
#gt_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
#val_idxs_path = "/REDACTED_PATH/KITTI/ImageSets/train.txt"

#KITTI-360
#prediction_path = "/REDACTED_PATH/k360_05_100_load/training/label_2/"
#gt_path = "/REDACTED_PATH/k360_05_100_load/training/labels_gt/"
#val_idxs_path = "/REDACTED_PATH/k360_05_100_load/ImageSets/train.txt"

prediction_path = "/REDACTED_PATH/test_kitti/training/label_pseudo/"
gt_path = "/REDACTED_PATH/test_kitti/training/label_2/"
val_idxs_path = "/REDACTED_PATH/test_kitti/ImageSets/train.txt"

print(f"Evaluating predictions in folder: {prediction_path}")

evaluator = eval(prediction_path, gt_path, val_idxs_path, None)

if prediction_path2 is not None:
    evaluator2 = eval(prediction_path2, gt_path, val_idxs_path, None)