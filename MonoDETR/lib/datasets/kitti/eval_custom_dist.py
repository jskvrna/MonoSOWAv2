import kitti_eval_python.kitti_common as kitti
from kitti_eval_python.eval import get_official_eval_result, get_distance_eval_result
import os
import datetime
import numpy as np

def filter_cars_by_distance(detections_data, min_distance, max_distance):
    """
    Filters detections based on object type and distance.

    Non-'Car' objects are always kept. 'Car' objects are kept only if their
    distance, calculated from x and z coordinates (sqrt(x^2 + z^2)) from the
    'location' array, is within the specified min_distance and max_distance (inclusive).

    Args:
        detections_data (dict): A dictionary where keys are attribute names
                                (e.g., 'name', 'location') and values are numpy arrays.
                                All arrays are assumed to have the same first dimension (number of objects).
        min_distance (float): The minimum distance for a 'Car' to be kept.
        max_distance (float): The maximum distance for a 'Car' to be kept.

    Returns:
        dict: A new dictionary with the filtered detections.
    """
    if not detections_data or 'name' not in detections_data or 'location' not in detections_data:
        return detections_data # Return original if essential keys are missing or data is empty

    names = detections_data['name']
    locations = detections_data['location']

    if names.ndim == 0 or locations.ndim < 2 or locations.shape[0] == 0: # Handle empty or scalar arrays
        return detections_data

    # Calculate distance from x and z coordinates
    # Assuming location is [x, y, z]
    x_coords = locations[:, 0]
    z_coords = locations[:, 2]
    distances_xz = np.sqrt(x_coords**2 + z_coords**2)

    # Mask for objects that are 'Car'
    is_car_mask = (names == 'Car')

    # Mask for 'Car' objects that are within the desired distance range
    cars_in_range_mask = (distances_xz >= min_distance) & (distances_xz <= max_distance)

    # Keep an object if:
    # 1. It's not a 'Car'
    # OR
    # 2. It IS a 'Car' AND it's within the specified distance range
    keep_mask = ~is_car_mask | (is_car_mask & cars_in_range_mask)

    filtered_detections = {}
    for key, value_array in detections_data.items():
        if isinstance(value_array, np.ndarray) and value_array.shape[0] == len(keep_mask):
            filtered_detections[key] = value_array[keep_mask]
        else:
            # If the array doesn't match the length (e.g. metadata), keep as is or handle as error
            # For simplicity, keeping as is; adjust if specific error handling is needed
            filtered_detections[key] = value_array

    return filtered_detections

def eval(results_dir, gt_dir, val_idxs_path, logger, writelist=None, class_mapping=None):
    """
    Evaluate 3D object detection results with distance filtering.
    
    Args:
        results_dir: Directory containing detection results
        gt_dir: Directory containing ground truth labels
        val_idxs_path: Path to validation indices file
        logger: Logger for output
        writelist: List of classes to evaluate (default: ['Car'])
        class_mapping: Dictionary mapping class names to indices (default: standard KITTI mapping)
    """

    print("==> Loading detections and GTs...")
    with open(val_idxs_path, 'r') as f:
        idx_list_lines = f.readlines()
        idx_list = [x.strip() for x in idx_list_lines]
    img_ids = [int(id) for id in idx_list]
    dt_annos_all = kitti.get_label_annos(results_dir, img_ids)
    gt_annos_all = kitti.get_label_annos(gt_dir, img_ids)
    
    # Use provided class_mapping or create default
    if class_mapping is None:
        # Standard KITTI classes
        class_mapping = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    
    if writelist is None:
        writelist = ['Car']

    print('==> Evaluating (official) ...')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_filename = os.path.join(results_dir, 'eval_dist_' + timestamp + '.txt')

    for start_dist in range(0, 100, 10):
        end_dist = start_dist + 10
        print(f"\n==> Evaluating Range: {start_dist}m - {end_dist}m")
        
        dt_annos = [filter_cars_by_distance(x, start_dist, end_dist) for x in dt_annos_all]
        gt_annos = [filter_cars_by_distance(x, start_dist, end_dist) for x in gt_annos_all]

        for category in writelist:
            if category not in class_mapping:
                print(f"Warning: Category {category} not found in class mapping, skipping")
                continue
            
            results_str, results_dict = get_distance_eval_result(gt_annos, dt_annos, class_mapping[category])
            print(results_str)
            
            with open(output_filename, 'a') as f:
                f.write(f"\n{'='*20}\nRange: {start_dist}m - {end_dist}m\n{'='*20}\n")
                f.write(results_str)

    return 0

prediction_path2 = None

#KITTI
#prediction_path = "/REDACTED_PATH/outputs/data/"
#prediction_path = "/REDACTED_PATH/labels_kitti_codetr_tfl_pseudo_newclass_scores/"
#prediction_path2 = "/REDACTED_PATH/labels_kitti_codetr/"
#prediction_path = "/REDACTED_PATH/kitti_pseudo_nofilt/training/label_2/"
#prediction_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
#gt_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
#val_idxs_path = "/REDACTED_PATH/KITTI/ImageSets/train.txt"

#KITTI-360
prediction_path = "/REDACTED_PATH/test_kitti/training/label_pseudo/"
gt_path = "/REDACTED_PATH/test_kitti/training/label_2/"
val_idxs_path = "/REDACTED_PATH/test_kitti/ImageSets/train.txt"

evaluator = eval(prediction_path, gt_path, val_idxs_path, None)

if prediction_path2 is not None:
    evaluator2 = eval(prediction_path2, gt_path, val_idxs_path, None)