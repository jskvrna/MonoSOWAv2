import numpy as np
import os
from collections import defaultdict
import copy

def get_annos_from_path(label_path, image_ids):
    """
    Loads annotations from label files.

    Args:
        label_path (str): Path to the directory containing label files.
        image_ids (list[int]): List of image indices to load.

    Returns:
        list[dict]: A list of annotation dictionaries, one for each image.
    """
    annos = []
    for img_id in image_ids:
        file_path = os.path.join(label_path, f"{img_id:06d}.txt")
        img_anno = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': []
        }
        if not os.path.exists(file_path):
            annos.append(img_anno)
            continue

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) < 15:
                continue # Skip malformed lines

            img_anno['name'].append(parts[0])
            img_anno['truncated'].append(float(parts[1]))
            img_anno['occluded'].append(int(parts[2]))
            img_anno['alpha'].append(float(parts[3]))
            img_anno['bbox'].append([float(p) for p in parts[4:8]])
            img_anno['dimensions'].append([float(p) for p in parts[8:11]])
            img_anno['location'].append([float(p) for p in parts[11:14]])
            img_anno['rotation_y'].append(float(parts[14]))
            if len(parts) > 15:
                img_anno['score'].append(float(parts[15]))
            else:
                # Ground truth doesn't have a score, use 1.0 for consistency
                img_anno['score'].append(1.0)
        
        for k, v in img_anno.items():
            img_anno[k] = np.array(v)
        
        annos.append(img_anno)
    return annos

def calculate_iou_2d(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Args:
        boxA (list or np.ndarray): [x1, y1, x2, y2]
        boxB (list or np.ndarray): [x1, y1, x2, y2]

    Returns:
        float: The IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    union_area = boxA_area + boxB_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def evaluate_2d(image_ids, gt_annos, pred_annos, classes=['Car', 'Pedestrian'], iou_thresholds=np.arange(0.0, 1.05, 0.1)):
    """
    Performs 2D object detection evaluation for multiple IoU thresholds.

    Args:
        image_ids (list[int]): List of image IDs to evaluate.
        gt_annos (list[dict]): Ground truth annotations.
        pred_annos (list[dict]): Prediction annotations.
        classes (list[str]): List of class names to evaluate.
        iou_thresholds (list or np.ndarray): A list of IoU thresholds for which to calculate AP.

    Returns:
        dict: A dictionary containing AP for each class at each IoU threshold.
    """
    results = {}
    for cls in classes:
        results[cls] = {}
        # Extract GT and predictions for the current class
        class_gt_template = defaultdict(list)
        class_preds = []
        total_gt_count = 0

        for i, img_id in enumerate(image_ids):
            gt_boxes = gt_annos[i]['bbox']
            gt_names = gt_annos[i]['name']
            pred_boxes = pred_annos[i]['bbox']
            pred_names = pred_annos[i]['name']
            pred_scores = pred_annos[i]['score']

            # Get GT for this class
            gt_for_img = [box for box, name in zip(gt_boxes, gt_names) if name == cls]
            class_gt_template[img_id] = {'boxes': np.array(gt_for_img), 'matched': np.zeros(len(gt_for_img))}
            total_gt_count += len(gt_for_img)

            # Get predictions for this class
            for box, name, score in zip(pred_boxes, pred_names, pred_scores):
                if name == cls:
                    class_preds.append({'img_id': img_id, 'box': box, 'score': score})

        # Sort predictions by confidence score
        class_preds.sort(key=lambda x: x['score'], reverse=True)

        if total_gt_count == 0:
            for iou_threshold in iou_thresholds:
                results[cls][f'AP@{iou_threshold:.1f}'] = 0.0
            continue

        for iou_threshold in iou_thresholds:
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            class_gt = copy.deepcopy(class_gt_template) # Reset matched status for each IoU run

            # Match predictions to ground truths
            for i, pred in enumerate(class_preds):
                gt_data = class_gt[pred['img_id']]
                gt_boxes = gt_data['boxes']
                
                best_iou = 0
                best_gt_idx = -1

                if gt_boxes.shape[0] > 0:
                    # Calculate IoU with all GT boxes in the image
                    ious = np.array([calculate_iou_2d(pred['box'], gt_box) for gt_box in gt_boxes])
                    best_iou = np.max(ious)
                    best_gt_idx = np.argmax(ious)

                # Only attempt to match if we actually have at least one GT box
                if best_gt_idx != -1 and best_iou >= iou_threshold:
                    # Count as TP only if this GT box hasn't been matched yet
                    if not gt_data['matched'][best_gt_idx]:
                        tp[i] = 1
                        gt_data['matched'][best_gt_idx] = True
                    else:  # Duplicate detection for the same GT box
                        fp[i] = 1
                else:  # No GT boxes in image or IoU below threshold
                    fp[i] = 1

            # Calculate precision and recall
            fp_cum = np.cumsum(fp)
            tp_cum = np.cumsum(tp)
            
            recall = tp_cum / total_gt_count
            precision = tp_cum / (tp_cum + fp_cum + 1e-8)

            # Calculate Average Precision (AP) using 11-point interpolation
            ap = 0.0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                ap += p / 11.0
            
            results[cls][f'AP@{iou_threshold:.1f}'] = ap * 100.0

    return results


if __name__ == '__main__':
    #KITTI
    #prediction_path = "/REDACTED_PATH/outputs/data/"
    #prediction_path = "/REDACTED_PATH/monosowa_kitti_labels_cars_060_v2/"
    prediction_path = "/REDACTED_PATH/kitti_labels_codetr2/"
    #prediction_path = "/REDACTED_PATH/kitti_pseudo_nofilt/training/label_2/"
    gt_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
    #prediction_path = gt_path
    val_idxs_path = "/REDACTED_PATH/KITTI/ImageSets/train.txt"
    
    # --- Loading Data ---
    print("==> Loading image IDs, GTs, and Detections...")
    with open(val_idxs_path, 'r') as f:
        image_ids = [int(line.strip()) for line in f.readlines()]

    gt_annotations = get_annos_from_path(gt_path, image_ids)
    pred_annotations = get_annos_from_path(prediction_path, image_ids)
    print(f"Loaded {len(image_ids)} images.")

    # --- Evaluation ---
    print("\n==> Starting 2D Evaluation...")
    # Define the IoU thresholds for evaluation
    iou_thresholds_to_test = np.arange(0.0, 1.05, 0.1)
    evaluation_results = evaluate_2d(image_ids, gt_annotations, pred_annotations, 
                                     classes=['Car', 'Pedestrian', 'Van', 'Cyclist'], 
                                     iou_thresholds=iou_thresholds_to_test)

    # --- Display Results ---
    print("\n--- 2D Evaluation Results (AP in %) ---")
    for cls, results_per_iou in evaluation_results.items():
        print(f"Class: {cls}")
        for ap_name, ap_value in results_per_iou.items():
            print(f"  {ap_name}: {ap_value:.2f}")
    print("--------------------------------------\n")

