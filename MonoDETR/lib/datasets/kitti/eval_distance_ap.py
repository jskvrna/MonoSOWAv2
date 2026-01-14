import os
import numpy as np
import kitti_eval_python.kitti_common as kitti
from kitti_eval_python.eval import calculate_iou_partly

def compute_ap_r40(precisions, recalls):
    """
    Compute Average Precision using R40 method (average of precisions at 40 recall positions).
    """
    ap = 0
    # R40 recall positions: 1/40, 2/40, ..., 1.0
    # Actually KITTI R40 uses 0, 1/40, 2/40 ... 1.0 (41 points)
    # Ref: https://github.com/traveller59/second.pytorch/blob/master/second/utils/eval.py
    
    # Smooth precision curve (monotonically decreasing)
    # But usually we compute max precision for recall >= r
    
    # 41 recall points [0, 1/40, ... , 1]
    recall_points = np.linspace(0, 1, 41)
    
    prec_at_recalls = []
    
    # Precompute max precisions for efficiency if needed, but simple loop is fine
    # For each recall point r, find max precision where recall >= r
    for r in recall_points:
        # masks where recall >= r
        mask = recalls >= r
        if np.any(mask):
            max_p = np.max(precisions[mask])
        else:
            max_p = 0.0
        prec_at_recalls.append(max_p)
        
    ap = np.mean(prec_at_recalls)
    return ap

def filter_annos_by_distance(annos, min_dist, max_dist, class_name):
    """
    Filter annotations (GT or Det) by distance and class name.
    Ignores Easy/Mod/Hard difficulty settings (occlusion/truncation/height).
    
    Args:
        annos: list of dicts (from kitti.get_label_annos)
        min_dist: float
        max_dist: float
        class_name: str (e.g., 'Car')
        
    Returns:
        filtered_annos: list of dicts, same length as input, containing only valid objects
    """
    filtered_annos = []
    
    for anno in annos:
        if anno is None or 'name' not in anno:
            filtered_annos.append({'name': np.array([]), 'bbox': np.zeros((0, 4)), 'location': np.zeros((0, 3)), 
                                  'dimensions': np.zeros((0, 3)), 'rotation_y': np.zeros((0,)), 'alpha': np.zeros((0,)),
                                  'score': np.zeros((0,))})
            continue

        names = anno['name']
        locations = anno['location']
        
        if len(names) == 0:
             filtered_annos.append(anno) # Empty, keep structure
             continue
             
        # Distance calculation
        x = locations[:, 0]
        z = locations[:, 2]
        dists = np.sqrt(x**2 + z**2)
        
        # Filter mask
        # 1. Class match
        # 2. Distance in range
        class_mask = (names == class_name)
        dist_mask = (dists >= min_dist) & (dists < max_dist)
        
        keep_mask = class_mask & dist_mask
        
        # Create new dict with filtered arrays
        new_anno = {}
        for key, val in anno.items():
            if isinstance(val, np.ndarray) and len(val) == len(keep_mask):
                new_anno[key] = val[keep_mask]
            else:
                new_anno[key] = val
        
        filtered_annos.append(new_anno)
        
    return filtered_annos

def eval_distance_step(
    gt_annos,
    dt_annos,
    min_dist,
    max_dist,
    class_name,
    iou_thresh,
    metric,
):
    """
    Evaluate AP for a specific distance range.
    """
    # 1. Filter data
    gt_filtered = filter_annos_by_distance(gt_annos, min_dist, max_dist, class_name)
    dt_filtered = filter_annos_by_distance(dt_annos, min_dist, max_dist, class_name)
    
    # 2. Count total GTs
    total_gt = sum([len(a['name']) for a in gt_filtered])
    
    if total_gt == 0:
        return 0.0, 0 # AP, num_gt
        
    # 3. Compute IoU for all images
    # metric: 0=bbox (2D), 1=bev, 2=3d
    # calculate_iou_partly returns list of Overlap Matrices (N_gt, N_dt) ? Or (N_dt, N_gt)?
    # Let's check kitti_eval_python logic or test. 
    # Usually underlying IoU returns (N, M). 
    # calculate_iou_partly calls image_box_overlap(gt, dt) which usually returns (N, M).
    # But wait, clean_data usually strips 'DontCare'. I should probably strip DontCares or handle them.
    # In my simplified logic, I'm only keeping 'Car' in range. So DontCare is ignored.
    # This is "Hard" evaluation (ignore nothing else).
    
    # Be careful: kitti_eval_python might expect numeric inputs for some fields.
    
    # If num_parts is default (50), it returns overlaps correctly split.
    num_parts = 50
    
    ret = calculate_iou_partly(gt_filtered, dt_filtered, metric=metric, num_parts=num_parts)
    overlaps_list = ret[0]
    
    # 4. Match and collection TP/FP
    # We need a list of all detections: (score, is_tp)
    all_dt_results = [] # list of (score, 1/0)
    
    for i in range(len(gt_filtered)):
        gt_anno = gt_filtered[i]
        dt_anno = dt_filtered[i]
        overlap = overlaps_list[i] # Shape (N_gt, N_dt) ? Or (N_dt, N_gt)? 
        # kitti_common usually returns (N, M) where N is boxes1, M is boxes2.
        # code: image_box_overlap(gt_boxes, dt_boxes) -> (N_gt, N_dt)
        
        num_gt = len(gt_anno['name'])
        num_dt = len(dt_anno['name'])
        
        if num_dt == 0:
            continue
            
        scores = dt_anno['score']
        
        # Sort/Match logic
        # Sort detections by score descending
        sorted_dt_indices = np.argsort(scores)[::-1]
        
        gt_detected = np.zeros(num_gt, dtype=bool)
        
        for dt_idx in sorted_dt_indices:
            score = scores[dt_idx]
            is_tp = 0
            
            if num_gt > 0:
                # Find best matching GT for this detection
                # overlap is (N_gt, N_dt)
                # We want overlap result for this dt: which is overlap[:, dt_idx]
                
                iou_vals = overlap[:, dt_idx]
                best_gt_idx = np.argmax(iou_vals)
                best_iou = iou_vals[best_gt_idx]
                
                if best_iou >= iou_thresh:
                    if not gt_detected[best_gt_idx]:
                        is_tp = 1
                        gt_detected[best_gt_idx] = True
                    else:
                        # Already matched this GT with a higher score detection -> FP (duplicate)
                        is_tp = 0
                else:
                    is_tp = 0
            
            all_dt_results.append((score, is_tp))
            
    # 5. Compute AP
    if not all_dt_results:
        return 0.0, total_gt
        
    # Sort all results by score
    all_dt_results.sort(key=lambda x: x[0], reverse=True)
    all_dt_results = np.array(all_dt_results)
    
    tp_list = all_dt_results[:, 1]
    fp_list = 1 - tp_list
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    ap = compute_ap_r40(precisions, recalls)
    
    return ap, total_gt

def main():
    # Configuration
    # Adjust paths as needed
    prediction_path = "/REDACTED_PATH/labels_kitti_codetr_tfl_pseudo_newclass_scores/"
    #prediction_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
    gt_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
    # val_idxs_path = "/REDACTED_PATH/KITTI/ImageSets/train.txt"
    val_idxs_path = "/REDACTED_PATH/pseudo_label_generator/train.txt"

    class_name = 'Car'

    # Metrics: 0=bbox (2D), 1=bev, 2=3d
    metrics = [
        ("bbox", 0),
        ("bev", 1),
        ("3d", 2),
    ]

    # IoU thresholds per metric (defaulting to 0.5 everywhere as requested)
    iou_thresholds = {
        "bbox": 0.5,
        "bev": 0.5,
        "3d": 0.5,
    }
    
    print(f"Loading data from {prediction_path} and {gt_path}...")
    
    # Load Image IDs
    with open(val_idxs_path, 'r') as f:
        idx_list = [x.strip() for x in f.readlines()]
    img_ids = [int(id) for id in idx_list]
    
    # Load Annos
    dt_annos = kitti.get_label_annos(prediction_path, img_ids)
    gt_annos = kitti.get_label_annos(gt_path, img_ids)
    
    print(f"Loaded {len(dt_annos)} dt annos and {len(gt_annos)} gt annos.")
    
    ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]

    for metric_name, metric_id in metrics:
        min_iou = iou_thresholds[metric_name]
        print(f"\nEvaluating AP ({metric_name.upper()}, IoU={min_iou}) for class '{class_name}' by distance:")
        print(f"{'Dist Range':<15} | {'AP (R40)':<10} | {'Num GT':<10}")
        print("-" * 45)

        for min_d, max_d in ranges:
            ap, num_gt = eval_distance_step(
                gt_annos,
                dt_annos,
                min_d,
                max_d,
                class_name,
                iou_thresh=min_iou,
                metric=metric_id,
            )
            print(f"{min_d:02d}m - {max_d:03d}m    | {ap*100:8.4f} % | {num_gt}")

if __name__ == '__main__':
    main()
