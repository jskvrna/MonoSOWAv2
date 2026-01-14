from lib.datasets.kitti.kitti_eval_python import kitti_common as kitti
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result, get_distance_eval_result
import os
import datetime
import numpy as np
from typing import Union

# -----------------------
# nuScenes-like metrics
# -----------------------

# Sanity-check configuration: apply constant metric offsets across all matches
SANITY_CHECK = False
SANITY_OFFSETS = {
    'dx': 1.0,           # meters added on x (ground plane)
    'dz': 0.5,           # meters added on z (ground plane)
    'dtheta_deg': 10.0,  # degrees added to yaw
    'ddims': [0.2, 0.1, 0.15],  # meters added to [l, w, h]
    'mode': 'gt_plus_offset',   # ensures constant ATE/AOE/ADE across pairs
}

def _angle_diff_deg(a, b):
    """Smallest absolute difference between two angles in radians, output in degrees.

    Args:
        a: float or np.array [rad]
        b: float or np.array [rad]
    Returns:
        abs angle difference in degrees in [0, 180].
    """
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    d = np.abs(d)
    # map from [0, pi] already, but numerical stability
    d = np.minimum(d, np.pi)
    return np.degrees(d)


def _center_distance_xy(p, q):
    """Euclidean distance on the ground plane (x,z) between KITTI camera coords.
    p, q: arrays like [x, y, z]
    Returns: float distance in meters using (x, z).
    """
    dx = float(p[0] - q[0])
    dz = float(p[2] - q[2])
    return float(np.hypot(dx, dz))


def _match_dets_to_gts_single_image(dt, gt, class_name, dist_thr):
    """Greedy one-to-one matching by center distance threshold on ground plane.

    dt, gt: annotation dicts for a single image from kitti.get_label_anno
    class_name: str
    dist_thr: float meters

    Returns:
        matches: list of tuples (dt_idx, gt_idx)
        unmatched_dt: list of dt idx
        unmatched_gt: list of gt idx
    """
    dt_mask = (dt['name'] == class_name)
    gt_mask = (gt['name'] == class_name)
    dt_idx = np.where(dt_mask)[0]
    gt_idx = np.where(gt_mask)[0]
    if dt_idx.size == 0 or gt_idx.size == 0:
        return [], list(dt_idx.tolist()), list(gt_idx.tolist())

    # Sort detections by score desc for stable matching similar to common evals
    scores = dt['score'][dt_idx] if 'score' in dt else np.ones(len(dt_idx))
    order = np.argsort(-scores)
    dt_idx = dt_idx[order]

    used_gt = set()
    matches = []
    for di in dt_idx:
        dloc = dt['location'][di]
        # find closest available gt within thr
        best_gi = -1
        best_dist = float('inf')
        for gi in gt_idx:
            if gi in used_gt:
                continue
            gloc = gt['location'][gi]
            dist = _center_distance_xy(dloc, gloc)
            if dist <= dist_thr and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            matches.append((di, best_gi))

    unmatched_dt = [di for di in dt_idx.tolist() if di not in [m[0] for m in matches]]
    unmatched_gt = [gi for gi in gt_idx.tolist() if gi not in [m[1] for m in matches]]
    return matches, unmatched_dt, unmatched_gt


def _compute_pair_metrics_2d_iou(dt_bbox, gt_bbox):
    """Compute 2D IoU for a pair of KITTI 2D boxes [ymin, xmin, ymax, xmax]."""
    # shape (1,1)
    iou_mat = kitti.iou(dt_bbox.reshape(1, 4), gt_bbox.reshape(1, 4), add1=False)
    return float(iou_mat[0, 0])


def eval_nusc_like(dt_annos, gt_annos, classes=('Car', 'Pedestrian', 'Cyclist'),
                   dist_thresholds=None, logger=None, sanity_print=5,
                   sanity_offsets: Union[dict, None] = None):
    """Compute nuScenes-inspired metrics AOE, ATE, ADE and 2D IoU on matched TPs.

    Matching: greedy one-to-one by center distance on ground plane (x,z).

        Args:
            dt_annos: list of annos dicts for each image (as from get_label_annos)
            gt_annos: list of annos dicts
            classes: tuple/list of class names to evaluate
            dist_thresholds: dict mapping class->meters; defaults: 5.0m for Car, Pedestrian, Cyclist
                    logger: optional callable for logging
                    sanity_print: deprecated; no-op.
                    sanity_offsets: optional dict to inject fixed errors into metrics calculation
                        without affecting matching. Keys:
                            - 'dx' (m), 'dz' (m): added to predicted center before computing ATE
                            - 'dtheta_deg' (deg): added to predicted rotation_y before AOE
                            - 'ddims' (m, list/array of [l,w,h]): added to predicted dimensions before ADE
                            - 'mode': 'add_to_pred' (default) or 'gt_plus_offset'
                                    'gt_plus_offset' uses pred := gt + offsets so AOE/ATE/ADE_abs are constant.

    Returns:
      summary: dict with per-class and overall metrics.
    """
    if dist_thresholds is None:
        dist_thresholds = {'Car': 5.0, 'Pedestrian': 5.0, 'Cyclist': 5.0}

    def log(msg):
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    metrics = {}
    overall_accums = {k: [] for k in ['aoe_deg', 'ate_m', 'ade_rel', 'ade_abs_m', 'iou2d']}
    overall_counts = {k: 0 for k in classes}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in classes:
        dthr = float(dist_thresholds.get(cls, 5.0))
        aoe_list = []
        ate_list = []
        ade_rel_list = []  # mean(|d_pred - d_gt| / d_gt) over (l,w,h)
        ade_abs_list = []  # mean absolute (meters)
        iou2d_list = []
        tp = 0
        fp = 0
        fn = 0
    # Note: sanity example collection removed

        # Precompute sanity offsets
        so = sanity_offsets or {}
        dx = float(so.get('dx', 0.0))
        dz = float(so.get('dz', 0.0))
        dtheta = np.radians(float(so.get('dtheta_deg', 0.0)))
        ddims = np.array(so.get('ddims', [0.0, 0.0, 0.0]), dtype=float).reshape(3,)
        mode = so.get('mode', 'add_to_pred')

        for img_idx in range(len(gt_annos)):
            gt = gt_annos[img_idx]
            dt = dt_annos[img_idx] if img_idx < len(dt_annos) else {'name': np.array([])}
            matches, u_dt, u_gt = _match_dets_to_gts_single_image(dt, gt, cls, dthr)
            tp += len(matches)
            fp += len(u_dt)
            fn += len(u_gt)
            for di, gi in matches:
                # Set synthetic prediction values for sanity offsets (metrics only)
                if sanity_offsets is None:
                    ry_pred = dt['rotation_y'][di]
                    loc_pred = dt['location'][di]
                    d_pred = dt['dimensions'][di]
                else:
                    if mode == 'gt_plus_offset':
                        ry_pred = gt['rotation_y'][gi] + dtheta
                        loc_pred = gt['location'][gi].copy()
                        loc_pred = np.array(loc_pred, dtype=float)
                        loc_pred[0] += dx
                        loc_pred[2] += dz
                        d_pred = gt['dimensions'][gi] + ddims
                    else:  # add_to_pred
                        ry_pred = dt['rotation_y'][di] + dtheta
                        loc_pred = dt['location'][di].copy()
                        loc_pred = np.array(loc_pred, dtype=float)
                        loc_pred[0] += dx
                        loc_pred[2] += dz
                        d_pred = dt['dimensions'][di] + ddims

                # Orientation error
                aoe = _angle_diff_deg(ry_pred, gt['rotation_y'][gi])
                # Translation error on ground plane
                ate = _center_distance_xy(loc_pred, gt['location'][gi])
                # Dimension error
                d_gt = gt['dimensions'][gi]
                abs_err = np.abs(d_pred - d_gt)
                rel_err = abs_err / np.maximum(np.abs(d_gt), 1e-6)
                ade_abs = float(np.mean(abs_err))
                ade_rel = float(np.mean(rel_err))
                # 2D IoU
                iou2d = _compute_pair_metrics_2d_iou(dt['bbox'][di], gt['bbox'][gi])

                # sanity bounds removed

                aoe_list.append(float(aoe))
                ate_list.append(float(ate))
                ade_rel_list.append(float(ade_rel))
                ade_abs_list.append(float(ade_abs))
                iou2d_list.append(float(iou2d))

                # sanity example logging removed

        # aggregate per class
        cls_metrics = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'AOE(deg)': float(np.mean(aoe_list)) if len(aoe_list) else float('nan'),
            'ATE(m)': float(np.mean(ate_list)) if len(ate_list) else float('nan'),
            'ADE_rel(mean(|d-p|/p))': float(np.mean(ade_rel_list)) if len(ade_rel_list) else float('nan'),
            'ADE_abs(m)': float(np.mean(ade_abs_list)) if len(ade_abs_list) else float('nan'),
            'IoU2D': float(np.mean(iou2d_list)) if len(iou2d_list) else float('nan'),
        }
        metrics[cls] = cls_metrics
        overall_counts[cls] = tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        # extend overall arrays for overall mean over all matches
        for v in aoe_list:
            overall_accums['aoe_deg'].append(v)
        for v in ate_list:
            overall_accums['ate_m'].append(v)
        for v in ade_rel_list:
            overall_accums['ade_rel'].append(v)
        for v in ade_abs_list:
            overall_accums['ade_abs_m'].append(v)
        for v in iou2d_list:
            overall_accums['iou2d'].append(v)

    # sanity example printing removed

    overall = {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'AOE(deg)': float(np.mean(overall_accums['aoe_deg'])) if overall_accums['aoe_deg'] else float('nan'),
        'ATE(m)': float(np.mean(overall_accums['ate_m'])) if overall_accums['ate_m'] else float('nan'),
        'ADE_rel(mean(|d-p|/p))': float(np.mean(overall_accums['ade_rel'])) if overall_accums['ade_rel'] else float('nan'),
        'ADE_abs(m)': float(np.mean(overall_accums['ade_abs_m'])) if overall_accums['ade_abs_m'] else float('nan'),
        'IoU2D': float(np.mean(overall_accums['iou2d'])) if overall_accums['iou2d'] else float('nan'),
    }

    return {'per_class': metrics, 'overall': overall}

def eval(results_dir, gt_dir, val_idxs_path, logger):
    print("==> Loading detections and GTs...")
    with open(val_idxs_path, 'r') as f:
        idx_list_lines = f.readlines()
        idx_list = [x.strip() for x in idx_list_lines]
    img_ids = [int(id) for id in idx_list]
    dt_annos = kitti.get_label_annos(results_dir, img_ids)
    gt_annos = kitti.get_label_annos(gt_dir, img_ids)

    test_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}

    print('==> Evaluating (official) ...')
    car_moderate = 0
    writelist= ['Car', 'Pedestrian']
    for category in writelist:
        results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
        if category == 'Car':
            car_moderate = mAP3d_R40
        print(results_str)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        nusc_summary = eval_nusc_like(
            dt_annos, gt_annos,
            classes=('Car', 'Pedestrian', 'Cyclist'),
            sanity_offsets=(SANITY_OFFSETS if SANITY_CHECK else None)
        )
        
        def fmt(v):
            if v != v:  # NaN check
                return 'nan'
            return f"{v:.4f}"

        nusc_lines = []
        nusc_lines.append('\n\n==> nuScenes-like metrics\n')
        for cls, m in nusc_summary['per_class'].items():
            nusc_lines.append(f"[nusc-like] {cls}: TP={m['TP']} FP={m['FP']} FN={m['FN']} | "
              f"AOE(deg)={fmt(m['AOE(deg)'])} ATE(m)={fmt(m['ATE(m)'])} "
              f"ADE_rel={fmt(m['ADE_rel(mean(|d-p|/p))'])} ADE_abs(m)={fmt(m['ADE_abs(m)'])} IoU2D={fmt(m['IoU2D'])}")
        o = nusc_summary['overall']
        nusc_lines.append(f"[nusc-like] OVERALL: TP={o['TP']} FP={o['FP']} FN={o['FN']} | "
              f"AOE(deg)={fmt(o['AOE(deg)'])} ATE(m)={fmt(o['ATE(m)'])} "
              f"ADE_rel={fmt(o['ADE_rel(mean(|d-p|/p))'])} ADE_abs(m)={fmt(o['ADE_abs(m)'])} IoU2D={fmt(o['IoU2D'])}")
        
        nusc_results_str = "\n".join(nusc_lines)

        with open('eval_' + category + '_' + timestamp + '.txt', 'w') as f:
            f.write(results_str)
            f.write(nusc_results_str)

        # Print once (first category) to avoid recomputing/duplicating later
        if category == writelist[0]:
            print('==> Evaluating (nuScenes-like metrics) ...')
            print(nusc_results_str)

    # Removed redundant post-loop nuScenes-like evaluation/print

    return car_moderate

if __name__ == "__main__":
    prediction_path2 = None

    #KITTI
    #prediction_path = "/REDACTED_PATH/outputs/data/"
    #prediction_path = "/REDACTED_PATH/monosowa_kitti_labels_cars_060_v2/"
    #prediction_path2 = "/REDACTED_PATH/labels_kitti_codetr/"
    #prediction_path = "/REDACTED_PATH/kitti_pseudo_nofilt/training/label_2/"
    gt_path = "/REDACTED_PATH/Public_datasets/KITTI/object_detection/training/label_2/"
    prediction_path = gt_path
    val_idxs_path = "/REDACTED_PATH/KITTI/ImageSets/train.txt"

    #KITTI-360
    #prediction_path = "/REDACTED_PATH/k360_05_100_load/training/label_2/"
    #gt_path = "/REDACTED_PATH/k360_05_100_load/training/labels_gt/"
    #val_idxs_path = "/REDACTED_PATH/k360_05_100_load/ImageSets/train.txt"

    #prediction_path = "/REDACTED_PATH/test_kitti/training/labels_pseudo/"
    #gt_path = "/REDACTED_PATH/test_kitti/training/labels_gt/"
    #val_idxs_path = "/REDACTED_PATH/test_kitti/ImageSets/train.txt"

    print(f"Evaluating predictions in folder: {prediction_path}")

    evaluator = eval(prediction_path, gt_path, val_idxs_path, None)

    if prediction_path2 is not None:
        evaluator2 = eval(prediction_path2, gt_path, val_idxs_path, None)