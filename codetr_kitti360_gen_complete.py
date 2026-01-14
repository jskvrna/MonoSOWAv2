import os
import glob
import random
import time
import tqdm
import numpy as np
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector

"""Generate Co-DETR mask + visualization outputs (car + pedestrian masks/images) for KITTI-360 sequences.

Matches the reference output pattern:
    masks_raw_cars_<THR>/<sequence_parts>/<frame>.zstd   (compressed car instance masks above threshold)
    masks_raw_pedestrians_<THR>/<sequence_parts>/<frame>.zstd   (compressed pedestrian masks + cyclist flags)
    masks_raw_image_<THR>/<sequence_parts>/<frame>.png   (visualization image)

Enhancements vs. KITTI complete generator:
    - Automatically detects valid image_00 camera directories (data*, data_rect*, etc.).
    - Supports multi-level KITTI-360 hierarchy by mirroring the relative sequence path in outputs.
    - Preserves the same handling of cars (red) and pedestrians/cyclists (green/yellow) with two score thresholds.
"""

# --- Configuration (edit if needed) ---
config_file = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py'
checkpoint_file = 'pytorch_model.pth'
dataset_path = '/path/to/KITTI/'  # KITTI-360 base directory (redacted)
output_path_global = '/path/to/output/monosowa_k360_2/'  # base output dir for KITTI-360 (redacted)

# Thresholds (0.4 and 0.5)
SCORE_THRESHOLDS = [0.4, 0.5]

# Candidate camera/image directory patterns to probe under each sequence (image_00 only)
CAMERA_DIR_CANDIDATES = [
    ('image_00', 'data'),
    ('image_00', 'data_rect'),
    ('image_00', 'data_rgb'),
    ('image_00', 'data_ego'),
]

# Deterministic-ish shuffle with time+PID seed for sequences/images
def shuffle_with_seed(items):
    items = list(items)
    unique_seed = int(time.time() * 1000) + os.getpid()
    random.seed(unique_seed)
    random.shuffle(items)
    return items


def resolve_image_directory(sequence_path):
    """Return the directory containing RGB frames for a KITTI-360 sequence."""
    for relative_parts in CAMERA_DIR_CANDIDATES:
        candidate = os.path.join(sequence_path, *relative_parts)
        if os.path.isdir(candidate):
            return candidate

    # fallback: first directory matching image_00/data*
    fallback = sorted(glob.glob(os.path.join(sequence_path, 'image_00', 'data*')))
    for candidate in fallback:
        if os.path.isdir(candidate):
            return candidate
    return None


def collect_sequence_roots(root_path):
    """Gather all sequence roots that contain camera imagery."""
    sequence_roots = []
    if not os.path.isdir(root_path):
        return []
    top_level_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    for top in top_level_dirs:
        top_path = os.path.join(root_path, top)
        img_dir = resolve_image_directory(top_path)
        if img_dir:
            sequence_roots.append(top_path)
            continue

        # One more level deep (e.g., original KITTI date -> drive structure)
        sub_dirs = [d for d in os.listdir(top_path) if os.path.isdir(os.path.join(top_path, d))]
        for sub in sub_dirs:
            sub_path = os.path.join(top_path, sub)
            img_dir = resolve_image_directory(sub_path)
            if img_dir:
                sequence_roots.append(sub_path)

    return sequence_roots


# COCO class indices
CAR_CLASS_IDX = 2
PERSON_CLASS_IDX = 0
BICYCLE_CLASS_IDX = 1  # COCO 'bicycle'
CYCLIST_IOU_THRESHOLD = 0.1  # IoU between person and bicycle to mark cyclist boolean


# --- Initialization ---
print("Initializing detector...")
try:
    from mmdet.core import DatasetEnum
    model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')
except ImportError:
    print("Warning: DatasetEnum not found or needed. Initializing model without it.")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("Detector initialized.")

sequence_roots = collect_sequence_roots(dataset_path)
if not sequence_roots:
    print(f"Warning: No KITTI-360 sequences with imagery found under {dataset_path}")
else:
    print(f"Found {len(sequence_roots)} sequence roots in {dataset_path}")

for sequence_root in shuffle_with_seed(sequence_roots):
    relative_parts = os.path.relpath(sequence_root, dataset_path).split(os.sep)
    relative_parts = [part for part in relative_parts if part not in ('', '.')]  # sanity cleanup
    sequence_label = '/'.join(relative_parts)
    image_dir = resolve_image_directory(sequence_root)
    if not image_dir:
        print(f"\nWarning: No image directory for sequence {sequence_label}. Skipping.")
        continue

    print(f"\nProcessing Sequence: {sequence_label}")
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    print(f" Found {len(image_paths)} images in {image_dir}.")
    if not image_paths:
        continue

    shuffled_image_paths = shuffle_with_seed(image_paths)

    for image_path in tqdm.tqdm(shuffled_image_paths, desc=f"Processing {sequence_label}", unit="image"):
        base_filename = os.path.basename(image_path)
        base_filename_mask = base_filename.replace('.png', '.zstd')

        # Pre-check: skip if ALL required outputs exist across thresholds
        all_exist = True
        for thr in SCORE_THRESHOLDS:
            t_str = f"{int(thr * 100):03d}"
            out_parts = [output_path_global, f"masks_raw_cars_{t_str}"] + relative_parts
            out_cars_dir = os.path.join(*out_parts)
            out_img_dir = os.path.join(output_path_global, f"masks_raw_image_{t_str}", *relative_parts)
            out_peds_dir = os.path.join(output_path_global, f"masks_raw_pedestrians_{t_str}", *relative_parts)
            
            if not (
                os.path.exists(os.path.join(out_cars_dir, base_filename_mask))
                and os.path.exists(os.path.join(out_img_dir, base_filename))
                and os.path.exists(os.path.join(out_peds_dir, base_filename_mask))
            ):
                all_exist = False
                break
        if all_exist:
            continue

        # Inference
        try:
            result = inference_detector(model, image_path)
        except Exception as exc:
            print(f"\nError during inference for {image_path}: {exc}")
            continue

        # Parse
        try:
            if not isinstance(result, tuple) or len(result) != 2:
                print(f"\nWarning: Unexpected result structure for {image_path}. Skipping.")
                continue
            bbox_results, segm_results = result
            num_classes = len(bbox_results)
            cars_bboxes = bbox_results[CAR_CLASS_IDX] if CAR_CLASS_IDX < num_classes else []
            cars_masks = segm_results[0][CAR_CLASS_IDX] if CAR_CLASS_IDX < num_classes else []
            persons_bboxes = bbox_results[PERSON_CLASS_IDX] if PERSON_CLASS_IDX < num_classes else []
            persons_masks = segm_results[0][PERSON_CLASS_IDX] if PERSON_CLASS_IDX < num_classes else []
            bicycles_bboxes = bbox_results[BICYCLE_CLASS_IDX] if BICYCLE_CLASS_IDX < num_classes else []
            bicycles_masks = segm_results[0][BICYCLE_CLASS_IDX] if BICYCLE_CLASS_IDX < num_classes else []
        except Exception as exc:
            print(f"\nError parsing inference results for {image_path}: {exc}")
            continue

        if len(cars_bboxes) != len(cars_masks):
            print(f"\nWarning: Car bbox/mask length mismatch for {image_path}. Skipping cars.")
            cars_bboxes = np.empty((0, 5))
            cars_masks = []
        if len(persons_bboxes) != len(persons_masks):
            print(f"\nWarning: Person bbox/mask length mismatch for {image_path}. Pedestrian masks disabled.")
            persons_masks = []

        def iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a[:4]
            bx1, by1, bx2, by2 = box_b[:4]
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter_area / max(area_a + area_b - inter_area, 1e-6)

        for thr in SCORE_THRESHOLDS:
            t_str = f"{int(thr * 100):03d}"
            out_cars_dir = os.path.join(output_path_global, f"masks_raw_cars_{t_str}", *relative_parts)
            out_peds_dir = os.path.join(output_path_global, f"masks_raw_pedestrians_{t_str}", *relative_parts)
            out_img_dir = os.path.join(output_path_global, f"masks_raw_image_{t_str}", *relative_parts)
            os.makedirs(out_cars_dir, exist_ok=True)
            os.makedirs(out_peds_dir, exist_ok=True)
            os.makedirs(out_img_dir, exist_ok=True)
            masks_out_file = os.path.join(out_cars_dir, base_filename_mask)
            peds_out_file = os.path.join(out_peds_dir, base_filename_mask)
            img_out_file = os.path.join(out_img_dir, base_filename)

            need_masks = not os.path.exists(masks_out_file)
            need_peds = not os.path.exists(peds_out_file)
            need_img = not os.path.exists(img_out_file)
            if not (need_masks or need_peds or need_img):
                continue

            # Filter cars for this threshold
            filtered_car_indices = [idx for idx, bbox in enumerate(cars_bboxes) if bbox[4] > thr]
            filtered_car_masks = [cars_masks[idx] for idx in filtered_car_indices]
            filtered_car_bboxes = [cars_bboxes[idx] for idx in filtered_car_indices]

            # Pedestrians + cyclist flag
            bicycle_active = [b for b in bicycles_bboxes if b[4] > thr]
            ped_entries = []  # (bbox, score, cyclist_bool, mask)
            for idx, bbox in enumerate(persons_bboxes):
                if bbox[4] > thr:
                    is_cyclist = any(iou(bbox, bike_bbox) >= CYCLIST_IOU_THRESHOLD for bike_bbox in bicycle_active)
                    mask = persons_masks[idx] if idx < len(persons_masks) else None
                    ped_entries.append((bbox, float(bbox[4]), bool(is_cyclist), mask))

            if need_masks:
                try:
                    import pickle
                    import zstd
                    processed_masks = []
                    for mask in filtered_car_masks:
                        try:
                            processed_masks.append(np.asarray(mask).transpose())
                        except Exception:
                            processed_masks.append(np.asarray(mask))
                    masks_array = np.array(processed_masks) if processed_masks else np.array([])
                    
                    # Extract scores
                    scores_list = [b[4] for b in filtered_car_bboxes]
                    scores_array = np.array(scores_list, dtype=np.float32)

                    payload = {
                        'masks': masks_array,
                        'scores': scores_array
                    }

                    blob = zstd.compress(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))
                    with open(masks_out_file, 'wb') as file_obj:
                        file_obj.write(blob)
                except Exception as exc:
                    print(f"\nError saving car masks {masks_out_file}: {exc}")

            if need_peds:
                try:
                    import pickle
                    import zstd
                    ped_masks_processed = []
                    cyclist_flags = []
                    ped_scores = []
                    for _bbox, _score, cyc, mask in ped_entries:
                        if mask is None:
                            continue
                        try:
                            ped_masks_processed.append(np.asarray(mask).transpose())
                        except Exception:
                            ped_masks_processed.append(np.asarray(mask))
                        cyclist_flags.append(1 if cyc else 0)
                        ped_scores.append(_score)

                    payload = {
                        'masks': np.array(ped_masks_processed) if ped_masks_processed else np.array([]),
                        'flags': np.array(cyclist_flags, dtype=np.uint8),
                        'scores': np.array(ped_scores, dtype=np.float32)
                    }
                    blob = zstd.compress(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))
                    with open(peds_out_file, 'wb') as file_obj:
                        file_obj.write(blob)
                except Exception as exc:
                    print(f"\nError saving pedestrian data {peds_out_file}: {exc}")

            if need_img:
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        raise RuntimeError('cv2.imread returned None')
                    # Draw cars
                    for bbox in filtered_car_bboxes:
                        x1, y1, x2, y2, score = bbox
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f"Car {score:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # Draw pedestrians
                    for bbox, score, cyc, _ in ped_entries:
                        x1, y1, x2, y2, _ = bbox
                        color = (0, 255, 0) if not cyc else (0, 255, 255)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"Ped {score:.2f} C={1 if cyc else 0}"
                        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.imwrite(img_out_file, img)
                except Exception as exc:
                    print(f"\nError saving image {img_out_file}: {exc}")

print("\nProcessing finished.")
