# This script processes KITTI complete sequences using Co-DETR model to generate:
# - Compressed car instance masks (zstd) for detection scores above thresholds (0.4 and 0.5)
# - Pedestrian masks with cyclist flags based on bicycle overlap
# - Visualization images with bounding boxes and labels for cars (red) and pedestrians (green/yellow for cyclists)
# Outputs are organized by threshold, folder, and subfolder to match reference patterns.
# No KITTI label files are generated; focuses on mask and image outputs for further processing.

import os
import glob
import random
import time
import tqdm
import numpy as np
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector

"""Generate Co-DETR mask + visualization outputs (car masks + images) for KITTI complete_sequences.

Matches the reference output pattern:
    masks_raw_cars_<THR>/folder/subfolder/<frame>.zstd   (compressed car instance masks above threshold)
    masks_raw_image_<THR>/folder/subfolder/<frame>.png   (visualization image)

Enhancements:
    - Also detects Pedestrians; Cyclists are NOT a separate class. A Pedestrian overlapping a bicycle (IoU>=0.1) is considered cyclist=True.
    - Pedestrians are drawn (green) with label: Ped <score> C=<0|1>.
    - Cars drawn (red) with Car <score>.
    - Only thresholds 0.4 and 0.5 are processed (040 / 050).

No KITTI text label files are written.
"""

# --- Configuration (edit if needed) ---
config_file = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_instance.py'
checkpoint_file = 'pytorch_model.pth'
dataset_path = '/path/to/datasets/KITTI/complete_sequences/'
output_path_global = '/path/to/output/monosowa_kitti_3/'  # base output dir (redacted)

# Thresholds (0.4 and 0.5)
SCORE_THRESHOLDS = [0.4, 0.5]

# Deterministic-ish shuffle with time+PID seed (like reference) for folders/subfolders/images
def shuffle_with_seed(items):
        unique_seed = int(time.time() * 1000) + os.getpid()
        random.seed(unique_seed)
        random.shuffle(items)
        return items

# COCO class indices
CAR_CLASS_IDX = 2
PERSON_CLASS_IDX = 0
BICYCLE_CLASS_IDX = 1  # COCO 'bicycle'
CYCLIST_IOU_THRESHOLD = 0.1  # IoU between person and bicycle to mark cyclist boolean

# --- Initialization ---
print("Initializing detector...")
try:
    # For newer MMDetection versions
    from mmdet.core import DatasetEnum
    model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')
except ImportError:
    # For older MMDetection versions
    print("Warning: DatasetEnum not found or needed. Initializing model without it.")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
print("Detector initialized.")

all_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print(f"Found {len(all_folders)} folders in {dataset_path}")

for folder in shuffle_with_seed(all_folders):
    folder_path = os.path.join(dataset_path, folder)
    print(f"\nProcessing Folder: {folder}")
    subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    print(f" Found {len(subfolders)} subfolders.")

    for subfolder in shuffle_with_seed(subfolders):
        subfolder_path = os.path.join(folder_path, subfolder)
        print(f" Processing Subfolder: {subfolder}")
        path_to_imgs = os.path.join(subfolder_path, 'image_02', 'data')
        if not os.path.isdir(path_to_imgs):
            print(f"  Warning: Image directory not found: {path_to_imgs}")
            continue
        image_paths = sorted(glob.glob(os.path.join(path_to_imgs, '*.png')))
        print(f"  Found {len(image_paths)} images.")
        if not image_paths:
            continue
        shuffled_image_paths = shuffle_with_seed(list(image_paths))

        for image_path in tqdm.tqdm(shuffled_image_paths, desc=f"Processing {folder}/{subfolder}", unit="image"):
            base_filename = os.path.basename(image_path)
            base_filename_mask = base_filename.replace('.png', '.zstd')

            # Pre-check: skip if ALL required outputs exist across thresholds
            all_exist = True
            for t in SCORE_THRESHOLDS:
                t_str = f"{int(t*100):03d}"
                out_cars_dir = os.path.join(output_path_global, f"masks_raw_cars_{t_str}", folder, subfolder)
                out_img_dir = os.path.join(output_path_global, f"masks_raw_image_{t_str}", folder, subfolder)
                if not (os.path.exists(os.path.join(out_cars_dir, base_filename_mask)) and os.path.exists(os.path.join(out_img_dir, base_filename))):
                    all_exist = False
                    break
            if all_exist:
                continue

            # Inference
            try:
                result = inference_detector(model, image_path)
            except Exception as e:
                print(f"\nError during inference for {image_path}: {e}")
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
            except Exception as e:
                print(f"\nError parsing inference results for {image_path}: {e}")
                continue

            if len(cars_bboxes) != len(cars_masks):
                print(f"\nWarning: Car bbox/mask length mismatch for {image_path}. Skipping cars.")
                cars_bboxes = np.empty((0,5))
                cars_masks = []
            if len(persons_bboxes) != len(persons_masks):
                print(f"\nWarning: Person bbox/mask length mismatch for {image_path}. Skipping persons masks save.")
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

            # Re-use bicycles for cyclist determination per threshold

            for thr in SCORE_THRESHOLDS:
                t_str = f"{int(thr*100):03d}"
                out_cars_dir = os.path.join(output_path_global, f"masks_raw_cars_{t_str}", folder, subfolder)
                out_peds_dir = os.path.join(output_path_global, f"masks_raw_pedestrians_{t_str}", folder, subfolder)
                out_img_dir = os.path.join(output_path_global, f"masks_raw_image_{t_str}", folder, subfolder)
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
                filtered_car_indices = [i for i, b in enumerate(cars_bboxes) if b[4] > thr]
                filtered_car_masks = [cars_masks[i] for i in filtered_car_indices]
                filtered_car_bboxes = [cars_bboxes[i] for i in filtered_car_indices]

                # Pedestrians + cyclist flag (also gather masks)
                bicycle_active = [b for b in bicycles_bboxes if b[4] > thr]
                ped_entries = []  # (bbox, score, cyclist_bool, mask)
                for idx, bbox in enumerate(persons_bboxes):
                    if bbox[4] > thr:
                        is_cyclist = any(iou(bbox, b) >= CYCLIST_IOU_THRESHOLD for b in bicycle_active)
                        mask = persons_masks[idx] if idx < len(persons_masks) else None
                        ped_entries.append((bbox, float(bbox[4]), bool(is_cyclist), mask))

                if need_masks:
                    # Save car masks AND scores
                    try:
                        import pickle, zstd
                        processed_masks = []
                        for m in filtered_car_masks:
                            try:
                                processed_masks.append(np.asarray(m).transpose())
                            except Exception:
                                processed_masks.append(np.asarray(m))
                        masks_array = np.array(processed_masks) if processed_masks else np.array([])
                        
                        # Extract scores
                        scores_list = [b[4] for b in filtered_car_bboxes]
                        scores_array = np.array(scores_list, dtype=np.float32)

                        payload = {
                            'masks': masks_array,
                            'scores': scores_array
                        }

                        blob = zstd.compress(pickle.dumps(payload, pickle.HIGHEST_PROTOCOL))
                        with open(masks_out_file, 'wb') as f:
                            f.write(blob)
                    except Exception as e:
                        print(f"\nError saving masks {masks_out_file}: {e}")

                if need_peds:
                    try:
                        import pickle, zstd
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
                        with open(peds_out_file, 'wb') as f:
                            f.write(blob)
                    except Exception as e:
                        print(f"\nError saving pedestrian data {peds_out_file}: {e}")

                if need_img:
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            raise RuntimeError('cv2.imread returned None')
                        # Draw cars
                        for bbox in filtered_car_bboxes:
                            x1,y1,x2,y2,score = bbox
                            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
                            label = f"Car {score:.2f}"
                            cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        # Draw pedestrians
                        for bbox, score, cyc, _ in ped_entries:
                            x1,y1,x2,y2,_ = bbox
                            color = (0,255,0) if not cyc else (0,255,255)  # cyclist highlighted yellow-ish
                            cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
                            label = f"Ped {score:.2f} C={1 if cyc else 0}"
                            cv2.putText(img, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        cv2.imwrite(img_out_file, img)
                    except Exception as e:
                        print(f"\nError saving image {img_out_file}: {e}")

print("\nProcessing finished.")
