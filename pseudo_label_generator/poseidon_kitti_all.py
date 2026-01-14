#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KITTI Dataset Processing with Object Detection and Pose Estimation
-----------------------------------------------------------------
This script processes the KITTI dataset. It performs the following steps:
1. Iterates through sequences of images in the KITTI dataset.
2. Loads pre-computed instance segmentation masks for pedestrians in each image.
3. For each detected pedestrian, it uses a multi-frame pose estimator (Poseidon)
    to predict 2D keypoints.
4. Saves the detection results (mask, bbox) along with the predicted
    keypoints into compressed files for each image.
"""
import argparse
import glob
import os
import pickle
import random
import time

import cv2
import numpy as np
import torch
import yaml
import zstd
from torchvision import transforms
from tqdm import tqdm

from models.best.Poseidon import Poseidon

# ─────────────────────── Poseidon Utilities ───────────────────────

# Keypoint configuration from the original Poseidon script
USED_KP_IDX = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
USED_KP_COLORS = [
     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
     (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 128),
     (255, 165, 0), (128, 128, 0), (0, 128, 128), (75, 0, 130),
     (255, 105, 180), (0, 191, 255), (255, 223, 0), (165, 42, 42),
     (34, 139, 34)
]

def load_pose_cfg(path: str):
    data = yaml.safe_load(open(path, "r"))

    class C:
        pass
    cfg = C()
    cfg.SEED = data.get("SEED", 42)

    cfg.MODEL = C()
    cfg.MODEL.METHOD = data["MODEL"]["METHOD"]
    cfg.MODEL.NUM_JOINTS = data["MODEL"]["NUM_JOINTS"]
    cfg.MODEL.IMAGE_SIZE = tuple(data["MODEL"]["IMAGE_SIZE"])
    cfg.MODEL.CONFIG_FILE = data["MODEL"].get("CONFIG_FILE")
    cfg.MODEL.CHECKPOINT_FILE = data["MODEL"].get("CHECKPOINT_FILE")
    cfg.MODEL.EMBED_DIM = data["MODEL"].get("EMBED_DIM", 256)
    cfg.WINDOWS_SIZE = data["MODEL"].get("WINDOWS_SIZE", 5)
    cfg.MODEL.HEATMAP_SIZE = data["MODEL"].get("HEATMAP_SIZE", (96, 72))
    cfg.MODEL.FREEZE_WEIGHTS = data["MODEL"].get("FREEZE_WEIGHTS", False)

    cfg.DATASET = C()
    cfg.DATASET.BBOX_ENLARGE_FACTOR = data["DATASET"].get(
        "BBOX_ENLARGE_FACTOR", 1.25)
    return cfg

def preprocess_frame(frame, size):
     """Prepares a single image frame for the Poseidon model."""
     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     frame = cv2.resize(frame, size)
     tfm = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     return tfm(frame)

def extract_kps(heatmaps, h_crop, w_crop):
     """Extracts keypoint coordinates from the model's output heatmaps."""
     _, _, H, W = heatmaps.shape
     hm = heatmaps[0, USED_KP_IDX].view(len(USED_KP_IDX), -1)
     _, idx = hm.max(dim=1)
     ys = (idx // W).float() * (h_crop / H)
     xs = (idx % W).float() * (w_crop / W)
     return torch.stack([xs, ys], dim=1).cpu().numpy()

# ─────────────────────── Main Processing Logic ───────────────────────

def parse_args():
     p = argparse.ArgumentParser(description="KITTI Detection and Pose Estimation")
     # Dataset and paths
    p.add_argument('--dataset_path', default='/path/to/datasets/KITTI/complete_sequences/', help='Path to KITTI dataset')
     p.add_argument('--mask_path', required=True, help='Path to root directory of pre-computed masks')
    p.add_argument('--output_path', default='/path/to/output/kitti_pose_detections', help='Global output path')
     # Pose estimation args
     p.add_argument('--pose_config', required=True, help='Path to Poseidon YAML config')
     p.add_argument('--pose_weights', required=True, help='Path to Poseidon .pt checkpoint')
     p.add_argument('--window_size', type=int, default=5, help='Sliding window size for pose estimation')
     p.add_argument('--window_step', type=int, default=1, help='Frame stride for sliding window')
     # General args
     p.add_argument('--gpu', type=int, default=0, help='CUDA device index')
     return p.parse_args()

def save_results(data_to_save, output_filename):
     """Compresses and saves detection/pose data using zstd and pickle."""
     if os.path.exists(output_filename):
          return
     try:
          compressed_data = zstd.compress(pickle.dumps(data_to_save, pickle.HIGHEST_PROTOCOL))
          with open(output_filename, 'wb') as f:
                f.write(compressed_data)
     except Exception as e:
          print(f"Error saving file {output_filename}: {e}")

def main():
    args = parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Initialize Pose Estimator ---
    print("Initializing pose estimator...")
    pose_cfg = load_pose_cfg(args.pose_config)
    pose_model = Poseidon(pose_cfg, phase="test", device=device)
    ckpt = torch.load(args.pose_weights, map_location=device)
    pose_model.load_state_dict(ckpt["model_state_dict"])
    pose_model.to(device).eval()
    print("Pose estimator initialized successfully.")

    # --- Create Output Directories ---
    output_path_data = os.path.join(args.output_path, "pedestrian_data")
    output_path_viz = os.path.join(args.output_path, "visualization")
    os.makedirs(output_path_data, exist_ok=True)
    os.makedirs(output_path_viz, exist_ok=True)

    # --- Process Dataset ---
    date_folders = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    random.shuffle(date_folders)
    print(f"Found {len(date_folders)} date folders in {args.dataset_path}")

    for date_folder in date_folders:
        date_folder_path = os.path.join(args.dataset_path, date_folder)
        seq_folders = [d for d in os.listdir(date_folder_path) if os.path.isdir(os.path.join(date_folder_path, d))]
        random.shuffle(seq_folders)

        for seq_folder in seq_folders:
             print(f"\nProcessing Sequence: {date_folder}/{seq_folder}")
             img_dir = os.path.join(date_folder_path, seq_folder, 'image_02/data/')
             if not os.path.isdir(img_dir):
                 print(f"  Warning: Image directory not found: {img_dir}")
                 continue

             image_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
             if not image_paths:
                 continue

             total_frames = len(image_paths)
             H_img, W_img, _ = cv2.imread(image_paths[0]).shape

             # Create sequence-specific output directories
             output_dir_data_seq = os.path.join(output_path_data, date_folder, seq_folder)
             output_dir_viz_seq = os.path.join(output_path_viz, date_folder, seq_folder)
             os.makedirs(output_dir_data_seq, exist_ok=True)
             os.makedirs(output_dir_viz_seq, exist_ok=True)

             for frame_idx, center_image_path in enumerate(tqdm(image_paths, desc=f"  {date_folder}/{seq_folder}", unit="frame")):
                 base_filename = os.path.basename(center_image_path)
                 output_file_data = os.path.join(output_dir_data_seq, base_filename.replace('.png', '.zstd'))
                 output_file_viz = os.path.join(output_dir_viz_seq, base_filename)

                 # Skip if both output files for this image already exist
                 if os.path.exists(output_file_data) and os.path.exists(output_file_viz):
                     continue

                 center_frame = cv2.imread(center_image_path)
                 if center_frame is None:
                    print(f"\nWarning: Could not read image {center_image_path}, skipping.")
                    continue

                 # --- 1. Load Pre-computed Masks (and optional cyclist flags) from .zstd file ---
                 mask_file_path = os.path.join(args.mask_path, date_folder, seq_folder, base_filename.replace('.png', '.zstd'))
                 masks = []
                 cyclist_flags = []  # 1 = cyclist, 0 = pedestrian
                 try:
                    with open(mask_file_path, 'rb') as f:
                        decompressed_data = zstd.decompress(f.read())
                        payload = pickle.loads(decompressed_data)
                        # Two possible formats:
                        #  (A) legacy: iterable/array of masks only
                        #  (B) new: dict { 'masks': <array/list>, 'flags': <array/list uint8 0|1> }
                        if isinstance(payload, dict):
                            raw_masks = payload.get('masks', [])
                            raw_flags = payload.get('flags', None)
                            # Normalize raw_masks to list
                            if isinstance(raw_masks, np.ndarray):
                                # Expect shape (N,H,W) or empty
                                if raw_masks.ndim == 3:
                                    raw_masks_list = [raw_masks[i] for i in range(raw_masks.shape[0])]
                                else:
                                    raw_masks_list = []
                            else:
                                raw_masks_list = list(raw_masks)
                            # Normalize flags
                            if raw_flags is None:
                                cyclist_flags = [0] * len(raw_masks_list)
                            else:
                                if isinstance(raw_flags, np.ndarray):
                                    cyclist_flags = raw_flags.tolist()
                                else:
                                    cyclist_flags = list(raw_flags)
                                # Length safeguard
                                if len(cyclist_flags) != len(raw_masks_list):
                                    cyclist_flags = [0] * len(raw_masks_list)
                            loaded_masks = raw_masks_list
                        else:
                            # Legacy format (list/array of masks)
                            if isinstance(payload, np.ndarray):
                                if payload.ndim == 3:
                                    loaded_masks = [payload[i] for i in range(payload.shape[0])]
                                else:
                                    loaded_masks = []
                            else:
                                loaded_masks = list(payload)
                            cyclist_flags = [0] * len(loaded_masks)

                        # Ensure masks orientation (transpose if width/height swapped)
                        masks = []
                        for m in loaded_masks:
                            try:
                                if m.shape[0] == W_img and m.shape[1] == H_img:
                                    # Transposed
                                    masks.append(m.T)
                                else:
                                    masks.append(m)
                            except Exception:
                                continue
                        # Align flags list length to masks
                        if len(cyclist_flags) != len(masks):
                            cyclist_flags = [0] * len(masks)
                 except (FileNotFoundError, pickle.UnpicklingError, zstd.ZstdError):
                    # Expected if a frame has no pedestrians
                    pass
                 except Exception as e:
                    print(f"\nWarning: Could not load or process masks from {mask_file_path}. Skipping. Error: {e}")

                 if not masks:
                     # Save empty results for frames with no detections to prevent reprocessing
                     if not os.path.exists(output_file_data):
                          save_results([], output_file_data)
                     if not os.path.exists(output_file_viz):
                          cv2.imwrite(output_file_viz, center_frame)
                     continue

                 # --- 2. Prepare Sliding Window for Pose Estimation ---
                 window_indices = []
                 radius = (args.window_size // 2) * args.window_step
                 for i in range(-radius, radius + 1, args.window_step):
                     padded_idx = max(0, min(frame_idx + i, total_frames - 1))
                     window_indices.append(padded_idx)

                 sampled_frames = []
                 for idx in window_indices:
                    img_path = image_paths[idx]
                    frame = cv2.imread(img_path)
                    if frame is None:
                        print(f"\nWarning: Could not read frame {img_path}, using center frame as fallback.")
                        frame = center_frame.copy()
                    sampled_frames.append(frame)

                 # --- 3. Pose Estimation for each detected pedestrian ---
                 annotated_frame = center_frame.copy()
                 pedestrian_results = []

                 for mask, cyc_flag in zip(masks, cyclist_flags):
                     rows, cols = np.where(mask)
                     if len(rows) == 0: continue

                     x1, y1 = int(np.min(cols)), int(np.min(rows))
                     x2, y2 = int(np.max(cols)), int(np.max(rows))

                     # Enlarge bounding box for cropping
                     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                     w = (x2 - x1) * pose_cfg.DATASET.BBOX_ENLARGE_FACTOR
                     h = (y2 - y1) * pose_cfg.DATASET.BBOX_ENLARGE_FACTOR
                     x1c = int(max(cx - w / 2, 0))
                     y1c = int(max(cy - h / 2, 0))
                     x2c = int(min(cx + w / 2, W_img))
                     y2c = int(min(cy + h / 2, H_img))
                     w_crop, h_crop = x2c - x1c, y2c - y1c

                     if w_crop <= 1 or h_crop <= 1: continue

                     # Crop person from all frames in the window
                     crops = [
                          preprocess_frame(f[y1c:y2c, x1c:x2c], pose_cfg.MODEL.IMAGE_SIZE)
                          for f in sampled_frames
                     ]
                     inp = torch.stack(crops).unsqueeze(0).to(device)

                     # Run pose inference
                     with torch.no_grad():
                          heatmaps = pose_model(inp)

                     keypoints = extract_kps(heatmaps, h_crop, w_crop)
                     # Convert crop-relative keypoints to global image coordinates
                     keypoints_global = keypoints.astype(np.float32) + np.array([x1c, y1c], dtype=np.float32)

                     # Store results: keypoints in global image coordinates
                     person_data = {
                         "bbox": np.array([x1, y1, x2, y2]),
                         "mask": mask,
                         "keypoints": keypoints_global,
                         "cyclist_flag": int(cyc_flag)  # 1 if cyclist, else 0
                     }
                     pedestrian_results.append(person_data)

                     # --- 4. Visualization ---
                     # Draw mask overlay
                     overlay = annotated_frame.copy()
                     # Highlight cyclists differently (yellow) else random color
                     if int(cyc_flag) == 1:
                         color = (0, 255, 255)
                     else:
                         color = random.choice(USED_KP_COLORS)
                     overlay[mask] = color
                     annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)
                     # Draw keypoints for this person using global coordinates
                     for i, (px, py) in enumerate(keypoints_global):
                         cv2.circle(annotated_frame, (int(px), int(py)), 3, USED_KP_COLORS[i], -1)

                 # --- 5. Save Results ---
                 save_results(pedestrian_results, output_file_data)
                 cv2.imwrite(output_file_viz, annotated_frame)

    print("\nProcessing finished.")

if __name__ == "__main__":
     main()
