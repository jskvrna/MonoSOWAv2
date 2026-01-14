#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poseidon KITTI Pose Extraction (v2 simplified)
---------------------------------------------
Minimal interface for new pedestrian mask format.

Expected directory layout (masks already threshold-filtered):
    --mask_path /
        <date>/<seq>/<frame>.zstd

Each pedestrian .zstd file: pickled dict
    { 'masks': np.ndarray (N,H,W) or list of N (H,W) masks,
      'flags': np.ndarray/list length N (0 pedestrian, 1 cyclist-like) }

Call example:
    python poseidon_kitti_all_v2.py \
        --mask_path /path/to/masks_raw_pedestrians/ \
        --pose_config configs/posetrack21/configPoseidonVitH.yaml \
        --pose_weights models/vith_model.pt

Optional (defaults):
    --dataset_path /path/to/KITTI/complete_sequences/
    --output_path  /path/to/output/kitti_pose_detections_v2

Outputs:
  <output_path>/pedestrian_data/<date>/<seq>/<frame>.zstd      (pose results list)
  <output_path>/visualization/<date>/<seq>/<frame>.png         (annotated frame)
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml
import zstd
from torchvision import transforms
from tqdm import tqdm

from models.best.Poseidon import Poseidon  # noqa: F401 (external dependency expected in environment)

# ---------------- Configuration (static keypoint subset from original) ---------------- #
USED_KP_IDX = [0, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
USED_KP_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 128),
    (255, 165, 0), (128, 128, 0), (0, 128, 128), (75, 0, 130),
    (255, 105, 180), (0, 191, 255), (255, 223, 0), (165, 42, 42),
    (34, 139, 34)
]


# ---------------- Utility Functions ---------------- #
def load_pose_cfg(path: str):
    data = yaml.safe_load(open(path, "r"))

    class C:  # minimal dot-access config
        pass

    cfg = C()
    cfg.SEED = data.get("SEED", 42)
    cfg.MODEL = C()
    m = data["MODEL"]
    cfg.MODEL.METHOD = m["METHOD"]
    cfg.MODEL.NUM_JOINTS = m["NUM_JOINTS"]
    cfg.MODEL.IMAGE_SIZE = tuple(m["IMAGE_SIZE"])
    cfg.MODEL.CONFIG_FILE = m.get("CONFIG_FILE")
    cfg.MODEL.CHECKPOINT_FILE = m.get("CHECKPOINT_FILE")
    cfg.MODEL.EMBED_DIM = m.get("EMBED_DIM", 256)
    cfg.WINDOWS_SIZE = m.get("WINDOWS_SIZE", 5)
    cfg.MODEL.HEATMAP_SIZE = m.get("HEATMAP_SIZE", (96, 72))
    cfg.MODEL.FREEZE_WEIGHTS = m.get("FREEZE_WEIGHTS", False)
    cfg.DATASET = C()
    cfg.DATASET.BBOX_ENLARGE_FACTOR = data["DATASET"].get("BBOX_ENLARGE_FACTOR", 1.25)
    return cfg


_POSE_TFM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_frame(frame, size: Tuple[int, int]):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    return _POSE_TFM(frame)


def extract_kps(heatmaps: torch.Tensor, h_crop: int, w_crop: int) -> np.ndarray:
    _, _, H, W = heatmaps.shape
    hm = heatmaps[0, USED_KP_IDX].view(len(USED_KP_IDX), -1)
    _, idx = hm.max(dim=1)
    ys = (idx // W).float() * (h_crop / H)
    xs = (idx % W).float() * (w_crop / W)
    return torch.stack([xs, ys], dim=1).cpu().numpy().astype(np.float32)


def save_zstd_pickle(obj, out_file: str):
    if os.path.exists(out_file):
        return
    try:
        blob = zstd.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, 'wb') as f:
            f.write(blob)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error saving {out_file}: {e}")


def load_ped_masks(file_path: str, target_hw: Tuple[int, int]) -> Tuple[List[np.ndarray], List[int]]:
    """Load new-format pedestrian masks & cyclist flags.
    Returns (masks_list, flags_list). Masks are HxW uint8 (bool-like).
    """
    H_img, W_img = target_hw
    try:
        with open(file_path, 'rb') as f:
            payload = pickle.loads(zstd.decompress(f.read()))
        raw_masks = payload.get('masks', [])
        raw_flags = payload.get('flags', [])
        if isinstance(raw_masks, np.ndarray):
            if raw_masks.ndim == 3:
                masks_iter = [raw_masks[i] for i in range(raw_masks.shape[0])]
            else:
                masks_iter = []
        else:
            masks_iter = list(raw_masks)
        if isinstance(raw_flags, np.ndarray):
            flags = raw_flags.tolist()
        else:
            flags = list(raw_flags)
        if len(flags) != len(masks_iter):
            flags = [0] * len(masks_iter)
        fixed_masks: List[np.ndarray] = []
        for m in masks_iter:
            try:
                arr = np.asarray(m)
                if arr.shape == (W_img, H_img):  # transposed
                    arr = arr.T
                fixed_masks.append(arr.astype(np.uint8))
            except Exception:  # noqa: BLE001
                continue
        if len(flags) != len(fixed_masks):
            flags = [0] * len(fixed_masks)
        return fixed_masks, flags
    except FileNotFoundError:
        return [], []
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed loading {file_path}: {e}")
        return [], []


# ---------------- Argument Parsing ---------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Poseidon KITTI Pose Extraction v2 (simplified new format)")
    p.add_argument('--mask_path', required=True,
                   help='Path containing <date>/<seq>/<frame>.zstd pedestrian mask files (new format)')
    p.add_argument('--dataset_path', default='/path/to/KITTI/complete_sequences/',
                   help='Root KITTI path containing date folders (default: %(default)s)')
    p.add_argument('--output_path', default='/path/to/output/kitti_pose_detections_v2',
                   help='Base output directory (default: %(default)s)')
    p.add_argument('--pose_config', required=True, help='Poseidon YAML config file')
    p.add_argument('--pose_weights', required=True, help='Poseidon checkpoint (.pt)')
    p.add_argument('--window_size', type=int, default=5, help='Temporal window size (odd)')
    p.add_argument('--window_step', type=int, default=1, help='Step between frames in window')
    p.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    p.add_argument('--skip_existing', action='store_true', default=False,
                   help='Skip frames whose output data & viz already exist')
    return p.parse_args()


# ---------------- Main Logic ---------------- #
def build_window_indices(center_idx: int, total: int, window_size: int, step: int) -> List[int]:
    radius = (window_size // 2) * step
    idxs = []
    for offset in range(-radius, radius + 1, step):
        idxs.append(max(0, min(center_idx + offset, total - 1)))
    return idxs


def main():  # noqa: C901 (complexity acceptable for orchestrator)
    args = parse_args()
    if args.window_size % 2 == 0:
        raise SystemExit("--window_size must be odd")

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Poseidon
    print("Initializing Poseidon model ...")
    pose_cfg = load_pose_cfg(args.pose_config)
    pose_model = Poseidon(pose_cfg, phase="test", device=device)
    ckpt = torch.load(args.pose_weights, map_location=device)
    pose_model.load_state_dict(ckpt["model_state_dict"])
    pose_model.to(device).eval()
    print("Poseidon ready.")

    ped_base = args.mask_path.rstrip('/')
    if not os.path.isdir(ped_base):
        raise SystemExit(f"--mask_path directory not found: {ped_base}")
    print(f"Using pedestrian masks from: {ped_base}")

    out_data_root = os.path.join(args.output_path, 'pedestrian_data')
    out_viz_root = os.path.join(args.output_path, 'visualization')
    os.makedirs(out_data_root, exist_ok=True)
    os.makedirs(out_viz_root, exist_ok=True)

    date_folders = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    random.shuffle(date_folders)
    print(f"Found {len(date_folders)} date folders.")

    for date_folder in date_folders:
        date_path = os.path.join(args.dataset_path, date_folder)
        seq_folders = [d for d in os.listdir(date_path) if os.path.isdir(os.path.join(date_path, d))]
        random.shuffle(seq_folders)
        for seq_folder in seq_folders:
            print(f"\nSequence: {date_folder}/{seq_folder}")
            img_dir = os.path.join(date_path, seq_folder, 'image_02', 'data')
            if not os.path.isdir(img_dir):
                print(f"  Missing image dir: {img_dir}")
                continue
            image_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            if not image_paths:
                continue
            first_img = cv2.imread(image_paths[0])
            if first_img is None:
                print("  Warning: Could not read first frame; skipping sequence")
                continue
            H_img, W_img, _ = first_img.shape

            seq_out_data = os.path.join(out_data_root, date_folder, seq_folder)
            seq_out_viz = os.path.join(out_viz_root, date_folder, seq_folder)
            os.makedirs(seq_out_data, exist_ok=True)
            os.makedirs(seq_out_viz, exist_ok=True)

            for frame_idx, img_path in enumerate(tqdm(image_paths, desc=f"  {date_folder}/{seq_folder}", unit='frame')):
                base_name = os.path.basename(img_path)
                out_data_file = os.path.join(seq_out_data, base_name.replace('.png', '.zstd'))
                out_viz_file = os.path.join(seq_out_viz, base_name)
                # Unconditional fast skip: if data file already exists we assume this frame was processed.
                # (Covers both full & empty results; avoids unnecessary disk + model work.)
                if os.path.exists(out_data_file):
                    # Unconditionally skip; do not regenerate visualization.
                    continue

                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"  Warning: Cannot read frame {img_path}")
                    continue

                mask_file = os.path.join(ped_base, date_folder, seq_folder, base_name.replace('.png', '.zstd'))
                masks, flags = load_ped_masks(mask_file, (H_img, W_img))
                if not masks:
                    # Save empty (so we can skip later runs) + raw frame viz
                    save_zstd_pickle([], out_data_file)
                    if not os.path.exists(out_viz_file):
                        cv2.imwrite(out_viz_file, frame)
                    continue

                window_idxs = build_window_indices(frame_idx, len(image_paths), args.window_size, args.window_step)
                window_frames = []
                for wi in window_idxs:
                    f = cv2.imread(image_paths[wi])
                    if f is None:
                        f = frame  # fallback
                    window_frames.append(f)

                annotated = frame.copy()
                persons_out = []

                for mask, cyc_flag in zip(masks, flags):
                    rows, cols = np.where(mask)
                    if rows.size == 0:
                        continue
                    x1, y1 = int(cols.min()), int(rows.min())
                    x2, y2 = int(cols.max()), int(rows.max())
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    w = (x2 - x1) * pose_cfg.DATASET.BBOX_ENLARGE_FACTOR
                    h = (y2 - y1) * pose_cfg.DATASET.BBOX_ENLARGE_FACTOR
                    x1c = int(max(cx - w / 2, 0))
                    y1c = int(max(cy - h / 2, 0))
                    x2c = int(min(cx + w / 2, W_img))
                    y2c = int(min(cy + h / 2, H_img))
                    w_crop, h_crop = x2c - x1c, y2c - y1c
                    if w_crop <= 1 or h_crop <= 1:
                        continue
                    crops = [preprocess_frame(f[y1c:y2c, x1c:x2c], pose_cfg.MODEL.IMAGE_SIZE) for f in window_frames]
                    inp = torch.stack(crops).unsqueeze(0).to(device)
                    with torch.no_grad():
                        heatmaps = pose_model(inp)
                    kps_rel = extract_kps(heatmaps, h_crop, w_crop)
                    kps_global = kps_rel + np.array([x1c, y1c], dtype=np.float32)
                    persons_out.append({
                        'bbox': np.array([x1, y1, x2, y2], dtype=np.int32),
                        'mask': mask,  # original orientation ensured
                        'keypoints': kps_global.astype(np.float32),
                        'cyclist_flag': int(cyc_flag)
                    })
                    # Visualization: mask overlay & keypoints
                    overlay = annotated.copy()
                    color = (0, 255, 255) if int(cyc_flag) == 1 else random.choice(USED_KP_COLORS)
                    overlay[mask.astype(bool)] = color
                    annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)
                    for i, (px, py) in enumerate(kps_global):
                        if i < len(USED_KP_COLORS):
                            cv2.circle(annotated, (int(px), int(py)), 3, USED_KP_COLORS[i], -1)

                save_zstd_pickle(persons_out, out_data_file)
                cv2.imwrite(out_viz_file, annotated)

    print("\nProcessing finished (v2).")


if __name__ == '__main__':
    main()
