#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poseidon KITTI-360 Pose Extraction (v2 simplified)
--------------------------------------------------
Adaptation of KITTI Poseidon pipeline for KITTI-360 sequences.

Expected inputs (masks already threshold-filtered via Co-DETR pipeline):
    --mask_path /
        <relative sequence path>/<frame>.zstd

Each pedestrian .zstd file: pickled dict created by codetr_kitti360_gen.py
    {
        'masks': np.ndarray (N,H,W) or list of N (H,W) masks,
        'flags': np.ndarray/list length N (0 pedestrian, 1 cyclist-like)
    }

Call example:
    python poseidon_kitti360_all_v2.py \
        --mask_path /path/to/masks_raw_pedestrians/ \
        --pose_config configs/posetrack21/configPoseidonVitH.yaml \
        --pose_weights models/vith_model.pt

Optional (defaults):
    --dataset_path /path/to/KITTI/
    --output_path  /path/to/output/kitti360_pose_detections_v2

Outputs mirror KITTI version:
  <output_path>/pedestrian_data/<relative seq>/<frame>.zstd    (pose results list)
  <output_path>/visualization/<relative seq>/<frame>.png       (annotated frame)
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import time
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

# Candidate camera directories that may contain RGB data for KITTI-360
CAMERA_DIR_CANDIDATES = [
    ('image_00', 'data'),
    ('image_00', 'data_rect'),
    ('image_00', 'data_rgb'),
    ('image_00', 'data_ego'),
]

# ---------------- Utility Functions ---------------- #
def shuffle_with_seed(items):
    items = list(items)
    unique_seed = int(time.time() * 1000) + os.getpid()
    random.seed(unique_seed)
    random.shuffle(items)
    return items


def resolve_image_directory(sequence_path: str) -> str | None:
    """Return the directory containing RGB frames for a KITTI-360 sequence."""
    for rel_parts in CAMERA_DIR_CANDIDATES:
        candidate = os.path.join(sequence_path, *rel_parts)
        if os.path.isdir(candidate):
            return candidate

    fallback = sorted(glob.glob(os.path.join(sequence_path, 'image_00', 'data*')))
    for candidate in fallback:
        if os.path.isdir(candidate):
            return candidate
    return None


def collect_sequence_roots(root_path: str) -> List[str]:
    """Gather all sequence roots that contain camera imagery under root_path."""
    sequence_roots: List[str] = []
    if not os.path.isdir(root_path):
        return sequence_roots

    top_level_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    for top in top_level_dirs:
        top_path = os.path.join(root_path, top)
        img_dir = resolve_image_directory(top_path)
        if img_dir:
            sequence_roots.append(top_path)
            continue

        sub_dirs = [d for d in os.listdir(top_path) if os.path.isdir(os.path.join(top_path, d))]
        for sub in sub_dirs:
            sub_path = os.path.join(top_path, sub)
            img_dir = resolve_image_directory(sub_path)
            if img_dir:
                sequence_roots.append(sub_path)
    return sequence_roots


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
    """Load KITTI-360 pedestrian masks & cyclist flags produced by Co-DETR generator."""
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
    p = argparse.ArgumentParser(description="Poseidon KITTI-360 Pose Extraction v2 (simplified)")
    p.add_argument('--mask_path', required=True,
                   help='Path containing <relative seq>/<frame>.zstd pedestrian mask files (K360 format)')
    p.add_argument('--dataset_path', default='/path/to/KITTI/',
                   help='Root KITTI-360 path containing sequence folders (default: %(default)s)')
    p.add_argument('--output_path', default='/path/to/output/kitti360_pose_detections_v2',
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


def main():  # noqa: C901 (complex orchestration acceptable)
    args = parse_args()
    if args.window_size % 2 == 0:
        raise SystemExit("--window_size must be odd")

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Poseidon setup
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

    sequence_roots = collect_sequence_roots(args.dataset_path)
    if not sequence_roots:
        raise SystemExit(f"No KITTI-360 sequences with imagery found under {args.dataset_path}")
    sequence_roots = shuffle_with_seed(sequence_roots)
    print(f"Found {len(sequence_roots)} sequences.")

    for sequence_root in sequence_roots:
        relative_parts = os.path.relpath(sequence_root, args.dataset_path).split(os.sep)
        relative_parts = [p for p in relative_parts if p not in ('', '.')]
        sequence_label = '/'.join(relative_parts) if relative_parts else os.path.basename(sequence_root)
        print(f"\nSequence: {sequence_label}")

        image_dir = resolve_image_directory(sequence_root)
        if not image_dir:
            print(f"  Missing image directory under {sequence_root}; skipping.")
            continue

        image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if not image_paths:
            print("  No PNG frames found; skipping.")
            continue

        first_img = cv2.imread(image_paths[0])
        if first_img is None:
            print("  Warning: Could not read first frame; skipping sequence")
            continue
        H_img, W_img, _ = first_img.shape

        seq_out_data = os.path.join(out_data_root, *relative_parts)
        seq_out_viz = os.path.join(out_viz_root, *relative_parts)
        os.makedirs(seq_out_data, exist_ok=True)
        os.makedirs(seq_out_viz, exist_ok=True)

        for frame_idx, img_path in enumerate(tqdm(image_paths, desc=f"  {sequence_label}", unit='frame')):
            base_name = os.path.basename(img_path)
            data_filename = base_name.replace('.png', '.zstd')
            out_data_file = os.path.join(seq_out_data, data_filename)
            out_viz_file = os.path.join(seq_out_viz, base_name)

            if args.skip_existing and os.path.exists(out_data_file) and os.path.exists(out_viz_file):
                continue
            if not args.skip_existing and os.path.exists(out_data_file):
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  Warning: Cannot read frame {img_path}")
                continue

            mask_file = os.path.join(ped_base, *relative_parts, data_filename)
            masks, flags = load_ped_masks(mask_file, (H_img, W_img))
            if not masks:
                save_zstd_pickle([], out_data_file)
                if not os.path.exists(out_viz_file):
                    cv2.imwrite(out_viz_file, frame)
                continue

            window_idxs = build_window_indices(frame_idx, len(image_paths), args.window_size, args.window_step)
            window_frames = []
            for wi in window_idxs:
                ref_frame = cv2.imread(image_paths[wi])
                if ref_frame is None:
                    ref_frame = frame
                window_frames.append(ref_frame)

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
                    'mask': mask,
                    'keypoints': kps_global.astype(np.float32),
                    'cyclist_flag': int(cyc_flag)
                })

                overlay = annotated.copy()
                color = (0, 255, 255) if int(cyc_flag) == 1 else random.choice(USED_KP_COLORS)
                overlay[mask.astype(bool)] = color
                annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)
                for i, (px, py) in enumerate(kps_global):
                    if i < len(USED_KP_COLORS):
                        cv2.circle(annotated, (int(px), int(py)), 3, USED_KP_COLORS[i], -1)

            save_zstd_pickle(persons_out, out_data_file)
            cv2.imwrite(out_viz_file, annotated)

    print("\nProcessing finished (KITTI-360 v2).")


if __name__ == '__main__':
    main()
