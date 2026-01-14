import re
import wandb
import json
import argparse
import os
import math  # Import math to check for nan
from collections import defaultdict

def parse_log_file(log_content: str):
    """Parse training log and extract per-epoch metrics for Car / Pedestrian / Cyclist.

    The previous version only handled Pedestrian official metrics and relied on a
    checkpoint-loading pattern to delimit epochs. This version:
      * Uses 'Test Epoch <N>' lines to segment epochs (matches current log)
      * Extracts official AP blocks for Car, Pedestrian, Cyclist (skips _R40)
      * Names metrics as: official/<Class>/AP@<thr_thr_thr>/<metric>_AP/<difficulty>
      * Keeps existing nuScenes-like parsing (multi-class already supported)
    """
    # --- Epoch segmentation (supports two formats) ---
    lines = log_content.splitlines()
    test_epoch_re = re.compile(r"Test Epoch (\d+)")
    checkpoint_re = re.compile(r"Loading from checkpoint '.*?checkpoint_epoch_(\d+)\.pth'")

    epoch_indices = []  # list of (epoch_number, start_line_index)
    for idx, line in enumerate(lines):  # First try newer 'Test Epoch' markers
        m = test_epoch_re.search(line)
        if m:
            epoch_indices.append((int(m.group(1)), idx))

    if not epoch_indices:  # Fallback to older checkpoint-loading markers
        for idx, line in enumerate(lines):
            m = checkpoint_re.search(line)
            if m:
                epoch_indices.append((int(m.group(1)), idx))

    # If still none, return empty dict
    if not epoch_indices:
        return {}

    # Build (epoch, start, end) tuples
    epoch_indices_with_end = []
    for i, (ep, start) in enumerate(epoch_indices):
        end = epoch_indices[i + 1][1] if i + 1 < len(epoch_indices) else len(lines)
        epoch_indices_with_end.append((ep, start, end))

    # Regex for official metric lines (bbox / bev / 3d / aos)
    # Allow extra spaces after colon (some logs print 'bev  AP:')
    official_ap_pattern = re.compile(r"^\s*(bbox|bev|3d|aos)\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)")
    # Header for an official block (robust to leading timestamp / INFO prefix)
    # Example raw lines:
    #   2025-08-28 22:12:09,647   INFO  Car AP@0.50, 0.50, 0.50:
    # We therefore allow any leading chars before class name.
    official_header_re = re.compile(r"(Car|Pedestrian|Cyclist)\s+AP(_R40)?@([\d\.,\s]+):")
    # Regex for nuScenes-like metrics
    nusc_pattern = re.compile(
        r"\[nusc-like\]\s+(Car|Pedestrian|Cyclist|OVERALL):\s+"  # class
        r"TP=(\d+)\s+FP=(\d+)\s+FN=([\d\.]+)\s*\|\s*"  # counts
        r"AOE\(deg\)=([\d\.]+|nan)\s+ATE\(m\)=([\d\.]+|nan)\s+"  # angle/translation
        r"ADE_rel=([\d\.]+|nan)\s+ADE_abs\(m\)=([\d\.]+|nan)\s+"  # depth errors
        r"IoU2D=([\d\.]+|nan)"  # IoU2D
    )

    all_epochs = {}

    for epoch_num, start, end in epoch_indices_with_end:
        epoch_lines = lines[start:end]
        metrics = {}

        # --- Parse official blocks ---
        i = 0
        while i < len(epoch_lines):
            line = epoch_lines[i].strip()
            header_match = official_header_re.search(line)
            if not header_match:
                i += 1
                continue

            cls_name, r40_flag, thresh_raw = header_match.groups()
            # Advance to next line to start collecting block content
            i += 1

            # If we choose to skip R40 variants (current behavior)
            if r40_flag:
                # Skip until blank line or a new header occurrence
                while i < len(epoch_lines):
                    nxt = epoch_lines[i].strip()
                    if not nxt:
                        break
                    if official_header_re.search(nxt):
                        break
                    i += 1
                continue

            # Collect metric lines for this block
            block_lines = []
            while i < len(epoch_lines):
                l = epoch_lines[i].strip()
                if not l:
                    break
                if official_header_re.search(l):  # start of next block
                    break
                block_lines.append(l)
                i += 1

            # Normalize threshold string for key (KEEP dots for readability)
            # Original was e.g. '0.50, 0.50, 0.50'
            thresh_token = re.sub(r"[\s,]+", "_", thresh_raw.strip())  # -> '0.50_0.50_0.50'
            # If you prefer legacy style without dots, uncomment next line:
            # thresh_token = thresh_token.replace('.', '')

            for bl in block_lines:
                m_ap = official_ap_pattern.match(bl)
                if m_ap:
                    metric_type, v1, v2, v3 = m_ap.groups()
                    base = f"official/{cls_name}/AP@{thresh_token}/{metric_type}_AP"
                    metrics[f"{base}/easy"] = float(v1)
                    metrics[f"{base}/moderate"] = float(v2)
                    metrics[f"{base}/hard"] = float(v3)

        # --- Parse nuScenes-like metrics ---
        epoch_text = "\n".join(epoch_lines)
        for m in nusc_pattern.finditer(epoch_text):
            category, tp, fp, fn, aoe, ate, ade_rel, ade_abs, iou_2d = m.groups()

            def to_float_or_nan(v):
                return float(v) if v != "nan" else float("nan")

            metrics[f"nusc-like/{category}/TP"] = int(tp)
            metrics[f"nusc-like/{category}/FP"] = int(fp)
            metrics[f"nusc-like/{category}/FN"] = int(fn)
            metrics[f"nusc-like/{category}/AOE_deg"] = to_float_or_nan(aoe)
            metrics[f"nusc-like/{category}/ATE_m"] = to_float_or_nan(ate)
            metrics[f"nusc-like/{category}/ADE_rel"] = to_float_or_nan(ade_rel)
            metrics[f"nusc-like/{category}/ADE_abs_m"] = to_float_or_nan(ade_abs)
            metrics[f"nusc-like/{category}/IoU2D"] = to_float_or_nan(iou_2d)

        if metrics:
            all_epochs[epoch_num] = metrics

    return all_epochs


# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser with help messages
    parser = argparse.ArgumentParser(description="Parse a training log and upload metrics to Weights & Biases.")
    parser.add_argument("log_file", type=str, help="Required: The path to the training log file.")
    parser.add_argument("run_name", type=str, help="Required: A unique name for the wandb run.")
    parser.add_argument("--project", type=str, default="monodetr-multiclass-analysis", help="wandb project name (default: monodetr-multiclass-analysis)")
    args = parser.parse_args()

    # 1. Initialize a new wandb run
    wandb.init(project=args.project, name=args.run_name)
    
    # 2. Read the log file from the provided argument
    try:
        with open(args.log_file, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file not found at '{args.log_file}'")
        exit()

    # 3. Parse the log file to get all metrics
    parsed_data = parse_log_file(log_content)
    
    # 4. Log metrics for each epoch to wandb
    if not parsed_data:
        print("No epoch data found in the log file. Exiting.")
        wandb.finish()
        exit()
        
    print(f"Found data for {len(parsed_data)} epochs. Logging to wandb as run '{args.run_name}'...")
    
    sorted_epochs = sorted(parsed_data.keys())
    
    for epoch in sorted_epochs:
        raw_metrics = parsed_data[epoch]
        
        # NEW: Filter out any items where the value is 'nan' before logging
        metrics_to_log = {
            key: value for key, value in raw_metrics.items()
            if not (isinstance(value, float) and math.isnan(value))
        }

        wandb.log(metrics_to_log, step=epoch)
        print(f"Logged {len(metrics_to_log)} metrics for epoch {epoch}.")

    # 5. Finish the wandb run
    wandb.finish()
    
    print("\nDone. Check your wandb project for the new run.")
    
    # Optional: Save parsed data to a JSON file
    with open('parsed_metrics.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)
    print("Parsed data (including nan values) saved to 'parsed_metrics.json'")