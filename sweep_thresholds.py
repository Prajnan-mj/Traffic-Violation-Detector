"""
sweep_thresholds.py — Confidence threshold sweeper for Traffic Sentinel.

Two modes of operation:

  1. AUTOMATED SWEEP (CLI / importable)
     Runs every model at incremental confidence levels over a labeled dataset
     and reports Precision, Recall, F1 at each threshold.  The best F1 row
     is flagged and can be auto-applied to a config JSON.

  2. MANUAL / INTERACTIVE (CLI flag --manual)
     Prints current thresholds and lets you edit them one by one in the
     terminal, then writes the result to a config JSON that app.py can import.

Dataset format expected
-----------------------
<dataset_dir>/
    images/          ← JPEG / PNG images
    labels/          ← YOLO .txt annotations, same stem as image
                       Each line:  <class_id> <cx> <cy> <w> <h>

Class IDs (must match what your models were trained on):
    0  triple_riding
    1  helmet_non_compliance
    2  illegal_parking
    3  license_plate

Usage
-----
# Automated sweep for helmet model:
python sweep_thresholds.py \\
    --dataset /path/to/data \\
    --model helmet_non_compliance \\
    --lo 0.20 --hi 0.90 --step 0.05

# Interactive manual edit:
python sweep_thresholds.py --manual --config sentinel_config.json

# Sweep + auto-write best threshold to config:
python sweep_thresholds.py \\
    --dataset /path/to/data \\
    --model triple_riding \\
    --apply --config sentinel_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ── Class ID map ───────────────────────────────────────────────────────────────
CLASS_ID_MAP: dict[str, int] = {
    "triple_riding":         0,
    "helmet_non_compliance": 1,
    "illegal_parking":       2,
    "license_plate":         3,
}

# Default floors (must not be swept below these in production)
DEFAULT_CONF: dict[str, float] = {
    "triple_riding":         0.45,
    "helmet_non_compliance": 0.40,
    "illegal_parking":       0.40,
    "license_plate":         0.40,
}

# ── Geometry ───────────────────────────────────────────────────────────────────
def _iou(a: list, b: list) -> float:
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    ua = max(0,ax2-ax1)*max(0,ay2-ay1)
    ub = max(0,bx2-bx1)*max(0,by2-by1)
    union = ua+ub-inter
    return inter/union if union > 0 else 0.0

def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> list[int]:
    x1 = int((cx - w/2) * img_w)
    y1 = int((cy - h/2) * img_h)
    x2 = int((cx + w/2) * img_w)
    y2 = int((cy + h/2) * img_h)
    return [x1, y1, x2, y2]

# ── Dataset loader ─────────────────────────────────────────────────────────────
def _load_dataset(dataset_dir: str, class_id: int) -> list[dict]:
    """
    Returns a list of dicts:
        {'image_path': str, 'gt_boxes': [[x1,y1,x2,y2], ...]}
    Only includes images that have at least one annotation for the target class.
    """
    img_dir = Path(dataset_dir) / "images"
    lbl_dir = Path(dataset_dir) / "labels"

    if not img_dir.exists():
        # Try flat structure (images and labels in root)
        img_dir = Path(dataset_dir)
        lbl_dir = Path(dataset_dir)

    items = []
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        lbl_path = (lbl_dir / img_path.stem).with_suffix(".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid != class_id:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            gt_boxes.append(_yolo_to_xyxy(cx, cy, w, h, img_w, img_h))

        if gt_boxes:
            items.append({"image_path": str(img_path), "gt_boxes": gt_boxes})

    return items


# ── Single-threshold evaluation ────────────────────────────────────────────────
def _evaluate_threshold(
    model,
    dataset: list[dict],
    run_conf: float,
    filter_conf: float,
    iou_thresh: float = 0.50,
) -> dict[str, Any]:
    """
    Run `model` on every image in `dataset`, keep predictions above `filter_conf`,
    and compute TP / FP / FN vs ground-truth boxes.

    Returns:
        dict with keys: threshold, tp, fp, fn, precision, recall, f1
    """
    tp = fp = fn = 0

    for item in dataset:
        img = cv2.imread(item["image_path"])
        if img is None:
            continue
        results = model(img, conf=run_conf, verbose=False)[0]

        pred_boxes = []
        for box in results.boxes:
            if float(box.conf[0]) >= filter_conf:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                pred_boxes.append([x1,y1,x2,y2])

        gt_boxes   = list(item["gt_boxes"])
        matched_gt = set()

        for pb in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    return {
        "threshold": round(filter_conf, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ── Public API ─────────────────────────────────────────────────────────────────
def run_sweep(
    dataset_dir: str,
    model_key: str,
    lo: float = 0.20,
    hi: float = 0.90,
    step: float = 0.05,
    iou_thresh: float = 0.50,
    conf_thresholds: dict | None = None,
) -> dict[str, Any]:
    """
    Sweep `model_key` confidence threshold over [lo, hi] and return results.

    Args:
        dataset_dir:      Root of the labeled dataset (images/ + labels/ subdirs,
                          or flat structure).
        model_key:        One of 'helmet_non_compliance', 'triple_riding',
                          'illegal_parking', 'license_plate'.
        lo, hi, step:     Sweep range and step size.
        iou_thresh:       IoU threshold for TP matching.
        conf_thresholds:  Current threshold dict (used to determine run_conf floor).

    Returns:
        {
          'model_key': str,
          'rows': [{'threshold':float, 'tp':int, 'fp':int, 'fn':int,
                    'precision':float, 'recall':float, 'f1':float}, ...],
          'best': {same keys as one row},
        }
    """
    from ultralytics import YOLO  # deferred so module imports without torch

    MODEL_FILES = {
        "triple_riding":         "triple.pt",
        "helmet_non_compliance": "helmet.pt",
        "illegal_parking":       "illegal_parking.pt",
        "license_plate":         "license_plate_detector.pt",
    }

    if model_key not in MODEL_FILES:
        raise ValueError(f"Unknown model_key '{model_key}'. "
                         f"Choose from: {list(MODEL_FILES)}")

    model_file = MODEL_FILES[model_key]
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    class_id = CLASS_ID_MAP[model_key]
    dataset  = _load_dataset(dataset_dir, class_id)
    if not dataset:
        raise RuntimeError(f"No labeled samples found for class '{model_key}' in {dataset_dir}")

    model    = YOLO(model_file)
    run_conf = 0.10  # very low run conf — we filter by filter_conf per step

    thresholds = []
    t = lo
    while t <= hi + 1e-9:
        thresholds.append(round(t, 4))
        t += step

    rows = []
    print(f"\n{'─'*60}")
    print(f"  Sweep: {model_key}  |  dataset: {dataset_dir}")
    print(f"  {len(dataset)} labeled images  |  thresholds: {lo:.2f}→{hi:.2f} (step {step:.2f})")
    print(f"{'─'*60}")
    print(f"  {'Thresh':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>7}  {'R':>7}  {'F1':>7}")
    print(f"{'─'*60}")

    for thr in thresholds:
        row = _evaluate_threshold(model, dataset, run_conf, thr, iou_thresh)
        rows.append(row)
        print(f"  {row['threshold']:>7.3f}  {row['tp']:>5}  {row['fp']:>5}  {row['fn']:>5}"
              f"  {row['precision']:>7.3f}  {row['recall']:>7.3f}  {row['f1']:>7.3f}")

    best = max(rows, key=lambda r: r['f1'])
    print(f"{'─'*60}")
    print(f"  ★ Best F1 = {best['f1']:.3f} at threshold = {best['threshold']:.3f}")
    print(f"{'─'*60}\n")

    return {"model_key": model_key, "rows": rows, "best": best}


# ── Manual interactive editor ──────────────────────────────────────────────────
def manual_edit(config_path: str | None = None) -> dict:
    """
    Interactively prompt the user to edit thresholds in the terminal.
    Loads from config_path if it exists; otherwise starts from DEFAULT_CONF.
    """
    # Load existing config
    thresholds = dict(DEFAULT_CONF)
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        thresholds.update(cfg.get("conf_thresholds", {}))
        print(f"\n✅  Loaded existing config from {config_path}")
    else:
        print("\n⚠️   No config file found — starting from defaults.")

    print("\n" + "═"*52)
    print("  Traffic Sentinel — Interactive Threshold Editor")
    print("═"*52)
    print("  Press Enter to keep the current value.\n")

    for key in DEFAULT_CONF:
        current = thresholds[key]
        label   = key.replace("_", " ").title()
        while True:
            raw = input(f"  {label} [{current:.2f}]: ").strip()
            if raw == "":
                break  # keep current
            try:
                val = float(raw)
                if not 0.0 <= val <= 1.0:
                    raise ValueError("Must be between 0.0 and 1.0")
                thresholds[key] = round(val, 4)
                break
            except ValueError as e:
                print(f"    ✗ Invalid: {e}. Try again.")

    print("\n  ── Updated thresholds ──────────────────────────")
    for k, v in thresholds.items():
        print(f"  {k:<30}  {v:.4f}")
    print("  ────────────────────────────────────────────────\n")

    return thresholds


# ── CLI ────────────────────────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(
        description="Traffic Sentinel — Confidence Threshold Sweeper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset",  "-d",  default=None,
                        help="Path to labeled dataset directory.")
    parser.add_argument("--model",    "-m",  default="helmet_non_compliance",
                        choices=list(CLASS_ID_MAP.keys()),
                        help="Which violation model to sweep.")
    parser.add_argument("--lo",       type=float, default=0.20, help="Sweep lower bound.")
    parser.add_argument("--hi",       type=float, default=0.90, help="Sweep upper bound.")
    parser.add_argument("--step",     type=float, default=0.05, help="Threshold step size.")
    parser.add_argument("--iou",      type=float, default=0.50, help="IoU threshold for TP matching.")
    parser.add_argument("--apply",    action="store_true",
                        help="Auto-write the best threshold to --config file.")
    parser.add_argument("--config",   "-c",  default="sentinel_config.json",
                        help="Config JSON to read from / write to.")
    parser.add_argument("--manual",   action="store_true",
                        help="Skip sweep; interactively edit thresholds and save.")
    parser.add_argument("--all",      action="store_true",
                        help="Sweep all four models sequentially (requires --dataset).")
    args = parser.parse_args()

    # ── Manual edit mode ──────────────────────────────────────────────────────
    if args.manual:
        thresholds = manual_edit(args.config)
        save = input("  Save to config file? [Y/n]: ").strip().lower()
        if save in ("", "y", "yes"):
            existing = {}
            if os.path.exists(args.config):
                with open(args.config) as f:
                    existing = json.load(f)
            existing["conf_thresholds"] = thresholds
            with open(args.config, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  ✅  Saved to {args.config}")
        else:
            print("  (not saved)")
        return

    # ── Sweep mode ────────────────────────────────────────────────────────────
    if not args.dataset:
        parser.error("--dataset is required for sweep mode. Use --manual to skip it.")

    keys_to_sweep = list(CLASS_ID_MAP.keys()) if args.all else [args.model]
    all_best: dict[str, float] = {}

    for key in keys_to_sweep:
        try:
            result = run_sweep(
                dataset_dir=args.dataset,
                model_key=key,
                lo=args.lo, hi=args.hi, step=args.step,
                iou_thresh=args.iou,
            )
            all_best[key] = result["best"]["threshold"]
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ⚠️   Skipping {key}: {e}", file=sys.stderr)

    # ── Apply best thresholds to config ───────────────────────────────────────
    if args.apply and all_best:
        existing: dict[str, Any] = {}
        if os.path.exists(args.config):
            with open(args.config) as f:
                existing = json.load(f)
        existing.setdefault("conf_thresholds", dict(DEFAULT_CONF))
        existing["conf_thresholds"].update(all_best)
        with open(args.config, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"✅  Best thresholds written to {args.config}:")
        for k, v in all_best.items():
            print(f"    {k:<32} {v:.4f}")
    elif all_best:
        print("\nRun with --apply to write these thresholds to config automatically.")


if __name__ == "__main__":
    _cli()