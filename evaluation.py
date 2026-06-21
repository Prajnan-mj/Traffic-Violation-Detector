"""
evaluation.py
--------------
Performance Evaluation for Traffic Sentinel.

Two independent pieces:

1. Detection accuracy  — Precision, Recall, F1, frame-level Accuracy and
   mAP@0.5, computed by running `detect_utils.detect_frame` over a labelled
   validation set and matching predictions to ground truth via IoU.

2. Computational efficiency / scalability — per-frame inference latency,
   throughput (FPS), memory footprint, and how those numbers change as
   input resolution scales up (so you know if the pipeline can hold up on
   higher-res camera feeds or longer videos).

Ground-truth dataset layout expected:

    dataset_dir/
        images/   *.jpg | *.jpeg | *.png
        labels/   *.txt   (YOLO format: "class_id cx cy w h", normalized 0-1,
                           one line per object, same basename as the image)

`CLASS_ID_MAP` below maps a label file's class_id -> the internal violation
label string. Edit it to match whatever convention your annotation tool used.
"""

import os
import glob
import time
from collections import defaultdict

import cv2
import numpy as np

from detect_utils import detect_frame
from config import VIOLATION_STYLES

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Edit to match your label .txt class_id -> internal violation label string.
CLASS_ID_MAP = {

    0: 'helmet_non_compliance',  # whichever class means "no helmet"
}

IOU_MATCH_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────
def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]


def _load_ground_truth(label_path, img_w, img_h):
    """Returns list of {'label':..., 'bbox':[x1,y1,x2,y2]}."""
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            label = CLASS_ID_MAP.get(cls_id)
            if label is None:
                continue
            gts.append({'label': label, 'bbox': _yolo_to_xyxy(cx, cy, w, h, img_w, img_h)})
    return gts


# ──────────────────────────────────────────────────────────────────
# Matching: TP / FP / FN per class for a single frame
# ──────────────────────────────────────────────────────────────────
def _match_frame(preds, gts, iou_thr=IOU_MATCH_THRESHOLD):
    """
    Greedy matching — highest-confidence predictions matched first.
    Returns:
        results:    list of (label, is_tp, confidence) for each prediction
        fn_counts:  dict label -> count of unmatched ground-truth boxes
    """
    results = []
    gt_used = [False] * len(gts)

    preds_sorted = sorted(preds, key=lambda p: p['confidence'], reverse=True)
    for p in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if gt_used[j] or g['label'] != p['label']:
                continue
            iou = _iou(p['bbox'], g['bbox'])
            if iou > best_iou:
                best_iou, best_j = iou, j
        is_tp = best_iou >= iou_thr
        if is_tp:
            gt_used[best_j] = True
        results.append((p['label'], is_tp, p['confidence']))

    fn_counts = defaultdict(int)
    for j, g in enumerate(gts):
        if not gt_used[j]:
            fn_counts[g['label']] += 1

    return results, fn_counts


# ──────────────────────────────────────────────────────────────────
# Average Precision (per class), 101-point interpolation (COCO-style)
# ──────────────────────────────────────────────────────────────────
def _average_precision(matches, total_gt):
    """
    matches: list of (confidence, is_tp) for ONE class, across the dataset.
    total_gt: total ground-truth boxes for that class.
    """
    if total_gt == 0:
        return None
    if not matches:
        return 0.0

    matches = sorted(matches, key=lambda m: m[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in matches:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / total_gt)

    ap = 0.0
    for t in np.linspace(0, 1, 101):
        candidates = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(candidates) if candidates else 0.0
    return ap / 101


# ──────────────────────────────────────────────────────────────────
# Main accuracy evaluation routine
# ──────────────────────────────────────────────────────────────────
def evaluate_dataset(dataset_dir, conf_thresholds=None, iou_thr=IOU_MATCH_THRESHOLD, progress_cb=None):
    """
    Runs detect_frame() over every image in dataset_dir/images, compares
    against dataset_dir/labels, and returns:

    {
      'per_class': {
          'helmet_non_compliance': {'precision':.., 'recall':.., 'f1':..,
                                     'ap':.., 'tp':.., 'fp':.., 'fn':.., 'n_gt':..},
          ...
      },
      'overall': {'accuracy':.., 'precision':.., 'recall':.., 'f1':.., 'mAP':..},
      'n_images': int,
    }

    'accuracy' here is frame-level: the fraction of frames where the SET of
    predicted violation types exactly matches the SET of ground-truth
    violation types (standard "accuracy" doesn't have a clean per-box
    definition in object detection since there's no true negative box;
    precision/recall/F1/mAP are the metrics that matter for the model
    itself, accuracy is a coarser sanity check at the frame level).
    """
    img_dir = os.path.join(dataset_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, 'labels')
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.jpeg')) +
        glob.glob(os.path.join(img_dir, '*.png'))
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    class_matches = defaultdict(list)   # label -> [(conf, is_tp), ...]
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_gt_total = defaultdict(int)

    correct_frames, total_frames = 0, 0

    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(lbl_dir, stem + '.txt')
        gts = _load_ground_truth(label_path, img_w, img_h)
        for g in gts:
            class_gt_total[g['label']] += 1

        preds, _ = detect_frame(img, conf_thresholds)

        matches, fn_counts = _match_frame(preds, gts, iou_thr)
        for label, is_tp, conf in matches:
            class_matches[label].append((conf, is_tp))
            if is_tp:
                class_tp[label] += 1
            else:
                class_fp[label] += 1
        for label, n in fn_counts.items():
            class_fn[label] += n

        pred_labels = {p['label'] for p in preds}
        gt_labels = {g['label'] for g in gts}
        if pred_labels == gt_labels:
            correct_frames += 1
        total_frames += 1

        if progress_cb:
            progress_cb((idx + 1) / len(img_paths))

    per_class = {}
    for label in VIOLATION_STYLES.keys():
        tp, fp, fn = class_tp[label], class_fp[label], class_fn[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        ap = _average_precision(class_matches[label], class_gt_total[label])
        per_class[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'ap': round(ap, 4) if ap is not None else None,
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_gt': class_gt_total[label],
        }

    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)
                  if (overall_precision + overall_recall) > 0 else 0.0)
    valid_aps = [v['ap'] for v in per_class.values() if v['ap'] is not None]
    mAP = round(float(np.mean(valid_aps)), 4) if valid_aps else None
    frame_accuracy = correct_frames / total_frames if total_frames > 0 else 0.0

    return {
        'per_class': per_class,
        'overall': {
            'accuracy': round(frame_accuracy, 4),
            'precision': round(overall_precision, 4),
            'recall': round(overall_recall, 4),
            'f1': round(overall_f1, 4),
            'mAP': mAP,
        },
        'n_images': total_frames,
    }


# ──────────────────────────────────────────────────────────────────
# Computational efficiency & scalability benchmark
# ──────────────────────────────────────────────────────────────────
def benchmark_efficiency(sample_image_path, conf_thresholds=None, n_runs=20,
                          resolutions=(640, 1280, 1920), progress_cb=None):
    """
    Measures inference latency / FPS / memory at native resolution, plus
    how those numbers scale as the input resolution grows.

    Returns:
    {
      'native': {'avg_latency_ms':.., 'p95_latency_ms':.., 'fps':.., 'memory_mb':.., 'memory_delta_mb':..},
      'scalability': [
          {'resolution': 640,  'shape': (h, w), 'avg_latency_ms':.., 'fps':..},
          ...
      ]
    }
    """
    base_img = cv2.imread(sample_image_path)
    if base_img is None:
        raise FileNotFoundError(sample_image_path)

    process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    def _time_runs(img, n):
        detect_frame(img, conf_thresholds)  # warm-up, excluded from timing
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            detect_frame(img, conf_thresholds)
            times.append(time.perf_counter() - t0)
        return times

    mem_before = process.memory_info().rss / (1024 ** 2) if process else None
    native_times = _time_runs(base_img, n_runs)
    mem_after = process.memory_info().rss / (1024 ** 2) if process else None

    avg_latency = float(np.mean(native_times)) * 1000
    native_result = {
        'avg_latency_ms': round(avg_latency, 1),
        'p95_latency_ms': round(float(np.percentile(native_times, 95)) * 1000, 1),
        'fps': round(1000 / avg_latency, 2) if avg_latency > 0 else None,
        'memory_mb': round(mem_after, 1) if mem_after is not None else 'psutil not installed',
        'memory_delta_mb': round(mem_after - mem_before, 1) if (mem_after is not None and mem_before is not None) else None,
    }

    scalability = []
    for i, target_dim in enumerate(resolutions):
        h, w = base_img.shape[:2]
        scale = target_dim / max(h, w)
        resized = cv2.resize(base_img, (max(1, int(w * scale)), max(1, int(h * scale))))
        times = _time_runs(resized, max(5, n_runs // 2))
        avg = float(np.mean(times)) * 1000
        scalability.append({
            'resolution': target_dim,
            'shape': list(resized.shape[:2]),
            'avg_latency_ms': round(avg, 1),
            'fps': round(1000 / avg, 2) if avg > 0 else None,
        })
        if progress_cb:
            progress_cb((i + 1) / len(resolutions))

    return {'native': native_result, 'scalability': scalability}


# ──────────────────────────────────────────────────────────────────
# CLI entry point — lets you run an evaluation without Streamlit
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Traffic Sentinel performance evaluation")
    parser.add_argument("--dataset", help="Path to labelled dataset_dir (images/ + labels/)")
    parser.add_argument("--bench-image", help="Sample image path for efficiency benchmarking")
    parser.add_argument("--runs", type=int, default=20, help="Timed runs per resolution")
    args = parser.parse_args()

    if args.dataset:
        print(f"Evaluating accuracy on: {args.dataset}")
        result = evaluate_dataset(args.dataset)
        print(_json.dumps(result, indent=2))

    if args.bench_image:
        print(f"Benchmarking efficiency on: {args.bench_image}")
        result = benchmark_efficiency(args.bench_image, n_runs=args.runs)
        print(_json.dumps(result, indent=2))

    if not args.dataset and not args.bench_image:
        parser.print_help()
