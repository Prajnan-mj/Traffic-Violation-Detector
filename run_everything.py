"""
run_everything.py
------------------
One-command driver for Traffic Sentinel performance evaluation.
Run this from inside your project folder (where app.py, detect_utils.py,
config.py and your .pt model weights already live).

USAGE
-----
1) You have raw, UNLABELED images and want the fastest path to a first
   evaluation run:

    python run_everything.py --images path/to/raw_images --bootstrap

   This runs your own model over every image and writes its predictions
   out as YOLO-format label files -- a head start instead of drawing every
   box by hand. Open bootstrapped_dataset/ in LabelImg / CVAT / Roboflow,
   fix the wrong/missing boxes, THEN trust the accuracy numbers. Evaluating
   directly against unedited bootstrapped labels just measures the model
   against its own output and will look artificially close to perfect --
   fine as a smoke test that the pipeline runs end-to-end, not as a real
   score.

2) You already have a properly hand-labeled dataset_dir (images/ + labels/):

    python run_everything.py --dataset path/to/dataset_dir --bench-image path/to/one_image.jpg

WHAT IT DOES
------------
- (optional) --bootstrap: generates starter YOLO labels from detect_frame's
  own predictions.
- Runs evaluate_dataset() -> Precision / Recall / F1 / mAP / Accuracy.
- Runs benchmark_efficiency() -> latency / FPS / memory / scalability.
- Prints one consolidated summary and saves performance_report.json.
"""

import argparse
import json
import os
import sys

import cv2

from detect_utils import detect_frame
from evaluation import evaluate_dataset, benchmark_efficiency, CLASS_ID_MAP


def bootstrap_labels(images_dir, out_dataset_dir, conf_thresholds=None):
    """
    Run the live model over every image in `images_dir` and write its
    predictions out as YOLO-format .txt labels in out_dataset_dir/labels/,
    copying images into out_dataset_dir/images/.

    This is a head start for manual correction, NOT a substitute for it --
    evaluating against these labels unedited just checks the model agrees
    with itself.
    """
    img_out = os.path.join(out_dataset_dir, "images")
    lbl_out = os.path.join(out_dataset_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # Reverse of evaluation.CLASS_ID_MAP: label string -> class_id
    label_to_id = {v: k for k, v in CLASS_ID_MAP.items()}

    exts = (".jpg", ".jpeg", ".png")
    img_paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(exts)
    ]
    if not img_paths:
        sys.exit(f"No images found in {images_dir}")

    print(f"Bootstrapping pseudo-labels for {len(img_paths)} images...")
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        violations, _ = detect_frame(img, conf_thresholds)

        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(img_out, os.path.basename(path)), img)

        lines = []
        for v in violations:
            cls_id = label_to_id.get(v["label"])
            if cls_id is None:
                continue
            x1, y1, x2, y2 = v["bbox"]
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))

        print(f"  [{i+1}/{len(img_paths)}] {os.path.basename(path)} -> {len(lines)} boxes")

    print(f"\nDone. Pseudo-labeled dataset written to: {out_dataset_dir}")
    print("Fix the boxes in a labeling tool BEFORE trusting the accuracy numbers below.\n")
    return out_dataset_dir


def main():
    p = argparse.ArgumentParser(description="Run Traffic Sentinel performance evaluation end-to-end")
    p.add_argument("--images", help="Folder of raw, unlabeled images (used with --bootstrap)")
    p.add_argument("--bootstrap", action="store_true",
                   help="Generate starter labels from the model's own predictions")
    p.add_argument("--bootstrap-out", default="bootstrapped_dataset",
                   help="Where to write the bootstrapped images/+labels/ folder")
    p.add_argument("--dataset", help="Existing labeled dataset_dir (images/ + labels/)")
    p.add_argument("--bench-image", help="Sample image for the efficiency benchmark")
    p.add_argument("--runs", type=int, default=20, help="Timed runs per resolution")
    p.add_argument("--out", default="performance_report.json", help="Output report path")
    args = p.parse_args()

    report = {}
    dataset_dir = args.dataset

    if args.bootstrap:
        if not args.images:
            sys.exit("--bootstrap requires --images <folder>")
        dataset_dir = bootstrap_labels(args.images, args.bootstrap_out)
        report["note"] = (
            "Labels were auto-bootstrapped from the model's own predictions. "
            "Accuracy/mAP below are NOT a real holdout score until you correct "
            "the boxes (e.g. in LabelImg/CVAT/Roboflow)."
        )

    if dataset_dir:
        print(f"Running accuracy evaluation on: {dataset_dir}")
        report["accuracy"] = evaluate_dataset(
            dataset_dir,
            progress_cb=lambda f: print(f"  {f*100:.0f}%", end="\r"),
        )
        print("\nAccuracy evaluation done.\n")

    bench_image = args.bench_image
    if not bench_image and dataset_dir:
        img_dir = os.path.join(dataset_dir, "images")
        candidates = sorted(os.listdir(img_dir)) if os.path.isdir(img_dir) else []
        if candidates:
            bench_image = os.path.join(img_dir, candidates[0])

    if bench_image:
        print(f"Running efficiency benchmark on: {bench_image}")
        report["efficiency"] = benchmark_efficiency(bench_image, n_runs=args.runs)
        print("Efficiency benchmark done.\n")

    if not dataset_dir and not bench_image:
        sys.exit("Nothing to do -- pass --images/--bootstrap, --dataset, and/or --bench-image.")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "accuracy" in report:
        ov = report["accuracy"]["overall"]
        print(f"Accuracy:  {ov['accuracy']}")
        print(f"Precision: {ov['precision']}")
        print(f"Recall:    {ov['recall']}")
        print(f"F1:        {ov['f1']}")
        print(f"mAP@0.5:   {ov['mAP']}")
    if "efficiency" in report:
        nat = report["efficiency"]["native"]
        print(f"Latency:   {nat['avg_latency_ms']} ms  ({nat['fps']} FPS)")
        print(f"Memory:    {nat['memory_mb']} MB")
    print(f"\nFull report saved to: {args.out}")


if __name__ == "__main__":
    main()
