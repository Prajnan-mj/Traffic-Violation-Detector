# Traffic Sentinel

A modular, multi-page Streamlit application for real-time traffic violation detection using custom-trained YOLOv8 models and EasyOCR. Detects helmet non-compliance, triple riding, illegal parking, and reads license plates — with geometric gating to eliminate false positives.

**Live demo:** [trafficviolationdetector.streamlit.app](https://trafficviolationdetector.streamlit.app/)

---

## What it does

Most violation detectors run a model and trust the output blindly. This pipeline doesn't. It runs a COCO pass first to locate actual vehicles, projects a geometric "rider zone" around every two-wheeler, and only then runs the violation models — gating every detection against real spatial context before accepting it. The result is far fewer false positives from pedestrians, parked cars, or background clutter.

Four violation types are fully implemented:

| Violation | Model | Decision Logic |
|---|---|---|
| Helmet non-compliance | `helmet.pt` (YOLOv8 custom) | Only flags heads inside an active rider zone with direct two-wheeler overlap |
| Triple riding | `triple.pt` (YOLOv8 custom) | Requires rider-zone overlap + aspect-ratio filter + independent person-count cross-check (≥2 persons) |
| Illegal parking | `illegal_parking.pt` (YOLOv8 custom) | Two-wheelers: checks if rider is absent or has feet on the ground. Four-wheelers: uses frame-edge position + vehicle clustering density |
| License plate OCR | `license_plate_detector.pt` + EasyOCR | Plate must overlap a confirmed vehicle box; two-strategy OCR with character-level segmentation fallback |

---

## Pipeline overview

```
Input (image / video / webcam)
        │
        ▼
  Preprocessing
  ─ Resize to 1920px max
  ─ Bilateral filter (noise suppression)
  ─ CLAHE on LAB luminance (contrast, low-light)
        │
        ▼
  COCO Pass (YOLOv8n)
  ─ Detect all persons and vehicles
  ─ Classify two-wheelers vs. four-wheelers
  ─ Project rider zones above each two-wheeler
        │
        ▼
  Violation Models (run in parallel)
  ─ helmet.pt     → head crop → rider-zone gate
  ─ triple.pt     → aspect ratio + zone + person count
  ─ illegal_parking.pt → rider-presence + clustering logic
  ─ license_plate_detector.pt → vehicle-overlap gate → OCR
        │
        ▼
  IoU-based deduplication (NMS across all classes)
        │
        ▼
  Annotated output + evidence record (JSON)
```

---

## Codebase

```
├── app.py                   # Streamlit multi-page app (Overview, Violations, Analytics,
│                            #   Live Cameras, Evidence Records, AI Models, Performance Eval)
├── detect_utils.py          # Core detection pipeline: detect_frame(), annotate_frame(), detect_all()
├── ocr_utils.py             # License plate OCR: enhancement → EasyOCR → char segmentation fallback
├── config.py                # Shared constants: thresholds, class sets, IoU params
├── evaluation.py            # Accuracy evaluation: Precision, Recall, F1, mAP@0.5 on labelled sets
├── sweep_thresholds.py      # Threshold sweep tool: finds best F1 operating point per model
├── debug_triple.py          # Debug utility for triple-riding detection
├── run_everything.py        # CLI entry point for headless batch processing
│
├── helmet.pt                # Custom YOLOv8 helmet/no-helmet model
├── triple.pt                # Custom YOLOv8 triple-riding model
├── illegal_parking.pt       # Custom YOLOv8 parking violation model
├── license_plate_detector.pt # Custom YOLOv8 license plate localiser
├── yolov8n.pt               # Pretrained YOLOv8n (COCO) for vehicle/person gating
│
├── style.css                # Streamlit custom CSS (pastel flat-design theme)
├── requirements.txt
└── samples/                 # Sample images for in-app testing
```

### Key files in detail

**`detect_utils.py`** — the core detection logic lives here. The main entry point is `detect_frame(img, conf_thresholds)`, which runs the full pipeline on a single NumPy frame and returns a list of violation dicts and a list of plate reads. `app.py` calls this for both images and video frames. Geometry helpers (`_iou`, `_overlap_fraction`, `_rider_zone`, `_count_persons_in_box`) and NMS (`_nms`) are all local to this file.

**`ocr_utils.py`** — two-strategy OCR. Strategy 1: Sobel-edge enhancement → morphological close → contour validation → adaptive threshold → EasyOCR on the full plate. Strategy 2 (fallback): connected-component character segmentation (requires `scikit-image`) → per-character EasyOCR. Output is cleaned with conservative OCR-error fixes (e.g. `O→0`, `I→1` in the numeric suffix) and validated for minimum character count and alphanumeric mix.

**`config.py`** — single source of truth for all thresholds, class sets, and IoU parameters. Violation models intentionally run at a lower internal confidence (`MODEL_RUN_CONF`) and are then filtered up to the user-facing floor (`DEFAULT_CONF`) after geometric gating — this lets the geometry checks do the heavy lifting rather than relying purely on the model's score.

**`evaluation.py`** — runs the full pipeline over a YOLO-format labelled dataset and reports per-class and overall Precision, Recall, F1, Accuracy, and mAP@0.5. A separate benchmarking routine measures average and P95 latency, throughput (FPS), and memory footprint (RSS) across a range of input resolutions.

**`sweep_thresholds.py`** — scans a configurable confidence range against a labelled set and surfaces the threshold that maximises F1 for a given model. Results can be applied directly from the AI Models page in the UI.

---

## App pages

- **Overview** — upload an image or video (or pick a sample), run detection, view annotated output, violation table, vehicle inventory, OCR plates, and download a JSON telemetry bundle.
- **Violations** — filterable log of all violations detected across every session (by type, plate text, and confidence floor).
- **Analytics** — Plotly charts: violation type breakdown (donut), vehicle mix (bar), session history.
- **Live Cameras** — capture a single frame from a webcam or RTSP stream and run detection on it.
- **Evidence Records** — per-session evidence records, each expandable with full JSON and a download link.
- **AI Models** — loaded model status, per-model confidence sliders, and the threshold sweep tool.
- **Performance Eval** — run accuracy evaluation against a labelled dataset and benchmark inference efficiency.

---

## Getting started

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/Prajnan-mj/Traffic-Violation-Detector.git
cd Traffic-Violation-Detector

pip install -r requirements.txt

streamlit run app.py
```

The app loads all five models on first run and caches them with `@st.cache_resource`. On CPU this takes ~10–20 seconds; subsequent runs within the same session are instant.

> `scikit-image` is optional but recommended — without it the character-segmentation OCR fallback is disabled and hard-to-read plates may not be recognised.

---

## Running evaluation

Place your validation data in YOLO format:

```
dataset/
  images/   *.jpg
  labels/   *.txt   # class_id cx cy w h (normalised)
```

Open the **Performance Eval** page and point it at the dataset directory, or run headlessly:

```python
from evaluation import evaluate_dataset
metrics = evaluate_dataset("path/to/dataset", conf_thresholds={})
```

Edit `CLASS_ID_MAP` in `evaluation.py` to match your label convention before running.

---

## Confidence thresholds

Thresholds can be adjusted from the sidebar (quick controls), the AI Models page (fine-grained per-model sliders), or the threshold sweep tool (data-driven F1 maximisation).

| Model | Default floor | Internal run conf |
|---|---|---|
| Helmet non-compliance | 0.40 | 0.20 |
| Triple riding | 0.30 | 0.10 |
| Illegal parking | 0.40 | 0.40 |
| License plate | 0.40 | 0.25 |

The two-stage design (low run conf → geometric gate → floor filter) is intentional: it lets spatial logic reject bad detections that a high model threshold alone would miss or misclassify.

---

## Tech stack

- **Detection:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **OCR:** [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **Preprocessing:** OpenCV (bilateral filter, CLAHE, Sobel, morphology)
- **UI:** Streamlit, Plotly
- **Character segmentation:** scikit-image (optional)

---

## Roadmap

The four implemented violation types are detectable from a single still frame. The following require multi-frame tracking or per-camera calibration and are planned for a future phase:

- Seatbelt non-compliance (fine-tune on cabin-view dataset)
- Wrong-side driving (trajectory tracking + lane direction model)
- Stop-line violation (per-camera ROI calibration + position tracking)
- Red-light violation (signal-state classifier + stop-line crossing logic)
