Page 1 of 8
Traffic Violation Detector
https://trafficviolationdetector.streamlit.app/
Automated Photo Identification and Classification
for Traffic Violations Using Computer Vision
Live detection output — illegal no-helmet violations flagged with confidence scores on an unedited traffic frame.
Concept Note & Prototype Proposal
Submitted for: Automated Photo Identification and Classification for Traffic Violations Using Computer Vision
Page 2 of 8
Table of Contents
Table of Contents .......................................................................................................................................................2
1. Problem Statement & Our Approach .....................................................................................................................3
1.1 Design Principles ...............................................................................................................................................3
2. System Architecture ...............................................................................................................................................3
2.1 Stage 1 — Ingestion & Preprocessing ...............................................................................................................3
2.2 Stage 2 — Perception (Detection & Localisation) ............................................................................................4
2.3 Stage 3 — Violation Intelligence (Decision Logic) ............................................................................................4
2.4 Stage 4 — Output, Evidence & Analytics ..........................................................................................................4
3. License Plate Recognition .......................................................................................................................................5
4. Performance Evaluation .........................................................................................................................................5
4.1 Detection Accuracy ...........................................................................................................................................5
4.2 Computational Efficiency & Scalability .............................................................................................................5
4.3 Threshold Calibration Tooling ..........................................................................................................................5
5. Current Implementation Status ..............................................................................................................................6
6. Prototype in Action .................................................................................................................................................7
7. Deployment Roadmap ............................................................................................................................................7
8. Expected Impact .....................................................................................................................................................7
Page 3 of 8
1. Problem Statement & Our Approach
Traffic surveillance cameras across Indian cities generate thousands of images every day, but manual review of this footage to spot violations is slow, inconsistent, and cannot scale with camera density. Traffic Violation Detector is a computer-vision pipeline that watches this footage so a human doesn't have to look at every frame — it automatically detects vehicles and riders, flags violations, reads number plates, and produces a timestamped evidentiary record ready for enforcement review.
Our approach is deliberately practical: rather than presenting only a concept, we built a working prototype. The core detection pipeline, the violation-specific models for triple riding, helmet non-compliance and illegal parking, the OCR-based license plate reader, and a full performance-evaluation suite (Accuracy, Precision, Recall, F1, mAP, latency and throughput) are implemented and running today inside a Streamlit web application. This document describes that system, the engineering decisions behind it, and the roadmap to extend it to the remaining violation classes in the problem brief.
1.1 Design Principles
• Evidence-grade, not just detection-grade: every flagged violation carries a bounding box, a confidence score, a timestamp, and — where applicable — an OCR-read license plate, so output is directly usable as review evidence, not just a label.
• Context-aware gating, not raw model output: violation models are deliberately run at low internal thresholds and then filtered through geometric and contextual logic (rider-zone overlap, person-count cross-checks, vehicle clustering) before a detection is accepted. This suppresses false positives that a bare confidence threshold would let through.
• Modular and swappable: each violation type is its own YOLOv8 model behind a common interface, so new violation classes (seatbelt, wrong-side driving, red-light, stop-line) can be added without touching the rest of the pipeline.
• Measurable from day one: the system ships with its own evaluation harness so accuracy and runtime claims are reproducible against a labelled validation set, not just asserted.
2. System Architecture
The pipeline is organised into four sequential stages, mirroring the task structure of the problem brief.
2.1 Stage 1 — Ingestion & Preprocessing
Incoming images (or video frames) are first normalised: large frames are downscaled to a 1920px working dimension for consistent runtime, then passed through a bilateral filter to suppress sensor noise while preserving edges, followed by CLAHE (Contrast-Limited Adaptive Histogram Equalization) applied to the luminance channel in LAB color space. This combination specifically targets the brief's stated environmental challenges — low light, haze, and washed-out contrast from glare — without the over-smoothing that a simple Gaussian blur would introduce. For helmet detection specifically, a mild unsharp-mask sharpening pass is
Page 4 of 8
applied to cropped head regions before classification, since helmet edges are easily lost in motion-blurred two-wheeler footage.
2.2 Stage 2 — Perception (Detection & Localisation)
A general-purpose YOLOv8 model (trained on COCO) performs the first pass over every frame, localising all persons and vehicles and classifying vehicles into two-wheelers (motorcycle, bicycle) versus four-wheelers (car, bus, truck). This pass is the foundation the rest of the pipeline reasons on top of: a rider-zone is geometrically projected above and around each detected two-wheeler, and every subsequent violation check is gated against this zone rather than running blind over the whole frame.
Four specialised YOLOv8 models, fine-tuned on task-specific datasets, then run in parallel over the same frame:
• Triple-riding detector — trained to localise the riding cluster on a two-wheeler.
• Helmet non-compliance detector — a face/head-centric model that flags unprotected heads inside an active rider zone.
• Illegal-parking detector — flags stationary vehicles, cross-checked against vehicle clustering and frame position.
• License-plate detector — localises plate regions on any active vehicle for downstream OCR.
2.3 Stage 3 — Violation Intelligence (Decision Logic)
Raw model output is not trusted directly — this is the layer that turns detections into defensible violations. Examples of the logic implemented:
• Triple riding: a candidate box must (a) clear a deliberately low model-confidence floor, (b) pass an aspect-ratio sanity filter, (c) geometrically overlap a real two-wheeler's rider zone, and (d) be corroborated by an independent person-count check (≥2 person boxes overlapping the same region) before it is accepted. This last cross-check was added specifically after we found the model alone produced false negatives at default thresholds and false positives without it.
• Helmet non-compliance: only persons with significant bounding-box overlap with an actual two-wheeler are scanned (so pedestrians and bus passengers are never flagged); the scan region is then cropped to the top of the body to isolate the head.
• Illegal parking: a parked vehicle is distinguished from a vehicle simply stopped at a light using rider-presence checks for two-wheelers, and frame-edge position plus vehicle-clustering density for four-wheelers, with the decision threshold adapting to overall traffic density in the frame.
• License plates: a plate detection is only accepted if it overlaps an actual detected vehicle box, preventing spurious plate-shaped regions in the background from being read.
• Cross-class de-duplication: all accepted violations across all four models pass through a final IoU-based non-maximum-suppression pass so the same physical object is never reported twice under different labels.
2.4 Stage 4 — Output, Evidence & Analytics
Accepted violations are annotated directly onto the source image (colour-coded bounding box, violation label, confidence score) and logged with a timestamp into an evidence record. Where a license plate was read, the
Page 5 of 8
OCR text and its own confidence score are attached to the same record, linking a violation to a specific vehicle. The application's Analytics view aggregates this log into violation-type breakdowns and trends, and every record remains searchable and exportable for downstream enforcement workflows.
3. License Plate Recognition
Once a license-plate region is localised and confirmed to sit on a real vehicle, the cropped region is upscaled (small, distant plates are common in wide-angle CCTV footage and degrade OCR accuracy if read at native resolution) and passed to EasyOCR for text recognition. Recognised text is cleaned against expected plate-format constraints (character count, aspect ratio of the source region) and a minimum OCR-confidence floor before being accepted, so low-confidence garbled reads are not attached to the evidence record. The plate detector itself was fine-tuned on a custom-annotated dataset built and labelled specifically for this project, rather than relying on a generic pretrained detector.
4. Performance Evaluation
A dedicated evaluation module benchmarks the system on two independent axes, directly addressing Task 8 of the brief.
4.1 Detection Accuracy
Given a labelled validation set (YOLO-format bounding-box annotations), the evaluator runs the full detection pipeline end-to-end — not just the raw model — over every image, matches predictions to ground truth via IoU, and reports per-class and overall Precision, Recall, F1-score, frame-level Accuracy, and mAP@0.5. Results are broken down per violation class so weaknesses in any single model are visible rather than hidden inside an aggregate number.
4.2 Computational Efficiency & Scalability
A separate benchmarking routine measures average and P95 inference latency, throughput (FPS), and memory footprint (RSS) on representative hardware, then repeats the measurement across a sweep of input resolutions to characterise how the pipeline degrades — or holds up — as camera feed resolution increases. This was a deliberate design choice: a system that is accurate on a curated test image but collapses in throughput on a live 4K camera feed is not actually deployable, and we wanted that failure mode visible during development rather than discovered after rollout.
4.3 Threshold Calibration Tooling
Because confidence thresholds trade precision against recall differently for every violation class, the application includes a threshold-sweep tool that scans a configurable range of thresholds against a labelled dataset and surfaces the operating point that maximises F1 for a chosen model, which can then be applied to the live configuration directly from the interface — turning threshold tuning from a manual trial-and-error process into a repeatable, data-driven step.
Page 6 of 8
5. Current Implementation Status
We believe a working prototype, however partial, is stronger evidence of feasibility than a complete diagram of an unbuilt system. The table below is an honest account of what runs end-to-end today versus what is planned next. Violation Type Status Detection Method Notes
Helmet non-compliance
Implemented
Fine-tuned YOLOv8 + rider-zone gating
Head-centric crop pipeline
Triple riding
Implemented
Fine-tuned YOLOv8 + person-count cross-check
Geometric + count corroboration
Illegal parking
Implemented
Fine-tuned YOLOv8 + clustering/edge logic
Density-adaptive threshold
License plate detection + OCR
Implemented
Fine-tuned YOLOv8 + EasyOCR
Custom-annotated plate dataset
Seatbelt non-compliance
Planned — Phase 2
YOLOv8 fine-tune on cabin-view dataset
Same modular interface
Wrong-side driving
Planned — Phase 2
Trajectory tracking across frames + lane direction model
Requires video, not single-frame
Stop-line violation
Planned — Phase 2
Stop-line ROI calibration + vehicle position tracking
Requires per-camera calibration
Red-light violation
Planned — Phase 2
Signal-state classifier + stop-line crossing logic
Builds on stop-line module
The four implemented classes were prioritised because they are detectable from a single still frame — matching the photographic-evidence framing of the brief — and because they cover the highest-frequency violations in dense urban two-wheeler traffic. The three remaining classes (wrong-side driving, stop-line, red-light) are fundamentally multi-frame or calibration-dependent problems — they require tracking a vehicle's trajectory or position relative to a fixed reference over time, not just classifying a single photo — and are scoped for Phase 2 once video-sequence ingestion is added to the pipeline.
Page 7 of 8
6. Prototype in Action
The screenshot below is taken directly from the running Streamlit application, showing the violation-detail view after processing a live traffic frame: bounding boxes for the detected vehicle and rider, the no-helmet flag with its confidence score, and the structured evidence table (violation type, plate, confidence, review status) that gets written to the evidence log.
violation detail view with annotated frame and structured evidence table.
7. Deployment Roadmap
1. Phase 1 (current): Single-frame violation detection for helmet, triple riding, illegal parking, and license plate OCR, with full evaluation tooling — complete.
2. Phase 2: Extend ingestion to video sequences; add seatbelt detection; add trajectory-based wrong-side driving detection; add calibrated stop-line and signal-state logic for red-light violations.
3. Phase 3: Edge deployment for on-camera inference to reduce bandwidth and central compute load; integration with state e-challan systems via a REST API for automated ticket generation from confirmed evidence records.
4. Phase 4: Active-learning loop — low-confidence or contested detections routed to human reviewers, with corrections fed back into periodic model retraining to continuously improve accuracy on local traffic conditions.
8. Expected Impact
By automating the highest-volume, most visually verifiable violation types first, Traffic Violation Detector removes the bulk of manual screening effort from traffic enforcement teams while producing a structured, auditable evidence trail for every flagged incident. The same architecture scales horizontally — additional camera feeds simply mean additional frames through an already-built pipeline — and the built-in evaluation suite means accuracy and throughput claims can be continuously re-verified as the system is deployed against new locations and conditions, rather than degrading silently over time.
Page 8 of 8
— End of Document —
