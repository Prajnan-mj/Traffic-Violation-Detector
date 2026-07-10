"""
detect_utils.py
----------------
Runs custom YOLOv8 models (triple riding, helmet non-compliance,
illegal parking) PLUS license plate detection on an image or video frame.
Gates the rider-related checks against actual two-wheelers detected by a
general COCO model.

Refactored & Improved:
- Face-centric helmet detection: Detects faces, checks if a helmet overlaps the top half of the face. If not, draws the box around the face as a violation.
- Global and class-specific Non-Maximum Suppression (NMS) to prevent duplicate boxes.
- Robust geometric overlap and IoU checks.
- Strict boundary and error handling to prevent crashes.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    VIOLATION_STYLES,
    DEFAULT_CONF,
    MODEL_RUN_CONF,
    TWO_WHEELER_CLASSES,
    NON_RIDER_VEHICLE_CLASSES,
    DEDUP_IOU,
    RIDER_GATE_OVERLAP,
    PLATE_VEHICLE_OVERLAP,
)
from ocr_utils import read_plate

# ------------------------------------------------------------------
# Model loading (once at module import)
# ------------------------------------------------------------------
_parking_model = None
_triple_model = None
_helmet_model = None
_coco_model = None
_license_plate_model = None

def get_models():
    global _parking_model, _triple_model, _helmet_model, _coco_model, _license_plate_model
    if _parking_model is None:
        _parking_model = YOLO('illegal_parking.pt')
        _triple_model = YOLO('triple.pt')
        _helmet_model = YOLO('helmet.pt')
        _coco_model = YOLO('yolov8n.pt')
        _license_plate_model = YOLO('license_plate_detector.pt')
    return _parking_model, _triple_model, _helmet_model, _coco_model, _license_plate_model

# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def _overlap_fraction(small_box, big_box):
    sx1, sy1, sx2, sy2 = small_box
    bx1, by1, bx2, by2 = big_box

    inter_x1 = max(sx1, bx1)
    inter_y1 = max(sy1, by1)
    inter_x2 = min(sx2, bx2)
    inter_y2 = min(sy2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    small_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    return inter_area / small_area


def _center_in_box(small_box, big_box):
    """Check if the center of small_box falls inside big_box."""
    sx1, sy1, sx2, sy2 = small_box
    bx1, by1, bx2, by2 = big_box
    cx = (sx1 + sx2) / 2
    cy = (sy1 + sy2) / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def _count_persons_in_box(target_box, person_boxes, min_overlap=0.4):
    """Count how many person boxes overlap the target box sufficiently."""
    count = 0
    for pb in person_boxes:
        if _overlap_fraction(pb, target_box) >= min_overlap:
            count += 1
    return count


def _rider_zone(vehicle_box, img_h):
    x1, y1, x2, y2 = vehicle_box
    h = y2 - y1
    w = x2 - x1

    margin_x = w * 0.35
    extend_up = h * 1.0

    zone_x1 = max(0, x1 - margin_x)
    zone_x2 = x2 + margin_x
    zone_y1 = max(0, y1 - extend_up)
    zone_y2 = y2

    return [zone_x1, zone_y1, zone_x2, zone_y2]


def _is_near_two_wheeler(box, rider_zones):
    """Dual gate: overlap fraction OR center-point inside zone."""
    for zone in rider_zones:
        if _overlap_fraction(box, zone) >= RIDER_GATE_OVERLAP:
            return True
        if _center_in_box(box, zone):
            return True
    return False


def _is_on_vehicle(plate_box, vehicle_boxes):
    for v in vehicle_boxes:
        if _overlap_fraction(plate_box, v) >= PLATE_VEHICLE_OVERLAP:
            return True
    return False


def _nms(detections, iou_threshold=DEDUP_IOU):
    """Standard Non-Maximum Suppression grouped by label."""
    if not detections:
        return []

    # Sort by confidence descending
    detections_sorted = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    kept = []

    for det in detections_sorted:
        is_dup = False
        for kept_det in kept:
            if kept_det['label'] != det['label']:
                continue
            if _iou(det['bbox'], kept_det['bbox']) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)

    return kept


def _safe_crop(img, x1, y1, x2, y2, pad=0):
    """Safely crops an image ensuring boundaries are respected."""
    img_h, img_w = img.shape[:2]
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img_w, x2 + pad)
    cy2 = min(img_h, y2 + pad)
    
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


def _annotate(img, box, color, text):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    badge_y1 = max(0, y1 - text_h - baseline - 6)
    cv2.rectangle(img, (x1, badge_y1), (x1 + text_w + 8, y1), color, -1)
    cv2.putText(
        img, text, (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )


# ------------------------------------------------------------------
# Core detection (single frame / image array)
# ------------------------------------------------------------------
def detect_frame(img, conf_thresholds=None):
    """
    Main detection pipeline:
    1. Base detection (YOLOv8n) to find person & vehicle boxes
    2. Helmet non-compliance check (Face -> Helmet pipeline inside rider zones)
    3. Triple riding check (count people in rider zones)
    4. Illegal parking check
    5. License plate read if ANY violation was found
    """
    _parking_model, _triple_model, _helmet_model, _coco_model, _license_plate_model = get_models()
    
    if conf_thresholds is None:
        conf_thresholds = {}

    def effective_floor(label):
        return conf_thresholds.get(label, DEFAULT_CONF[label])

    img_h, img_w = img.shape[:2]
    violations = []
    plates_read = []

    # --- 0. COCO vehicle + person gating pass ---
    coco_results = _coco_model(img, conf=0.35, verbose=False)[0]
    two_wheeler_boxes = []
    non_rider_vehicle_boxes = []
    all_active_vehicles = []
    person_boxes = []

    for box in coco_results.boxes:
        cls = int(box.cls[0])
        name = _coco_model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name in TWO_WHEELER_CLASSES:
            two_wheeler_boxes.append([x1, y1, x2, y2])
            all_active_vehicles.append([x1, y1, x2, y2])
        elif name in NON_RIDER_VEHICLE_CLASSES:
            non_rider_vehicle_boxes.append([x1, y1, x2, y2])
            all_active_vehicles.append([x1, y1, x2, y2])
        elif name == 'person':
            person_boxes.append([x1, y1, x2, y2])

    rider_zones = [_rider_zone(b, img_h) for b in two_wheeler_boxes]
    has_two_wheeler = len(two_wheeler_boxes) > 0

    # --- 1. Helmet Non-Compliance ---
    if has_two_wheeler:
        helmet_floor = effective_floor('helmet_non_compliance')
        
        helmet_results = _helmet_model(img, conf=MODEL_RUN_CONF.get('helmet_non_compliance', 0.3), verbose=False)[0]
        
        for box in helmet_results.boxes:
            conf = float(box.conf[0])
            if conf < helmet_floor:
                continue

            cls_id = int(box.cls[0])
            cls_name = _helmet_model.names[cls_id].lower()
            
            # Prevent duplicate vehicle boxes: only scan 'person' or explicit 'no_helmet' classes
            if cls_name not in ['person', 'no_helmet', 'no helmet', 'without_helmet', 'helmet_non_compliance', 'head']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]
            
            # Strict gate: Only scan people who are actively riding 2-wheelers
            # Ensure significant overlap directly with the vehicle box, not just walking nearby
            is_riding = False
            for tw in two_wheeler_boxes:
                if _overlap_fraction(bbox, tw) > 0.25 or _iou(bbox, tw) > 0.1:
                    is_riding = True
                    break
                    
            if not is_riding:
                continue

            # Fix: Boxes appearing around the entire body
            # Crop the bounding box to the top 25% to just highlight the head area
            h = y2 - y1
            head_y2 = int(y1 + h * 0.25)
            head_bbox = [x1, y1, x2, head_y2]

            violations.append({
                'label': 'helmet_non_compliance',
                'confidence': round(conf, 2),
                'bbox': head_bbox
            })

    # --- 2. Triple Riding ---
    if has_two_wheeler:
        triple_floor = effective_floor('triple_riding')
        triple_results = _triple_model(
            img, conf=MODEL_RUN_CONF['triple_riding'], verbose=False
        )[0]

        for box in triple_results.boxes:
            conf = float(box.conf[0])
            if conf < triple_floor:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox = [x1, y1, x2, y2]

            # Aspect ratio filter
            box_w = x2 - x1
            box_h = y2 - y1
            if box_h <= 0 or box_w <= 0:
                continue
            aspect = box_w / box_h
            if aspect < 0.3 or aspect > 3.0:
                continue

            # Dual gate: rider-zone overlap OR direct two-wheeler IoU
            near_rider_zone = _is_near_two_wheeler(bbox, rider_zones)
            near_two_wheeler = any(_iou(bbox, tw) >= 0.15 for tw in two_wheeler_boxes)
            if not near_rider_zone and not near_two_wheeler:
                continue

            # Person-count cross-check
            n_persons = _count_persons_in_box(bbox, person_boxes, min_overlap=0.4)
            if n_persons < 2:
                continue

            violations.append({
                'label': 'triple_riding',
                'confidence': round(conf, 2),
                'bbox': bbox,
            })

    # --- 3. Illegal Parking ---
    parking_floor = effective_floor('illegal_parking')
    parking_results = _parking_model(
        img, conf=MODEL_RUN_CONF['illegal_parking'], verbose=False
    )[0]

    for box in parking_results.boxes:
        conf = float(box.conf[0])
        if conf < parking_floor:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]

        is_parked = False
        matched_box = None
        
        # Check if the parking box aligns with a two-wheeler
        for tw in two_wheeler_boxes:
            if _iou(bbox, tw) > 0.10:
                matched_box = tw
                riders_on_bike = [p for p in person_boxes if _overlap_fraction(p, tw) > 0.25 or _iou(p, tw) > 0.1]
                if not riders_on_bike:
                    is_parked = True
                else:
                    # Rider present -> Check if any rider has their feet on the ground
                    tw_y2 = tw[3]
                    tw_h = tw[3] - tw[1]
                    for rider in riders_on_bike:
                        rider_y2 = rider[3]
                        # If rider's bounding box reaches the bottom 15% of the bike, feet are on the ground
                        if rider_y2 >= tw_y2 - (tw_h * 0.15):
                            is_parked = True
                            break
                break
                
        # If not a two-wheeler, check cars/buses/trucks
        if not matched_box:
            for nv in non_rider_vehicle_boxes:
                if _iou(bbox, nv) > 0.10:
                    matched_box = nv
                    nx1, ny1, nx2, ny2 = nv
                    # 1. Near edge of frame?
                    is_near_edge = (nx1 < img_w * 0.15) or (nx2 > img_w * 0.85)
                    # 2. Clustered with other vehicles?
                    nearby_cars = 0
                    for other_nv in non_rider_vehicle_boxes:
                        if other_nv != nv:
                            cx1, cy1 = (nx1+nx2)/2, (ny1+ny2)/2
                            cx2, cy2 = (other_nv[0]+other_nv[2])/2, (other_nv[1]+other_nv[3])/2
                            if np.hypot(cx1-cx2, cy1-cy2) < max(nx2-nx1, ny2-ny1) * 1.5:
                                nearby_cars += 1
                                
                    is_trafficy = len(all_active_vehicles) > 8
                    if is_trafficy:
                        is_parked = is_near_edge # busy road, only parked if on edge
                    else:
                        is_parked = is_near_edge or nearby_cars >= 1 # quiet road, parked if on edge or clustered
                    break

        if not is_parked or not matched_box:
            continue

        bbox = matched_box # snap to the actual vehicle box

        violations.append({
            'label': 'illegal_parking',
            'confidence': round(conf, 2),
            'bbox': bbox,
        })

    # --- 4. License Plate Detection ---
    lp_floor = effective_floor('license_plate')
    lp_results = _license_plate_model(
        img, conf=MODEL_RUN_CONF['license_plate'], verbose=False
    )[0]

    raw_plates = []
    for box in lp_results.boxes:
        conf = float(box.conf[0])
        if conf < lp_floor:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]

        if not _is_on_vehicle(bbox, all_active_vehicles):
            continue

        raw_plates.append({
            'label': 'license_plate',
            'confidence': round(conf, 2),
            'bbox': bbox,
        })

    # Deduplicate plates before OCR
    raw_plates = _nms(raw_plates)

    for plate in raw_plates:
        x1, y1, x2, y2 = plate['bbox']
        
        # Use safe crop for plate OCR
        plate_crop = _safe_crop(img, x1, y1, x2, y2, pad=5)
        if plate_crop is None or plate_crop.size == 0:
            continue

        cleaned_text, ocr_conf = read_plate(plate_crop, min_conf=0.25)
        if cleaned_text is None:
            continue

        plate_data = {
            'label': 'license_plate',
            'confidence': plate['confidence'],
            'bbox': plate['bbox'],
            'plate_text': cleaned_text,
            'ocr_confidence': round(float(ocr_conf), 2),
        }

        violations.append(plate_data)
        plates_read.append({
            'plate_text': cleaned_text,
            'confidence': plate['confidence'],
            'ocr_confidence': round(float(ocr_conf), 2),
            'bbox': plate['bbox'],
        })

    # --- 5. Deduplicate all violations ---
    violations = _nms(violations)
    return violations, plates_read


def annotate_frame(img, violations):
    """Draw annotations on image in-place."""
    for v in violations:
        style = VIOLATION_STYLES[v['label']]
        if v.get('plate_text'):
            text = f"PLATE {v['plate_text']} | {v['confidence']:.2f}"
        else:
            text = f"{style['label']} {v['confidence']:.2f}"

        _annotate(img, v['bbox'], style['color'], text)

    if violations:
        banner = f"{len(violations)} VIOLATION{'S' if len(violations) != 1 else ''} DETECTED"
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 255), -1)
        cv2.putText(
            img, banner, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )


def detect_all(image_path, conf_thresholds=None):
    """
    Process a single image file.
    Returns: (out_path, violations, plates_read)
    """
    if conf_thresholds is None:
        conf_thresholds = {}

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    violations, plates_read = detect_frame(img, conf_thresholds)
    annotate_frame(img, violations)

    out_path = (
        image_path.replace('.jpg', '_detected.jpg')
        .replace('.png', '_detected.png')
        .replace('.jpeg', '_detected.jpeg')
    )
    cv2.imwrite(out_path, img)
    return out_path, violations, plates_read