"""
Traffic Sentinel — app.py
Full multi-page Streamlit application with working navigation.
"""

import io
import os
import copy
import json
import cv2
import datetime
import numpy as np
import streamlit as st
from collections import Counter
from ultralytics import YOLO
from ocr_utils import read_plate
from evaluation import evaluate_dataset, benchmark_efficiency

# ── Preprocessing ──────────────────────────────────────────────────────────────
PREPROCESS_MAX_DIM = 1920

def preprocess_image(img):
    if img is None:
        return img
    h, w = img.shape[:2]
    if max(h, w) > PREPROCESS_MAX_DIM:
        scale = PREPROCESS_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return img

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Traffic Violation Detector", layout="wide")

# Load external CSS (stripped of inline style block to rely fully on style.css)
if os.path.exists("style.css"):
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
# Map violation types to our flat semantic themes: rust, yellow, sage, teal
VIOLATION_STYLES = {
    'triple_riding':         {'theme': 'yellow', 'label': 'Triple Riding'},
    'helmet_non_compliance': {'theme': 'rust',   'label': 'No Helmet'},
    'illegal_parking':       {'theme': 'teal',   'label': 'Illegal Parking'},
    'license_plate':         {'theme': 'sage',   'label': 'License Plate'},
}
VEHICLE_STYLE = {'theme': 'teal', 'label': 'Vehicle'}

DEFAULT_CONF = {
    'triple_riding':         0.30,
    'helmet_non_compliance': 0.40,
    'illegal_parking':       0.40,
    'license_plate':         0.40,
}
MODEL_RUN_CONF = {
    'triple_riding':         0.10,
    'helmet_non_compliance': 0.20,
    'illegal_parking':       0.40,
    'license_plate':         0.25,
}
TWO_WHEELER_CLASSES   = {'motorcycle', 'bicycle'}
NON_RIDER_VEHICLE_CLASSES = {'car', 'bus', 'truck'}
DEDUP_IOU             = 0.45
RIDER_GATE_OVERLAP    = 0.15
OCR_MIN_CONF          = 0.35
PLATE_MIN_CHARS       = 4
VEHICLE_EMOJIS = {"motorcycle": "🏍️", "bicycle": "🚲", "car": "🚗", "bus": "🚌", "truck": "🚛"}

# Head -> Helmet pipeline tuning
HEAD_DETECT_CONF = 0.25   # run-time confidence floor for the head-detector pass
HEAD_CROP_PAD    = 0.30   # fractional padding added around a detected head box
                          # before running the helmet classifier, so the helmet's
                          # brim/edges don't get clipped by a tight head box.

# ── Session State Init ─────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "page":              "overview",
    "violation_history": [],      
    "plate_history":     [],      
    "vehicle_history":   [],      
    "evidence_log":      [],      
    "conf_thresholds":   DEFAULT_CONF, 
    "show_vehicle_boxes": True,
    "frame_skip":        5,
    "last_result":       None,    
    "sweep_results":     None,    
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        # Deepcopy prevents slider drags from modifying _SS_DEFAULTS
        st.session_state[_k] = copy.deepcopy(_v)

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import easyocr
    parking  = YOLO('illegal_parking.pt')
    triple   = YOLO('triple.pt')
    helmet   = YOLO('helmet.pt')
    coco     = YOLO('yolov8n.pt')
    lp       = YOLO('license_plate_detector.pt')
    return parking, triple, helmet, coco, lp

# ── Geometry helpers ───────────────────────────────────────────────────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    ua = max(0,ax2-ax1)*max(0,ay2-ay1)
    ub = max(0,bx2-bx1)*max(0,by2-by1)
    union = ua+ub-inter
    return inter/union if union>0 else 0.0

def _overlap_fraction(small, big):
    sx1,sy1,sx2,sy2 = small; bx1,by1,bx2,by2 = big
    ix1,iy1 = max(sx1,bx1),max(sy1,by1)
    ix2,iy2 = min(sx2,bx2),min(sy2,by2)
    iw,ih = max(0,ix2-ix1),max(0,iy2-iy1)
    inter = iw*ih
    sa = max(1,(sx2-sx1)*(sy2-sy1))
    return inter/sa

def _rider_zone(vbox, img_h):
    x1,y1,x2,y2 = vbox
    h,w = y2-y1, x2-x1
    mx = w*0.45; up = h*1.5
    return [max(0,x1-mx), max(0,y1-up), x2+mx, y2]

def _center_in_box(small_box, big_box):
    """Check if the center of small_box falls inside big_box."""
    sx1, sy1, sx2, sy2 = small_box
    bx1, by1, bx2, by2 = big_box
    cx = (sx1 + sx2) / 2
    cy = (sy1 + sy2) / 2
    return bx1 <= cx <= bx2 and by1 <= cy <= by2

def _sharpen_crop(crop):
    """Apply a mild unsharp-mask to sharpen a head crop before helmet inference."""
    if crop is None or crop.size == 0:
        return crop
    blurred = cv2.GaussianBlur(crop, (0, 0), 3)
    return cv2.addWeighted(crop, 1.5, blurred, -0.5, 0)

def _count_persons_in_box(target_box, person_boxes, min_overlap=0.3):
    count = 0
    tx1, ty1, tx2, ty2 = target_box
    th = ty2 - ty1
    if th <= 0: return 0
    for pb in person_boxes:
        px1, py1, px2, py2 = pb
        ph = py2 - py1
        if ph / th < 0.25:
            continue
        iou_val = _iou(pb, target_box)
        pb_in_target = _overlap_fraction(pb, target_box)
        if iou_val > 0.05 or pb_in_target > 0.3:
            count += 1
    return count

def _detect_heads_multiscale(head_model_ref, img, conf):
    """Run head detection at full scale + tiled for large images."""
    heads = []
    results = head_model_ref(img, conf=conf, verbose=False)[0]
    for hb in results.boxes:
        hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
        heads.append([hx1, hy1, hx2, hy2, float(hb.conf[0])])

    img_h, img_w = img.shape[:2]
    if img_w > 1280:
        tile_w = img_w // 2
        overlap = int(tile_w * 0.15)
        tile_starts = [0, tile_w - overlap]
        for tx in tile_starts:
            tx_end = min(tx + tile_w + overlap, img_w)
            tile = img[:, tx:tx_end]
            tile_results = head_model_ref(tile, conf=conf, verbose=False)[0]
            for hb in tile_results.boxes:
                hx1, hy1, hx2, hy2 = map(int, hb.xyxy[0])
                heads.append([hx1 + tx, hy1, hx2 + tx, hy2, float(hb.conf[0])])

    if len(heads) <= 1:
        return heads
    heads_sorted = sorted(heads, key=lambda h: h[4], reverse=True)
    kept = []
    for h in heads_sorted:
        if not any(_iou(h[:4], k[:4]) > 0.45 for k in kept):
            kept.append(h)
    return kept

def _deduplicate(detections):
    kept = []
    for det in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        if not any(kd['label']==det['label'] and _iou(det['bbox'],kd['bbox'])>DEDUP_IOU for kd in kept):
            kept.append(det)
    return kept

def _get_bgr_color(theme_name):
    # Mapping our strict palette to BGR for OpenCV
    color_map = {
        'rust':   (90, 90, 186),   # #BA5A5A
        'yellow': (155, 228, 247), # #F7E49B
        'sage':   (139, 206, 164), # #A4CE8B
        'teal':   (189, 188, 134)  # #86BCBD
    }
    return color_map.get(theme_name, (255, 255, 255))

def _annotate(img, box, theme_name, text):
    x1,y1,x2,y2 = [int(v) for v in box]
    color = _get_bgr_color(theme_name)
    # Flat design approach for boxes - slightly thicker, no gradient
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 4)
    (tw,th),bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    by1 = max(0, y1-th-bl-6)
    cv2.rectangle(img, (x1,by1), (x1+tw+8,y1), color, -1)
    # Dark text for lighter colors, white text for dark
    txt_color = (30,30,30) if theme_name in ['yellow', 'sage'] else (255,255,255)
    cv2.putText(img, text, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_color, 2)

# ── Vehicle detection ──────────────────────────────────────────────────────────
def detect_vehicles(image_input):
    parking_model, triple_model, helmet_model, coco_model, lp_model = load_models()
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None: return []
        orig_h,orig_w = img.shape[:2]
        img = preprocess_image(img)
        proc_h,proc_w = img.shape[:2]
        sx = orig_w/proc_w if proc_w>0 else 1.0
        sy = orig_h/proc_h if proc_h>0 else 1.0
    else:
        img = image_input; sx=sy=1.0

    boxes = []
    for box in coco_model(img, conf=0.35)[0].boxes:
        cls = coco_model.names[int(box.cls[0])]
        if cls in ('car','motorcycle','bus','truck','bicycle'):
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            boxes.append({'name':cls,'bbox':[int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)]})
    return boxes

# ── Core detection ─────────────────────────────────────────────────────────────
def _run_detection_on_frame(frame, conf_thresholds, show_vehicle_boxes):
    parking_model, triple_model, helmet_model, coco_model, lp_model = load_models()
    def floor(label):
        return conf_thresholds.get(label, DEFAULT_CONF[label])

    violations, plates_read, vehicle_detections = [], [], []
    vehicle_detections = detect_vehicles(frame)
    two_wheeler_boxes = [v['bbox'] for v in vehicle_detections if v['name'] in TWO_WHEELER_CLASSES]
    non_rider_vehicle_boxes = [v['bbox'] for v in vehicle_detections if v['name'] in NON_RIDER_VEHICLE_CLASSES]
    all_vehicles      = [v['bbox'] for v in vehicle_detections]
    has_tw = bool(two_wheeler_boxes)
    rzones = [_rider_zone(b, frame.shape[0]) for b in two_wheeler_boxes] if has_tw else []

    # Collect person boxes from COCO for triple-riding cross-check
    person_boxes = []
    for box in coco_model(frame, conf=0.35, verbose=False)[0].boxes:
        cls = coco_model.names[int(box.cls[0])]
        if cls == 'person':
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            person_boxes.append([x1,y1,x2,y2])

    # Helmet Non-Compliance
    if has_tw:
        hfloor = floor('helmet_non_compliance')
        helmet_run_conf = MODEL_RUN_CONF['helmet_non_compliance']

        helmet_results = helmet_model(frame, conf=helmet_run_conf, verbose=False)[0].boxes
        for box in helmet_results:
            conf = float(box.conf[0])
            if conf < hfloor:
                continue
                
            cls_id = int(box.cls[0])
            cls_name = helmet_model.names[cls_id].lower()
            
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

            violations.append({'label':'helmet_non_compliance','confidence':round(conf,2),'bbox':head_bbox})

    # Triple riding
    tfloor = floor('triple_riding')
    for box in triple_model(frame, conf=MODEL_RUN_CONF['triple_riding'], verbose=False)[0].boxes:
        conf = float(box.conf[0])
        if conf < tfloor: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        bbox = [x1,y1,x2,y2]

        # Aspect ratio filter
        box_w, box_h = x2 - x1, y2 - y1
        if box_h > 0:
            aspect = box_w / box_h
            if aspect < 0.3 or aspect > 3.0:
                continue

        # Dual gate: rider-zone overlap OR direct two-wheeler IoU
        near_rz = has_tw and any(_overlap_fraction(bbox, z) >= RIDER_GATE_OVERLAP for z in rzones)
        near_tw = has_tw and any(_iou(bbox, tw) >= 0.15 for tw in two_wheeler_boxes)
        if has_tw and not near_rz and not near_tw: continue

        # Person-count cross-check: require >= 2 persons inside the box
        n_persons = _count_persons_in_box(bbox, person_boxes, min_overlap=0.3)
        if n_persons < 2: continue

        violations.append({'label':'triple_riding','confidence':round(conf,2),'bbox':bbox})

    # Parking
    pfloor = floor('illegal_parking')
    for box in parking_model(frame, conf=MODEL_RUN_CONF['illegal_parking'], verbose=False)[0].boxes:
        conf = float(box.conf[0])
        if conf < pfloor: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        bbox = [x1,y1,x2,y2]
        
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
                    img_w = frame.shape[1]
                    is_near_edge = (nx1 < img_w * 0.15) or (nx2 > img_w * 0.85)
                    # 2. Clustered with other vehicles?
                    nearby_cars = 0
                    for other_nv in non_rider_vehicle_boxes:
                        if other_nv != nv:
                            cx1, cy1 = (nx1+nx2)/2, (ny1+ny2)/2
                            cx2, cy2 = (other_nv[0]+other_nv[2])/2, (other_nv[1]+other_nv[3])/2
                            if np.hypot(cx1-cx2, cy1-cy2) < max(nx2-nx1, ny2-ny1) * 1.5:
                                nearby_cars += 1
                                
                    is_trafficy = len(all_vehicles) > 8
                    if is_trafficy:
                        is_parked = is_near_edge # busy road, only parked if on edge
                    else:
                        is_parked = is_near_edge or nearby_cars >= 1 # quiet road, parked if on edge or clustered
                    break

        if not is_parked or not matched_box:
            continue

        violations.append({'label':'illegal_parking','confidence':round(conf,2),'bbox':matched_box})

    # License plate
    lpfloor = floor('license_plate')
    for box in lp_model(frame, conf=MODEL_RUN_CONF['license_plate'], verbose=False)[0].boxes:
        conf = float(box.conf[0])
        if conf < lpfloor: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        bbox = [x1,y1,x2,y2]
        if not any(_overlap_fraction(bbox,v)>=0.10 for v in all_vehicles): continue
        pad=5
        crop = frame[max(0,y1-pad):min(frame.shape[0],y2+pad), max(0,x1-pad):min(frame.shape[1],x2+pad)]
        if crop.size == 0: continue
        text, ocr_conf = read_plate(crop, min_conf=0.25)
        if text is None: continue
        entry = {'label':'license_plate','confidence':round(conf,2),'bbox':bbox,'plate_text':text,'ocr_confidence':round(float(ocr_conf),2)}
        violations.append(entry)
        plates_read.append({'plate_text':text,'confidence':round(conf,2),'ocr_confidence':round(float(ocr_conf),2),'bbox':bbox})

    violations = _deduplicate(violations)

    # Annotate
    for v in violations:
        style = VIOLATION_STYLES[v['label']]
        label_text = f"PLATE {v.get('plate_text','')} | {v['confidence']:.2f}" if v.get('plate_text') else f"{style['label']} {v['confidence']:.2f}"
        _annotate(frame, v['bbox'], style['theme'], label_text)
    if show_vehicle_boxes:
        for v in vehicle_detections:
            _annotate(frame, v['bbox'], VEHICLE_STYLE['theme'], f"{v['name'].upper()} | VEHICLE")

    return frame, violations, plates_read, vehicle_detections


def detect_all(image_path, conf_thresholds=None, show_vehicle_boxes=True):
    if conf_thresholds is None: conf_thresholds = {}
    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"Cannot read: {image_path}")
    img = preprocess_image(img)
    frame, violations, plates, vehicles = _run_detection_on_frame(img, conf_thresholds, show_vehicle_boxes)
    out_path = image_path.rsplit('.', 1)[0] + '_detected.' + image_path.rsplit('.', 1)[-1]
    cv2.imwrite(out_path, frame)
    return out_path, violations, plates, vehicles


def process_video(video_path, conf_thresholds=None, frame_skip=5, show_vehicle_boxes=True):
    if conf_thresholds is None: conf_thresholds = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {video_path}")
    fps         = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path    = video_path.rsplit('.', 1)[0] + '_detected.mp4'
    out_fps     = max(1, fps // frame_skip)
    out = None
    frame_idx   = 0
    all_violations, all_plates = [], []
    vtype_counts, viol_counts  = Counter(), Counter()
    total_vehicles = 0
    progress = st.progress(0)
    status   = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = preprocess_image(frame)
        if frame_idx % frame_skip != 0:
            frame_idx += 1; continue
        if out is None:
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w, h))
        frame, violations, plates, vehicles = _run_detection_on_frame(frame, conf_thresholds, show_vehicle_boxes)
        all_violations.extend(violations)
        all_plates.extend(plates)
        for v in violations:  viol_counts[v['label']] += 1
        for v in vehicles:    vtype_counts[v['name']] += 1
        total_vehicles += len(vehicles)
        out.write(frame)
        frame_idx += 1
        if total_frames > 0:
            progress.progress(min(1.0, frame_idx/total_frames))
            status.text(f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    if out: out.release()
    progress.empty(); status.empty()
    return out_path, len(all_violations), all_plates, vtype_counts, total_vehicles, viol_counts, all_violations


# ── Navigation helper ──────────────────────────────────────────────────────────
def _nav_btn(icon, label, page_key, badge=None):
    is_active = st.session_state.page == page_key
    # FIX: Plain text for the badge to avoid HTML rendering literally in the st.button
    badge_text = f" · {badge}" if badge else ""
    display = f"{icon}  {label}{badge_text}"
    btn_type = "primary" if is_active else "secondary"
    if st.button(display, key=f"nav_{page_key}", type=btn_type, use_container_width=True):
        st.session_state.page = page_key
        st.rerun()

def _go(page):
    st.session_state.page = page
    st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Options</div>', unsafe_allow_html=True)

    total_viol_count = len(st.session_state.violation_history)
    _nav_btn("", "Overview",         "overview")
    _nav_btn("", "Violations",       "violations", badge=total_viol_count if total_viol_count else None)
    _nav_btn("", "Analytics",        "analytics")
    _nav_btn("", "Live Cameras",     "live_cameras")
    st.markdown('<div class="nav-section">Management</div>', unsafe_allow_html=True)
    _nav_btn("", "Evidence Records", "evidence_records")
    _nav_btn("", "AI Models",        "ai_models")
    _nav_btn("", "Performance Eval", "performance_eval")
    _nav_btn("", "Settings",         "settings")
    st.markdown("<hr class='solid-divider'>", unsafe_allow_html=True)

    st.markdown("**Quick Controls**")
    st.session_state.show_vehicle_boxes = st.toggle(
        "Show Vehicle Boxes", value=st.session_state.show_vehicle_boxes)
    st.session_state.conf_thresholds['helmet_non_compliance'] = st.slider(
        "Helmet", 0.10, 1.0, st.session_state.conf_thresholds['helmet_non_compliance'], 0.05)
    st.session_state.conf_thresholds['triple_riding'] = st.slider(
        "Triple Riding", 0.10, 1.0, st.session_state.conf_thresholds['triple_riding'], 0.05)
    st.session_state.conf_thresholds['illegal_parking'] = st.slider(
        "Parking", 0.10, 1.0, st.session_state.conf_thresholds['illegal_parking'], 0.05)
    st.session_state.conf_thresholds['license_plate'] = st.slider(
        "License Plate", 0.10, 1.0, st.session_state.conf_thresholds['license_plate'], 0.05)
    st.session_state.frame_skip = st.slider(
        "Frame Skip", 1, 15, st.session_state.frame_skip)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
def page_overview():
    hcol, bcol = st.columns([4, 1])
    with hcol:
        st.markdown('<div class="main-header"><div class="page-title">Traffic Violation Detector</div>'
                    '<div class="page-subtitle">Upload an image or video to engage all detection sensors.</div></div>',
                    unsafe_allow_html=True)
    with bcol:
        st.markdown('<div style="padding-top:10px;"></div>', unsafe_allow_html=True)
        if st.button("＋ New Analysis", type="primary", use_container_width=True):
            st.session_state.last_result = None
            st.rerun()

    with st.expander("Data Source", expanded=True):
        col_s, col_u = st.columns(2)
        with col_s:
            sample = st.selectbox("Inject Sample Dataset", ["None","sample1.jpg","sample2.jpg","sample3.jpg"])
        with col_u:
            upload = st.file_uploader("Upload Feed Asset", type=['jpg','jpeg','png','mp4','avi','mov'])

    uploaded_file = None
    if upload:
        uploaded_file = upload
    elif sample != "None":
        sp = os.path.join("samples", sample)
        if os.path.exists(sp):
            with open(sp, "rb") as f:
                buf = io.BytesIO(f.read())
            buf.name = sample
            uploaded_file = buf

    if not uploaded_file:
        # Replaced hex colors with semantic CSS classes for the theme
        st.markdown("""
        <div class="standby-container">
            <div class="standby-icon">📡</div>
            <div class="standby-title">UPLINK STANDBY</div>
            <div class="standby-sub txt-muted">Provide an image or video asset above to engage sensors.</div>
        </div>""", unsafe_allow_html=True)
        return

    ext = uploaded_file.name.rsplit('.', 1)[-1].lower()
    is_video = ext in ['mp4','avi','mov']
    input_path = f"input_feed.{ext}"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("⚙️ Processing feed through violation models…"):
        if is_video:
            out_path, incident_count, plates, vtype_counts, total_vehicles, viol_counts, all_viol = process_video(
                input_path, conf_thresholds=st.session_state.conf_thresholds,
                frame_skip=st.session_state.frame_skip,
                show_vehicle_boxes=st.session_state.show_vehicle_boxes)
            violations, vehicle_detections = all_viol, []
            vehicle_type_counts = dict(vtype_counts)
            violation_type_counts = dict(viol_counts)
        else:
            out_path, violations, plates, vehicle_detections = detect_all(
                input_path, conf_thresholds=st.session_state.conf_thresholds,
                show_vehicle_boxes=st.session_state.show_vehicle_boxes)
            incident_count = len(violations)
            total_vehicles = len(vehicle_detections)
            vehicle_type_counts  = dict(Counter(v['name'] for v in vehicle_detections))
            violation_type_counts= dict(Counter(v['label'] for v in violations))

    ts = datetime.datetime.now().isoformat()
    for v in violations:
        v['timestamp'] = ts
        v['source'] = uploaded_file.name
    st.session_state.violation_history.extend(violations)
    st.session_state.plate_history.extend(plates)
    st.session_state.vehicle_history.extend(vehicle_detections)

    evidence_record = {
        "timestamp": ts, "source": uploaded_file.name, "is_video": is_video,
        "summary": {
            "helmet_violations":  violation_type_counts.get('helmet_non_compliance', 0),
            "triple_riding":      violation_type_counts.get('triple_riding', 0),
            "illegal_parking":    violation_type_counts.get('illegal_parking', 0),
            "plates_read":        len(plates),
            "vehicles_detected":  total_vehicles,
        },
        "violations": violations, "plates": plates,
        "vehicles": vehicle_detections,
        "vehicle_type_counts": vehicle_type_counts,
        "violation_type_counts": violation_type_counts,
        "out_path": out_path,
    }
    st.session_state.evidence_log.append(evidence_record)
    st.session_state.last_result = evidence_record

    # Metric cards dynamically themed
    hv = violation_type_counts.get('helmet_non_compliance', 0)
    tv = violation_type_counts.get('triple_riding', 0)
    pv = violation_type_counts.get('illegal_parking', 0)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, theme, header, val, sub in [
        (c1, "theme-rust",   "Helmet Violations",  hv,             "Non-compliance"),
        (c2, "theme-yellow", "Triple Riding",      tv,             "Over-capacity"),
        (c3, "theme-teal",   "Parking Deviation",  pv,             "Illegal parking"),
        (c4, "theme-teal",   "Vehicles Detected",  total_vehicles, "Active in feed"),
        (c5, "theme-sage",   "Plates Read",        len(plates),    "OCR acquisitions"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card {theme}"><div class="metric-card-header">{header}</div>'
                        f'<div class="metric-card-value">{val}</div>'
                        f'<div class="metric-card-sub txt-muted">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Feed Analysis</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2, gap="large")
    with fc1:
        st.caption("INPUT FEED")
        if is_video: st.video(input_path)
        else:        st.image(input_path, use_container_width=True)
    with fc2:
        st.caption("DETECTION OVERLAY")
        alert_theme = "safe-alert" if incident_count == 0 else "danger-alert"
        msg = f"{incident_count} INCIDENTS LOGGED" if incident_count > 0 else "SECURE // NO THREATS"
        st.markdown(f'<div class="neon-alert {alert_theme}">{msg}</div>', unsafe_allow_html=True)
        if is_video:
            st.video(out_path)
            with open(out_path, "rb") as vf:
                st.download_button("📥 Download Processed Video", vf, "evidence.mp4", "video/mp4", use_container_width=True)
        else:
            st.image(out_path, use_container_width=True)

    if violations:
        st.markdown('<div class="section-title">Detected Violations</div>', unsafe_allow_html=True)
        rows = ""
        for i, v in enumerate(violations[:10], 1):
            theme = VIOLATION_STYLES[v['label']]['theme']
            lbl   = VIOLATION_STYLES[v['label']]['label']
            pct   = int(v['confidence']*100)
            plate = v.get('plate_text', '—')
            
            rows += f"""<tr class="stagger-row delay-{i%5+1}">
                <td class="txt-muted">{i}</td>
                <td><span class="badge badge-{theme}">{lbl}</span></td>
                <td class="txt-primary mono">{plate}</td>
                <td><div class="conf-wrap">
                    <div class="conf-bar-bg"><div class="conf-bar-fill fill-{theme}" style="width:{pct}%;"></div></div>
                    <span class="txt-muted">{pct}%</span></div></td>
                <td><span class="badge badge-pending">Pending</span></td></tr>"""
        st.markdown(f"""<table class="detection-table"><thead><tr>
            <th>#</th><th>Violation</th><th>Plate</th><th>Confidence</th><th>Status</th>
            </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

    if vehicle_type_counts:
        st.markdown('<div class="section-title">Vehicle Inventory</div>', unsafe_allow_html=True)
        inv_cols = st.columns(len(vehicle_type_counts))
        for idx,(vtype,cnt) in enumerate(vehicle_type_counts.items()):
            with inv_cols[idx]:
                st.markdown(f'<div class="inventory-card theme-teal"><div class="inventory-emoji">{VEHICLE_EMOJIS.get(vtype,"🚙")}</div>'
                            f'<div class="inventory-count">{cnt}</div>'
                            f'<div class="inventory-label">{vtype.upper()}</div></div>', unsafe_allow_html=True)

    lc, rc = st.columns(2, gap="large")
    with lc:
        st.markdown('<div class="section-title">OCR Plates</div>', unsafe_allow_html=True)
        if plates:
            chips = "".join(f'<span class="license-plate animate-flipIn">{p["plate_text"]}</span>' for p in plates)
            st.markdown(f'<div class="plate-container">{chips}</div>', unsafe_allow_html=True)
        else:
            st.info("No plates isolated.")
    with rc:
        st.markdown('<div class="section-title">Incident Log</div>', unsafe_allow_html=True)
        if violations:
            for v in violations:
                theme = VIOLATION_STYLES[v['label']]['theme']
                st.markdown(f'<div class="violation-card viol-{theme}">'
                            f'<span class="font-weight-700">{VIOLATION_STYLES[v["label"]]["label"].upper()}</span>'
                            f'<span class="viol-pct">{int(v["confidence"]*100)}%</span></div>',
                            unsafe_allow_html=True)
        else:
            st.success("Log clear — no infractions.")

    st.download_button(
        "📥 Download Telemetry JSON",
        data=json.dumps(evidence_record, indent=2),
        file_name="telemetry.json", mime="application/json",
        use_container_width=True)


# ── VIOLATIONS ────────────────────────────────────────────────────────────────
def page_violations():
    st.markdown('<div class="main-header"><div class="page-title">Violations Log</div>'
                '<div class="page-subtitle">All violations detected across every session.</div></div>',
                unsafe_allow_html=True)

    history = st.session_state.violation_history
    if not history:
        st.info("No violations logged yet. Run a detection from the Overview page.")
        return

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_type = st.selectbox("Filter by Type", ["All"] + list(VIOLATION_STYLES.keys()),
                                   format_func=lambda x: VIOLATION_STYLES[x]['label'] if x != "All" else "All Types")
    with fc2:
        filter_plate = st.text_input("Search Plate", placeholder="e.g. MH12AB3456")
    with fc3:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

    filtered = [v for v in history
                if (filter_type == "All" or v['label'] == filter_type)
                and (not filter_plate or filter_plate.upper() in (v.get('plate_text') or ''))
                and v['confidence'] >= min_conf]

    st.markdown(f"**{len(filtered)}** violations shown" + (f" (filtered from {len(history)} total)" if len(filtered)!=len(history) else ""))

    if not filtered:
        st.warning("No violations match the current filters.")
        return

    rows = ""
    for i, v in enumerate(filtered, 1):
        theme = VIOLATION_STYLES[v['label']]['theme']
        lbl   = VIOLATION_STYLES[v['label']]['label']
        pct   = int(v['confidence']*100)
        plate = v.get('plate_text','—')
        ocr_c = f"{int(v.get('ocr_confidence',0)*100)}%" if v.get('ocr_confidence') else '—'
        src   = v.get('source','Unknown')
        ts    = v.get('timestamp','')[:19].replace('T',' ') if v.get('timestamp') else '—'
        
        rows += f"""<tr class="stagger-row delay-{i%5+1}">
            <td class="txt-muted">{i}</td>
            <td><span class="badge badge-{theme}">{lbl}</span></td>
            <td class="txt-primary mono">{plate}</td>
            <td>{ocr_c}</td>
            <td><div class="conf-wrap">
                <div class="conf-bar-bg"><div class="conf-bar-fill fill-{theme}" style="width:{pct}%;"></div></div>
                <span class="txt-muted">{pct}%</span></div></td>
            <td class="txt-muted">{src}</td>
            <td class="txt-faint">{ts}</td>
            <td><span class="badge badge-pending">Pending</span></td></tr>"""

    st.markdown(f"""<table class="detection-table"><thead><tr>
        <th>#</th><th>Type</th><th>Plate</th><th>OCR Conf</th>
        <th>Confidence</th><th>Source</th><th>Time</th><th>Status</th>
        </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)


# ── ANALYTICS ─────────────────────────────────────────────────────────────────
def page_analytics():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    st.markdown('<div class="main-header"><div class="page-title">Analytics</div>'
                '<div class="page-subtitle">Visual breakdown of all detection data across sessions.</div></div>',
                unsafe_allow_html=True)

    history  = st.session_state.violation_history
    vehicles = st.session_state.vehicle_history
    plates   = st.session_state.plate_history

    if not history and not vehicles:
        st.info("No data yet — run at least one detection from Overview.")
        return

    c1,c2,c3,c4 = st.columns(4)
    for col, title, val, sub in [
        (c1, "Total Violations",    len(history),  "across all sessions"),
        (c2, "Plates Captured",     len(plates),   "unique reads"),
        (c3, "Vehicles Processed",  len(vehicles), "all types"),
        (c4, "Sessions Run",        len(st.session_state.evidence_log), "evidence records"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-card-header">{title}</div>'
                        f'<div class="metric-card-value">{val}</div>'
                        f'<div class="metric-card-sub txt-muted">{sub}</div></div>', unsafe_allow_html=True)

    if not HAS_PLOTLY:
        st.warning("Please install Plotly for advanced visualisations (`pip install plotly`).")
        return

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.markdown('<div class="section-title">Violation Type Breakdown</div>', unsafe_allow_html=True)
        vcounts = Counter(v['label'] for v in history)
        if vcounts:
            # Map plotly colors to our flat palette
            hex_map = {'rust': '#BA5A5A', 'yellow': '#F7E49B', 'teal': '#86BCBD', 'sage': '#A4CE8B'}
            labels = [VIOLATION_STYLES[k]['label'] for k in vcounts]
            values = list(vcounts.values())
            colors = [hex_map[VIOLATION_STYLES[k]['theme']] for k in vcounts]
            fig = go.Figure(go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.45, textinfo='label+percent'))
            fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300, showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    with row1c2:
        st.markdown('<div class="section-title">Vehicle Mix</div>', unsafe_allow_html=True)
        vceh = Counter(v['name'] for v in vehicles)
        if vceh:
            fig2 = go.Figure(go.Bar(x=list(vceh.keys()), y=list(vceh.values()), marker_color='#86BCBD', text=list(vceh.values()), textposition='auto'))
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E2E8F0'))
            st.plotly_chart(fig2, use_container_width=True)


# ── LIVE CAMERAS ──────────────────────────────────────────────────────────────
def page_live_cameras():
    st.markdown('<div class="main-header"><div class="page-title">Live Cameras</div>'
                '<div class="page-subtitle">Connect to an RTSP stream or local webcam for real-time detection.</div></div>',
                unsafe_allow_html=True)

    src_type = st.radio("Source", ["Webcam", "RTSP URL"], horizontal=True)
    rtsp_url = st.text_input("RTSP Stream URL", placeholder="rtsp://...") if src_type == "RTSP URL" else ""

    col_cap, col_out = st.columns(2, gap="large")
    with col_cap:
        st.markdown('<div class="section-title">Live Preview</div>', unsafe_allow_html=True)
        if st.button("📸 Capture & Analyse", type="primary", use_container_width=True):
            source = 0 if src_type == "Webcam" else rtsp_url
            if src_type == "RTSP URL" and not rtsp_url: st.error("Please enter URL.")
            else:
                with st.spinner("Connecting…"):
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            st.session_state['_live_frame'] = preprocess_image(frame)
                            st.success("Frame captured.")
                    cap.release()
        if '_live_frame' in st.session_state:
            st.image(cv2.cvtColor(st.session_state['_live_frame'], cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_out:
        st.markdown('<div class="section-title">Detection Overlay</div>', unsafe_allow_html=True)
        if '_live_frame' in st.session_state:
            frame_copy = st.session_state['_live_frame'].copy()
            frame_copy, violations, plates, _ = _run_detection_on_frame(frame_copy, st.session_state.conf_thresholds, st.session_state.show_vehicle_boxes)
            st.image(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB), use_container_width=True)
            for v in violations:
                theme = VIOLATION_STYLES[v['label']]['theme']
                st.markdown(f'<div class="violation-card viol-{theme}"><b>{VIOLATION_STYLES[v["label"]]["label"]}</b> <span class="viol-pct">{int(v["confidence"]*100)}%</span></div>', unsafe_allow_html=True)


# ── EVIDENCE RECORDS ──────────────────────────────────────────────────────────
def page_evidence_records():
    st.markdown('<div class="main-header"><div class="page-title">Evidence Records</div>'
                '<div class="page-subtitle">All sessions with downloadable telemetry bundles.</div></div>',
                unsafe_allow_html=True)

    log = st.session_state.evidence_log
    if not log:
        st.info("No evidence records yet.")
        return

    for i, rec in enumerate(reversed(log), 1):
        ts  = rec.get('timestamp','')[:19].replace('T',' ')
        s   = rec.get('summary', {})
        with st.expander(f"Session {len(log)-i+1} — {ts} | **{sum(s.values())} captures**"):
            st.json(rec)


# ── AI MODELS ─────────────────────────────────────────────────────────────────
def page_ai_models():
    st.markdown('<div class="main-header"><div class="page-title">AI Models</div>'
                '<div class="page-subtitle">Loaded models, performance thresholds, and sweep tools.</div></div>',
                unsafe_allow_html=True)

    MODEL_INFO = [
        ("illegal_parking.pt",        "Parking Detector",   "illegal_parking",        "YOLOv8 custom — detects vehicles in no-parking zones.", "teal"),
        ("triple.pt",                 "Triple Riding",      "triple_riding",          "YOLOv8 custom — detects 3+ riders on a two-wheeler.", "yellow"),
        ("helmet.pt",                 "Helmet Detector",    "helmet_non_compliance",  "YOLOv8 custom — detects missing helmets on two-wheeler riders.", "rust"),
        ("license_plate_detector.pt", "License Plate OCR",  "license_plate",          "YOLOv8 custom + EasyOCR — plate detection and text reading.", "sage"),
    ]

    st.markdown('<div class="section-title">Loaded Models</div>', unsafe_allow_html=True)
    for fname, title, conf_key, desc, theme in MODEL_INFO:
        exists = os.path.exists(fname)
        status_html = '<span class="model-status safe">Loaded</span>' if exists else '<span class="model-status danger">Missing</span>'
        current_conf = st.session_state.conf_thresholds.get(conf_key, 'N/A')
        
        st.markdown(f"""<div class="model-card theme-{theme}">
            <div class="model-card-header"><span class="model-name">{title}</span>{status_html}</div>
            <div class="model-desc txt-muted">{desc}</div>
            <div class="model-meta">File: <code>{fname}</code> | Floor: <b>{current_conf}</b></div></div>""", unsafe_allow_html=True)

    # Threshold editor
    st.markdown('<div class="section-title">Threshold Editor</div>', unsafe_allow_html=True)
    edited_thresholds = {}
    tc1, tc2 = st.columns(2)
    with tc1:
        edited_thresholds['helmet_non_compliance'] = st.slider("Helmet Non-Compliance", 0.10, 1.0, st.session_state.conf_thresholds['helmet_non_compliance'], 0.01)
        edited_thresholds['triple_riding'] = st.slider("Triple Riding", 0.10, 1.0, st.session_state.conf_thresholds['triple_riding'], 0.01)
    with tc2:
        edited_thresholds['illegal_parking'] = st.slider("Illegal Parking", 0.10, 1.0, st.session_state.conf_thresholds['illegal_parking'], 0.01)
        edited_thresholds['license_plate'] = st.slider("License Plate", 0.10, 1.0, st.session_state.conf_thresholds['license_plate'], 0.01)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Apply Thresholds", type="primary"):
            st.session_state.conf_thresholds.update(edited_thresholds)
            st.success("Updated.")

    # Sweep Tool
    st.markdown('<div class="section-title">Threshold Sweep Tool</div>', unsafe_allow_html=True)
    with st.expander("Sweep Configuration"):
        sweep_dir   = st.text_input("Dataset directory")
        sweep_model = st.selectbox("Model to sweep", ["helmet_non_compliance","triple_riding","illegal_parking","license_plate"])
        if st.button("▶ Run Sweep", type="primary"):
            if not sweep_dir: st.error("Directory required.")
            else:
                from sweep_thresholds import run_sweep
                results = run_sweep(sweep_dir, sweep_model, 0.2, 0.9, 0.05, st.session_state.conf_thresholds)
                # FIX: Lock the target model_key in the state dictionary to prevent mismatched application
                results['model_key_used'] = sweep_model 
                st.session_state.sweep_results = results

    if st.session_state.sweep_results:
        res = st.session_state.sweep_results
        best = res['best']
        applied_model = res['model_key_used']
        
        st.success(f"Best for {applied_model}: {best['threshold']:.2f}")
        # Uses the locked key instead of the currently selected dropdown option!
        if st.button(f"Apply {best['threshold']:.2f} to {applied_model}"):
            st.session_state.conf_thresholds[applied_model] = best['threshold']
            st.success("Applied.")


# ── PERFORMANCE EVALUATION ────────────────────────────────────────────────────
def page_performance_eval():
    st.markdown('<div class="main-header"><div class="page-title">Performance Evaluation</div>'
                '<div class="page-subtitle">Accuracy, Precision, Recall, F1, mAP and runtime efficiency / scalability.</div></div>',
                unsafe_allow_html=True)

    tab_acc, tab_eff = st.tabs(["Detection Accuracy", "Computational Efficiency"])

    # ── Accuracy / mAP ──────────────────────────────────────────────────────
    with tab_acc:
        st.markdown('<div class="section-title">Dataset Evaluation</div>', unsafe_allow_html=True)
        st.caption("Expected layout: `dataset_dir/images/*.jpg` + `dataset_dir/labels/*.txt` "
                   "(YOLO format: `class_id cx cy w h`, normalized). Edit `CLASS_ID_MAP` in "
                   "`evaluation.py` to match your label convention. Runs against the "
                   "`detect_utils.detect_frame` pipeline.")
        dataset_dir = st.text_input("Ground-truth dataset directory", key="eval_dataset_dir")

        if st.button("▶ Run Evaluation", type="primary"):
            if not dataset_dir or not os.path.isdir(dataset_dir):
                st.error("Provide a valid dataset directory.")
            else:
                progress = st.progress(0)
                with st.spinner("Running detection over evaluation set…"):
                    try:
                        metrics = evaluate_dataset(
                            dataset_dir,
                            conf_thresholds=st.session_state.conf_thresholds,
                            progress_cb=lambda f: progress.progress(min(1.0, f)),
                        )
                        st.session_state['_eval_metrics'] = metrics
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                progress.empty()

        metrics = st.session_state.get('_eval_metrics')
        if metrics:
            ov = metrics['overall']
            c1, c2, c3, c4, c5 = st.columns(5)
            for col, title, val in [
                (c1, "Accuracy",  ov['accuracy']),
                (c2, "Precision", ov['precision']),
                (c3, "Recall",    ov['recall']),
                (c4, "F1-Score",  ov['f1']),
                (c5, "mAP@0.5",   ov['mAP']),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-card-header">{title}</div>'
                                f'<div class="metric-card-value">{val if val is not None else "—"}</div>'
                                f'<div class="metric-card-sub txt-muted">{metrics["n_images"]} images</div></div>',
                                unsafe_allow_html=True)

            st.markdown('<div class="section-title">Per-Class Breakdown</div>', unsafe_allow_html=True)
            rows = ""
            for label, m in metrics['per_class'].items():
                theme = VIOLATION_STYLES[label]['theme']
                lbl = VIOLATION_STYLES[label]['label']
                rows += f"""<tr><td><span class="badge badge-{theme}">{lbl}</span></td>
                    <td>{m['precision']}</td><td>{m['recall']}</td><td>{m['f1']}</td>
                    <td>{m['ap'] if m['ap'] is not None else '—'}</td>
                    <td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td><td>{m['n_gt']}</td></tr>"""
            st.markdown(f"""<table class="detection-table"><thead><tr>
                <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AP@0.5</th>
                <th>TP</th><th>FP</th><th>FN</th><th>GT Count</th>
                </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

            st.download_button("📥 Download Metrics JSON", data=json.dumps(metrics, indent=2),
                                file_name="performance_metrics.json", mime="application/json")

    # ── Computational Efficiency ────────────────────────────────────────────
    with tab_eff:
        st.markdown('<div class="section-title">Runtime Benchmark</div>', unsafe_allow_html=True)
        sample_path = st.text_input("Sample image path for benchmarking", key="eff_sample_path")
        n_runs = st.slider("Timed runs per resolution", 5, 50, 20)

        if st.button("▶ Run Benchmark", type="primary"):
            if not sample_path or not os.path.exists(sample_path):
                st.error("Provide a valid image path.")
            else:
                progress = st.progress(0)
                with st.spinner("Benchmarking inference latency / FPS / memory…"):
                    try:
                        bench = benchmark_efficiency(
                            sample_path,
                            conf_thresholds=st.session_state.conf_thresholds,
                            n_runs=n_runs,
                            progress_cb=lambda f: progress.progress(min(1.0, f)),
                        )
                        st.session_state['_eff_bench'] = bench
                    except Exception as e:
                        st.error(f"Benchmark failed: {e}")
                progress.empty()

        bench = st.session_state.get('_eff_bench')
        if bench:
            nat = bench['native']
            c1, c2, c3, c4 = st.columns(4)
            for col, title, val in [
                (c1, "Avg Latency",  f"{nat['avg_latency_ms']} ms"),
                (c2, "P95 Latency",  f"{nat['p95_latency_ms']} ms"),
                (c3, "Throughput",   f"{nat['fps']} FPS"),
                (c4, "Memory (RSS)", f"{nat['memory_mb']} MB"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-card-header">{title}</div>'
                                f'<div class="metric-card-value">{val}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Scalability Across Resolutions</div>', unsafe_allow_html=True)
            try:
                import plotly.graph_objects as go
                res_labels = [f"{s['resolution']}px" for s in bench['scalability']]
                fps_vals   = [s['fps'] for s in bench['scalability']]
                lat_vals   = [s['avg_latency_ms'] for s in bench['scalability']]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=res_labels, y=fps_vals, name="FPS", marker_color='#86BCBD'))
                fig.add_trace(go.Scatter(x=res_labels, y=lat_vals, name="Latency (ms)", yaxis="y2", marker_color='#BA5A5A'))
                fig.update_layout(
                    yaxis=dict(title="FPS"),
                    yaxis2=dict(title="Latency (ms)", overlaying='y', side='right'),
                    height=320, margin=dict(t=20,b=20,l=20,r=20),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                for s in bench['scalability']:
                    st.write(f"**{s['resolution']}px** → {s['fps']} FPS, {s['avg_latency_ms']} ms")

            st.download_button("📥 Download Benchmark JSON", data=json.dumps(bench, indent=2),
                                file_name="efficiency_benchmark.json", mime="application/json")


# ── SETTINGS ──────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown('<div class="main-header"><div class="page-title">Settings</div></div>', unsafe_allow_html=True)
    with st.expander("Data Management", expanded=True):
        if st.button("Reset All Settings", type="secondary"):
            for k,v in _SS_DEFAULTS.items():
                # Deepcopy restores proper safe defaults
                st.session_state[k] = copy.deepcopy(v)
            st.rerun()

PAGE_MAP = {"overview": page_overview, "violations": page_violations, "analytics": page_analytics, "live_cameras": page_live_cameras, "evidence_records": page_evidence_records, "ai_models": page_ai_models, "performance_eval": page_performance_eval, "settings": page_settings}
PAGE_MAP.get(st.session_state.page, page_overview)()