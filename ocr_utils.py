"""
ocr_utils.py
------------
Hybrid license plate OCR combining:
- PlateFinder-style traditional CV preprocessing (Sobel, morphology)
- Character segmentation fallback (requires scikit-image)
- EasyOCR for text recognition
"""

import cv2
import numpy as np
import re
import easyocr

try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: scikit-image not installed. Character segmentation fallback disabled.")
    print("Install with: pip install scikit-image")

# Lazy load EasyOCR
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _reader

# Config
OCR_MIN_CONF = 0.25
PLATE_MIN_CHARS = 4
PLATE_MAX_CHARS = 10

# Plate enhancement constraints
PLATE_RATIO_MIN = 2.5
PLATE_RATIO_MAX = 7.0
PLATE_AREA_MIN = 1500
PLATE_AREA_MAX = 50000


def _safe_resize(img, *, width=None, height=None):
    """Resize helper. Keyword-only arguments to prevent positional confusion."""
    if img is None or img.size == 0:
        return img
    if width is not None:
        h, w = img.shape[:2]
        if w == 0:
            return img
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    elif height is not None:
        h, w = img.shape[:2]
        if h == 0:
            return img
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        return img
    return cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)


def _enhance_plate(plate_crop):
    """
    Traditional CV preprocessing to isolate plate region before OCR:
    Gaussian blur -> Sobel X edges -> Otsu threshold -> Morphological close
    -> Validate plate contour -> Adaptive threshold -> Return cleaned plate
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    # Upscale small plates
    h, w = plate_crop.shape[:2]
    if h < 40:
        plate_crop = _safe_resize(plate_crop, height=100)

    # 1. Blur + grayscale
    blurred = cv2.GaussianBlur(plate_crop, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 2. Sobel X for vertical edges
    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

    # 3. Otsu threshold
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours to validate plate region
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return plate_crop  # fallback

    best_crop = None
    best_score = -1
    ph, pw = plate_crop.shape[:2]

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        ratio = float(cw) / float(ch) if ch > 0 else 0
        if ratio < 1:
            ratio = 1.0 / ratio

        if (PLATE_AREA_MIN < area < PLATE_AREA_MAX and 
            PLATE_RATIO_MIN < ratio < PLATE_RATIO_MAX):
            x = max(0, x)
            y = max(0, y)
            cw = min(cw, pw - x)
            ch = min(ch, ph - y)
            candidate = plate_crop[y:y+ch, x:x+cw]
            if candidate.size > 0 and area > best_score:
                best_score = area
                best_crop = candidate

    if best_crop is not None and best_crop.size > 0:
        plate_crop = best_crop

    # 6. Final adaptive threshold
    gray_final = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    cleaned_thresh = cv2.adaptiveThreshold(
        gray_final, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return cv2.cvtColor(cleaned_thresh, cv2.COLOR_GRAY2BGR)


def _segment_chars(plate_img):
    """
    Character segmentation fallback using connected components.
    Returns list of individual character images.
    """
    if not HAS_SKIMAGE or plate_img is None or plate_img.size == 0:
        return []

    plate_resized = _safe_resize(plate_img, width=400)
    if plate_resized is None or plate_resized.size == 0:
        return []

    # HSV Value channel
    hsv = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2HSV)
    V = cv2.split(hsv)[2]

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(V, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Connected components
    labels = measure.label(thresh, background=0)
    char_candidates = np.zeros(thresh.shape, dtype='uint8')

    for label in np.unique(labels):
        if label == 0:
            continue

        label_mask = np.zeros(thresh.shape, dtype='uint8')
        label_mask[labels == label] = 255

        cnts = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if len(cnts) == 0:
            continue

        c = max(cnts, key=cv2.contourArea)
        boxX, boxY, boxW, boxH = cv2.boundingRect(c)

        aspect_ratio = boxW / float(boxH) if boxH > 0 else 0
        solidity = cv2.contourArea(c) / float(boxW * boxH) if boxW*boxH > 0 else 0
        height_ratio = boxH / float(plate_resized.shape[0])

        # Filter character contours
        if (aspect_ratio < 1.0 and 
            solidity > 0.15 and 
            0.3 < height_ratio < 0.95 and 
            boxW > 10):
            hull = cv2.convexHull(c)
            cv2.drawContours(char_candidates, [hull], -1, 255, -1)

    # Extract sorted characters
    cnts = cv2.findContours(char_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return []

    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    sorted_pairs = sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][0])
    cnts = [p[0] for p in sorted_pairs]

    chars = []
    add_pixel = 4
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        x = max(0, x - add_pixel)
        y = max(0, y - add_pixel)
        w = w + add_pixel * 2
        h = h + add_pixel * 2

        x = min(x, plate_resized.shape[1]-1)
        y = min(y, plate_resized.shape[0]-1)
        w = min(w, plate_resized.shape[1] - x)
        h = min(h, plate_resized.shape[0] - y)

        char_img = plate_resized[y:y+h, x:x+w]
        if char_img.size > 0:
            chars.append(char_img)

    return chars


def _clean_text(raw_text):
    """Normalize OCR output: keep only alphanumerics, uppercase, fix common OCR errors."""
    if not raw_text:
        return ""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

    # Conservative OCR error fixes (only last 4 chars usually numbers)
    fixes = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'Q': '0', 'D': '0'}
    result = []
    for i, c in enumerate(cleaned):
        if i >= len(cleaned) - 4 and c in fixes:
            result.append(fixes[c])
        else:
            result.append(c)
    return ''.join(result)


def _is_valid_plate(text):
    """Validate plate text: length, alphanumeric mix."""
    if not text or len(text) < PLATE_MIN_CHARS or len(text) > PLATE_MAX_CHARS:
        return False
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    return has_letter and has_digit


# Backward compatibility alias
_looks_like_plate = _is_valid_plate


def read_plate(plate_crop_bgr, min_conf=0.25):
    """
    Main OCR function.
    Strategy 1: EasyOCR on PlateFinder-enhanced full plate.
    Strategy 2: Character segmentation + individual EasyOCR if strategy 1 fails.
    """
    if plate_crop_bgr is None or plate_crop_bgr.size == 0:
        return None, 0

    # Strategy 1: Full plate with enhancement
    enhanced = _enhance_plate(plate_crop_bgr)
    if enhanced is not None and enhanced.size > 0:
        results = get_reader().readtext(enhanced, detail=1, paragraph=False,
                                  allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        for r in results:
            _, text, conf = r
            if conf >= min_conf:
                cleaned = _clean_text(text)
                if _is_valid_plate(cleaned):
                    return cleaned, conf

    # Strategy 2: Segment characters and OCR individually
    chars = _segment_chars(plate_crop_bgr)
    if len(chars) >= PLATE_MIN_CHARS:
        plate_text = ""
        total_conf = 0
        valid_count = 0

        for char_img in chars:
            char_img = cv2.resize(char_img, (64, 64), interpolation=cv2.INTER_CUBIC)
            results = get_reader().readtext(char_img, detail=1, paragraph=False,
                                      allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if results:
                _, text, conf = results[0]
                if conf >= min_conf * 0.6:
                    c = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if c:
                        plate_text += c
                        total_conf += conf
                        valid_count += 1

        if valid_count >= PLATE_MIN_CHARS:
            avg_conf = total_conf / valid_count
            if _is_valid_plate(plate_text):
                return plate_text, avg_conf

    return None, 0