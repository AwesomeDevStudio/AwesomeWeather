"""
card_zone_extractor.py

But :
- Détecter automatiquement le rectangle exact de la carte météo dans une image.
- Normaliser la carte (resize) pour rendre les zones % stables.
- Extraire pour chaque zone (12 stations) Tmin/Tmax avec pré-traitement et heuristiques de fallback.
- Validation désactivée : accepte toutes les valeurs détectées par OCR.

Dépendances :
- opencv-python
- numpy
- pytesseract (et Tesseract installé sur la machine)
"""

import cv2
import numpy as np
import pytesseract
import re
from datetime import datetime
from typing import Tuple, Dict, Optional

# Configuration Tesseract (adapte selon ton installation)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Zones en % (coordonnées élargies 2x)
ZONES_STANDARD = {
    'Dori': (32, 0, 83, 25),
    'Ouahigouya': (7, 2, 63, 43),
    'Bogandé': (55, 5, 105, 55),
    'Dédougou': (0, 15, 50, 70),
    'Ouagadougou': (15, 15, 85, 70),
    "Fada N'Gourma": (55, 15, 110, 70),
    'Koudougou': (5, 25, 75, 90),
    'Boromo': (0, 25, 60, 90),
    'Tenkodogo': (30, 25, 100, 90),
    'Bobo-Dioulasso': (0, 40, 80, 115),
    'Pô': (10, 40, 100, 115),
    'Gaoua': (0, 70, 60, 130),
}

def get_zone_coords(zone_percent: tuple, img_width: int, img_height: int) -> tuple:
    x_min_pct, y_min_pct, x_max_pct, y_max_pct = zone_percent
    x_min = int(img_width * x_min_pct / 100)
    y_min = int(img_height * y_min_pct / 100)
    x_max = int(img_width * x_max_pct / 100)
    y_max = int(img_height * y_max_pct / 100)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_width, x_max), min(img_height, y_max)
    return (x_min, y_min, x_max, y_max)

def detect_card_bbox(img: np.ndarray, pad_px: int = 10) -> Tuple[int, int, int, int]:
    """Détection robuste du cadre de la carte par détection des zones non-blanches"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, cw, ch = cv2.boundingRect(contours[0])
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(w, x + cw + pad_px)
    y2 = min(h, y + ch + pad_px)
    return (x1, y1, x2, y2)

def normalize_card(card_img: np.ndarray, target_width: int = 1024, target_height: int = None) -> np.ndarray:
    """Resize la carte pour une taille stable"""
    h, w = card_img.shape[:2]
    if target_height is None:
        scale = target_width / w
        target_height = int(h * scale)
    resized = cv2.resize(card_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return resized

def preprocess_zone_for_ocr(zone_img: np.ndarray, upscale: int = 3) -> np.ndarray:
    """Preprocessing avancé pour OCR (upscale augmenté à 3)"""
    if upscale > 1:
        zone_img = cv2.resize(zone_img, (zone_img.shape[1] * upscale, zone_img.shape[0] * upscale),
                              interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    th = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    denoised = cv2.medianBlur(th, 3)
    gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=3)
    unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
    return unsharp

def sanitize_ocr_text(raw: str) -> str:
    s = raw.strip()
    s = s.replace('O', '0').replace('o', '0')
    s = s.replace('l', '1').replace('I', '1').replace('|', '1')
    s = s.replace('S', '5').replace('s', '5')
    s = re.sub(r'[^0-9/\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_temperature_from_zone(image: np.ndarray, zone_coords: tuple, debug: bool = False) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Extraction température dans une zone - VALIDATION DÉSACTIVÉE"""
    x_min, y_min, x_max, y_max = zone_coords
    h, w = image.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    x_max = max(0, min(w, x_max))
    y_min = max(0, min(h - 1, y_min))
    y_max = max(0, min(h, y_max))
    zone_img = image[y_min:y_max, x_min:x_max]
    prep = preprocess_zone_for_ocr(zone_img, upscale=3)
    config_ocr = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/ '
    try:
        raw = pytesseract.image_to_string(prep, config=config_ocr)
    except Exception:
        raw = ""
    raw_s = sanitize_ocr_text(raw)
    match = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})', raw_s)
    if not match:
        m2 = re.findall(r'\d{1,2}', raw_s)
        if len(m2) >= 2:
            temp_min = int(m2[0])
            temp_max = int(m2[1])
        else:
            temp_min = None
            temp_max = None
    else:
        temp_min = int(match.group(1))
        temp_max = int(match.group(2))
    
    # ✅ VALIDATION DÉSACTIVÉE + Auto-correction inversion
    if temp_min is not None and temp_max is not None:
        # Correction automatique si min > max (inversion OCR)
        if temp_min > temp_max:
            if debug:
                print(f"⚠️ Inversion détectée {temp_min}/{temp_max} → corrigé en {temp_max}/{temp_min}")
            temp_min, temp_max = temp_max, temp_min
        elif debug:
            print(f"✅ Température acceptée : {temp_min}/{temp_max}")
    
    return (temp_min, temp_max, raw_s)

def global_ocr_and_map(card_img: np.ndarray, zones_px: Dict[str, tuple]) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """Fallback OCR global + association spatiale"""
    results = {k: (None, None) for k in zones_px.keys()}
    config_ocr = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/ '
    try:
        raw = pytesseract.image_to_string(card_img, config=config_ocr)
    except Exception:
        raw = ""
    raw_s = sanitize_ocr_text(raw)
    found = re.findall(r'(\d{1,2})\s*/\s*(\d{1,2})', raw_s)
    if not found:
        return results
    try:
        data = pytesseract.image_to_data(card_img, config=config_ocr, output_type=pytesseract.Output.DICT)
    except Exception:
        data = None
    pairs = []
    if data:
        n = len(data['text'])
        for i in range(n):
            text = sanitize_ocr_text(data['text'][i])
            m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})', text)
            if m:
                x = int(data['left'][i] + data['width'][i] / 2)
                y = int(data['top'][i] + data['height'][i] / 2)
                pairs.append((int(m.group(1)), int(m.group(2)), x, y))
    else:
        for a, b in found:
            pairs.append((int(a), int(b), None, None))
    centers = {}
    for name, (x1, y1, x2, y2) in zones_px.items():
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers[name] = (cx, cy)
    for tup in pairs:
        a, b, x, y = tup
        if x is None:
            for nm, val in results.items():
                if val == (None, None):
                    results[nm] = (a, b)
                    break
            continue
        best = None
        bestd = 1e9
        for nm, (cx, cy) in centers.items():
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < bestd:
                bestd = d
                best = nm
        if results[best] == (None, None):
            results[best] = (a, b)
    return results

def process_card_image(card_img: np.ndarray,
                       zones_percent: Dict[str, tuple] = ZONES_STANDARD,
                       normalize_width: int = 1024,
                       debug: bool = False) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """Pipeline principal pour UNE carte"""
    import config
    x1, y1, x2, y2 = detect_card_bbox(card_img, pad_px=6)
    card_crop = card_img[y1:y2, x1:x2]
    norm = normalize_card(card_crop, target_width=normalize_width)
    
    # DEBUG : Sauvegarder l'image normalisée
    if debug:
        debug_path = config.IMAGE_DIR / f"DEBUG_normalized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(debug_path), norm)
        print(f"✅ Image normalisée sauvegardée : {debug_path}")
    
    H, W = norm.shape[:2]
    zones_px = {}
    for name, pct in zones_percent.items():
        pxcoords = get_zone_coords(pct, W, H)
        zones_px[name] = pxcoords
    
    results = {}
    raw_texts = {}
    for name, px in zones_px.items():
        tmin, tmax, raw = extract_temperature_from_zone(norm, px, debug=debug)
        results[name] = (tmin, tmax)
        raw_texts[name] = raw
        
        # DEBUG : Sauvegarder chaque zone découpée
        if debug:
            x_min, y_min, x_max, y_max = px
            zone_crop = norm[y_min:y_max, x_min:x_max]
            zone_debug_path = config.IMAGE_DIR / f"DEBUG_zone_{name.replace(' ', '_').replace("'", '')}_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(str(zone_debug_path), zone_crop)
    
    none_count = sum(1 for v in results.values() if v == (None, None))
    if none_count > max(2, int(0.25 * len(results))):
        if debug:
            print(f"[fallback] {none_count} zones empty -> running global OCR mapping")
        fallback = global_ocr_and_map(norm, zones_px)
        for k, v in fallback.items():
            if results[k] == (None, None) and v != (None, None):
                results[k] = v
    return results

def process_page(page_img: np.ndarray,
                 zones_percent: Dict[str, tuple] = ZONES_STANDARD,
                 normalize_width: int = 1024,
                 debug: bool = False) -> Dict[str, Dict[str, Tuple[Optional[int], Optional[int]]]]:
    """Process page complète (top/bottom split)"""
    h, w = page_img.shape[:2]
    half = h // 2
    top = page_img[0:half, :]
    bottom = page_img[half:, :]
    obs = process_card_image(top, zones_percent=zones_percent, normalize_width=normalize_width, debug=debug)
    prev = process_card_image(bottom, zones_percent=zones_percent, normalize_width=normalize_width, debug=debug)
    return {'observations': obs, 'previsions': prev}
