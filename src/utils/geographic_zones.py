"""
extract_zones_fixed.py

Solution complète : Détection, recalibration automatique de la carte + OCR localisé amélioré.
Dépendances :
  - opencv-python
  - numpy
  - pytesseract
  - Pillow (PIL)
Installe : pip install opencv-python pytesseract pillow numpy
Assure-toi que tesseract est installé et accessible dans le PATH.
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Tuple, Dict, Optional
from datetime import datetime

# ---------------------------
# Zones (en % sur la carte normalisée)
# "Deux fois plus larges" que valeurs classiques
# ---------------------------
ZONES_STD = {
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

def get_zone_coords(zone_percent: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[int,int,int,int]:
    x_min_pct, y_min_pct, x_max_pct, y_max_pct = zone_percent
    x_min = int(img_width * x_min_pct / 100.0)
    y_min = int(img_height * y_min_pct / 100.0)
    x_max = int(img_width * x_max_pct / 100.0)
    y_max = int(img_height * y_max_pct / 100.0)
    x_min = max(0, min(x_min, img_width-1))
    y_min = max(0, min(y_min, img_height-1))
    x_max = max(0, min(x_max, img_width))
    y_max = max(0, min(y_max, img_height))
    return (x_min, y_min, x_max, y_max)

def detect_map_bbox(half_img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    h, w = half_img.shape[:2]
    gray = cv2.cvtColor(half_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    candidates = []
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        area = ww * hh
        if area < (w*h)*0.01:
            continue
        center_x = x + ww/2
        rightness = center_x / w
        score = area * (1 + max(0, (rightness-0.5))*2)
        candidates.append((score, (x,y,ww,hh)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, (x,y,ww,hh) = candidates[0]
    pad_x = int(ww * 0.03); pad_y = int(hh * 0.03)
    x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
    x1 = min(w, x + ww + pad_x); y1 = min(h, y + hh + pad_y)
    return (x0, y0, x1, y1)

def normalize_map(map_img: np.ndarray, size: Tuple[int,int]=(1024,1024)) -> np.ndarray:
    return cv2.resize(map_img, size, interpolation=cv2.INTER_CUBIC)

def preprocess_for_ocr(zone_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
    scale = 3
    h, w = gray.shape[:2]
    gray_up = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.medianBlur(gray_up, 3)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return processed

def ocr_zone_read(processed_img) -> Optional[Tuple[int,int]]:
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/'
    text = pytesseract.image_to_string(processed_img, config=config)
    if not text:
        return None
    text = text.strip().replace(' ', '').replace('\n','/')
    m = re.search(r'(\d{1,2})\s*[\/\\]\s*(\d{1,2})', text)
    if not m:
        text_corr = text.replace('O','0').replace('o','0').replace('l','1').replace('I','1')
        m = re.search(r'(\d{1,2})[\/\\](\d{1,2})', text_corr)
    if m:
        try:
            tmin = int(m.group(1))
            tmax = int(m.group(2))
            if 10 <= tmin <= 40 and 20 <= tmax <= 50 and tmin < tmax:
                return (tmin, tmax)
        except ValueError:
            return None
    return None

def ocr_global_and_associate(map_img: np.ndarray, zones_pct: Dict[str, Tuple[float,float,float,float]]):
    h, w = map_img.shape[:2]
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d = pytesseract.image_to_data(th, config=r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/ ', output_type=pytesseract.Output.DICT)
    results = []
    n = len(d['text'])
    for i in range(n):
        txt = d['text'][i].strip()
        if not txt:
            continue
        if re.search(r'\d{1,2}[\/\\]\d{1,2}', txt):
            x = d['left'][i] + d['width'][i]//2
            y = d['top'][i] + d['height'][i]//2
            results.append((x,y,txt))
    associations = {}
    zone_centers = {}
    for name, pct in zones_pct.items():
        x0,y0,x1,y1 = get_zone_coords(pct, w, h)
        cx = (x0+x1)//2; cy = (y0+y1)//2
        zone_centers[name] = (cx,cy)
    for x,y,txt in results:
        t = re.search(r'(\d{1,2})[\/\\](\d{1,2})', txt)
        if not t:
            continue
        tmin = int(t.group(1)); tmax = int(t.group(2))
        best = None; best_dist = float('inf')
        for name, (cx,cy) in zone_centers.items():
            d2 = (cx-x)**2 + (cy-y)**2
            if d2 < best_dist:
                best = name; best_dist = d2
        if best:
            associations[best] = (tmin, tmax)
    return associations

def extract_temperatures_from_map(map_img: np.ndarray,
                                  zones_pct: Dict[str, Tuple[float,float,float,float]],
                                  normalized_size: Tuple[int,int]=(1024,1024)) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    map_norm = normalize_map(map_img, size=normalized_size)
    h, w = map_norm.shape[:2]
    results = {}
    for name, pct in zones_pct.items():
        x0,y0,x1,y1 = get_zone_coords(pct, w, h)
        zone = map_norm[y0:y1, x0:x1]
        if zone.size == 0:
            results[name] = (None, None)
            continue
        proc = preprocess_for_ocr(zone)
        ocr_res = ocr_zone_read(proc)
        results[name] = ocr_res or (None, None)
    none_count = sum(1 for v in results.values() if v == (None, None))
    if none_count > len(results)*0.35:
        fallback = ocr_global_and_associate(map_norm, zones_pct)
        for k,v in fallback.items():
            results[k] = v
    return results

def process_page_halves(full_page_img: np.ndarray,
                        split_horizontal: bool = True,
                        zones_pct: Dict[str, Tuple[float,float,float,float]] = ZONES_STD):
    h,w = full_page_img.shape[:2]
    if split_horizontal:
        mid = h // 2
        top = full_page_img[0:mid, :]
        bottom = full_page_img[mid:h, :]
    else:
        top = full_page_img
        bottom = full_page_img
    results = {}
    for label, half in [('observations', top), ('previsions', bottom)]:
        bbox = detect_map_bbox(half)
        if bbox is None:
            x0 = int(w*0.5); y0 = 0; x1 = w; y1 = half.shape[0]
            map_crop = half[y0:y1, x0:x1]
        else:
            x0,y0,x1,y1 = bbox
            map_crop = half[y0:y1, x0:x1]
        temps = extract_temperatures_from_map(map_crop, zones_pct)
        results[label] = temps
    return results

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    from PIL import Image
    if len(sys.argv) < 2:
        print("Usage: python extract_zones_fixed.py <page_image.png>")
        sys.exit(1)
    img_path = sys.argv[1]
    bgr = cv2.imread(img_path)
    if bgr is None:
        pil = Image.open(img_path).convert('RGB')
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    output = process_page_halves(bgr, split_horizontal=True, zones_pct=ZONES_STD)
    print("Extraction résultats :")
    import json
    print(json.dumps(output, indent=2, ensure_ascii=False))
