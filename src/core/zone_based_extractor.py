import cv2
import pytesseract
from pathlib import Path
from typing import List, Dict
import sys
import re
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
from src.utils.geographic_zones import (
    ZONES_OBSERVATIONS, 
    ZONES_PREVISIONS,
    get_zone_coords
)

# Configuration Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_zone(zone_img):
    """Preprocessing simple mais efficace pour petites zones"""
    
    # Upscale x2
    h, w = zone_img.shape[:2]
    zone_upscaled = cv2.resize(zone_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    
    # Grayscale si couleur
    if len(zone_upscaled.shape) == 3:
        gray = cv2.cvtColor(zone_upscaled, cv2.COLOR_BGR2GRAY)
    else:
        gray = zone_upscaled
    
    # Augmenter contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Binarisation
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Vérifier orientation (texte noir sur fond blanc)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    
    return binary


def extract_by_geographic_zones(image_path: Path, type_data: str = "observation", debug: bool = False) -> List[Dict]:
    """Extraction basée sur zones géographiques fixes"""
    
    print(f"🗺️ Extraction par zones géographiques : {type_data}")
    
    # Charger image ORIGINALE (pas preprocessée globalement)
    img_path_str = str(image_path)
    img = cv2.imread(img_path_str)
    
    if img is None:
        print(f"❌ Impossible de charger image")
        return []
    
    height, width = img.shape[:2]
    print(f"📐 Image : {width}x{height} pixels")
    
    # Choisir zones selon type
    zones = ZONES_OBSERVATIONS if type_data == "observation" else ZONES_PREVISIONS
    
    data = []
    
    # Créer dossier debug si nécessaire
    if debug:
        debug_dir = image_path.parent / "debug_zones"
        debug_dir.mkdir(exist_ok=True)
    
    # Extraire température pour chaque zone
    for station, zone_pct in zones.items():
        # Convertir % en pixels
        zone_coords = get_zone_coords(zone_pct, width, height)
        x_min, y_min, x_max, y_max = zone_coords
        
        print(f"🔍 Zone {station} : ({x_min},{y_min}) → ({x_max},{y_max})")
        
        # Découper zone de l'image ORIGINALE
        zone_img = img[y_min:y_max, x_min:x_max]
        
        # Preprocessing de la zone
        zone_processed = preprocess_zone(zone_img)
        
        # Sauvegarder zone debug
        if debug:
            zone_filename = f"{station.replace(' ', '_').replace("'", '')}_{type_data}.png"
            cv2.imwrite(str(debug_dir / zone_filename), zone_processed)
        
        # OCR ciblé avec plusieurs configs
        configs_to_try = [
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/ ',  # Ligne unique
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789/ ',  # Mot unique
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789/ ', # Texte brut
        ]
        
        best_text = ""
        
        for config_ocr in configs_to_try:
            try:
                text = pytesseract.image_to_string(zone_processed, config=config_ocr).strip()
                if len(text) > len(best_text):
                    best_text = text
            except:
                continue
        
        # Chercher température dans le texte
        match = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})', best_text)
        
        if match:
            temp_min = int(match.group(1))
            temp_max = int(match.group(2))
            
            # Validation
            if 15 <= temp_min <= 35 and 25 <= temp_max <= 45 and temp_min < temp_max:
                data.append({
                    'station': station,
                    'temp_min': temp_min,
                    'temp_max': temp_max,
                    'type': type_data,
                    'confidence': 'high',
                    'method': 'zone_geographic',
                    'raw_text': best_text
                })
                print(f"   ✅ {station}: {temp_min}/{temp_max}°C")
            else:
                print(f"   ⚠️ {station}: températures invalides ({temp_min}/{temp_max}) - texte: '{best_text}'")
        else:
            print(f"   ❌ {station}: pas de température détectée - texte: '{best_text}'")
    
    print(f"📊 Total extrait : {len(data)}/{len(zones)} stations")
    
    return data
