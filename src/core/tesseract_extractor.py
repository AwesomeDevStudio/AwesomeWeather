import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
from pathlib import Path
from typing import List, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Configuration Tesseract (chemin Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_tesseract(image_path: Path, save_debug: bool = False) -> str:
    """Extrait texte avec config optimale pour cartes météo"""
    
    from src.utils.image_preprocessing import enhance_image_for_ocr
    
    processed_img = enhance_image_for_ocr(image_path, save_debug=save_debug)
    
    if processed_img is None:
        return ""
    
    # Config spéciale pour cartes (texte épars)
    # --psm 11 : Texte épars sans ordre particulier (PARFAIT pour cartes)
    # --psm 6 : Bloc uniforme (pour texte normal)
    
    configs = [
        r'--oem 3 --psm 11 -l fra -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzàâäéèêëïîôùûüç0123456789/-:° ',
        r'--oem 3 --psm 6 -l fra',
        r'--oem 3 --psm 12 -l fra'  # Texte épars avec OSD
    ]
    
    best_text = ""
    best_score = 0
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(processed_img, config=config)
            
            # Scorer : compter combien de patterns XX/YY on trouve
            temp_patterns = len(re.findall(r'\d{1,2}/\d{1,2}', text))
            
            if temp_patterns > best_score:
                best_score = temp_patterns
                best_text = text
                print(f"✅ Config trouvée {temp_patterns} températures")
        
        except Exception as e:
            continue
    
    print(f"📝 Meilleur résultat : {len(best_text)} caractères, {best_score} températures")
    
    return best_text


def parse_temperatures_from_text(text: str, type_data: str = "observation") -> List[Dict]:
    """Parse avec détection intelligente de proximité ville-température"""
    
    import re
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import config
    
    data = []
    
    # Étape 1 : Extraire toutes les températures avec leur position
    temp_pattern = r'(\d{1,2})/(\d{1,2})'
    temperatures = []
    
    for match in re.finditer(temp_pattern, text):
        temp_min = int(match.group(1))
        temp_max = int(match.group(2))
        position = match.start()
        
        # Validation
        if 15 <= temp_min <= 35 and 25 <= temp_max <= 45 and temp_min < temp_max:
            temperatures.append({
                'min': temp_min,
                'max': temp_max,
                'position': position,
                'matched': False
            })
            print(f"🌡️ Température trouvée : {temp_min}/{temp_max} à position {position}")
    
    if not temperatures:
        print("⚠️ Aucune température valide détectée")
        return []
    
    # Étape 2 : Chercher les noms de villes avec leur position
    stations_found = []
    
    for station in config.STATIONS_BF:
        # Chercher station (insensible à la casse, espaces flexibles)
        station_pattern = r'\b' + re.escape(station).replace(r'\ ', r'[\s-]*') + r'\b'
        
        for match in re.finditer(station_pattern, text, re.IGNORECASE):
            stations_found.append({
                'name': station,
                'position': match.start(),
                'matched': False
            })
            print(f"📍 Station trouvée : {station} à position {match.start()}")
    
    if not stations_found:
        print("⚠️ Aucune station reconnue, recherche de noms génériques...")
        
        # Fallback : chercher mots en majuscules (potentiels noms de villes)
        generic_pattern = r'\b[A-Z]{3,}(?:\s+[A-Z]{3,})?\b'
        for match in re.finditer(generic_pattern, text):
            name = match.group(0).strip()
            if len(name) >= 4 and name not in ['ANAM', 'BULLETIN', 'MARS', 'LEGENDZ']:
                stations_found.append({
                    'name': name.title(),
                    'position': match.start(),
                    'matched': False
                })
                print(f"📍 Nom potentiel trouvé : {name} à position {match.start()}")
    
    # Étape 3 : Associer stations et températures par proximité
    for station in stations_found:
        if station['matched']:
            continue
        
        # Chercher la température la plus proche (dans les 200 caractères)
        min_distance = float('inf')
        closest_temp = None
        
        for temp in temperatures:
            if temp['matched']:
                continue
            
            distance = abs(temp['position'] - station['position'])
            
            if distance < min_distance and distance < 200:  # Seuil de proximité
                min_distance = distance
                closest_temp = temp
        
        if closest_temp:
            data.append({
                'station': station['name'],
                'temp_min': closest_temp['min'],
                'temp_max': closest_temp['max'],
                'type': type_data,
                'confidence': 'high' if min_distance < 50 else 'medium',
                'method': 'tesseract',
                'raw_text': f"{station['name']} {closest_temp['min']}/{closest_temp['max']}"
            })
            
            # Marquer comme appariés
            station['matched'] = True
            closest_temp['matched'] = True
            
            print(f"✅ Appariement : {station['name']} → {closest_temp['min']}/{closest_temp['max']} (distance: {min_distance})")
    
    # Étape 4 : Températures orphelines (pas de station proche)
    orphan_temps = [t for t in temperatures if not t['matched']]
    
    if orphan_temps:
        print(f"⚠️ {len(orphan_temps)} température(s) orpheline(s)")
        
        for idx, temp in enumerate(orphan_temps):
            data.append({
                'station': f"Station_inconnue_{idx+1}",
                'temp_min': temp['min'],
                'temp_max': temp['max'],
                'type': type_data,
                'confidence': 'low',
                'method': 'tesseract_orphan',
                'raw_text': f"{temp['min']}/{temp['max']}"
            })
    
    # Dédoublonner
    seen = set()
    unique_data = []
    for entry in data:
        key = (entry['station'], entry['temp_min'], entry['temp_max'])
        if key not in seen:
            seen.add(key)
            unique_data.append(entry)
    
    print(f"📊 Total : {len(unique_data)} station(s) extraite(s)")
    
    return unique_data



def extract_temperatures_tesseract(image_path: Path, type_data: str = "observation", debug: bool = False) -> List[Dict]:
    """
    Fonction principale : extraction complète avec Tesseract
    
    Args:
        image_path: Chemin vers l'image
        type_data: "observation" ou "prevision"
        debug: Si True, sauvegarde les images de preprocessing
    """
    
    print(f"🔍 Extraction Tesseract : {image_path.name}")
    
    # Étape 1 : OCR avec preprocessing
    text = extract_text_tesseract(image_path, save_debug=debug)
    
    if len(text) < 20:
        print(f"⚠️ Très peu de texte extrait ({len(text)} car)")
        return []
    
    print(f"📄 Texte brut extrait ({len(text)} caractères):")
    print(text[:500])
    
    # Étape 2 : Parsing
    data = parse_temperatures_from_text(text, type_data)
    
    print(f"✅ {len(data)} stations extraites")
    
    if len(data) == 0:
        print(f"⚠️ AUCUNE donnée extraite !")
        print(f"Texte complet :")
        print(text)
    
    return data
