from typing import List, Dict, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

def validate_temperature_data(data: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Valide et nettoie les données de température"""
    
    validated = []
    errors = []
    
    for entry in data:
        station = entry['station']
        temp_min = entry['temp_min']
        temp_max = entry['temp_max']
        
        # Vérifications
        if not (config.TEMP_MIN_RANGE[0] <= temp_min <= config.TEMP_MIN_RANGE[1]):
            errors.append(f"❌ {station}: Temp min {temp_min}°C hors plage {config.TEMP_MIN_RANGE}")
            continue
        
        if not (config.TEMP_MAX_RANGE[0] <= temp_max <= config.TEMP_MAX_RANGE[1]):
            errors.append(f"❌ {station}: Temp max {temp_max}°C hors plage {config.TEMP_MAX_RANGE}")
            continue
        
        if temp_min >= temp_max:
            errors.append(f"❌ {station}: Min ({temp_min}°C) >= Max ({temp_max}°C)")
            continue
        
        entry['validated'] = True
        validated.append(entry)
    
    return validated, errors


def deduplicate_stations(data: List[Dict]) -> List[Dict]:
    """Supprime les doublons (garde la dernière occurrence)"""
    unique = {}
    for entry in data:
        unique[entry['station']] = entry
    return list(unique.values())
