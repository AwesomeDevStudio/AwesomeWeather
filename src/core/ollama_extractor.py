import ollama
import re
from pathlib import Path
from typing import Dict, List
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Configuration optimisée pour GPU 4GB
OLLAMA_OPTIONS_GPU_4GB = {
    'temperature': 0.0,
    'top_k': 1,
    'top_p': 0.1,
    'num_predict': 500,
    'num_ctx': 2048,
    'num_gpu': 35,
    'num_thread': 2,
    'repeat_penalty': 1.1
}

def extract_observations(image_path: Path) -> str:
    """Extrait les températures OBSERVÉES (carte HAUTE)"""
    
    prompt = """
Tu es un expert en lecture de cartes météorologiques du Burkina Faso.

Cette image montre la partie SUPÉRIEURE d'un bulletin météo avec des températures OBSERVÉES.

TÂCHE : Extraire TOUTES les villes visibles avec leurs températures au format XX/YY.

FORMAT DE SORTIE STRICT (une ligne par ville, texte brut, PAS de markdown) :
Ouagadougou: 23/35
Bobo-Dioulasso: 21/33
Dédougou: 22/34
Ouahigouya: 20/36
Fada N'Gourma: 24/37
Koudougou: 20/36

RÈGLES :
- Format EXACT : NomVille: XX/YY
- PAS de *, PAS de **, PAS de tirets
- Une ville par ligne
- N'invente rien
"""
    
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(image_path)]
            }],
            options=OLLAMA_OPTIONS_GPU_4GB
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Erreur Ollama (observations): {e}")
        return ""


def extract_previsions(image_path: Path) -> str:
    """Extrait les températures PRÉVUES (carte BASSE)"""
    
    prompt = """
Tu es un expert en lecture de cartes météorologiques du Burkina Faso.

Cette image montre la partie INFÉRIEURE d'un bulletin météo avec des températures PRÉVUES.

TÂCHE : Extraire TOUTES les villes visibles avec leurs températures au format XX/YY.

FORMAT DE SORTIE STRICT (une ligne par ville, texte brut, PAS de markdown) :
Ouagadougou: 19/37
Bobo-Dioulasso: 20/36
Dédougou: 18/38
Ouahigouya: 17/39
Fada N'Gourma: 21/38
Koudougou: 18/36

RÈGLES :
- Format EXACT : NomVille: XX/YY
- PAS de *, PAS de **, PAS de tirets
- Une ville par ligne
- N'invente rien
"""
    
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(image_path)]
            }],
            options=OLLAMA_OPTIONS_GPU_4GB
        )
        return response['message']['content']
    except Exception as e:
        print(f"❌ Erreur Ollama (prévisions): {e}")
        return ""


def parse_ollama_response(response: str, data_type: str = "observation") -> List[Dict]:
    """Parse la réponse d'Ollama et extrait les données structurées"""
    
    data = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Nettoyer markdown (* et **)
        line = line.replace('*', '').replace('**', '').strip()
        
        # Pattern : Ville: XX/YY
        match = re.search(
            r'([A-Za-zéèêàâôûùçï\s\'-]+)[:\s]+(\d{1,2})/(\d{1,2})', 
            line
        )
        
        if match:
            station_raw = match.group(1).strip()
            temp_min = int(match.group(2))
            temp_max = int(match.group(3))
            
            # Valider le nom de station
            station_found = None
            for valid_station in config.STATIONS_BF:
                if valid_station.lower() in station_raw.lower():
                    station_found = valid_station
                    break
            
            if station_found:
                data.append({
                    'station': station_found,
                    'temp_min': temp_min,
                    'temp_max': temp_max,
                    'type': data_type,
                    'raw_text': line
                })
            else:
                # Station non reconnue mais valide
                if 15 <= temp_min <= 35 and 25 <= temp_max <= 45 and temp_min < temp_max:
                    # Ignorer stations fictives répétées
                    if station_raw.lower() not in ['boukou', 'koumbi', 'tansarga']:
                        data.append({
                            'station': station_raw,
                            'temp_min': temp_min,
                            'temp_max': temp_max,
                            'type': data_type,
                            'raw_text': line
                        })
    
    # Dédoublonner
    seen = set()
    unique_data = []
    for entry in data:
        key = (entry['station'], entry['temp_min'], entry['temp_max'])
        if key not in seen:
            seen.add(key)
            unique_data.append(entry)
    
    return unique_data
