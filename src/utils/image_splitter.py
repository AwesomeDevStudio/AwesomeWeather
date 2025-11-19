from PIL import Image
from pathlib import Path
import unicodedata
import re


def normalize_filename(filename: str) -> str:
    """Nettoie le nom de fichier (enlève accents)"""
    nfkd = unicodedata.normalize('NFKD', filename)
    filename_ascii = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    filename_clean = re.sub(r'[^\w\s.-]', '_', filename_ascii)
    filename_clean = re.sub(r'\s+', '_', filename_clean)
    return filename_clean


def split_bulletin_image(image_path: Path) -> tuple:
    """Découpe le bulletin pour isoler les CARTES"""
    
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Nettoyer le nom de base
        stem_clean = normalize_filename(image_path.stem)
        
        # Observations : 30% à 60% de la hauteur
        obs_top = int(height * 0.30)
        obs_bottom = int(height * 0.60)
        obs_img = img.crop((0, obs_top, width, obs_bottom))
        obs_path = image_path.parent / f"{stem_clean}_OBS_carte.png"
        obs_img.save(obs_path, 'PNG')
        
        # Prévisions : 60% à 95% de la hauteur
        prev_top = int(height * 0.60)
        prev_bottom = int(height * 0.95)
        prev_img = img.crop((0, prev_top, width, prev_bottom))
        prev_path = image_path.parent / f"{stem_clean}_PREV_carte.png"
        prev_img.save(prev_path, 'PNG')
        
        print(f"✂️ Zones cartes isolées : {image_path.name}")
        print(f"   📊 Observations sauvegardé: {obs_path.name}")
        print(f"   🔮 Prévisions sauvegardé: {prev_path.name}")
        
        return obs_path, prev_path
    
    except Exception as e:
        print(f"❌ Erreur découpage: {e}")
        return image_path, image_path
