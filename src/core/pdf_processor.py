from pdf2image import convert_from_path
from pathlib import Path
from typing import List
import sys
import os
import unicodedata
import re

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def normalize_filename(filename: str) -> str:
    """
    Nettoie le nom de fichier :
    - Enlève accents
    - Remplace espaces par underscores
    - Enlève caractères spéciaux
    """
    # Enlever accents
    nfkd = unicodedata.normalize('NFKD', filename)
    filename_ascii = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    
    # Remplacer caractères spéciaux par underscore
    filename_clean = re.sub(r'[^\w\s.-]', '_', filename_ascii)
    
    # Remplacer espaces multiples par un seul underscore
    filename_clean = re.sub(r'\s+', '_', filename_clean)
    
    return filename_clean


def pdf_to_images(pdf_path: str, output_folder: Path = None) -> List[Path]:
    """Convertit PDF en images PNG haute résolution"""
    
    if output_folder is None:
        output_folder = config.IMAGE_DIR
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Nettoyer le nom du PDF
    pdf_name_original = Path(pdf_path).stem
    pdf_name_clean = normalize_filename(pdf_name_original)
    
    print(f"📄 PDF original: {pdf_name_original}")
    print(f"📄 PDF nettoyé: {pdf_name_clean}")
    
    try:
        images = convert_from_path(
            pdf_path, 
            dpi=200,  # Résolution optimale
            fmt='png',
            poppler_path=config.POPPLER_PATH
        )
        
        image_paths = []
        for i, image in enumerate(images):
            # Utiliser le nom nettoyé
            img_path = output_folder / f"{pdf_name_clean}_page_{i+1}.png"
            image.save(img_path, 'PNG', quality=95)
            image_paths.append(img_path)
            print(f"✅ Sauvegardé: {img_path.name}")
        
        return image_paths
    
    except Exception as e:
        print(f"❌ Erreur conversion PDF: {e}")
        print(f"Chemin Poppler utilisé: {config.POPPLER_PATH}")
        raise
