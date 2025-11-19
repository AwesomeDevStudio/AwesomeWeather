import os
from pathlib import Path

# ---- Chemins de base
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
IMAGE_DIR = PROCESSED_DIR / "images"
DATA_DIR = PROCESSED_DIR / "data"
DB_PATH = BASE_DIR / "database" / "meteo.db"

# ---- POPPLER PATH (pour pdf2image)
POPPLER_PATH = r"C:\poppler-25.07.0\Library\bin"

# ---- Créer tous les dossiers nécessaires au démarrage
for dir_path in [UPLOAD_DIR, IMAGE_DIR, DATA_DIR, DB_PATH.parent]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ---- Configuration Ollama (pour IA locale optionnelle)
OLLAMA_MODEL = "llava:7b"
OLLAMA_HOST = "http://localhost:11434"

# ---- Stations météorologiques du Burkina Faso
STATIONS_BF = [
    "Ouagadougou", "Bobo-Dioulasso", "Ouahigouya", 
    "Dédougou", "Fada N'Gourma", "Banfora", 
    "Koudougou", "Tenkodogo", "Diapaga", "Tougan",
    "Gaoua", "Dori", "Bogandé", "Kantchari", "Pô", "Boromo"
]

# ---- Plages de températures valides pour le Burkina Faso (°C)
TEMP_MIN_RANGE = (15, 35)  # Température minimale
TEMP_MAX_RANGE = (25, 45)  # Température maximale

# ---- Configuration OCR Tesseract (si besoin de spécifier le chemin)
# Décommente et adapte si Tesseract n'est pas dans le PATH système
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Configuration logging (optionnel)
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = BASE_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
