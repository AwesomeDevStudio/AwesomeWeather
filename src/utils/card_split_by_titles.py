import cv2
import pytesseract
from pathlib import Path


def decoupe_par_titres(img_path_or_arr, lang="fra"):
    """
    Détecte les titres 'Le temps des dernières 24 heures' et 
    'Prévisions valables jusqu’à demain 12 heures' pour découper la page en deux zones images :
    observations (haut) et prévisions (bas).
    Args:
        img_path_or_arr: chemin vers image ou np.ndarray déjà chargée
        lang: langue tesseract
    Returns:
        obs_img, prev_img (cv2 images)
    """
    # Charger image
    if isinstance(img_path_or_arr, str) or isinstance(img_path_or_arr, Path):
        img = cv2.imread(str(img_path_or_arr))
    else:
        img = img_path_or_arr

    h, w = img.shape[:2]

    # OCR de la structure et récupération des positions
    d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang=lang, config="--psm 6")
    nboxes = len(d['level'])
    
    y_obs_start, y_prev_start, y_obs_end = 0, h//2, h//2  # valeurs par défaut si on ne trouve pas les titres
    found_obs, found_prev = False, False
    for i in range(nboxes):
        line = d['text'][i].strip().lower()
        # On tolère des variantes de guillemets ou apostrophes
        if "le temps des dernières 24 heures" in line:
            y_obs_start = d['top'][i]
            found_obs = True
        if ("prévisions valables jusqu’à demain" in line or "prévisions valables jusqu'a demain" in line):
            y_obs_end = d['top'][i]
            y_prev_start = d['top'][i]
            found_prev = True
            break

    # Si jamais un titre n'est pas trouvé
    if not found_obs:
        y_obs_start = 0
    if not found_prev:
        y_obs_end = h // 2
        y_prev_start = h // 2

    obs_img = img[y_obs_start:y_obs_end, :]
    prev_img = img[y_prev_start:h, :]

    return obs_img, prev_img
