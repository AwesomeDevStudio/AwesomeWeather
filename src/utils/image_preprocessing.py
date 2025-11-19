import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def enhance_image_for_ocr(image_path: Path, save_debug: bool = False) -> np.ndarray:
    """
    Pipeline complet de preprocessing pour cartes météo :
    - Upscaling pour petit texte
    - Isolation du texte noir
    - Enhancement contraste
    - Binarisation
    - Nettoyage bruit
    
    Args:
        image_path: Chemin vers l'image à traiter
        save_debug: Si True, sauvegarde les étapes intermédiaires
    
    Returns:
        Image preprocessée (numpy array) ou None si erreur
    """
    
    # Convertir Path en string pour compatibilité OpenCV Windows
    img_path_str = str(image_path)
    
    # Charger image
    img = cv2.imread(img_path_str)
    
    if img is None:
        print(f"❌ Impossible de charger l'image: {image_path.name}")
        print(f"   Chemin: {img_path_str}")
        print(f"   Existe? {image_path.exists()}")
        return None
    
    print(f"✅ Image chargée: {img.shape[1]}x{img.shape[0]} pixels")
    
    # 1. Upscale x3 (crucial pour texte petit sur cartes)
    height, width = img.shape[:2]
    img_upscaled = cv2.resize(
        img, 
        (width * 3, height * 3), 
        interpolation=cv2.INTER_CUBIC
    )
    
    # 2. Conversion HSV pour isoler texte noir
    hsv = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2HSV)
    
    # 3. Masque pour isoler texte noir (températures en noir sur cartes)
    # HSV: Hue, Saturation, Value
    # Texte noir = faible Value, toute Saturation
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])  # V < 100 = sombre
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 4. Inverser masque (texte devient blanc sur fond noir)
    mask_inv = cv2.bitwise_not(mask)
    
    # 5. Conversion grayscale classique
    gray = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2GRAY)
    
    # 6. Appliquer masque pour isoler texte
    text_only = cv2.bitwise_and(gray, gray, mask=mask_inv)
    
    # 7. Augmenter contraste fortement avec CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(text_only)
    
    # 8. Réduire bruit avant binarisation
    denoised = cv2.fastNlMeansDenoising(
        enhanced, 
        None, 
        h=10, 
        templateWindowSize=7, 
        searchWindowSize=21
    )
    
    # 9. Binarisation avec seuil adaptatif (meilleur pour fond complexe)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=4
    )
    
    # 10. Morphologie : Fermer petits trous dans les lettres
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 11. Dilater légèrement pour épaissir le texte
    dilated = cv2.dilate(morph, kernel, iterations=1)
    
    # 12. Nettoyage final
    final = cv2.medianBlur(dilated, 3)
    
    # 13. Vérifier orientation (texte doit être noir sur blanc)
    white_pixels = np.sum(final == 255)
    black_pixels = np.sum(final == 0)
    
    if white_pixels < black_pixels:
        # Inverser si nécessaire
        final = cv2.bitwise_not(final)
        print("🔄 Image inversée (texte noir sur fond blanc)")
    
    # Sauvegarder images debug
    if save_debug:
        debug_dir = image_path.parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        
        print(f"💾 Sauvegarde images debug dans: {debug_dir}")
        
        # Sauvegarder chaque étape (utiliser str() pour Windows)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_1_upscaled.png"), img_upscaled)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_2_mask.png"), mask)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_3_text_only.png"), text_only)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_4_enhanced.png"), enhanced)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_5_denoised.png"), denoised)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_6_binary.png"), binary)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_7_morph.png"), morph)
        cv2.imwrite(str(debug_dir / f"{image_path.stem}_8_FINAL.png"), final)
        
        print(f"✅ 8 images debug sauvegardées")
    
    return final


def extract_text_regions(image_path: Path) -> list:
    """
    Détecte et extrait les régions contenant du texte
    (Alternative avancée pour découpe intelligente)
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        Liste de dictionnaires avec images de régions et coordonnées
    """
    
    img_path_str = str(image_path)
    img = cv2.imread(img_path_str)
    
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détection de contours
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Trouver contours
    contours, _ = cv2.findContours(
        binary, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filtrer contours de taille texte
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtrer par taille (ajuster selon PDFs)
        if 10 < w < 300 and 10 < h < 100:
            # Extraire région
            region = img[y:y+h, x:x+w]
            text_regions.append({
                'image': region,
                'bbox': (x, y, w, h)
            })
    
    return text_regions


def apply_sharpening(image: np.ndarray, strength: int = 1) -> np.ndarray:
    """
    Applique un filtre de netteté à l'image
    
    Args:
        image: Image numpy array
        strength: Intensité (1-3)
    
    Returns:
        Image avec netteté augmentée
    """
    
    # Kernel de netteté
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    
    sharpened = image.copy()
    for _ in range(strength):
        sharpened = cv2.filter2D(sharpened, -1, kernel)
    
    return sharpened


def remove_background_color(image_path: Path, target_color_bgr: tuple = (255, 200, 200)) -> np.ndarray:
    """
    Enlève une couleur d'arrière-plan spécifique (ex: fond bleu des cartes)
    
    Args:
        image_path: Chemin vers l'image
        target_color_bgr: Couleur BGR à enlever
    
    Returns:
        Image avec fond blanc
    """
    
    img = cv2.imread(str(image_path))
    
    if img is None:
        return None
    
    # Convertir en HSV pour mieux cibler couleur
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Définir range de couleur à enlever (ex: bleu)
    # Ajuster selon la couleur de vos cartes
    lower = np.array([100, 50, 50])   # Bleu foncé
    upper = np.array([130, 255, 255]) # Bleu clair
    
    # Créer masque
    mask = cv2.inRange(hsv, lower, upper)
    
    # Remplacer par blanc
    img[mask > 0] = [255, 255, 255]
    
    return img


def auto_rotate_text(image: np.ndarray) -> np.ndarray:
    """
    Détecte et corrige l'orientation du texte automatiquement
    
    Args:
        image: Image numpy array
    
    Returns:
        Image réorientée
    """
    
    # Utiliser pytesseract pour détecter orientation
    try:
        import pytesseract
        from PIL import Image as PILImage
        
        # Convertir numpy -> PIL
        pil_img = PILImage.fromarray(image)
        
        # Détecter orientation
        osd = pytesseract.image_to_osd(pil_img)
        
        # Extraire angle de rotation
        angle = int(osd.split('\n')[2].split(':')[1].strip())
        
        if angle != 0:
            print(f"🔄 Rotation détectée: {angle}°")
            
            # Créer matrice de rotation
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Appliquer rotation
            rotated = cv2.warpAffine(image, M, (w, h), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
    
    except Exception as e:
        print(f"⚠️ Auto-rotation échouée: {e}")
    
    return image
