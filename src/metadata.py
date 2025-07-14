import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy

def compute_entropy(gray_img):
    hist, _ = np.histogram(
        gray_img.flatten(), bins=256, range=(0,255), density=True
    )
    return float(scipy_entropy(hist + 1e-12))

def extract_metadata(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        'brightness': float(gray.mean()),
        'contrast':   float(gray.std()),
        'sharpness':  float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'entropy':    compute_entropy(gray),
        'height':     img.shape[0],
        'width':      img.shape[1],
    }
