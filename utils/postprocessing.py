import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def refine_mask(prob_map, min_area=None, use_otsu=False, threshold=None):
    """
    Refines probability map into binary mask.
    1. Thresholding (Adaptive Otsu or fixed Config threshold)
    2. Morphological Opening (remove noise speckles)
    3. Morphological Closing (bridge internal gaps)
    4. Connected Component Area Filtering
    """
    if prob_map is None:
        return None
        
    min_area = min_area if min_area is not None else Config.MIN_AREA
    
    # 1. Thresholding
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    if use_otsu:
        # Otsu computes the optimal threshold dynamically based on histogram
        _, mask_binary = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Fixed thresholding
        thresh = threshold if threshold is not None else Config.THRESHOLD
        thresh_val = int(thresh * 255)
        _, mask_binary = cv2.threshold(prob_uint8, thresh_val, 255, cv2.THRESH_BINARY)
        
    # Morphological element
    kernel = np.ones((Config.MORPH_KERNEL_SIZE, Config.MORPH_KERNEL_SIZE), np.uint8)
    
    # 2. Opening: Erosion then Dilation (Remove isolated noise points)
    mask_opened = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
    
    # 3. Closing: Dilation then Erosion (Bridge small filaments)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

    # 4. Area Thresholding (Remove remaining small blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    clean_mask = np.zeros_like(mask_closed)
    for i in range(1, num_labels): # Start from 1 to skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask
