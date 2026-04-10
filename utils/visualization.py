import cv2
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def overlay_mask(rgb_img, mask_img, color=(0, 0, 255), alpha=0.3):
    """
    Overlays a binary mask onto an RGB image.
    """
    target_shape = Config.IMAGE_SIZE
    
    if rgb_img.shape[:2] != target_shape:
        rgb_img = cv2.resize(rgb_img, target_shape)
    if mask_img.shape[:2] != target_shape:
        mask_img = cv2.resize(mask_img, target_shape, interpolation=cv2.INTER_NEAREST)

    overlay = rgb_img.copy()
    
    # Assign color to masked pixels
    is_masked = mask_img > 127
    overlay[is_masked] = color

    # Blend original with overlay
    blended = cv2.addWeighted(rgb_img, 1.0 - alpha, overlay, alpha, 0)
    
    return blended

def heatmap_visualization(prob_map):
    """
    Generates a Jet-colormapped heatmap from raw 0..1 probability tensors.
    """
    prob_scaled = (prob_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_scaled, cv2.COLORMAP_JET)
    return heatmap

def side_by_side_comparison(rgb_img, sar_img, binary_mask, overlay_img):
    """
    Creates a horizontally stacked visualization of the analysis pipeline.
    rgb_img | sar_img | binary_mask | overlay_img
    """
    target_shape = Config.IMAGE_SIZE

    def _ensure_bgr(img):
        """Helper to equalize shapes and dimensions to BGR."""
        if img is None:
            return np.zeros(target_shape + (3,), dtype=np.uint8)
        img_resized = cv2.resize(img, target_shape)
        
        # If it's a 1-channel mask/SAR, convert to BGR visually
        if len(img_resized.shape) == 2:
            return cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        return img_resized

    vis_rgb = _ensure_bgr(rgb_img)
    vis_sar = _ensure_bgr(sar_img)
    vis_mask = _ensure_bgr(binary_mask)
    vis_overlay = _ensure_bgr(overlay_img)
    
    # Render final matrix horizontally
    return np.hstack([vis_rgb, vis_sar, vis_mask, vis_overlay])

