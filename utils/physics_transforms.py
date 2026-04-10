import cv2
import numpy as np

def compute_physics_channels(img_bgr):
    """
    Computes rigorous physics-guided features for SAR imagery to capture 
    biological slick characteristics vs rough water topology.
    Returns a (H, W, 2) numpy array containing normalized physical maps.
    """
    # Convert BGR to grayscale for uniform physical intensity
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # Ensure float representation for transformations
    gray_f = gray.astype(np.float32)

    # 1. Log Transform: Enhances low-intensity scattering (dark oil bands)
    # Log scale strictly compresses the dynamic range of sea clutters.
    # We add 1.0 to avoid log(0)
    max_val = np.max(gray_f)
    if max_val > 0:
        c = 255.0 / np.log(1 + max_val)
        log_transformed = c * (np.log(1 + gray_f))
    else:
        log_transformed = np.zeros_like(gray_f)
    log_transformed = log_transformed.astype(np.float32) / 255.0

    # 2. Local Variance (Laplacian Gradient): Differentiates biological 
    # lookalikes (smooth transitions) from mechanical spills (sharp boundaries)
    # Using 3x3 Sobel-based laplacian kernel
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian = np.abs(laplacian)
    
    # Normalize Laplacian to [0, 1] securely
    lap_max = np.max(laplacian)
    if lap_max > 0:
        laplacian = laplacian / lap_max
        
    # Stack into a physical feature map (H, W, 2)
    physics_map = np.stack([log_transformed, laplacian], axis=-1)
    
    # 3. Secure Feature Normalization
    mean = np.mean(physics_map, axis=(0, 1), keepdims=True)
    std = np.std(physics_map, axis=(0, 1), keepdims=True)
    physics_map = (physics_map - mean) / (std + 1e-8)
    
    return physics_map
