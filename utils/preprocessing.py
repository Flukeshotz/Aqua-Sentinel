import cv2
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def preprocess_inputs(rgb_image, sar_image):
    """
    Preprocesses RGB and SAR images using central Config.
    Returns tensors with batch dimension added (1, C, H, W).
    """
    target_size = Config.IMAGE_SIZE

    # Resize
    if rgb_image.shape[:2] != target_size:
        rgb_image = cv2.resize(rgb_image, target_size)
    if sar_image.shape[:2] != target_size:
        sar_image = cv2.resize(sar_image, target_size)

    # Normalize RGB [0, 1]
    rgb_norm = rgb_image.astype(np.float32) / 255.0
    
    # Normalize SAR [0, 1]
    sar_norm = sar_image.astype(np.float32) / 255.0

    # Convert to Tensors: HWC to CHW
    rgb_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).float()
    sar_tensor = torch.from_numpy(sar_norm).permute(2, 0, 1).float()

    # Add batch dimension
    rgb_tensor = rgb_tensor.unsqueeze(0)
    sar_tensor = sar_tensor.unsqueeze(0)

    return rgb_tensor, sar_tensor
