import os
import torch

class Config:
    # Model Configuration
    WEIGHTS_PATH = os.path.join("checkpoints", "best_model_v8.pth")
    IMAGE_SIZE = (256, 256)
    
    # Inference Parameters
    THRESHOLD = 0.5
    
    # Postprocessing
    MIN_AREA = 500
    MORPH_KERNEL_SIZE = 5
    
    # Hardware (Auto-detect)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        USE_HALF_PRECISION = True
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        USE_HALF_PRECISION = False # MPS standardly prefers fp32
    else:
        DEVICE = torch.device("cpu")
        USE_HALF_PRECISION = False

    # Output structure
    OUTPUT_BASE = "outputs"
    OUTPUT_MASKS = os.path.join(OUTPUT_BASE, "masks")
    OUTPUT_OVERLAYS = os.path.join(OUTPUT_BASE, "overlays")
    OUTPUT_HEATMAPS = os.path.join(OUTPUT_BASE, "heatmaps")
    OUTPUT_COMPS = os.path.join(OUTPUT_BASE, "comparisons")
    OUTPUT_RAW = os.path.join(OUTPUT_BASE, "raw")
    OUTPUT_BEST_WORST = os.path.join(OUTPUT_BASE, "best_worst")

    @classmethod
    def ensure_dirs(cls):
        for directory in [cls.OUTPUT_MASKS, cls.OUTPUT_OVERLAYS, cls.OUTPUT_HEATMAPS, 
                          cls.OUTPUT_COMPS, cls.OUTPUT_RAW, cls.OUTPUT_BEST_WORST]:
            os.makedirs(directory, exist_ok=True)
        
# Initialize directories on import
Config.ensure_dirs()
