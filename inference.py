import os
import torch
import cv2
import numpy as np
import sys

# Ensure models directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from physics_fusion_v9 import PhysicsGuidedDualEncoder
from utils.physics_transforms import compute_physics_channels

class V9InferenceEngine:
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing V9 Engine on {self.device}...")
        
        # Rigorous path resolution for standalone and app contexts
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if base_dir not in sys.path:
            sys.path.append(base_dir)
            
        self.model = PhysicsGuidedDualEncoder(use_physics=True, use_fp_head=True).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            # Try absolute path within repo
            checkpoint_path = os.path.join(base_dir, checkpoint_path)
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("Engine Ready.")

    def preprocess(self, img_bgr):
        # Resize to 256x256
        img_resized = cv2.resize(img_bgr, (256, 256))
        
        # 1. Normal Image Tensor (3, 256, 256)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 2. Physics Channels (Log + Laplacian)
        phys_map = compute_physics_channels(img_resized)
        phys_tensor = torch.from_numpy(phys_map).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor, phys_tensor, img_resized

    def run_prediction(self, image_path, threshold=0.3):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        img_tensor, phys_tensor, img_resized = self.preprocess(img_bgr)
        
        with torch.no_grad():
            # In V9, we use RGB for both streams if only one is provided, 
            # or in this case we simulate a multimodal pair from the same input.
            seg_logit, fp_logit = self.model(img_tensor, img_tensor, physics_tensor=phys_tensor)
            
            # Application of FP Suppression
            seg_prob = torch.sigmoid(seg_logit)
            fp_prob = torch.sigmoid(fp_logit).view(-1, 1, 1, 1)
            
            # Joint probability gating
            final_prob = seg_prob * (1 - fp_prob)
            
            # Mask generation
            mask_bin = (final_prob > threshold).float().squeeze().cpu().numpy()
            
        # Post-process for visualization
        mask_vis = (mask_bin * 255).astype(np.uint8)
        heatmap = (final_prob.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # Overlay
        overlay = img_resized.copy()
        overlay[mask_bin > 0.5] = [0, 0, 255] # Red spill highlight
        cv2.addWeighted(overlay, 0.4, img_resized, 0.6, 0, overlay)
        
        return {
            'mask': mask_vis,
            'heatmap': heatmap,
            'overlay': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            'fp_suppression_score': fp_prob.item(),
            'original': cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        }

if __name__ == "__main__":
    # Test initialization
    engine = V9InferenceEngine("checkpoints/best_model_v9.pth")
    print("Test: Initialization Successful.")
