import os, cv2, torch, json, random, logging, sys, traceback, time
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Task 8: Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
from physics_fusion_v9 import PhysicsGuidedDualEncoder

from utils.physics_transforms import compute_physics_channels

LOG_FILE = '/root/aqua_clean_pipeline/logs/train_v9.log'
os.makedirs('/root/aqua_clean_pipeline/logs', exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}')
    logging.info(msg)
    sys.stdout.flush()

BASE = '/root/aqua_clean_pipeline/dataset_multimodal_v6'
SAVE_DIR = '/root/aqua_clean_pipeline/checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

class DeterministicFusionDatasetV9(Dataset):
    def __init__(self, img_dir, gt_dir, use_physics=True):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.use_physics = use_physics
        
        img_files = set([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))])
        gt_files = set([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))])
        
        self.files = sorted(list(img_files.intersection(gt_files)))
        
    def __len__(self): 
        return len(self.files)

    def __getitem__(self, i):
        try:
            name = self.files[i]
            img_bgr = cv2.imread(os.path.join(self.img_dir, name))
            img = cv2.resize(img_bgr, (256, 256)) / 255.0
            
            mask = cv2.imread(os.path.join(self.gt_dir, name), 0) / 255.0
            mask = cv2.resize(mask, (256, 256))
            
            img_tensor = torch.tensor(img).permute(2,0,1).float()
            mask_tensor = torch.tensor(mask).unsqueeze(0).float()
            
            # Determine if sample is a false positive (empty mask)
            fp_label = float(mask_tensor.sum() == 0)

            physics_tensor = torch.zeros((2, 256, 256))
            if self.use_physics:
                phys_map = compute_physics_channels(cv2.resize(img_bgr, (256, 256)))
                physics_tensor = torch.tensor(phys_map).permute(2,0,1).float()
            
            return img_tensor, physics_tensor, mask_tensor, fp_label
        except Exception:
            return torch.zeros((3,256,256)), torch.zeros((2,256,256)), torch.zeros((1,256,256)), 0.0

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        return (self.alpha * (1-pt)**self.gamma * BCE_loss).mean()

class JointLossV9(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.fp_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target_mask, fp_logit, fp_target):
        p = torch.sigmoid(pred)
        inter = (p * target_mask).sum()
        l_dice = 1 - (2*inter + 1) / (p.sum() + target_mask.sum() + 1)
        
        seg_loss = (0.5 * l_dice) + (0.3 * self.bce(pred, target_mask)) + (0.2 * self.focal(pred, target_mask))
        
        # Train Jointly with Auxiliary False Positive Suppression
        fp_loss = self.fp_criterion(fp_logit, fp_target)
        
        return seg_loss + (0.3 * fp_loss)

def set_encoder_requires_grad(model, requires_grad):
    """
    Toggles gradient updates for native encoder branches.
    """
    for layer in [model.rgb_init, model.rgb_l1, model.rgb_l2, model.rgb_l3, model.rgb_l4,
                  model.sar_init, model.sar_l1, model.sar_l2, model.sar_l3, model.sar_l4]:
        for param in layer.parameters():
            param.requires_grad = requires_grad

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhysicsGuidedDualEncoder(use_physics=True, use_fp_head=True).to(device)
    
    # 5. Load compatible V8 weights dynamically
    w_p = f'{SAVE_DIR}/best_model_v8.pth'
    if os.path.exists(w_p): 
        model.load_state_dict(torch.load(w_p, map_location=device), strict=False)
        log("DEBUG: Partially resumed structural weights from best_model_v8.pth")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn = JointLossV9()
    
    img_dir = f'{BASE}/images'
    gt_dir = f'{BASE}/masks'
    dataset = DeterministicFusionDatasetV9(img_dir, gt_dir, use_physics=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    with open('/root/aqua_clean_pipeline/logs/train_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss", "dice", "iou"])

    best_dice = 0.0
    for epoch in range(40):
        # 4. Phase-Specific Training Strategy
        if epoch < 10:
            if epoch == 0:
                log("--- PHASE 1: Freezing Encoders ---")
                set_encoder_requires_grad(model, False)
        elif epoch == 10:
            log("--- PHASE 2: Unfreezing All Layers ---")
            set_encoder_requires_grad(model, True)
            
        log(f"DEBUG: Starting V9 Training Epoch {epoch}")
        model.train()
        epoch_losses = []
        epoch_dices = []
        epoch_ious = []
        
        for i, (img, phys, mask, fp_label) in enumerate(loader):
            img, phys, mask, fp_label = img.to(device), phys.to(device), mask.to(device), fp_label.to(device).unsqueeze(1)
            optimizer.zero_grad()
            
            # Identical injection mapping simulating V8 parameters
            pred, fp_logit = model(img, img, physics_tensor=phys)
            
            loss = loss_fn(pred, mask, fp_logit, fp_label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            torch.cuda.empty_cache()
            epoch_losses.append(loss.item())
            
            # Metric Tracking 
            p = torch.sigmoid(pred) * (1 - torch.sigmoid(fp_logit).view(-1, 1, 1, 1))
            p_bin = (p > 0.5).float()
            inter = (p_bin * mask).sum()
            dice = (2 * inter) / (p_bin.sum() + mask.sum() + 1e-8)
            iou = inter / (p_bin.sum() + mask.sum() - inter + 1e-8)
            
            epoch_dices.append(dice.item())
            epoch_ious.append(iou.item())
            
            if i % 100 == 0: 
                log(f"Batch {i} Done | Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        avg_dice = np.mean(epoch_dices)
        avg_iou = np.mean(epoch_ious)
        
        log(f"✅ Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        with open('/root/aqua_clean_pipeline/logs/train_metrics.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, avg_loss, avg_dice, avg_iou])
        
        scheduler.step(avg_loss)
        
        torch.save(model.state_dict(), f'{SAVE_DIR}/last_model_v9.pth')
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f'{SAVE_DIR}/best_model_v9.pth')
            log(f"⭐ New Best Model Checkpoint Saved! (Dice: {best_dice:.4f})")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        log(f"❌ CRITICAL V9 INCIDENT: {e}")
        traceback.print_exc()
        sys.exit(1)
