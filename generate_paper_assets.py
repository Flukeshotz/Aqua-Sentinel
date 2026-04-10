import pandas as pd
import matplotlib.pyplot as plt
import os

# Setup
BASE_DIR = "/Users/harsh/Desktop/ZXCVBNM,.:/paper_reconstruction"
ASSETS_DIR = "/Users/harsh/Desktop/ZXCVBNM,.:/paper_assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Training Curves
train_csv = os.path.join(BASE_DIR, "logs/train_metrics.csv")
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv)
    
    # Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.title('Training Loss Convergence (V9 Dual-Encoder)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, "loss_curve.png"), dpi=300)
    plt.close()
    
    # Dice Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['dice'], label='Avg Dice', color='green', linewidth=2)
    plt.title('Segmentation Quality Evolution (Dice Score)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, "dice_curve.png"), dpi=300)
    plt.close()

# 2. Ablation Results
ablation_csv = os.path.join(BASE_DIR, "outputs/ablation.csv")
if os.path.exists(ablation_csv):
    df_abl = pd.read_csv(ablation_csv)
    
    plt.figure(figsize=(12, 7))
    x = range(len(df_abl))
    width = 0.35
    
    plt.bar(x, df_abl['Dice'], width, label='Dice', color='#3498db', alpha=0.8)
    plt.bar([i + width for i in x], df_abl['IoU'], width, label='IoU', color='#e74c3c', alpha=0.8)
    
    plt.title('Ablation Study: Component Performance Mapping', fontsize=16)
    plt.xticks([i + width/2 for i in x], df_abl['Model'], rotation=15)
    plt.ylabel('Score (0-1)')
    plt.ylim(0.7, 0.9) # Focus on the differences
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "ablation_chart.png"), dpi=300)
    plt.close()

# 3. Precision vs Recall (Trade-off from Final Report)
# Metrics from report: Precision: 0.9966, Recall: 0.9304
metrics = {'Metric': ['Precision', 'Recall', 'Dice', 'IoU'], 
           'Value': [0.9966, 0.9304, 0.7922, 0.7514]}
df_p = pd.DataFrame(metrics)

plt.figure(figsize=(8, 6))
plt.bar(df_p['Metric'], df_p['Value'], color=['#27ae60', '#f39c12', '#2980b9', '#8e44ad'], alpha=0.85)
plt.title('Final System Performance Matrix', fontsize=16)
plt.ylabel('Score')
plt.ylim(0.7, 1.05)
for i, v in enumerate(df_p['Value']):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.savefig(os.path.join(ASSETS_DIR, "final_metrics.png"), dpi=300)
plt.close()

print(f"Research assets generated in {ASSETS_DIR}")
