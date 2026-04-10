import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
RECON_DIR = "/Users/harsh/Desktop/ZXCVBNM,.:/paper_reconstruction"
SUBMISSION_DIR = "/Users/harsh/Desktop/ZXCVBNM,.:/paper_submission"
FIG_DIR = os.path.join(SUBMISSION_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Training Curves (from logs/train_metrics.csv)
train_csv = os.path.join(RECON_DIR, "logs/train_metrics.csv")
if os.path.exists(train_csv):
    df = pd.read_csv(train_csv)
    
    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Training Loss')
    plt.title('Training Loss Convergence (V9 Architecture)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # Dice Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['dice'], 'g-', linewidth=2, label='Avg Dice Score')
    plt.title('Segmentation Accuracy Evolution', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "dice_curve.png"), dpi=300)
    plt.close()

# 2. Ablation Comparison (from outputs/ablation.csv)
ablation_csv = os.path.join(RECON_DIR, "outputs/ablation.csv")
if os.path.exists(ablation_csv):
    df_abl = pd.read_csv(ablation_csv)
    
    # Filter or Pivot if needed - currently it has Model,Dice,IoU
    plt.figure(figsize=(12, 7))
    x = range(len(df_abl))
    width = 0.35
    
    plt.bar(x, df_abl['Dice'], width, label='Dice', color='#3498db', alpha=0.8)
    plt.bar([i + width for i in x], df_abl['IoU'], width, label='IoU', color='#e74c3c', alpha=0.8)
    
    plt.title('Ablation Study: Architecture Contribution Analysis', fontsize=16)
    plt.xticks([i + width/2 for i in x], df_abl['Model'], rotation=15)
    plt.ylabel('Score')
    plt.ylim(0.7, 0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "ablation_chart.png"), dpi=300)
    plt.close()

# 3. Final Metrics Table Data Extraction
metrics_csv = os.path.join(RECON_DIR, "outputs/metrics.csv")
if os.path.exists(metrics_csv):
    df_met = pd.read_csv(metrics_csv)
    # Header was 'Threshold,Avg Dice' in the latest run
    peak_row = df_met[df_met['Threshold'] == 0.30]
    print(f"PEAK METRICS: {peak_row.to_dict('records')}")

# 4. Visual Evidence Extraction
VISUALS_DIR = os.path.join(RECON_DIR, "outputs/visuals")
if os.path.exists(VISUALS_DIR):
    import shutil
    # Select Best/Worst Samples
    # Picking rank 0 for Best, rank 0 for Worst
    best_panel = [f for f in os.listdir(VISUALS_DIR) if "panel_best_rank_0" in f][0]
    worst_panel = [f for f in os.listdir(VISUALS_DIR) if "panel_worst_rank_0" in f][0]
    heat_sample = [f for f in os.listdir(VISUALS_DIR) if "heat_best_rank_0" in f][0]

    shutil.copy(os.path.join(VISUALS_DIR, best_panel), os.path.join(FIG_DIR, "panel_best.png"))
    shutil.copy(os.path.join(VISUALS_DIR, worst_panel), os.path.join(FIG_DIR, "panel_worst.png"))
    shutil.copy(os.path.join(VISUALS_DIR, heat_sample), os.path.join(FIG_DIR, "heatmap_sample.png"))
    print("Visual panels extracted to submission figures.")

print(f"Scientific assets finalized in {FIG_DIR}")
