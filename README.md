# AQUA-SENTINEL V9: Physics-Guided Oceanic Oil Spill Surveillance

**AQUA-SENTINEL V9** is a production-grade AI system designed for high-precision maritime oil spill detection. It combines a **Deployment-Centric Physics-Guided Dual-Encoder (PGDE)** neural architecture with a **Post-Inference Environmental Constraint Layer (PIECL)** to achieve industry-leading operational reliability.

## 🚀 Key Features
- **Neural Intelligence**: Symmetric parallel encoders (SAR/RGB) with Adaptive Log-Intensity Mapping and Laplacian Structural Priors.
- **Physical Reasoning (PIECL)**: A modality-aware deployment layer that enforces water-only constraints and high-frequency texture suppression.
- **Vessel/Hull Exclusion**: Domain-driven architectural logic that eliminates false positives on ships, docks, and coastal infrastructure.
- **Performance**:
  - **Precision**: 99.66% (Operational threshold)
  - **Dice Score**: 0.7922
  - **Latency**: ~14ms per 256x256 patch (NVIDIA L4)

## 📁 Repository Structure
- `app.py`: Streamlit-based surveillance dashboard.
- `inference.py`: Core V9 inference engine and PIECL logic.
- `checkpoints/`: Model weights directory (Note: `.pth` files excluded from Git due to size).
- `models/`: PGDE architecture and gating modules.
- `utils/`: Physics-priors and preprocessing transforms.
- `paper/`: IEEE manuscript LaTeX source and diagnostic figures.

## 🛠️ Installation & Usage

### 1. Requirements
```bash
pip install torch opencv-python streamlit numpy pillow
```

### 2. Restore Model Weights
The production weights (`best_model_v9.pth` at 169MB) exceed GitHub's file size limits and are excluded from this repository. 
- **Setup**: Manually place the `best_model_v9.pth` checkpoint into the `checkpoints/` directory to enable the inference app.

### 3. Launch Surveillance Dashboard
```bash
streamlit run app.py
```

## 📄 Manuscript
The system is presented in the research paper: *"AQUA-SENTINEL: Deployment-Centric Physics-Guided Dual-Encoder for Precise Maritime Oil Spill Detection."* The LaTeX source as well as finalized PDF assets are available in the `/paper` directory.

---
**Note**: This repository contains the production-grade inference system. For raw training datasets or experimental benchmarks, please contact the primary investigators.
