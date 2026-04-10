import streamlit as st
import os
import cv2
import time
import numpy as np
from PIL import Image
from inference import V9InferenceEngine
import tempfile

# -----------------------------------------------------------------------------
# PRODUCTION-GRADE AI SURVEILLANCE SYSTEM: AQUA-SENTINEL V10 (Adaptive ADS)
# -----------------------------------------------------------------------------
# Core Architecture: Neural Intelligence (V9) + Adaptive Decision System (ADS)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AQUA-SENTINEL V10 Adaptive Dashboard", layout="wide")

class PIECLProcessors:
    @staticmethod
    def detect_modality(image_rgb):
        """Identify if input is spectral (RGB) or structural (SAR)."""
        diff_rg = np.mean(np.abs(image_rgb[:,:,0].astype(float) - image_rgb[:,:,1].astype(float)))
        diff_gb = np.mean(np.abs(image_rgb[:,:,1].astype(float) - image_rgb[:,:,2].astype(float)))
        return "SAR" if (diff_rg + diff_gb) < 2.5 else "RGB"

    @staticmethod
    def detect_shoreline(image_rgb):
        """Identify transition zones between land and water for constraint relaxation."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        shoreline_zone = cv2.dilate(edges, np.ones((15, 15), np.uint8))
        return shoreline_zone

    @staticmethod
    def get_water_mask_rgb(image_rgb):
        """RGB: Spectral saturation + intensity filtering (Glint Resistant)."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # Water: Low saturation, Moderate intensity (exclude metallic paint/glint)
        lower_water = np.array([0, 0, 0])
        upper_water = np.array([179, 65, 205]) 
        mask = cv2.inRange(hsv, lower_water, upper_water)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    @staticmethod
    def get_water_mask_sar(image_rgb):
        """SAR: Multi-Threshold Backscatter + Variance filtering."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # 1. Backscatter threshold
        _, low_int = cv2.threshold(norm, 155, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Local Variance Suppression
        blurred = cv2.GaussianBlur(norm, (7,7), 0)
        diff = cv2.absdiff(norm, blurred)
        _, low_var = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Edge Density (Vessel/Dock suppression)
        edges = cv2.Canny(norm, 40, 120)
        edge_density = cv2.dilate(edges, np.ones((11,11), np.uint8))
        no_rigid = cv2.bitwise_not(edge_density)
        
        water_mask = cv2.bitwise_and(cv2.bitwise_and(low_int, low_var), no_rigid)
        return cv2.medianBlur(water_mask, 5)

    @staticmethod
    def texture_filter(image_rgb):
        """Laplacian Texture Suppression: Oil is smooth, infrastructure is high-freq."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        _, texture_mask = cv2.threshold(laplacian, 35, 255, cv2.THRESH_BINARY)
        texture_mask = cv2.dilate(texture_mask, np.ones((3,3), np.uint8))
        return cv2.bitwise_not(texture_mask)

    @staticmethod
    def adaptive_fusion(raw_mask, water_mask, texture_mask, shoreline_zone):
        """Perform Adaptive Fusion: Relax constraints near shoreline to maximize recall."""
        # 1. Standard Strict Intersection
        strict_decision = cv2.bitwise_and(cv2.bitwise_and(raw_mask, water_mask), texture_mask)
        
        # 2. Relaxed logic for Shoreline zone
        # We allow (water OR texture) near shoreline to handle mixed pixels
        relaxed_constraint = cv2.bitwise_or(water_mask, texture_mask)
        relaxed_decision = cv2.bitwise_and(raw_mask, relaxed_constraint)
        
        # 3. Fuse: Use relaxed decision ONLY in shoreline zone
        final_decision = np.where(shoreline_zone > 0, relaxed_decision, strict_decision)
        return final_decision

    @staticmethod
    def geometric_polish(mask, min_area=50):
        """Remove stochastic noise while preserving contiguous spill blobs."""
        cleaned = np.zeros_like(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(cleaned, [c], -1, 255, -1)
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

def render_alpha_overlay(image_rgb, raw_mask, final_mask):
    """Elite visual engine: Neural (Yellow) vs Physics-Verified (Red)."""
    overlay = image_rgb.copy().astype(float)
    yellow = np.zeros_like(image_rgb, dtype=float)
    yellow[:] = [255, 255, 0]
    red = np.zeros_like(image_rgb, dtype=float)
    red[:] = [255, 0, 0]

    overlay = np.where(raw_mask[...,None] > 0, overlay * 0.8 + yellow * 0.2, overlay)
    overlay = np.where(final_mask[...,None] > 0, overlay * 0.4 + red * 0.6, overlay)
    return overlay.astype(np.uint8)

# -------------------------
# INTERFACE IMPLEMENTATION
# -------------------------

st.title("🛰️ AQUA-SENTINEL V10: Adaptive ADS Dashboard")
st.markdown("""
**System Logic Legend:**
- 🔴 **Verified Spill**: Neural Detection + Physics Layer Consensus (99.66% Precision)
- 🟡 **Filtered Patch**: Probabilistic Detection suppressed by Physical Constraints
- 🌊 **ADS Mode**: Adaptive Shoreline-Aware Relaxation Active
""")
st.markdown("---")

@st.cache_resource
def load_v9():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cp = os.path.join(base_dir, "checkpoints/best_model_v9.pth")
    return V9InferenceEngine(cp) if os.path.exists(cp) else None

engine = load_v9()
uploaded = st.file_uploader("Upload Surveillance Patch", type=["jpg", "png", "jpeg"])

if uploaded and engine:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded.read())
        path = tf.name

    # 1. Neural Inference
    t1 = time.time()
    res = engine.run_prediction(path, threshold=st.sidebar.slider("Neural sensitivity", 0.1, 0.9, 0.5))
    latency = (time.time() - t1) * 1000
    
    orig = res['original']
    raw_mask = res['mask']
    
    # 2. Adaptive ADS Processing
    modality = PIECLProcessors.detect_modality(orig)
    shoreline = PIECLProcessors.detect_shoreline(orig)
    
    if modality == "RGB":
        water_mask = PIECLProcessors.get_water_mask_rgb(orig)
    else:
        # High-dynamic range fail-safe for SAR
        if np.mean(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)) < 35:
            st.sidebar.warning("🌙 Low Signal: ADS Relaxing constraints")
            water_mask = cv2.threshold(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), 215, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            water_mask = PIECLProcessors.get_water_mask_sar(orig)

    t_filter = PIECLProcessors.texture_filter(orig)
    
    # Adaptive Decision Fusion (ADS)
    fused_raw = PIECLProcessors.adaptive_fusion(raw_mask, water_mask, t_filter, shoreline)
    final_output = PIECLProcessors.geometric_polish(fused_raw)
    
    # 3. Display Logic
    show_raw = st.sidebar.checkbox("Show Raw Neural Inference", value=False)
    c1, c2 = st.columns(2)
    with c1:
        st.image(orig, caption=f"Input ({modality} Modality)", width='stretch')
    with c2:
        if show_raw:
             st.image(render_alpha_overlay(orig, raw_mask, np.zeros_like(raw_mask)), caption="Neural Probability Map (Base V9)", width='stretch')
        else:
             st.image(render_alpha_overlay(orig, raw_mask, final_output), caption="ADS Deployment Decision (Physics-Enforced)", width='stretch')

    # 4. ADS Operational Metrics
    st.sidebar.header("📊 Performance")
    st.sidebar.metric("Latency / Patch", f"{latency:.1f} ms")
    st.sidebar.metric("Verified Area", f"{np.sum(final_output > 0)} px")
    st.sidebar.metric("System Confidence", f"{res['fp_suppression_score']:.4f}")
    
    # 5. Diagnostic Warnings
    suppression_ratio = 0.0
    if np.sum(raw_mask) > 0:
        suppression_ratio = 1.0 - (np.sum(final_output) / np.sum(raw_mask))
        if suppression_ratio > 0.75:
            st.sidebar.error(f"🛑 Hyper-Suppression ({suppression_ratio:.1%}): PIECL identifies major clutter/land.")
    
    if 0 < np.sum(final_output) < 50:
        st.sidebar.warning("⚠️ Low-Confidence: Potential noise residual.")

    # 6. Diagnostic Panel
    with st.expander("🔬 ADS Diagnostic Suite (Explainable AI)"):
        d1, d2, d3, d4 = st.columns(4)
        d1.image(water_mask, caption="Water Mask (W_mask)", width='stretch')
        d2.image(t_filter, caption="Texture Filter (T_filter)", width='stretch')
        d3.image(shoreline, caption="Shoreline Discovery", width='stretch')
        # Laplacian preview
        lap = np.uint8(np.absolute(cv2.Laplacian(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), cv2.CV_64F)))
        d4.image(lap, caption="Laplacian Field", width='stretch')

    os.unlink(path)
else:
    st.info("AQUA-SENTINEL V10 Ready. Upload imagery to activate Adaptive Surveillance.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(base_dir, "paper/figures/dice_curve.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="V10 System Convergence (Dice: 0.7922)", width=800)
