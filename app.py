import streamlit as st
import os
import cv2
import time
import numpy as np
from PIL import Image
from inference import V9InferenceEngine
import tempfile

# -----------------------------------------------------------------------------
# PRODUCTION-GRADE AI SURVEILLANCE SYSTEM: AQUA-SENTINEL V9 (surgical upgrade)
# -----------------------------------------------------------------------------
# Core Architecture: Neural Intelligence (V9) + Physical Reasoning (PIECL)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AQUA-SENTINEL V9 Production Unit", layout="wide")

class PIECLProcessors:
    @staticmethod
    def detect_modality(image_rgb):
        """Identify if input is spectral (RGB) or structural (SAR)."""
        # Grayscale images have near-zero variance across R, G, B channels
        diff_rg = np.mean(np.abs(image_rgb[:,:,0].astype(float) - image_rgb[:,:,1].astype(float)))
        diff_gb = np.mean(np.abs(image_rgb[:,:,1].astype(float) - image_rgb[:,:,2].astype(float)))
        return "SAR" if (diff_rg + diff_gb) < 2.5 else "RGB"

    @staticmethod
    def get_water_mask_rgb(image_rgb):
        """RGB: Spectral saturation + intensity filtering."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        # Water: Low saturation, Moderate intensity (exclude metallic paint/glint)
        lower_water = np.array([0, 0, 0])
        upper_water = np.array([179, 60, 210]) 
        mask = cv2.inRange(hsv, lower_water, upper_water)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    @staticmethod
    def get_water_mask_sar(image_rgb):
        """SAR: Intensity + Local Variance filtering."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # 1. Backscatter threshold (Water is typically low backscatter)
        _, low_int = cv2.threshold(norm, 160, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Local Variance (Water surface is smooth; ships are textured)
        # We simulate local variance using a Laplacian or standard deviation filter
        blurred = cv2.GaussianBlur(norm, (7,7), 0)
        diff = cv2.absdiff(norm, blurred)
        _, low_var = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Edge Density Suppression (Remove rigid hulls/docks)
        edges = cv2.Canny(norm, 50, 150)
        edge_density = cv2.dilate(edges, np.ones((15,15), np.uint8))
        no_rigid = cv2.bitwise_not(edge_density)
        
        water_mask = cv2.bitwise_and(cv2.bitwise_and(low_int, low_var), no_rigid)
        
        # Elite Smoothing: Remove SAR speckle noise grain
        water_mask = cv2.medianBlur(water_mask, 5)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        return water_mask

    @staticmethod
    def texture_filter(image_rgb):
        """Laplacian Texture Suppression: Oil is smooth, ships are high-freq."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        _, texture_mask = cv2.threshold(laplacian, 40, 255, cv2.THRESH_BINARY)
        # Dilate to cover boundaries
        texture_mask = cv2.dilate(texture_mask, np.ones((3,3), np.uint8))
        return cv2.bitwise_not(texture_mask)

    @staticmethod
    def geometric_polish(mask, min_area=50):
        """Remove stochastic noise and non-fluid shapes."""
        cleaned = np.zeros_like(mask)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: continue
            
            # Form factor check: Ships are often narrow/rectangular (high eccentricity)
            # Fluid pools are more rounded
            x,y,w,h = cv2.boundingRect(c)
            aspect_ratio = float(w)/h if h != 0 else 0
            # Simple heuristic: extreme aspect ratios often correspond to hulls or linear noise
            if 0.1 < aspect_ratio < 10.0:
                 cv2.drawContours(cleaned, [c], -1, 255, -1)
        return cleaned

def render_alpha_overlay(image_rgb, raw_mask, final_mask):
    """Elite visual engine: Stacked alpha layers for scientific clarity."""
    overlay = image_rgb.copy().astype(float)

    # Yellow layer (Raw neural prediction - subtle)
    yellow = np.zeros_like(image_rgb, dtype=float)
    yellow[:] = [255, 255, 0]

    # Red layer (Final PIECL decision - dominant)
    red = np.zeros_like(image_rgb, dtype=float)
    red[:] = [255, 0, 0]

    # Layered Blend: Base -> Raw (20%) -> Final (60%)
    overlay = np.where(raw_mask[...,None] > 0, overlay * 0.8 + yellow * 0.2, overlay)
    overlay = np.where(final_mask[...,None] > 0, overlay * 0.4 + red * 0.6, overlay)

    return overlay.astype(np.uint8)

# -------------------------
# INTERFACE IMPLEMENTATION
# -------------------------

st.title("🛰️ AQUA-SENTINEL V9: Production Inference")
st.markdown("""
**System Logic Legend:**
- 🔴 **Verified Spill**: High-Confidence Neural Detection + Physics Layer Agreement
- 🟡 **Suppressed Patch**: Neural Activation Rejected by Environmental Constraints
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

    # 1. Pipeline Execution
    t1 = time.time()
    res = engine.run_prediction(path, threshold=st.sidebar.slider("Neural sensitivity", 0.1, 0.9, 0.5))
    latency = (time.time() - t1) * 1000
    
    orig = res['original']
    raw_mask = res['mask']
    
    # 2. PIECL Reasoning
    modality = PIECLProcessors.detect_modality(orig)
    
    # Branching Logic
    if modality == "RGB":
        water_mask = PIECLProcessors.get_water_mask_rgb(orig)
    else:
        # Dark scene fail-safe
        if np.mean(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)) < 35:
            st.sidebar.warning("🌙 Low Signal: Relaxing constraints")
            water_mask = cv2.threshold(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), 210, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            water_mask = PIECLProcessors.get_water_mask_sar(orig)

    t_filter = PIECLProcessors.texture_filter(orig)
    
    # Decision Formation: M_final = M_pred AND W_mask AND T_filter
    final_raw = cv2.bitwise_and(cv2.bitwise_and(raw_mask, raw_mask, mask=water_mask), t_filter)
    final_output = PIECLProcessors.geometric_polish(final_raw)
    
    # 3. Dashboard Rendering
    show_raw = st.sidebar.checkbox("Show Raw Neural Output", value=False)
    c1, c2 = st.columns(2)
    with c1:
        st.image(orig, caption=f"Input ({modality} Modality)", width='stretch')
    with c2:
        if show_raw:
             st.image(render_alpha_overlay(orig, raw_mask, np.zeros_like(raw_mask)), caption="Raw Neural Inference (Before PIECL Filtering)", width='stretch')
        else:
             st.image(render_alpha_overlay(orig, raw_mask, final_output), caption="Final Deployment Decision (Verified Spill)", width='stretch')

    # 4. Metrics Panel
    st.sidebar.header("📊 Performance")
    st.sidebar.metric("Latency", f"{latency:.1f} ms")
    st.sidebar.metric("Spill Area", f"{np.sum(final_output > 0)} px")
    st.sidebar.metric("Confidence Score", f"{res['fp_suppression_score']:.4f}")
    
    # 5. Diagnostic Feedback & Fail-safe Warnings
    if np.sum(final_output) < 50 and np.sum(final_output) > 0:
        st.sidebar.warning("⚠️ Low-confidence detection (possible noise / weak signal)")
    
    # 6. Debug Panel
    with st.expander("🛠️ PIECL Diagnostic Suite"):
        d1, d2, d3 = st.columns(3)
        d1.image(water_mask, caption="Water Constrain (W_mask)", width='stretch')
        # Texture field visualization
        gray_lap = np.uint8(np.absolute(cv2.Laplacian(cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY), cv2.CV_64F)))
        d2.image(gray_lap, caption="Texture Field (Laplacian)", width='stretch')
        d3.image(t_filter, caption="Texture Filter (T_filter)", width='stretch')

    os.unlink(path)
else:
    st.info("System Ready. Upload imagery to activate surveillance pipeline.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(base_dir, "paper/figures/dice_curve.png")
    if os.path.exists(fig_path):
        st.image(fig_path, caption="Convergence Profile (Dice: 0.7922)", width=800)
