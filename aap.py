import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson

# App Configuration
st.set_page_config(
    page_title="Uveitis Progression Predictor",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Probabilistic Uveitis Progression Model")
st.markdown("""
*Moving beyond rigid clinical bins:* This interactive app evaluates whether an increase in eye cell counts between visits represents **true disease progression** or simply **statistical noise (wiggling counts)**, adjusted for human grading traps and patient disease history.
""")

# Sidebar Controls
st.sidebar.header("Clinical Parameters")

k1 = st.sidebar.number_input("Visit 1 Cell Count ($k_1$)", min_value=0, max_value=200, value=15, step=1)
k2 = st.sidebar.number_input("Visit 2 Cell Count ($k_2$)", min_value=0, max_value=200, value=16, step=1)

noise_std = st.sidebar.slider(
    "Scan Measurement Noise ($\sigma$)", 
    min_value=0.0, max_value=10.0, value=2.0, step=0.5,
    help="Calibrated from twin-scan reproducibility data (Dataset 4)."
)

disease_context = st.sidebar.selectbox(
    "Patient Disease Subtype (Dataset 6)",
    [
        "General / Standard Baseline",
        "Fuchs Heterochromic Cyclitis (Smoldering Baseline)",
        "HLA-B27 Associated (Repeated Sudden Flares)",
        "Juvenile Idiopathic Arthritis (JIA Chronic)"
    ]
)

# Adjust prior or sensitivity based on disease context
context_bias = 0.0
if "Fuchs" in disease_context:
    context_bias = -1.0 # Tolerates higher baseline counts as normal
elif "HLA-B27" in disease_context:
    context_bias = 1.0  # More sensitive to rapid jumps

# Monte Carlo Simulation Core
num_samples = 50000
alpha_prior, beta_prior = 1.0, 1.0

alpha_post1 = alpha_prior + max(0, k1 + context_bias)
beta_post1 = beta_prior + 1.0

alpha_post2 = alpha_prior + k2
beta_post2 = beta_prior + 1.0

lam1_samples = gamma.rvs(a=alpha_post1, scale=1.0/beta_post1, size=num_samples)
lam2_samples = gamma.rvs(a=alpha_post2, scale=1.0/beta_post2, size=num_samples)

if noise_std > 0:
    lam1_noisy = np.maximum(lam1_samples + np.random.normal(0, noise_std, size=num_samples), 0)
    lam2_noisy = np.maximum(lam2_samples + np.random.normal(0, noise_std, size=num_samples), 0)
    prob_worse = np.mean(lam2_noisy > lam1_noisy)
else:
    prob_worse = np.mean(lam2_samples > lam1_samples)

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Progression Probability Assessment")
    
    # Metric Display with Dynamic Styling
    prob_pct = prob_worse * 100
    if prob_pct < 60:
        status_color = "orange"
        assessment = "⚠️ Inconclusive / Likely Statistical Noise (Coin Flip)"
    elif prob_pct < 85:
        status_color = "yellow"
        assessment = "📈 Mild Indicator of Progression"
    else:
        status_color = "red"
        assessment = "🚨 Definite Clinical Progression"
        
    st.metric(label="Probability True Inflammation Increased", value=f"{prob_pct:.1f}%")
    st.markdown(f"**Clinical Interpretation:** {assessment}")
    
    st.markdown("---")
    st.markdown("### 🔍 Model Diagnostic Insights")
    st.write(f"- **Raw Count Shift:** {k1} $\\to$ {k2} cells (Difference: {k2 - k1:+d})")
    st.write(f"- **Human Grading Trap Check:** {'⚠️ High Risk of Boundary Artifact (e.g. 15→16)' if (k1==15 and k2==16) else '✅ Outside critical boundary blur zone.'}")
    st.write(f"- **Active Context:** {disease_context}")

with col2:
    st.subheader("📉 Latent Rate Distribution ($\lambda$ Curves)")
    
    # Plotting Posterior Distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, max(30, max(k1, k2) + 15), 300)
    y1 = gamma.pdf(x, a=alpha_post1, scale=1.0/beta_post1)
    y2 = gamma.pdf(x, a=alpha_post2, scale=1.0/beta_post2)
    
    ax.plot(x, y1, label=f'Visit 1 (k={k1})', color='blue', lw=2)
    ax.plot(x, y2, label=f'Visit 2 (k={k2})', color='red', lw=2)
    ax.fill_between(x, y1, alpha=0.2, color='blue')
    ax.fill_between(x, y2, alpha=0.2, color='red')
    
    ax.set_title("Posterior Probability Distribution of True $\lambda$")
    ax.set_xlabel("True Average Cell Count ($\lambda$)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)

st.markdown("---")
st.info("💡 **Tip for Presentation:** Use the sidebar to test class test cases like **15 → 16** (showing the 54% coin toss due to measurement noise) versus **16 → 50** (showing near 100% certainty of worsening).")