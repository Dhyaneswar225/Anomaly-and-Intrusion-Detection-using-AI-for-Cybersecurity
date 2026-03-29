import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import json
from pathlib import Path
from src.DenseAutoEncoderModel import DenseAutoencoder
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Specifically target unpickling version warnings
warnings.filterwarnings("ignore", category=UserWarning)
# ================= PAGE CONFIG =================
st.set_page_config(page_title="Cyber IDS", layout="wide", page_icon="🛡️")

# ================= CUSTOM CSS (THE STYLING) =================
st.markdown("""
    <style>
    /* Main background and font */
    .main {
        background-color: #0e1117;
    }
    /* Card-like containers for sections */
    div[data-testid="stVerticalBlock"] > div:has(div.stSubheader) {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    /* Custom Title Style */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    /* Analyze Button Styling */
    div.stButton > button {
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #008fb3;
        color: white;
        box-shadow: 0 0 15px #00d4ff;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        color: #00d4ff;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= CONFIG & PATHS =================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD ARTIFACTS =================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
    feature_names = list(scaler.feature_names_in_)
    with open(DATA_DIR / "label_mappings.json") as f:
        mappings = json.load(f)

    model = DenseAutoencoder(input_dim=len(feature_names), bottleneck=32, dropout=0.2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "dense_autoencoder_best.pth", map_location=DEVICE))
    model.eval()

    threshold = float(np.load(MODEL_DIR / "dense_ae_threshold.npy"))
    clf = joblib.load(MODEL_DIR / "attack_classifier_xgb.pkl")
    le = joblib.load(MODEL_DIR / "attack_label_encoder.pkl")
    
    train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
    mean_values = train_df[feature_names].mean().to_dict()

    return scaler, feature_names, mappings, model, threshold, clf, le, mean_values

scaler, FEATURE_NAMES, mappings, model, THRESHOLD, clf, le, MEAN_VALUES = load_artifacts()

# --- COMPACT HEADER (FLUSH LEFT) ---
header_html = """
<div style="display: flex; align-items: center; justify-content: space-between; background: #00d4ff; padding: 15px 30px; border-radius: 10px; border-left: 5px solid #00d4ff; margin-bottom: 20px;">
    <div style="display: flex; align-items: center;">
        <img src="https://img.freepik.com/premium-photo/cyber-security-icon-with-shield-keyhole-vector-illustration_1048419-578.jpg" 
     width="90" 
     height="90" 
     style="margin-right: 20px; border-radius: 50%; transform: scale(1.1); transform-origin: center center; object-fit: cover;">
        <div>
            <h2 style="color: black; font-family: sans-serif; margin: 0; font-size: 1.8rem; letter-spacing: 2px;">
                ANOMALY AND INTRUSION DETECTION SYSTEM
            </h2>
        </div>
    </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)

# ================= PREPROCESS FUNCTION =================
def preprocess(ui_raw_dict):
    row = MEAN_VALUES.copy()
    encoded_proto = mappings["protocol_type"].get(ui_raw_dict["protocol"], 0)
    encoded_service = mappings["service"].get(ui_raw_dict["service"], 0)
    encoded_flag = mappings["flag"].get(ui_raw_dict["flag"], 0)

    updates = {
        "protocol_type": encoded_proto, "service": encoded_service, "flag": encoded_flag,
        "duration": ui_raw_dict["duration"], "src_bytes": ui_raw_dict["src_bytes"],
        "dst_bytes": ui_raw_dict["dst_bytes"], "logged_in": int(ui_raw_dict["logged_in"]),
        "num_failed_logins": ui_raw_dict["num_failed_logins"], "is_guest_login": int(ui_raw_dict["is_guest_login"]),
        "root_shell": int(ui_raw_dict["root_shell"]), "num_shells": ui_raw_dict["num_shells"],
        "num_compromised": ui_raw_dict["num_compromised"], "hot": ui_raw_dict["hot"],
        "wrong_fragment": ui_raw_dict["wrong_fragment"], "count": ui_raw_dict["count"],
        "srv_count": ui_raw_dict["srv_count"], "serror_rate": ui_raw_dict["serror_rate"],
        "rerror_rate": ui_raw_dict["rerror_rate"], "same_srv_rate": ui_raw_dict["same_srv_rate"],
        "diff_srv_rate": ui_raw_dict["diff_srv_rate"], "srv_diff_host_rate": ui_raw_dict["srv_diff_host_rate"],
        "dst_host_count": ui_raw_dict["dst_host_count"], "dst_host_srv_count": ui_raw_dict["dst_host_srv_count"],
        "dst_host_same_srv_rate": ui_raw_dict["dst_host_same_srv_rate"], "dst_host_diff_srv_rate": ui_raw_dict["dst_host_diff_srv_rate"],
        "dst_host_same_src_port_rate": ui_raw_dict["dst_host_same_src_port_rate"], "dst_host_srv_diff_host_rate": ui_raw_dict["dst_host_srv_diff_host_rate"],
        "dst_host_serror_rate": ui_raw_dict["dst_host_serror_rate"], "dst_host_rerror_rate": ui_raw_dict["dst_host_rerror_rate"]
    }
    for k, v in updates.items():
        if k in row: row[k] = v

    df_input = pd.DataFrame([row], columns=FEATURE_NAMES)
    scaled = scaler.transform(df_input).astype(np.float32)[0]
    return scaled

# ================= UI MAIN LAYOUT =================

# 1. Custom Header with Large Logo and Styled Text
# Create the header string as a variable to avoid quote confusion

# Use Columns to create a sidebar-like control panel on the left or just center it
# Here we use the Card style containers defined in CSS
with st.container():
    st.subheader("📡 Connection Intelligence")
    c1, c2, c3 = st.columns(3)
    with c1:
        protocol = st.selectbox("Protocol", list(mappings["protocol_type"].keys()))
        duration = st.number_input("Duration (sec)", 0.0, format="%.2f")
    with c2:
        service = st.selectbox("Service", list(mappings["service"].keys()))
        src_bytes = st.number_input("Source Bytes", 0.0, format="%.0f")
    with c3:
        flag = st.selectbox("Flag", list(mappings["flag"].keys()))
        dst_bytes = st.number_input("Destination Bytes", 0.0, format="%.0f")

with st.container():
    st.subheader("🔑 Content & Authentication")
    c1, c2 = st.columns(2)
    with c1:
        logged_in = st.toggle("User Logged In")
        is_guest_login = st.toggle("Guest Account")
        root_shell = st.toggle("Root Shell")
    with c2:
        num_failed_logins = st.number_input("Failed Logins", 0)
        num_shells = st.number_input("Number of Shells", 0)
        hot = st.number_input("Hot Indicators", 0)

with st.container():
    st.subheader("📊 Network Flow Behavior")
    c1, c2 = st.columns(2)
    with c1:
        count = st.number_input("Connections to Host (Count)", 0.0)
        srv_count = st.number_input("Connections to Service (srv_count)", 0.0)
        num_compromised = st.number_input("Compromised Conditions", 0)
        wrong_fragment = st.number_input("Wrong Fragments", 0)
    with c2:
        serror_rate = st.slider("S-Error Rate", 0.0, 1.0, 0.0)
        rerror_rate = st.slider("R-Error Rate", 0.0, 1.0, 0.0)
        same_srv_rate = st.slider("Same Srv Rate", 0.0, 1.0, 0.0)
        diff_srv_rate = st.slider("Diff Srv Rate", 0.0, 1.0, 0.0)
        srv_diff_host_rate = st.slider("Srv Diff Host Rate", 0.0, 1.0, 0.0)

with st.expander("🛠️ Advanced Host Metrics (Deep Inspection)"):
    c1, c2 = st.columns(2)
    with c1:
        dst_host_count = st.number_input("Destination Host Count", 0.0)
        dst_host_srv_count = st.number_input("Destination Host Srv Count", 0.0)
        dst_host_same_srv_rate = st.slider("Destination Host Same Srv Rate", 0.0, 1.0, 0.0)
        dst_host_diff_srv_rate = st.slider("Destination Host Diff Srv Rate", 0.0, 1.0, 0.0)
    with c2:
        dst_host_same_src_port_rate = st.slider("Destination Host Same Src Port Rate", 0.0, 1.0, 0.0)
        dst_host_srv_diff_host_rate = st.slider("Destination Host Srv Diff Host Rate", 0.0, 1.0, 0.0)
        dst_host_serror_rate = st.slider("Destination Host S-Error Rate", 0.0, 1.0, 0.0)
        dst_host_rerror_rate = st.slider("Destination Host R-Error Rate", 0.0, 1.0, 0.0)

# ================= PREDICTION EXECUTION =================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 INITIATE SECURITY SCAN", use_container_width=True):
    ui_raw = {
        "protocol": protocol, "service": service, "flag": flag, "src_bytes": src_bytes, 
        "dst_bytes": dst_bytes, "duration": duration, "logged_in": logged_in, 
        "num_failed_logins": num_failed_logins, "is_guest_login": is_guest_login, 
        "root_shell": root_shell, "num_shells": num_shells, "num_compromised": num_compromised,
        "hot": hot, "wrong_fragment": wrong_fragment, "count": count, "srv_count": srv_count, 
        "serror_rate": serror_rate, "rerror_rate": rerror_rate, "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate, "srv_diff_host_rate": srv_diff_host_rate,
        "dst_host_count": dst_host_count, "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate, "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": dst_host_same_src_port_rate, "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": dst_host_serror_rate, "dst_host_rerror_rate": dst_host_rerror_rate
    }

    with st.spinner("Analyzing traffic patterns via Neural Engine..."):
        x_scaled = preprocess(ui_raw)
        x_tensor = torch.tensor(x_scaled).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            reconstruction = model(x_tensor)
            mse_error = torch.mean((reconstruction - x_tensor) ** 2).item()

    # Results Section
    st.divider()
    res_c1, res_c2 = st.columns(2)
    with res_c1:
        st.metric("Neural Reconstruction Error", f"{mse_error:.6f}")
    with res_c2:
        status = "CRITICAL ANOMALY" if mse_error >= THRESHOLD else "SECURE"
        st.metric("Traffic Status", status)

    if mse_error < THRESHOLD:
        st.success("🟢 **SYSTEM SECURE**: No unusual patterns detected in this connection.")
    else:
        attack_id = clf.predict(x_scaled.reshape(1, -1))[0]
        attack_label = le.inverse_transform([attack_id])[0]
        st.error(f"🔴 **INTRUSION DETECTED**: Classified as **{attack_label.upper()}**")

st.markdown('<div class="sub-title" style="margin-top:50px; padding-bottom: 50px;">© 2026 Master Thesis Project | Secure Hybrid IDS Architecture</div>', unsafe_allow_html=True)