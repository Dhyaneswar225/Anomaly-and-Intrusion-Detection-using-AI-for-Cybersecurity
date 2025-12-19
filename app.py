# app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch
import joblib
import json
from pathlib import Path
from src.lstm_model import LSTMAutoencoder

# ================= CONFIG =================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
ANOMALY_THRESHOLD = 0.365428   # 0.28

# ================= LOAD ARTIFACTS =================
@st.cache_resource
def load_artifacts():
    # ----- scaler & features -----
    scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
    feature_names = list(scaler.feature_names_in_)

    # ----- categorical mappings -----
    with open(DATA_DIR / "label_mappings.json") as f:
        mappings = json.load(f)

    # ----- LSTM autoencoder -----
    lstm_model = LSTMAutoencoder(input_dim=len(feature_names)).to(DEVICE)
    lstm_model.load_state_dict(
        torch.load(MODEL_DIR / "lstm_autoencoder.pth", map_location=DEVICE)
    )
    lstm_model.eval()

    # ----- attack classifier -----
    clf = joblib.load(MODEL_DIR / "attack_classifier_xgb.pkl")
    le = joblib.load(MODEL_DIR / "attack_label_encoder.pkl")

    return scaler, feature_names, mappings, lstm_model, clf, le

scaler, FEATURE_NAMES, mappings, lstm_model, clf, le = load_artifacts()

# ================= PREPROCESS =================
def preprocess(raw):
    row = {f: 0.0 for f in FEATURE_NAMES}

    # numeric values
    for k in raw:
        if k in row:
            row[k] = raw[k]

    # categorical encoding
    row["protocol_type"] = mappings["protocol_type"].get(raw["protocol"], 0)
    row["service"] = mappings["service"].get(raw["service"], 0)
    row["flag"] = mappings["flag"].get(raw["flag"], 0)
    row["logged_in"] = int(raw["logged_in"])

    df = pd.DataFrame([row], columns=FEATURE_NAMES)

    # scaled vector (for both models)
    scaled_vec = scaler.transform(df).astype(np.float32)[0]

    # LSTM sequence
    seq = np.repeat(scaled_vec.reshape(1, 1, -1), SEQ_LEN, axis=1)

    return scaled_vec, seq

# ================= UI =================
st.title("🔐 Hybrid IDS: LSTM Autoencoder + XGBoost")

raw = {
    "duration": st.number_input("Duration", 0.0),
    "src_bytes": st.number_input("Source Bytes", 0.0),
    "dst_bytes": st.number_input("Destination Bytes", 0.0),
    "count": st.number_input("Count", 0.0),
    "srv_count": st.number_input("Service Count", 0.0),
    "protocol": st.selectbox("Protocol", mappings["protocol_type"].keys()),
    "service": st.selectbox("Service", mappings["service"].keys()),
    "flag": st.selectbox("Flag", mappings["flag"].keys()),
    "logged_in": st.checkbox("Logged In")
}

if st.button("Analyze Traffic"):
    scaled_vec, seq = preprocess(raw)

    x = torch.tensor(seq).to(DEVICE)

    with torch.no_grad():
        recon = lstm_model(x)
        error = torch.mean((recon[:, -1] - x[:, -1]) ** 2).item()

    st.metric("Reconstruction Error", f"{error:.6f}")

    # ================= DECISION LOGIC =================
    if error < ANOMALY_THRESHOLD:
        st.success("✅ NORMAL TRAFFIC")
    else:
        # Stage-2 attack classification
        attack_id = clf.predict(scaled_vec.reshape(1, -1))[0]
        attack_name = le.inverse_transform([attack_id])[0]
        st.error(f"🚨 INTRUSION DETECTED ({attack_name})")
