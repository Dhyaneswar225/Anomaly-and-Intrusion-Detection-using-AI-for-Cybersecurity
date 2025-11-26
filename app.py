# ======================================================================
# app.py — FINAL FIXED VERSION
# LSTM Autoencoder + Attack Classifier
#
# Thresholds (same for single-flow & batch):
#     NORMAL < 5
#     SUSPICIOUS 5–20
#     INTRUSION > 20
#
# Scaling rules:
#   - RAW CSV → scale using scaler
#   - SCALED CSV → use directly (NO RESCALING)
#   - Classifier always gets RAW (inverse-transform if needed)
# ======================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from typing import Tuple

st.set_page_config(page_title="AI IDS 2025", layout="centered", page_icon="🛡️")

# -------------------------------- UI STYLE --------------------------------
st.markdown("""
<style>
.big {font-size: 90px !important; text-align: center; margin: 30px 0;}
.med {font-size: 40px !important; text-align: center; font-weight: bold;}
.normal {color: #00ff44;}
.attack {color: #ff0066;}
.susp {color: #ffaa00;}
</style>
""", unsafe_allow_html=True)

st.title("🛡 AI Intrusion Detection System — LSTM + Attack Classifier")
st.markdown("---")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Read checkpoint helpers
# ======================================================================
def read_state_dict(path):
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
        return sd
    return sd


def infer_model_meta(sd) -> Tuple[int, int]:
    if sd is None:
        return 41, 1
    input_dim = None
    for k, v in sd.items():
        if k.endswith("fc.weight"):
            input_dim = v.shape[0]
            break
    if input_dim is None:
        for k, v in sd.items():
            if k.endswith("fc.bias"):
                input_dim = v.shape[0]
                break
    has_l1 = any("weight_ih_l1" in k for k in sd.keys())
    return input_dim, (2 if has_l1 else 1)


# ======================================================================
# LSTM Autoencoder definition
# ======================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return self.fc(out)


# ======================================================================
# Load all components
# ======================================================================
@st.cache_resource
def load_everything():
    ckpt = "models/lstm_autoencoder_best.pth"
    sd = read_state_dict(ckpt)
    input_dim, num_layers = infer_model_meta(sd)
    model = LSTMAutoencoder(input_dim, 64, num_layers).to(DEVICE)
    model.load_state_dict(sd, strict=False)

    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)

    scaler = joblib.load("models/vae_scaler.pkl")

    clf = None
    enc = None
    if os.path.exists("models/attack_classifier_xgb.pkl"):
        clf = joblib.load("models/attack_classifier_xgb.pkl")
        enc = joblib.load("models/attack_label_encoder.pkl")

    try:
        df_tmp = pd.read_csv("data/processed/train_processed.csv", nrows=1)
        drop_cols = {"label", "label_attack", "label_binary"}
        features = [c for c in df_tmp.columns if c not in drop_cols]
    except:
        features = None

    return model, mappings, scaler, input_dim, clf, enc, features


ae_model, mappings, scaler, MODEL_INPUT_DIM, attack_clf, attack_encoder, FEATURE_NAMES = load_everything()


# ======================================================================
# Preprocess single flow
# ======================================================================
def preprocess_single_flow(raw: dict):
    vec = np.zeros(MODEL_INPUT_DIM, dtype=float)

    vec[1] = mappings["protocol_type"].get(raw["protocol"], 0)
    vec[2] = mappings["service"].get(raw["service"], 0)
    vec[3] = mappings["flag"].get(raw["flag"], 0)
    vec[11] = int(raw["logged_in"])

    vec[0] = raw["duration"]
    vec[4] = raw["src_bytes"] / 1e6
    vec[5] = raw["dst_bytes"] / 1e6
    vec[22] = raw["count"] / 1e3
    vec[23] = raw["srv_count"] / 1e3

    scaled = scaler.transform(vec.reshape(1, -1))[0]
    return scaled.astype(np.float32), vec.astype(np.float32)


# ======================================================================
# LSTM Evaluation (same thresholds everywhere)
# ======================================================================
def evaluate_lstm(scaled_vec):
    x = torch.tensor(scaled_vec).unsqueeze(0).to(DEVICE)
    seq = x.unsqueeze(1).repeat(1, 10, 1)

    with torch.no_grad():
        recon = ae_model(seq)
        score = torch.mean((recon[:, -1, :] - x) ** 2).item()

    if score < 5:
        return "🟢 NORMAL", "Legitimate traffic", score
    elif score < 20:
        return "🟡 SUSPICIOUS", "Unusual behaviour", score
    else:
        return "🔴 INTRUSION DETECTED", "High anomaly — possible attack", score


# ======================================================================
# Attack name prediction
# ======================================================================
def predict_attack_name(raw_vec):
    if attack_clf is None or attack_encoder is None:
        return None

    if FEATURE_NAMES and len(FEATURE_NAMES) == len(raw_vec):
        df = pd.DataFrame([raw_vec], columns=FEATURE_NAMES)
    else:
        df = pd.DataFrame([raw_vec])

    pred = attack_clf.predict(df)[0]
    return attack_encoder.inverse_transform([pred])[0]


# ======================================================================
# UI — SINGLE FLOW
# ======================================================================
tab1, tab2 = st.tabs(["🔍 Single Flow", "📊 Batch CSV"])

with tab1:
    st.subheader("Single Flow Detection")

    with st.form("single"):
        c1, c2 = st.columns(2)

        with c1:
            duration = st.number_input("Duration", 0, 100000, 0)
            protocol = st.selectbox("Protocol", list(mappings["protocol_type"].keys()))
            service = st.selectbox("Service", list(mappings["service"].keys()))
            flag = st.selectbox("Flag", list(mappings["flag"].keys()))

        with c2:
            src_bytes = st.number_input("Src Bytes", 0, 1_000_000_000, 215)
            dst_bytes = st.number_input("Dst Bytes", 0, 1_000_000_000, 4500)
            logged_in = st.checkbox("Logged In", False)
            count = st.slider("Count", 0, 511, 0)
            srv_count = st.slider("Srv Count", 0, 511, 0)

        if st.form_submit_button("🚨 DETECT"):
            raw = dict(
                duration=duration,
                protocol=protocol,
                service=service,
                flag=flag,
                src_bytes=src_bytes,
                dst_bytes=dst_bytes,
                logged_in=logged_in,
                count=count,
                srv_count=srv_count,
            )

            scaled_vec, raw_vec = preprocess_single_flow(raw)
            verdict, desc, score = evaluate_lstm(scaled_vec)

            color = "normal" if "NORMAL" in verdict else "attack" if "INTRUSION" in verdict else "susp"

            st.markdown(f'<p class="big {color}">{verdict}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="med">{desc}</p>', unsafe_allow_html=True)
            st.info(f"Score: {score:.6f}")

            if "INTRUSION" in verdict:
                atk = predict_attack_name(raw_vec)
                st.error(f"Attack: **{atk}**" if atk and atk.lower() != "normal" else "Attack: **Unknown**")


# ======================================================================
# UI — BATCH CSV (FINAL FIXED VERSION)
# ======================================================================
with tab2:
    st.subheader("Batch CSV Detection")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        results = []

        # ------------------------------------------------------------
        # CASE A → RAW processed CSV
        # ------------------------------------------------------------
        if FEATURE_NAMES and set(FEATURE_NAMES) <= set(df.columns):
            X_raw = df[FEATURE_NAMES].values.astype(np.float32)
            X_scaled = scaler.transform(X_raw)

            for raw_vec, scaled_vec in zip(X_raw, X_scaled):
                verdict, desc, score = evaluate_lstm(scaled_vec)

                if "INTRUSION" in verdict:
                    atk = predict_attack_name(raw_vec)
                    atk = atk if atk and atk.lower() != "normal" else "Unknown"
                else:
                    atk = ""

                results.append([verdict, desc, score, atk])

        # ------------------------------------------------------------
        # CASE B → SCALED CSV
        # ------------------------------------------------------------
        elif df.shape[1] == MODEL_INPUT_DIM:
            st.info("Detected SCALED CSV → using directly, no rescaling.")
            X_scaled = df.values.astype(np.float32)
            X_raw = scaler.inverse_transform(X_scaled)

            for scaled_vec, raw_vec in zip(X_scaled, X_raw):
                verdict, desc, score = evaluate_lstm(scaled_vec)

                if "INTRUSION" in verdict:
                    atk = predict_attack_name(raw_vec)
                    atk = atk if atk and atk.lower() != "normal" else "Unknown"
                else:
                    atk = ""

                results.append([verdict, desc, score, atk])

        else:
            st.error(
                "CSV not recognized.\nUpload either:\n"
                "1. RAW processed CSV with original feature columns\n"
                "2. SCALED CSV with exactly model input columns"
            )
            st.stop()

        out = pd.DataFrame(results, columns=["Verdict", "Description", "Score", "Attack_Name"])
        final = pd.concat([df.reset_index(drop=True), out], axis=1)

        st.dataframe(final)
        st.download_button("Download Results", final.to_csv(index=False), "ids_results_with_attacks.csv")
