# app.py — FINAL VERSION with StandardScaler + Full Debugger + LSTM Autoencoder
# Uses scaler.pkl (your uploaded vae_scaler.pkl — which you confirmed is same as lstm scaler)
# Input dimension automatically detected from checkpoint (41 features)

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

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

st.title("🛡️ AI Intrusion Detection System")
st.markdown("---")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================================================================
# 🔍 FIXED DEBUGGER — SAFE PRINTING (NO .cpu() ERRORS)
# ========================================================================
def debug_single_flow(raw, encoded, scaled, final_vec, score, recon_last):
    print("\n" + "=" * 120)
    print("🔍 DEBUG TRACE — SINGLE FLOW")
    print("=" * 120)

    print("\n📌 RAW INPUT:")
    for k, v in raw.items():
        print(f"   {k:15} : {v}")

    print("\n📌 ENCODED VALUES:")
    for k, v in encoded.items():
        print(f"   {k:15} : {v}")

    print("\n📌 SCALED VALUES (from StandardScaler):")
    for k, v in scaled.items():
        print(f"   {k:15} : {v}")

    print(f"\n📌 FINAL FEATURE VECTOR (len={len(final_vec)}):")
    print(final_vec)

    print("\n📌 MODEL RECONSTRUCTION (last timestep):")
    try:
        if isinstance(recon_last, torch.Tensor):
            print(recon_last.detach().cpu().numpy())
        else:
            print("[WARNING] recon_last is not tensor:", recon_last)
    except Exception as e:
        print("[ERROR printing recon_last]:", e)

    print("\n📌 FINAL ANOMALY SCORE:")
    print(score)

    print("=" * 120 + "\n")


# ========================================================================
# READ CHECKPOINT + DETECT INPUT DIM
# ========================================================================
def read_state_dict(path):
    if not os.path.exists(path): 
        return None
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
        if all(isinstance(k, str) for k in sd.keys()):
            return sd
    return sd


def infer_model_meta(sd):
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
    num_layers = 2 if has_l1 else 1

    return input_dim, num_layers


# ========================================================================
# MODEL DEFINITION
# ========================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return self.fc(out)


# ========================================================================
# LOAD EVERYTHING (MODEL + MAPPINGS + SCALER)
# ========================================================================
@st.cache_resource
def load_everything():
    ckpt = "models/lstm_autoencoder_best.pth"
    sd = read_state_dict(ckpt)

    input_dim, num_layers = infer_model_meta(sd)
    model = LSTMAutoencoder(input_dim, 64, num_layers).to(DEVICE)

    try:
        model.load_state_dict(sd, strict=True)
    except:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    
    # --- load mappings ---
    with open("data/processed/label_mappings.json", "r") as f:
        mappings = json.load(f)

    # --- load StandardScaler ---
    scaler = joblib.load("models/vae_scaler.pkl")   # you said it's the same scaler

    return model, mappings, scaler, input_dim


ae_model, mappings, scaler, MODEL_INPUT_DIM = load_everything()


# ========================================================================
# PREPROCESS USING SCALER.PKL
# ========================================================================
def preprocess_single_flow(raw):
    vec = np.zeros(MODEL_INPUT_DIM, dtype=float)

    # Categorical encoding
    protocol_enc = mappings["protocol_type"].get(raw["protocol"], 0)
    service_enc  = mappings["service"].get(raw["service"], 0)
    flag_enc     = mappings["flag"].get(raw["flag"], 0)
    logged_enc   = int(raw["logged_in"])

    # Fill feature vector
    vec[0] = raw["duration"]
    vec[1] = protocol_enc
    vec[2] = service_enc
    vec[3] = flag_enc
    vec[11] = logged_enc
    vec[22] = raw["count"] / 1e3
    vec[23] = raw["srv_count"] / 1e3
    vec[4]  = raw["src_bytes"] /1e6
    vec[5]  = raw["dst_bytes"] / 1e6

    # Scale using StandardScaler
    scaled_vec = scaler.transform(vec.reshape(1, -1))[0]

    encoded = dict(protocol_enc=protocol_enc,
                   service_enc=service_enc,
                   flag_enc=flag_enc,
                   logged_in_enc=logged_enc)

    scaled_debug = dict(
        src_scaled=scaled_vec[4],
        dst_scaled=scaled_vec[5],
        count_scaled=scaled_vec[22],
        srv_scaled=scaled_vec[23],
    )

    return scaled_vec.astype(np.float32), encoded, scaled_debug


# ========================================================================
# EVALUATE VECTOR + DEBUGGER
# ========================================================================
def evaluate_vector(vec, raw=None, encoded=None, scaled=None):
    try:
        x = torch.tensor(vec).unsqueeze(0).to(DEVICE)
        seq = x.unsqueeze(1).repeat(1, 10, 1)

        with torch.no_grad():
            recon = ae_model(seq)
            recon_last = recon[:, -1, :]

            score = torch.mean((recon_last - x) ** 2).item()

        if raw:
            debug_single_flow(raw, encoded, scaled, vec, score, recon_last)

        if score < 5:
            return "🟢 NORMAL", "Legitimate traffic", score
        elif 5 < score < 20:
            return "🟡 SUSPICIOUS", "Unusual behaviour",score
        else:
            return "🔴 INTRUSION DETECTED", "High anomaly — possible attack", score

    except Exception as e:
        print("Error during preprocessing/evaluation:", e)
        return "❌ ERROR", str(e), 0.0
    
def evaluate_vector_batch(vec, raw=None, encoded=None, scaled=None):
    try:
        x = torch.tensor(vec).unsqueeze(0).to(DEVICE)
        seq = x.unsqueeze(1).repeat(1, 10, 1)

        with torch.no_grad():
            recon = ae_model(seq)
            recon_last = recon[:, -1, :]

            score = torch.mean((recon_last - x) ** 2).item()

        if raw:
            debug_single_flow(raw, encoded, scaled, vec, score, recon_last)

        if score < 0.05:
            return "🟢 NORMAL", "Legitimate traffic", score
        elif score > 0.1:
            return "🔴 INTRUSION DETECTED", "High anomaly — possible attack", score
        else:
            return "🟡 SUSPICIOUS", "Unusual behaviour", score

    except Exception as e:
        print("Error during preprocessing/evaluation:", e)
        return "❌ ERROR", str(e), 0.0


# ========================================================================
# UI — SINGLE FLOW
# ========================================================================
tab1, tab2 = st.tabs(["🔍 Single Flow (StandardScaler)", "📊 Batch CSV"])

with tab1:
    st.subheader("Single Flow — Raw Inputs (scaled using scaler.pkl)")

    with st.form("single"):
        c1, c2 = st.columns(2)

        with c1:
            duration = st.number_input("Duration", 0, 100000, 0)
            protocol = st.selectbox("Protocol", list(mappings["protocol_type"].keys()))
            service = st.selectbox("Service", list(mappings["service"].keys()))
            flag = st.selectbox("Flag", list(mappings["flag"].keys()))
            src_bytes = st.number_input("Src Bytes", 0, 10**9, 215)
            dst_bytes = st.number_input("Dst Bytes", 0, 10**9, 4500)

        with c2:
            logged_in = st.checkbox("Logged In")
            count = st.slider("Count", 0, 511, 0)
            srv_count = st.slider("Srv Count", 0, 511, 0)

        if st.form_submit_button("🚨 DETECT INTRUSION"):
            raw = dict(
                duration=duration,
                protocol=protocol,
                service=service,
                flag=flag,
                src_bytes=src_bytes,
                dst_bytes=dst_bytes,
                logged_in=logged_in,
                count=count,
                srv_count=srv_count
            )

            vec, encoded, scaled_debug = preprocess_single_flow(raw)
            verdict, desc, score = evaluate_vector(vec, raw, encoded, scaled_debug)

            color = "normal" if "NORMAL" in verdict else "attack" if "INTRUSION" in verdict else "susp"

            st.markdown(f'<p class="big {color}">{verdict}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="med">{desc}</p>', unsafe_allow_html=True)
            st.info(f"Anomaly Score: **{score:.6f}**")


# ========================================================================
# UI — BATCH CSV
# ========================================================================
with tab2:
    upload = st.file_uploader("Upload CSV", type="csv")

    if upload:
        df = pd.read_csv(upload)

        if df.shape[1] == MODEL_INPUT_DIM:
            scaled_df = df.values.astype(np.float32)
        else:
            st.error(f"CSV must have exactly {MODEL_INPUT_DIM} columns.")

        results = []
        for row in scaled_df:
            verdict, desc, score = evaluate_vector_batch(row)
            results.append([verdict, desc, score])

        out = pd.DataFrame(results, columns=["Verdict", "Description", "Score"])
        final = pd.concat([df, out], axis=1)

        st.dataframe(final)
        st.download_button("Download Results", final.to_csv(index=False), "ids_results.csv")


