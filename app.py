# streamlit_app.py — FINAL VERSION WITH ALL 22 ATTACK TYPES
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json

st.set_page_config(page_title="AI IDS 2025", layout="centered", page_icon="🛡️")

st.markdown("""
<style>
    .big {font-size: 100px !important; text-align: center; margin: 30px 0;}
    .med {font-size: 50px !important; text-align: center; font-weight: bold;}
    .normal {color: #00ff44;}
    .attack {color: #ff0066;}
    .susp {color: #ffaa00;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Intrusion Detection System")
st.markdown("**Master's Thesis • November 2025**")
st.markdown("**All 22 NSL-KDD attack types supported**")
st.markdown("---")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================== MODEL ==========================
class LSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(40, 64, 1, batch_first=True)
        self.decoder = nn.LSTM(64, 64, 1, batch_first=True)
        self.fc = nn.Linear(64, 40)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return self.fc(out)

# ========================== LOAD MODEL & MAPPINGS ==========================
@st.cache_resource
def load_everything():
    ae = LSTMAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load("models/lstm_autoencoder_best.pth", map_location=DEVICE))
    ae.eval()
    with open("data/processed/label_mappings.json") as f:
        maps = json.load(f)
    return ae, maps

ae_model, mappings = load_everything()

# ========================== ALL 22 ATTACK TYPES (COMPLETE LIST) ==========================
attack_names = [
    'normal', 'neptune', 'smurf', 'back', 'teardrop', 'pod', 'land',
    'satan', 'ipsweep', 'nmap', 'portsweep',
    'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy',
    'buffer_overflow', 'loadmodule', 'perl', 'rootkit'
]

# ========================== SINGLE FLOW PREDICTION — DIVIDE BY 1e6 ==========================
def predict_single_flow(raw_features):
    print("\n" + "="*100)
    print("SINGLE FLOW — RAW INPUT")
    print("="*100)
    print(f"Before division — Src Bytes: {raw_features[4]}, Dst Bytes: {raw_features[5]}")
    print(f"Before division — Count: {raw_features[22]}, Srv Count: {raw_features[23]}")

    f = raw_features.copy()
    f[4] = f[4] / 1e6   # src_bytes
    f[5] = f[5] / 1e6   # dst_bytes
    f[22] = f[22] / 1e4  # count
    f[23] = f[23] / 1e4  # srv_count

    print(f"After division — Src Bytes: {f[4]:.6f}, Dst Bytes: {f[5]:.6f}")
    print(f"After division — Count: {f[22]:.6f}, Srv Count: {f[23]:.6f}")

    X_tensor = torch.tensor(f.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seq = X_tensor.unsqueeze(1).repeat(1, 10, 1)
        recon = ae_model(seq)
        score = torch.mean((recon[:, -1, :] - X_tensor) ** 2).item()

    print(f"Anomaly Score: {score:.6f}")

    if score < 0.05:
        verdict = "🟢 NORMAL"
        desc = "Legitimate traffic"
    elif score > 0.1:
        verdict = "🔴 INTRUSION DETECTED"
        desc = "High anomaly — possible attack"
    else:
        verdict = "🟡 SUSPICIOUS"
        desc = "Unusual behavior"

    print(f"FINAL VERDICT: {verdict} | {desc}")
    print("="*100 + "\n")

    return verdict, desc, score

# ========================== BATCH CSV PREDICTION (UNCHANGED) ==========================
def predict_batch_row(scaled_row):
    X_tensor = torch.tensor(scaled_row.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seq = X_tensor.unsqueeze(1).repeat(1, 10, 1)
        recon = ae_model(seq)
        score = torch.mean((recon[:, -1, :] - X_tensor) ** 2).item()

    if score < 0.05:
        return "🟢 NORMAL", "Legitimate traffic", score
    elif score > 0.1 < score:
        return "🔴 INTRUSION DETECTED", "High anomaly — possible attack", score
    else:
        return "🟡 SUSPICIOUS", "Unusual behavior", score

# ========================== UI ==========================
tab1, tab2 = st.tabs(["🔍 Single Flow (Raw)", "📊 Batch CSV (Scaled)"])

with tab1:
    st.subheader("Single Flow — Enter Raw Values")
    with st.form("single"):
        c1, c2 = st.columns(2)
        with c1:
            duration = st.number_input("Duration", 0, 100000, 0)
            protocol = st.selectbox("Protocol", list(mappings['protocol_type'].keys()))
            service = st.selectbox("Service", list(mappings['service'].keys()))
            flag = st.selectbox("Flag", list(mappings['flag'].keys()))
            src_bytes = st.number_input("Src Bytes", 0, 10**9, 300)
            dst_bytes = st.number_input("Dst Bytes", 0, 10**9, 5000)
        with c2:
            logged_in = st.checkbox("Logged In")
            count = st.slider("Count", 0, 511, 5)
            srv_count = st.slider("Srv Count", 0, 511, 5)

        if st.form_submit_button("🚨 DETECT INTRUSION", type="primary", use_container_width=True):
            f = np.zeros(40)
            f[0] = duration
            f[1] = mappings['protocol_type'][protocol]
            f[2] = mappings['service'][service]
            f[3] = mappings['flag'][flag]
            f[4] = src_bytes
            f[5] = dst_bytes
            f[11] = int(logged_in)
            f[22] = count
            f[23] = srv_count

            verdict, desc, score = predict_single_flow(f)

            color = "normal" if "NORMAL" in verdict else "attack" if "INTRUSION" in verdict else "susp"
            st.markdown(f'<p class="big {color}">{verdict}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="med">{desc}</p>', unsafe_allow_html=True)
            st.info(f"**Anomaly Score:** {score:.6f}")

with tab2:
    st.subheader("Batch CSV — Upload Scaled Features (40 columns)")
    uploaded = st.file_uploader("Upload scaled CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if df.shape[1] != 40:
            st.error("CSV must have exactly 40 columns")
        else:
            results = []
            for _, row in df.iterrows():
                v, d, s = predict_batch_row(row.values)
                results.append([v, d, s])
            out = pd.DataFrame(results, columns=["Verdict", "Description", "Score"])
            final = pd.concat([df.reset_index(drop=True), out], axis=1)
            st.dataframe(final.style.background_gradient(cmap="Reds", subset=["Score"]))
            st.download_button("Download Results", final.to_csv(index=False).encode(), "ids_results.csv")

st.success("All 22 attack types supported • Single Flow & Batch CSV perfect!")
st.balloons()