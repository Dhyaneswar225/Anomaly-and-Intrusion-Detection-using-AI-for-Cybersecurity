# DenseAETest_UI_Simulation.py

import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score
)

from src.DenseAutoEncoderModel import DenseAutoencoder


# ================= CONFIG =================
BASE_DIR  = Path("F:/Master Thesis/anomaly-ids")
RAW_PATH  = BASE_DIR / "data/raw/nsl-kdd/KDDTrain+.txt"
DATA_DIR  = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "data/generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ================= LOAD =================
print("\nLoading artefacts...")

scaler        = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

ANOMALY_THRESHOLD = float(np.load(MODEL_DIR / "dense_ae_threshold.npy"))

print(f"✅ Features          : {len(FEATURE_NAMES)}")
print(f"✅ Anomaly threshold : {ANOMALY_THRESHOLD:.6f}")

with open(DATA_DIR / "label_mappings.json") as f:
    mappings = json.load(f)

# ================= MODEL =================
model = DenseAutoencoder(
    input_dim=len(FEATURE_NAMES),
    bottleneck=32,
    dropout=0.2
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_DIR / "dense_autoencoder_best.pth", map_location=DEVICE)
)
model.eval()
print("✅ Dense Autoencoder loaded")

clf = joblib.load(MODEL_DIR / "attack_classifier_xgb.pkl")
le  = joblib.load(MODEL_DIR / "attack_label_encoder.pkl")
print("✅ Attack classifier loaded")


# ================= DATA =================
COLUMNS = [
'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
'num_compromised','root_shell','su_attempted','num_root',
'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login','is_guest_login','count','srv_count','serror_rate',
'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
'dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
'dst_host_srv_rerror_rate','label','difficulty'
]

print("\nLoading dataset...")

df = pd.read_csv(RAW_PATH, names=COLUMNS)
df["label_attack"] = df["label"]

print(f"Total rows : {len(df)}")

# ================= PREPROCESS =================
df["protocol_type"] = df["protocol_type"].map(mappings["protocol_type"])
df["service"]       = df["service"].map(mappings["service"])
df["flag"]          = df["flag"].map(mappings["flag"])

df = df.fillna(0)

# ================= 🔥 UI SIMULATION =================
print("\nApplying UI-like simulation (26 features + mean fill)...")

IMPORTANT_FEATURES = [
    "protocol_type",
    "flag",
    "src_bytes",
    "logged_in",
    "num_failed_logins",
    "serror_rate",
    "rerror_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "service",
    "dst_bytes",
    "duration",
    "count",
    "srv_count",
    "dst_host_same_srv_rate",
    "same_srv_rate",
    "dst_host_rerror_rate",
    "dst_host_serror_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",

    # 🔥 NEWLY ADDED FEATURES
    "srv_diff_host_rate",
    "dst_host_srv_diff_host_rate",
    "num_compromised",
    "hot",
    "is_guest_login",
    "root_shell",
    "num_shells",
    "wrong_fragment"
]

# mean values
mean_values = df[FEATURE_NAMES].mean()

# simulate UI input
df_simulated = df[FEATURE_NAMES].copy()

for col in FEATURE_NAMES:
    if col not in IMPORTANT_FEATURES:
        df_simulated[col] = mean_values[col]

print(f"Kept features     : {len(IMPORTANT_FEATURES)}")
print(f"Replaced features : {len(FEATURE_NAMES) - len(IMPORTANT_FEATURES)}")

# ================= SCALE =================
X_scaled = scaler.transform(df_simulated.values.astype(np.float32))


# ================= AE =================
print("\nRunning Dense AE...")

X_tensor = torch.tensor(X_scaled).to(DEVICE)

errors = []
BATCH = 1024

with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH):
        batch = X_tensor[i:i+BATCH]
        recon = model(batch)
        err   = torch.mean((recon - batch) ** 2, dim=1)
        errors.extend(err.cpu().numpy())

errors = np.array(errors)

# ================= PREDICT =================
print("Generating predictions...")

is_anomaly = errors > ANOMALY_THRESHOLD

attack_rows   = X_scaled[is_anomaly]
attack_ids    = clf.predict(attack_rows) if len(attack_rows) > 0 else []
attack_labels = le.inverse_transform(attack_ids) if len(attack_ids) > 0 else []

predicted_labels = np.where(is_anomaly, "attack_placeholder", "normal")
predicted_labels[is_anomaly] = attack_labels

results_df = pd.DataFrame({
    "actual_label": df["label_attack"],
    "predicted_label": predicted_labels,
    "error": errors,
    "flagged_as_anomaly"   : is_anomaly.astype(int)
})

# ================= METRICS =================
y_true = (results_df["actual_label"] != "normal").astype(int)
y_pred = (results_df["predicted_label"] != "normal").astype(int)
tp = int(((y_true == 1) & (y_pred == 1)).sum())
tn = int(((y_true == 0) & (y_pred == 0)).sum())
fp = int(((y_true == 0) & (y_pred == 1)).sum())
fn = int(((y_true == 1) & (y_pred == 0)).sum())

print("\n===== RESULTS (UI SIMULATION) =====")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, errors))

print("\nConfusion Matrix:")
print(f"True  Positives (attacks caught)  : {tp:>6}")
print(f"True  Negatives (normal correct)  : {tn:>6}")
print(f"False Positives (false alarms)    : {fp:>6}")
print(f"False Negatives (missed attacks)  : {fn:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# EXACT LABEL MATCH  (normal / specific attack type)
# ══════════════════════════════════════════════════════════════════════════════

exact_correct = (results_df["actual_label"] == results_df["predicted_label"]).sum()
exact_wrong   = len(results_df) - exact_correct

print("\n" + "=" * 45)
print("EXACT LABEL MATCH  (multi-class)")
print("=" * 45)
print(f"Correct : {exact_correct}  ({exact_correct/len(results_df):.2%})")
print(f"Wrong   : {exact_wrong}  ({exact_wrong/len(results_df):.2%})")


# ══════════════════════════════════════════════════════════════════════════════
# PER ATTACK TYPE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 45)
print("PER ATTACK TYPE DETECTION RATE")
print("=" * 45)

attack_types = results_df[results_df["actual_label"] != "normal"]["actual_label"].unique()

for attack in sorted(attack_types):
    mask      = results_df["actual_label"] == attack
    total     = mask.sum()
    detected  = (results_df.loc[mask, "flagged_as_anomaly"] == 1).sum()
    rate      = detected / total if total > 0 else 0
    print(f"  {attack:<20} detected {detected:>5}/{total:<5}  ({rate:.2%})")

print("\nDone.")