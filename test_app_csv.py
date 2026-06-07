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
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
BASE_DIR  = Path("F:/Master Thesis/anomaly-ids")

TRAIN_PATH = BASE_DIR / "data/raw/nsl-kdd/KDDTrain+.txt"
TEST_PATH  = BASE_DIR / "data/raw/nsl-kdd/KDDTest+.txt"

DATA_DIR  = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "data/generated"
RES_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ================= LOAD =================
print("\nLoading artifacts...")

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

clf = joblib.load(MODEL_DIR / "attack_classifier_full_xgb.pkl")
le  = joblib.load(MODEL_DIR / "attack_label_encoder_full.pkl")
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

print("\nLoading FULL Train + Test Data...")

train_df = pd.read_csv(TRAIN_PATH, names=COLUMNS)
test_df  = pd.read_csv(TEST_PATH, names=COLUMNS)

print(f"Train rows: {len(train_df)}")
print(f"Test rows : {len(test_df)}")

# ✅ Use FULL test dataset (no filtering)
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
df["label_attack"] = df["label"]

print(f"Total rows : {len(df)}")

# ================= PREPROCESS =================
df["protocol_type"] = df["protocol_type"].map(mappings["protocol_type"])
df["service"]       = df["service"].map(mappings["service"])
df["flag"]          = df["flag"].map(mappings["flag"])

df = df.fillna(0)

# ================= UI SIMULATION =================
print("\nApplying UI-like simulation...")

IMPORTANT_FEATURES = [
    # Basic connection features (keep most of yours)
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
     "wrong_fragment",

    # Login / privilege related (critical for R2L & U2R)
    "logged_in", "num_failed_logins", "is_guest_login",
    "root_shell",
    "num_shells", "num_access_files",

    # Error and rate features (good for DoS/Probe)
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate",

    # Host-based traffic features
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

print(f"✅ Features  used: {len(IMPORTANT_FEATURES)}")
mean_values = df[FEATURE_NAMES].mean()

df_simulated = df[FEATURE_NAMES].copy()

for col in FEATURE_NAMES:
    if col not in IMPORTANT_FEATURES:
        df_simulated[col] = mean_values[col]

# ================= SCALE (FIXED) =================
X_scaled = scaler.transform(df_simulated).astype(np.float32)

# ================= AE =================
print("\nRunning Dense AE...")

X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)

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

attack_rows = X_scaled[is_anomaly]

if len(attack_rows) > 0:
    attack_ids = clf.predict(attack_rows)
    attack_labels = le.inverse_transform(attack_ids)
else:
    attack_labels = []

predicted_labels = np.array(
    ["normal"] * len(df),
    dtype=object
)
predicted_labels[is_anomaly] = attack_labels

results_df = pd.DataFrame({
    "actual_label": df["label_attack"],
    "predicted_label": predicted_labels,
    "error": errors,
    "flagged_as_anomaly": is_anomaly.astype(int)
})

# ================= METRICS =================
y_true = (results_df["actual_label"] != "normal").astype(int)
y_pred = (results_df["predicted_label"] != "normal").astype(int)

print("\n===== RESULTS =====")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, errors))

# ================= CONFUSION MATRIX =================
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ----------- 1. BINARY CONFUSION MATRIX -----------
print("\n=== BINARY CONFUSION MATRIX ===")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

# Plot Binary CM
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Binary Confusion Matrix")
plt.tight_layout()
plt.savefig(RES_DIR / "confusion_matrix_binary_test_app.png")


# ----------- 2. MULTI-CLASS CONFUSION MATRIX -----------
print("\n=== MULTI-CLASS CONFUSION MATRIX ===")

labels = sorted(results_df["actual_label"].unique())
print("\nActual labels:")
print(sorted(results_df["actual_label"].unique()))

print("\nPredicted labels:")
print(sorted(results_df["predicted_label"].unique()))

print(
    results_df["predicted_label"].value_counts()
)

print(
    (results_df["predicted_label"] == "neptune").sum()
)

cm_multi = confusion_matrix(
    results_df["actual_label"],
    results_df["predicted_label"],
    labels=labels
)

print("Matrix shape:", cm_multi.shape)

# Plot Multi-class CM
plt.figure(figsize=(14,12))
sns.heatmap(
    cm_multi,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multi-Class Confusion Matrix (All Attacks)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RES_DIR / "confusion_matrix_multiclass_test_app.png")


# ----------- 3. SAVE CONFUSION MATRIX -----------
cm_df = pd.DataFrame(cm_multi, index=labels, columns=labels)
cm_df.to_csv(RES_DIR / "confusion_matrix_38_classes_test_app.csv")
print("\nNeptune row:")
print(cm_df.loc["neptune"])

print("\nNeptune -> Neptune:")
print(cm_df.loc["neptune", "neptune"])

print("\nNeptune row sum:")
print(cm_df.loc["neptune"].sum())

print("✅ Confusion matrix saved to CSV")

# ================= PER ATTACK =================
print("\n=== PER ATTACK DETECTION ===")

attack_types = results_df[results_df["actual_label"] != "normal"]["actual_label"].unique()

for attack in sorted(attack_types):
    mask = results_df["actual_label"] == attack
    total = mask.sum()
    detected = (results_df.loc[mask, "flagged_as_anomaly"] == 1).sum()
    rate = detected / total if total > 0 else 0
    print(f"{attack:<20} {detected}/{total} ({rate:.2%})")

print("\n✅ Done.")