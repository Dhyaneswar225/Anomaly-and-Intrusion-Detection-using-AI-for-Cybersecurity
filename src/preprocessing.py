import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import json
import joblib

# -----------------------------
# Paths
# -----------------------------
DATA_RAW = Path("data/raw/nsl-kdd")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Column names for NSL-KDD
# -----------------------------
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]

print("🔹 Loading NSL-KDD data...")

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv(DATA_RAW / "KDDTrain+.txt", names=columns)
test_df  = pd.read_csv(DATA_RAW / "KDDTest+.txt", names=columns)

# -----------------------------
# Preserve original attack labels
# -----------------------------
train_df["label_attack"] = train_df["label"]
test_df["label_attack"] = test_df["label"]

# Binary label
train_df["label_binary"] = train_df["label"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)
test_df["label_binary"] = test_df["label"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)

# -----------------------------
# Drop difficulty column
# -----------------------------
train_df.drop(columns=["difficulty"], inplace=True)
test_df.drop(columns=["difficulty"], inplace=True)

# -----------------------------
# Remove duplicates & missing values
# -----------------------------
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# -----------------------------
# Identify feature types
# -----------------------------
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [
    c for c in train_df.columns
    if c not in categorical_cols + ['label', 'label_attack', 'label_binary']
]

# -----------------------------
# Encode categorical features
# (fit on train, apply on test)
# -----------------------------
print("🔹 Encoding categorical features...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

    label_encoders[col] = {
        cls: int(val)
        for cls, val in zip(le.classes_, le.transform(le.classes_))
    }

# -----------------------------
# Scale numeric features (NO LEAKAGE)
# -----------------------------
print("🔹 Scaling numeric features with StandardScaler...")

scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols]  = scaler.transform(test_df[numeric_cols])

# -----------------------------
# Save processed datasets
# -----------------------------
train_df.to_csv(DATA_PROCESSED / "train_processed.csv", index=False)
test_df.to_csv(DATA_PROCESSED / "test_processed.csv", index=False)

# -----------------------------
# Save encoders & scaler
# -----------------------------
with open(DATA_PROCESSED / "label_mappings.json", "w") as f:
    json.dump(label_encoders, f, indent=4)

joblib.dump(scaler, DATA_PROCESSED / "standard_scaler.pkl")

print("\n🎯 Preprocessing complete!")
print(f"Train shape: {train_df.shape}")
print(f"Test shape : {test_df.shape}")
print("Saved to:", DATA_PROCESSED)
print("🔐 Files saved:")
print("  - train_processed.csv")
print("  - test_processed.csv")
print("  - label_mappings.json")
print("  - standard_scaler.pkl")
