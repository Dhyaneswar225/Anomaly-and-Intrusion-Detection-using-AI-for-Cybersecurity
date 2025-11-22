import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import json

# Paths
DATA_RAW = Path("data/raw/nsl-kdd")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Column names for NSL-KDD dataset
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

# Load original files
train_df = pd.read_csv(DATA_RAW / "KDDTrain+.txt", names=columns)
test_df  = pd.read_csv(DATA_RAW / "KDDTest+.txt", names=columns)

# Keep original attack names in a new column
train_df["label_attack"] = train_df["label"]
test_df["label_attack"] = test_df["label"]

# Convert to binary: normal vs attack
train_df["label_binary"] = train_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")
test_df["label_binary"] = test_df["label"].apply(lambda x: "normal" if x == "normal" else "attack")

# Combine for encoding consistency
df = pd.concat([train_df, test_df], axis=0)
print(f"✅ Loaded dataset: {df.shape}")

# Remove difficulty column
df = df.drop(columns=["difficulty"])

# Clean — remove duplicates and missing values if any
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Identify categorical and numeric columns
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in df.columns if c not in categorical_cols + ['label', 'label_attack', 'label_binary']]

# Encode categorical values
print("🔹 Encoding categorical features...")
mappings = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mappings[col] = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}

# Scale numeric columns
print("🔹 Scaling numeric features...")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split back to train/test
train_processed = df.iloc[:len(train_df)]
test_processed = df.iloc[len(train_df):]

# Save
train_processed.to_csv(DATA_PROCESSED / "train_processed.csv", index=False)
test_processed.to_csv(DATA_PROCESSED / "test_processed.csv", index=False)

# Save label mappings
with open(DATA_PROCESSED / "label_mappings.json", "w") as f:
    json.dump(mappings, f, indent=4)

print("\n🎯 Updated preprocessing complete!")
print(f"Train: {train_processed.shape}, Test: {test_processed.shape}")
print("Saved to:", DATA_PROCESSED)
print("\n🔐 New columns added: label_attack & label_binary")
