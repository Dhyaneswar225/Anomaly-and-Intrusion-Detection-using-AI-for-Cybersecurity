import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import json

# Paths
DATA_RAW = Path("data/raw/nsl-kdd")
DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Column names for NSL-KDD (41 features + label + difficulty)
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

# Load data
train_df = pd.read_csv(DATA_RAW / "KDDTrain+.txt", names=columns)
test_df  = pd.read_csv(DATA_RAW / "KDDTest+.txt", names=columns)

# Combine for consistent encoding
df = pd.concat([train_df, test_df], axis=0)
print(f"✅ Loaded dataset with shape: {df.shape}")

# --- Cleaning Step ---
print("🔹 Cleaning data...")

# 1. Check for missing values
missing = df.isnull().sum()
if missing.any():
    print("⚠️ Missing values found:\n", missing[missing > 0])
    # Option: drop rows with missing values or impute
    df = df.dropna()
    print(f"✅ Dropped rows with missing values. New shape: {df.shape}")
else:
    print("✅ No missing values found.")

# 2. Remove constant columns
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"⚠️ Constant columns found: {constant_cols}")
    df = df.drop(columns=constant_cols)
    print(f"✅ Dropped constant columns. New shape: {df.shape}")
else:
    print("✅ No constant columns found.")

# 3. Remove duplicates
initial_rows = df.shape[0]
df = df.drop_duplicates()
print(f"✅ Removed {initial_rows - df.shape[0]} duplicate rows. New shape: {df.shape}")

# Convert label to binary: normal vs attack
df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Identify categorical and numeric columns
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [c for c in df.columns if c not in categorical_cols + ['label', 'difficulty']]

# Encode categorical columns and store mappings
print("🔹 Encoding categorical features...")
mappings = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    mappings[col] = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}

# Display mappings
print("\n🔍 Label Encoding Mappings:\n")
for col, mapping in mappings.items():
    print(f"{col} mapping:")
    for k, v in mapping.items():
        print(f"  {k:<25} → {v}")
    print()

# Drop difficulty column
df = df.drop(columns=['difficulty'])

# Scale numeric features
print("🔹 Scaling numeric features...")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Split back into train/test
train_len = len(train_df)
train_processed = df.iloc[:train_len].copy()
test_processed  = df.iloc[train_len:].copy()

# Save processed data
train_processed.to_csv(DATA_PROCESSED / "train_processed.csv", index=False)
test_processed.to_csv(DATA_PROCESSED / "test_processed.csv", index=False)

# Convert mappings to JSON-serializable format
mappings_clean = {
    col: {str(k): int(v) for k, v in mapping.items()}
    for col, mapping in mappings.items()
}

# Save mappings
with open(DATA_PROCESSED / "label_mappings.json", "w") as f:
    json.dump(mappings_clean, f, indent=4)
print("\n💾 Saved label mappings to:", DATA_PROCESSED / "label_mappings.json")

print("\n Preprocessing complete!")
print(f"Train processed shape: {train_processed.shape}")
print(f"Test processed shape: {test_processed.shape}")
print(f"Saved processed files to: {DATA_PROCESSED}")
