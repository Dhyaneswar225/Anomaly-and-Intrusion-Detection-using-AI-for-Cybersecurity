import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ============================
# PATHS
# ============================
DATA_PATH = "data/processed/train_processed.csv"
SCALER_PATH = "data/processed/standard_scaler.pkl"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# LOAD DATA
# ============================
train = pd.read_csv(DATA_PATH)

if "label_attack" not in train.columns:
    raise ValueError("train_processed.csv must contain 'label_attack' column")

# ============================
# LOAD SCALER (SINGLE SOURCE OF TRUTH)
# ============================
scaler = joblib.load(SCALER_PATH)
FEATURE_NAMES = list(scaler.feature_names_in_)

print(f"Using {len(FEATURE_NAMES)} features (aligned with LSTM & scaler)")

# ============================
# KEEP ONLY ATTACK SAMPLES
# ============================
train_attacks = train[train["label_attack"] != "normal"].copy()

print(f"Attack samples: {len(train_attacks)}")

# ============================
# FEATURE MATRIX + LABELS
# ============================
X = train_attacks[FEATURE_NAMES].values.astype(np.float32)
y = train_attacks["label_attack"]

# ============================
# LABEL ENCODING
# ============================
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Attack classes:", list(le.classes_))
print("Total attack classes:", len(le.classes_))

# ============================
# TRAIN XGBOOST CLASSIFIER
# ============================
clf = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

clf.fit(X, y_enc)

# ============================
# SAFETY CHECK (CRITICAL)
# ============================
assert clf.n_features_in_ == len(FEATURE_NAMES), (
    f"Model expects {clf.n_features_in_} features, "
    f"but scaler provides {len(FEATURE_NAMES)}"
)

# ============================
# SAVE MODEL + ENCODER
# ============================
joblib.dump(clf, f"{MODEL_DIR}/attack_classifier_xgb.pkl")
joblib.dump(le, f"{MODEL_DIR}/attack_label_encoder.pkl")

print("✅ Attack classifier training complete")
print("✅ Models saved:")
print("   - models/attack_classifier_xgb.pkl")
print("   - models/attack_label_encoder.pkl")
