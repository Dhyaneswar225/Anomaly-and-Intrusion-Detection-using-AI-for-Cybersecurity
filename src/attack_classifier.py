import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================
# PATHS
# ============================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")

TRAIN_PATH = BASE_DIR / "data/processed/train_processed.csv"
TEST_PATH  = BASE_DIR / "data/processed/test_processed.csv"
SCALER_PATH = BASE_DIR / "data/processed/standard_scaler.pkl"
MODEL_DIR = BASE_DIR / "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================
# MISSING ATTACKS (FROM TEST)
# ============================
MISSING_ATTACKS = [
    'saint','mscan','apache2','snmpgetattack','processtable',
    'httptunnel','ps','snmpguess','mailbomb','named','sendmail',
    'sqlattack','udpstorm','worm','xlock','xsnoop','xterm'
]

# ============================
# LOAD DATA
# ============================
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

if "label_attack" not in train.columns or "label_attack" not in test.columns:
    raise ValueError("Both datasets must contain 'label_attack' column")

# ============================
# LOAD SCALER
# ============================
scaler = joblib.load(SCALER_PATH)
FEATURE_NAMES = list(scaler.feature_names_in_)

print(f"Using {len(FEATURE_NAMES)} features")

# ============================
# FILTER TRAIN (ATTACKS ONLY)
# ============================
train_attacks = train[train["label_attack"] != "normal"].copy()

# ============================
# FILTER TEST (ONLY MISSING ATTACKS)
# ============================
test_filtered = test[test["label_attack"].isin(MISSING_ATTACKS)].copy()

print(f"Selected test samples: {len(test_filtered)}")
print(f"New attack classes from test: {test_filtered['label_attack'].nunique()}")

# ============================
# ENSURE FEATURE ALIGNMENT
# ============================
train_attacks = train_attacks[FEATURE_NAMES + ["label_attack"]]
test_filtered = test_filtered[FEATURE_NAMES + ["label_attack"]]

# ============================
# MERGE DATA
# ============================
combined = pd.concat([train_attacks, test_filtered], axis=0)

print(f"Total samples: {len(combined)}")
print(f"Total attack classes: {combined['label_attack'].nunique()}")

# ============================
# FEATURE MATRIX
# ============================
X = combined[FEATURE_NAMES].values.astype(np.float32)
y = combined["label_attack"]

# ============================
# LABEL ENCODING (FULL CLASSES)
# ============================
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Final attack classes:", sorted(le.classes_))
print("Total classes:", len(le.classes_))

# ============================
# TRAIN MODEL
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
# SAFETY CHECK
# ============================
assert clf.n_features_in_ == len(FEATURE_NAMES)

# ============================
# SAVE MODEL
# ============================
joblib.dump(clf, MODEL_DIR / "attack_classifier_full_xgb.pkl")
joblib.dump(le, MODEL_DIR / "attack_label_encoder_full.pkl")

print("\n✅ Model trained with additional test attacks")
print("✅ Total classes learned:", len(le.classes_))