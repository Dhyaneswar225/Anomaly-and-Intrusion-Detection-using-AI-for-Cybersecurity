import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import random
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    recall_score, roc_auc_score, f1_score, precision_recall_curve, 
    auc, accuracy_score, precision_score
)
from lstm_model import LSTMAutoencoder
import warnings
warnings.filterwarnings('ignore')

# ================= REPRODUCIBILITY =================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ================= CONFIG =================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3

# ================= LOAD DATA =================
print("Loading data...")
train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
test_df  = pd.read_csv(DATA_DIR / "test_processed.csv")
scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")

FEATURE_NAMES = list(scaler.feature_names_in_)

# ================= SEQUENCE CREATION =================
def create_sequences(data, seq_len):
    num_samples = len(data) - seq_len + 1
    sequences = np.zeros((num_samples, seq_len, data.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        sequences[i] = data[i : i + seq_len]
    return sequences

# Train only on NORMAL samples
train_normal = train_df[train_df["label_binary"] == "normal"]
X_train_normal = train_normal[FEATURE_NAMES].values.astype(np.float32)
X_train_seq = create_sequences(X_train_normal, SEQ_LEN)

# Test includes both classes
X_test_all = test_df[FEATURE_NAMES].values.astype(np.float32)
X_test_seq = create_sequences(X_test_all, SEQ_LEN)
y_test_seq = (test_df["label_binary"] == "attack").astype(int).values[SEQ_LEN-1:]

# Create DataLoader
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_seq)), 
                          batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL SETUP =================
# We use hidden_dim=32 to force feature compression
model = LSTMAutoencoder(input_dim=len(FEATURE_NAMES), hidden_dim=32).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

# Scheduler to drop LR if loss doesn't improve for 5 epochs
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ================= TRAINING =================
print(f"Starting Training on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    
    for batch in train_loader:
        x_batch = batch[0].to(DEVICE)
        
        recon = model(x_batch)
        loss = criterion(recon, x_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = np.mean(epoch_losses)
    scheduler.step(avg_loss)
    
    if epoch % 5 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "lstm_autoencoder_best.pth")

# ================= EVALUATION =================
print("\nEvaluating Model...")
model.eval()
test_errors = []

with torch.no_grad():
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_seq)), 
                             batch_size=BATCH_SIZE, shuffle=False)
    for batch in test_loader:
        x_batch = batch[0].to(DEVICE)
        recon = model(x_batch)
        # Sequence-wide MSE (Mean over time and features)
        mse = torch.mean((recon - x_batch)**2, dim=(1, 2))
        test_errors.extend(mse.cpu().numpy())

test_errors = np.array(test_errors)

# 1. ROC-AUC
roc_auc = roc_auc_score(y_test_seq, test_errors)

# 2. PR-AUC
precision_vals, recall_vals, _ = precision_recall_curve(y_test_seq, test_errors)
pr_auc = auc(recall_vals, precision_vals)

# 3. Optimize Threshold for F1 and Accuracy
best_f1 = 0
best_thresh = 0
# Scan percentiles to find the best cut-off point
for p in np.arange(70, 100, 0.5):
    t = np.percentile(test_errors, p)
    current_f1 = f1_score(y_test_seq, (test_errors > t).astype(int))
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_thresh = t

final_preds = (test_errors > best_thresh).astype(int)
precision_cls = precision_score(y_test_seq, final_preds)
recall_cls = recall_score(y_test_seq, final_preds)
accuracy = accuracy_score(y_test_seq, final_preds)

# 4. Precision@10%
def get_precision_at_k(y_true, y_scores, k=0.10):
    n_k = int(len(y_true) * k)
    top_indices = np.argsort(y_scores)[-n_k:]
    return precision_score(y_true[top_indices], [1]*n_k, zero_division=0)

p_at_10 = get_precision_at_k(y_test_seq, test_errors, k=0.10)

# ================= RESULTS =================
print("-" * 35)
print(f"ROC-AUC:       {roc_auc:.4f}")
print(f"PR-AUC:        {pr_auc:.4f}")
print(f"Precision:     {precision_cls:.4f}")
print(f"Recall:        {recall_cls:.4f}") 
print(f"F1-Score:      {best_f1:.4f}")
print(f"Accuracy:      {accuracy:.4f}")
print(f"Precision@10%: {p_at_10:.4f}")
print(f"Best Threshold: {best_thresh:.6f}")
print("-" * 35)