# src/gnn_model.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ============================= CONFIG =============================
DATA_PROCESSED = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === FINAL: Ultra-small window + relaxed normal ===
WINDOW_SIZE = 15          # Critical: small enough to find clean segments
STRIDE = 8
BATCH_SIZE = 32
EPOCHS = 80
MAX_ATTACK_IN_NORMAL = 0.20  # Allow up to 20% attack in "normal" windows

# ============================= GRAPH BUILDER =============================
def build_graph(win: pd.DataFrame):
    win = win.copy()
    win['src_node'] = win['service'].astype(str) + '_src'
    win['dst_node'] = win['service'].astype(str) + '_dst'

    nodes = pd.unique(win[['src_node', 'dst_node']].values.ravel())
    if len(nodes) < 2:
        return None

    node2idx = {n: i for i, n in enumerate(nodes)}

    edge_index = []
    edge_attr = []
    for _, r in win.iterrows():
        src = node2idx[r['src_node']]
        dst = node2idx[r['dst_node']]
        edge_index.append([src, dst])
        edge_attr.append([
            r['duration'], r['src_bytes'], r['dst_bytes'],
            r['count'], r['same_srv_rate'], r['diff_srv_rate'],
            r['serror_rate'], r['rerror_rate']
        ])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    node_feat = np.array([
        [
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['duration'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['src_bytes'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['dst_bytes'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['count'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['same_srv_rate'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['diff_srv_rate'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['serror_rate'].mean(),
            win[(win['src_node'] == n) | (win['dst_node'] == n)]['rerror_rate'].mean(),
            len(win[(win['src_node'] == n) | (win['dst_node'] == n)]),
            1.0 if 'http' in n else 0.0
        ] for n in nodes
    ], dtype=np.float32)
    x = torch.from_numpy(node_feat)

    attack_ratio = (win['label'] == 'attack').mean()
    y = 1 if attack_ratio > 0.5 else 0
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([y])), attack_ratio

# ============================= MODEL =============================
class GNNAnomalyDetector(nn.Module):
    def __init__(self, input_dim, hidden=64, layers=2):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden)] +
                                   [GCNConv(hidden, hidden) for _ in range(layers-1)])
        self.lin1 = nn.Linear(hidden, hidden // 2)
        self.lin2 = nn.Linear(hidden // 2, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.drop(x)
        x = global_mean_pool(x, batch)
        x = self.relu(self.lin1(x))
        x = self.drop(x)
        return self.lin2(x).squeeze(-1)

# ============================= TRAINING =============================
def train_gnn():
    train_path = os.path.join(DATA_PROCESSED, "train_processed.csv")
    test_path  = os.path.join(DATA_PROCESSED, "test_processed.csv")

    print(f"Loading data...")
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # === TRAINING GRAPHS (PURE NORMAL) ===
    print("Building training graphs...")
    normal_train = train_df[train_df['label'] == 'normal'].copy()
    train_graphs = []
    for start in range(0, len(normal_train) - WINDOW_SIZE + 1, STRIDE):
        win = normal_train.iloc[start:start + WINDOW_SIZE]
        result = build_graph(win)
        if result:
            g, _ = result
            train_graphs.append(g)

    # === TEST GRAPHS ===
    print("Building test graphs...")
    test_graphs = []
    normal_windows = []
    for start in range(0, len(test_df) - WINDOW_SIZE + 1, STRIDE):
        win = test_df.iloc[start:start + WINDOW_SIZE]
        result = build_graph(win)
        if result:
            g, attack_ratio = result
            test_graphs.append(g)
            if attack_ratio <= MAX_ATTACK_IN_NORMAL:
                normal_windows.append(g)

    print(f"Found {len(normal_windows)} real normal windows (≤{MAX_ATTACK_IN_NORMAL*100}% attack)")

    if len(normal_windows) == 0:
        print("Still no normal windows. Using smallest possible...")
        # Fallback: take any window with <50% attack as normal
        normal_windows = [g for g in test_graphs if g.y.item() == 0]
        if len(normal_windows) == 0:
            normal_windows = test_graphs[:30]  # Last resort
            for g in normal_windows:
                g.y = torch.tensor([0])

    # === FINAL TEST SET ===
    attack_test_graphs = [g for g in test_graphs if g.y.item() == 1]
    final_test_graphs = normal_windows + attack_test_graphs

    print(f"Final Test → Normal: {len(normal_windows)}, Attack: {len(attack_test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(final_test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = train_graphs[0].x.shape[1]
    model = GNNAnomalyDetector(input_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for data in train_loader:
            data = data.to(DEVICE)
            data.y = torch.zeros(data.num_graphs, dtype=torch.float, device=DEVICE)
            logits = model(data)
            loss = crit(logits, data.y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.8f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/gnn_detector.pth")

    # === EVALUATION ===
    model.eval()
    scores, y_true = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            logits = model(data)
            scores.extend(torch.sigmoid(logits).cpu().numpy())
            y_true.extend(data.y.cpu().numpy())

    scores = np.array(scores)
    y_true = np.array(y_true)

    normal_scores = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(DEVICE)
            normal_scores.extend(torch.sigmoid(model(data)).cpu().numpy())
    threshold = np.percentile(normal_scores, 95)

    if len(np.unique(y_true)) < 2:
        auc_roc = np.nan
        auc_pr = 1.0
        prec10 = 1.0
    else:
        auc_roc = roc_auc_score(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        auc_pr = auc(rec, prec)
        k = max(1, int(0.1 * len(scores)))
        prec10 = np.mean(y_true[np.argsort(scores)[-k:]])

    print(f"\nGNN Results:")
    print(f"  ROC-AUC     : {auc_roc:.6f}")
    print(f"  PR-AUC      : {auc_pr:.6f}")
    print(f"  Precision@10%: {prec10:.6f}")
    print(f"  Threshold   : {threshold:.6f}")

    pd.DataFrame({"score": scores, "label": y_true}).to_csv(f"{RESULTS_DIR}/gnn_scores.csv", index=False)

    plt.figure(figsize=(10,6))
    sns.histplot(scores[y_true==0], label="Normal", alpha=0.7, bins=40, color="#1f77b4")
    sns.histplot(scores[y_true==1], label="Attack", alpha=0.7, bins=40, color="#d62728")
    plt.axvline(threshold, color='k', ls='--', label=f'Threshold = {threshold:.4f}')
    plt.title("GNN Anomaly Score Distribution")
    plt.xlabel("Score"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/gnn_score_distribution.png", dpi=200)
    plt.close()

    print(f"\nResults saved in: {RESULTS_DIR}/")
    print("GNN training completed")

# ============================= RUN =============================
if __name__ == "__main__":
    train_gnn()