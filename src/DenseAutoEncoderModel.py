import torch
import torch.nn as nn


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck=32, dropout=0.2):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, bottleneck),   # bottleneck: 16 → 32 (fix #3)
            nn.BatchNorm1d(bottleneck),
            nn.ReLU()
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, input_dim)   # no activation at output
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        """Return bottleneck representation."""
        return self.encoder(x)