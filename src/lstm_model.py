# src/lstm_model.py

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_last = h[-1]
        repeated = h_last.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(repeated)
        out = self.output_layer(dec_out)
        return out