import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Simple Attention Layer
        self.attention = nn.Linear(hidden_dim, 1)

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Encoder output: (batch, seq_len, hidden_dim)
        enc_out, (h, _) = self.encoder(x)
        
        # Calculate Attention Weights
        attn_weights = F.softmax(self.attention(enc_out).squeeze(2), dim=1) # (batch, seq_len)
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out) # (batch, 1, hidden_dim)
        
        # Repeat context for decoder
        repeated = context.repeat(1, seq_len, 1)
        
        dec_out, _ = self.decoder(repeated)
        return self.output_layer(dec_out)