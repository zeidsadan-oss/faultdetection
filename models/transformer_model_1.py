import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  
        self.register_buffer('pe', pe)

    def forward(self, x):
     
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=4, ff_dim=256,
                 dropout=0.1, max_len=500):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, ff_dim)

        self.pos_encoder = PositionalEncoding(ff_dim, dropout=dropout, max_len=max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(ff_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ff_dim, num_classes)

    def forward(self, x):
     
        x = self.input_proj(x)  
        x = x.permute(1, 0, 2) 
        x = self.pos_encoder(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(1, 2, 0) 
        x = self.pool(x).squeeze(-1) 
        x = self.fc(x)  
        return x
