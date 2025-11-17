import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding as described in the Transformer architecture.

    Args:
        d_model (int): Dimension of the model/embedding.
        max_len (int): Maximum sequence length expected.

    Attributes:
        pe (torch.Tensor): Precomputed positional encoding matrix of shape (max_len, 1, d_model).
    """
    def __init__(self, d_model, max_len=800):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            torch.Tensor: Positionally encoded tensor of the same shape as input.
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerAutoencoder(nn.Module):
    """
    Transformer-based autoencoder for sequence reconstruction.

    Args:
        input_dim (int): Dimensionality of input features.
        seq_len (int): Length of the input sequence.
        n_heads (int): Number of attention heads in each layer.
        ff_dim (int): Hidden layer size in feedforward sub-networks.
        dropout (float): Dropout probability.

    Attributes:
        encoder (nn.TransformerEncoder): Transformer encoder stack.
        decoder (nn.TransformerDecoder): Transformer decoder stack.
        positional_encoding (PositionalEncoding): Sinusoidal positional encoding module.
    """
    def __init__(self, input_dim, seq_len, n_heads=4, ff_dim=128, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)

        self.positional_encoding = PositionalEncoding(input_dim, max_len=seq_len)

    def forward(self, x):
        """
        Forward pass of the Transformer autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            torch.Tensor: Reconstructed output of shape (batch_size, seq_len, input_dim)
        """
        # Apply positional encoding (requires [seq_len, batch_size, input_dim])
        src = self.positional_encoding(x.permute(1, 0, 2)).permute(1, 0, 2)
        memory = self.encoder(src)            # Encode the input
        out = self.decoder(src, memory)       # Decode using original input as tgt
        return out

