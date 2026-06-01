"""
Transformers can preserve the context but can't preserve the order
+---+----------------------------------------------+--------------------------------------------------+
|   | RNN                                          | Transformer                                      |
+---+----------------------------------------------+--------------------------------------------------+
| 1 | Preserves context and order.                 | Preserves context  but cannot natively           |
|   |                                              | preserve order.                                  |
+---+----------------------------------------------+--------------------------------------------------+
| 2 | Processes one token per timestep, passing    | Uses self-attention: many tokens are handled     |
|   | hidden state forward → slow & costly on      | in parallel (with GPU support), so long          |
|   | long sequences.                              | contexts train faster and cheaper.               |
+---+----------------------------------------------+--------------------------------------------------+

# Mayukh kills Lion
# Lion kills Mayukh

In the above 2 sentences, attention for kills will be the same in arch,
but they mean entirely opposite just because of positioning.
"""

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer architectures.

    This class implements the sinusoidal positional encoding described in
    "Attention Is All You Need". Since transformers process all positions in parallel
    and have no inherent notion of sequence order, positional encodings are added
    to input embeddings to provide position information.

    The encoding uses sine and cosine functions of different frequencies:
    - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Attributes:
        d_model (int): Hidden dimension size (embedding dimension).
        seq_len (int): Maximum sequence length supported.
        dropout (nn.Dropout): Dropout layer applied to the final output.
    """

    def __init__(self, max_seq_len: int, hidden_size: int, dropout: float):
        """
        Initialize positional encoding with precomputed position embeddings.

        Precomputes positional encodings for all positions up to max_seq_len using
        sinusoidal functions. The encodings are stored in a buffer so they don't
        require gradients and are automatically moved to the correct device.

        Args:
            max_seq_len (int): Maximum sequence length for which to precompute
                positional encodings.
            hidden_size (int): Hidden dimension size (must match embedding dimension).
            dropout (float): Dropout probability applied to the output.
        """
        super().__init__()
        self.d_model = hidden_size
        self.seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # We can have at-most seq_len number of positions for the input
        # because the size of the input can't be more than seq_len
        pos_encodings = torch.zeros(self.seq_len, self.d_model, dtype=torch.float32)
        log_factor = math.log(10000.0)
        exponent_constant = -torch.div(log_factor, self.d_model)
        exponent = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * exponent_constant
        )
        # Now fill in the even indices with sin
        pos = torch.arange(0, self.seq_len, dtype=torch.float32).reshape(
            self.seq_len, 1
        )  # Make it column-vector
        pos_encodings[:, 0::2] = torch.sin(pos * exponent)
        # And odd indices with cos
        pos_encodings[:, 1::2] = torch.cos(pos * exponent)

        # Add a batch dimension with value 1
        pos_encodings = pos_encodings.reshape(1, self.seq_len, self.d_model)

        self.register_buffer("pe", pos_encodings)

    def forward(self, x) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Embeddings with positional encodings added, same shape as input.
                Output has dropout applied for regularization.

        Note:
            The method automatically handles variable sequence lengths by only using
            the necessary portion of the precomputed positional encodings.
        """
        actual_seq_len = x.size(1)
        # Take all the batch, only up to seq len and for all the columns (hidden_size)
        # We are doing this because the length of an actual input sequence can be less than max_seq_len
        x = x + self.pe[:, :actual_seq_len, :]  # type: ignore
        return self.dropout(x)  # [BATCH, SEQ_LEN, HIDDEN_SIZE]
