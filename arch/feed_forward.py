import torch
import torch.nn as nn
from typing import Callable


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used in transformer architectures.

    This module implements the feed-forward network described in "Attention Is All You Need".
    It consists of two linear transformations with a ReLU activation in between,
    followed by dropout for regularization. This is applied to each position separately
    and identically in transformer layers.

    Attributes:
        W_1 (nn.Linear): First linear transformation that expands the hidden dimension
            to the feed-forward dimension.
        W_2 (nn.Linear): Second linear transformation that projects back to the
            original hidden dimension.
        relu (nn.ReLU): ReLU activation function applied between the linear layers.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, hidden_size: int, ff_hidden_size: int, dropout_pe: float):
        """
        Initialize the feed-forward network with specified dimensions.

        Args:
            hidden_size (int): Dimension of the input and output features.
            ff_hidden_size (int): Dimension of the intermediate hidden layer
                (typically 4x the hidden_size in standard transformers).
            dropout_pe (float): Dropout probability applied after the ReLU activation.
        """
        super().__init__()
        self.W_1 = nn.Linear(hidden_size, ff_hidden_size)
        self.W_2 = nn.Linear(ff_hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        x = self.W_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.W_2(x)

    def _init_layer(self, initializer: Callable, init_bias: bool):
        initializer(self.W_1.weight)
        initializer(self.W_2.weight)
        if init_bias:
            nn.init.zeros_(self.W_1.bias)
            nn.init.zeros_(self.W_2.bias)
