import math
from collections.abc import Callable

import torch
from torch import nn


class CrossAttention(nn.Module):
    """
    Implements the Cross-Attention mechanism for sequence-to-sequence models.

    The CrossAttention class is a module designed to compute cross-attention,
    which allows a sequence in one context to attend to a sequence in another
    context effectively. This is typically used in tasks that require correlating
    information between two separate sequences, such as in transformer architectures.
    """

    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        d_k: int,
        dropout_pe: float,
        masking: bool = False,
    ):
        """
        Initializes the Cross-Attention instance with specified parameters.

        Creates linear transformation layers for query, key, and value computations,
        along with softmax and dropout layers for attention computation.

        Args:
            hidden_size (int): Hidden size of the input sequence representations.
            max_seq_len (int): Maximum sequence length (unused in cross-attention).
            d_k (int): Dimensionality of the query/key/value vectors.
            dropout_pe (float): Dropout probability applied to attention weights.
            masking (bool): Masking parameter (unused in cross-attention). Defaults to False.

        Note:
            Masking is not required in cross-attention, so the masking parameter is ignored.
        """
        super().__init__()
        self.d_model = hidden_size
        self.d_k = d_k

        # Create 3 matrices for Key, Value, Query
        self.W_q = nn.Linear(hidden_size, d_k)  # Query matrix
        self.W_k = nn.Linear(hidden_size, d_k)  # Key matrix
        self.W_v = nn.Linear(hidden_size, d_k)  # Value matrix

        # -1 represents the last index of [BATCH, SEQ_LEN, SEQ_LEN]
        self.softmax = nn.Softmax(dim=-1)  # Used in attention scores

        self.attn_dropout = nn.Dropout(p=dropout_pe)

        # Masking is not required in cross-attention
        _ = masking

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass of the cross-attention mechanism.

        Computes query vectors from decoder input and key/value vectors from encoder output,
        calculates attention scores, and returns the weighted value vectors.

        Args:
            x (torch.Tensor): Decoder input tensor of shape [BATCH, SEQ_LEN, HIDDEN_SIZE].
            encoder_output (torch.Tensor): Encoder output tensor for key and value computation.

        Returns:
            torch.Tensor: Output tensor of shape [BATCH, SEQ_LEN, d_k].

        Note:
            Query vectors are computed from decoder input, while key and value vectors
            are computed from encoder output.
        """
        # // NOTE: We pass encoder output to the value and key vector
        Q = self.W_q(x)
        K = self.W_k(encoder_output)
        V = self.W_v(encoder_output)

        # Perform Q.K
        # Q -> [BATCH, SEQ_LEN, d_k]
        # K -> [BATCH, SEQ_LEN, d_k]
        # So we need to transpose K to calculate dot product
        K_t = CrossAttention.transpose(K)
        scores = Q @ K_t

        # Now scale the attention scores
        scores = scores / math.sqrt(self.d_k)

        # Apply Softmax
        scores = self.softmax(scores)  # [BATCH, SEQ_LEN, SEQ_LEN]

        # Apply dropout
        W = self.attn_dropout(scores)

        # Linear Projection
        return W @ V

    @staticmethod
    def transpose(key: torch.Tensor) -> torch.Tensor:
        """
        Transposes the input tensor along specific dimensions.

        This function takes a tensor and swaps its second and third dimensions.
        It is useful for preparing the key tensor in attention mechanisms.

        Args:
            key (torch.Tensor): A tensor to be transposed of shape [BATCH, SEQ_LEN, d_k].

        Returns:
            torch.Tensor: A tensor with its second and third dimensions swapped.
        """
        # Take the key vector of dim [BATCH, SEQ_LEN, d_k]
        # Swap dim 1 and dim 2
        return key.transpose(1, 2)

    def _init_layer(self, initializer: Callable, init_bias: bool):
        initializer(self.W_q.weight)
        initializer(self.W_k.weight)
        initializer(self.W_v.weight)
        if init_bias:
            nn.init.zeros_(self.W_q.bias)
            nn.init.zeros_(self.W_k.bias)
            nn.init.zeros_(self.W_v.bias)
