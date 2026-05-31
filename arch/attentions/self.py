import torch
import torch.nn as nn
import math
from typing import Callable


class SelfAttention(nn.Module):
    """
    Implements a Self-Attention mechanism for use in neural network architectures.

    This class defines a self-attention mechanism in which each element in the input
    sequence attends to all other elements, creating a representation of the input
    that considers relationships between sequence elements. It employs separate
    learnable weight matrices for query, key, and value computations.
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
        Initializes the Self-Attention instance with specified parameters.

        This includes the creation of linear layers for key, value, and query matrices
        used in attention mechanisms, as well as a dropout layer and optional causal masking.

        Args:
            hidden_size (int): The size of the input hidden feature dimension.
            max_seq_len (int): Maximum sequence length for creating causal mask.
            d_k (int): Dimensionality of the key, value, and query representations.
            dropout_pe (float): The dropout probability to be applied to the attention
                mechanism outputs to help regularize the model.
            masking (bool): Whether to apply causal masking. Defaults to False.
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

        # Create the masking buffer
        self.masking = masking
        causal_mask: torch.Tensor = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=True)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Performs forward pass of the self-attention mechanism.

        Computes query, key, and value vectors from input, calculates attention scores,
        applies masking if specified, and returns the weighted value vectors.

        Args:
            x (torch.Tensor): Input tensor of shape [BATCH, SEQ_LEN, HIDDEN_SIZE].
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [BATCH, SEQ_LEN, d_k].
        """
        # [BATCH, SEQ_LEN, d_k]
        Q = self.W_q(x)  # Query Vector
        K = self.W_k(x)  # Key Vector
        V = self.W_v(x)  # Value Vector

        # Perform Q.K
        # Q -> [BATCH, SEQ_LEN, d_k]
        # K -> [BATCH, SEQ_LEN, d_k]
        # So we need to transpose K to calculate dot product
        K_t = SelfAttention.transpose(K)
        scores = (
            Q @ K_t
        )  # [BATCH, SEQ_LEN, SEQ_LEN], a perfect square matrix for each batch

        # Now scale the attention scores
        scores = scores / math.sqrt(self.d_k)

        neg_inf = torch.finfo(scores.dtype).min

        # Apply padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.to(scores.device)
            scores = scores.masked_fill(padding_mask.unsqueeze(1), neg_inf)

        # Apply masking
        if self.masking:
            _, SeqLen, _ = scores.shape
            mask: torch.Tensor = self.causal_mask[:SeqLen, :SeqLen]
            scores = scores.masked_fill(mask, neg_inf)

        # Apply Softmax
        scores = self.softmax(scores)  # [BATCH, SEQ_LEN, SEQ_LEN]

        # Apply dropout
        W = self.attn_dropout(scores)

        if padding_mask is not None:
            #  Zero PAD QUERIES (rows) AFTER softmax
            W = W.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Linear Projection
        return W @ V  # [BATCH, SEQ_LEN, d_k]

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
