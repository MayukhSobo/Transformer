from collections.abc import Callable

import torch
from torch import nn

from arch.attentions.multihead import MultiHeadAttention
from arch.attentions.self import SelfAttention
from arch.feed_forward import FeedForward
from arch.residual_add_norm import ResidualAddNorm


class EncoderLayer(nn.Module):
    """
    A single encoder layer implementing the transformer encoder architecture.
    This layer consists of a multi-head self-attention mechanism followed by a position-wise
    feed-forward network, with residual connections and layer normalization applied around
    each sub-layer. The layer follows the "Pre-LN" transformer variant where layer
    normalization is applied before the sub-layers.
    The encoder layer processes input sequences and outputs representations of the same
    dimensionality, making it suitable for stacking multiple layers to form a complete
    transformer encoder.
    Attributes:
        multi_head_attn (MultiHeadAttention): Multi-head self-attention mechanism that
            allows the model to attend to different representation subspaces.
        residual_add_norm_attn (ResidualAddNorm): Residual connection and layer normalization
            applied around the attention sub-layer.
        residual_add_norm_ff (ResidualAddNorm): Residual connection and layer normalization
            applied around the feed-forward sub-layer.
        feed_forward (FeedForward): Position-wise feed-forward network that processes
            each position independently.
    Example:
        >>> encoder_layer = EncoderLayer(
        ...     hidden_size=512,
        ...     max_seq_len=128,
        ...     dropout_pe=0.1,
        ...     n_heads=8,
        ...     ff_hidden_size=2048
        ... )
        >>> input_tensor = torch.randn(32, 128, 512)  # (batch_size, seq_len, hidden_size)
        >>> output = encoder_layer(input_tensor)
        >>> output.shape
        torch.Size([32, 128, 512])
    """

    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        dropout_pe: float,
        n_heads: int,
        ff_hidden_size: int,
        d_k: int | None = None,
    ):
        """
        Initializes the encoder layer with a multi-head attention mechanism, residual connections,
        and a feed-forward network. This layer does not include token embeddings or positional
        encoding as those are handled at the model level.

        :param hidden_size: Dimensionality of the input embeddings and model hidden states.
        :param max_seq_len: Maximum sequence length for an attention mechanism.
        :param dropout_pe: Dropout rate for multi-head attention and feed-forward layers.
        :param n_heads: Number of attention heads in the multi-head attention mechanism.
        :param ff_hidden_size: Dimensionality of the hidden layer in the feed-forward network.
        :param d_k: Dimensionality of the key (and query) vectors in attention. If None,
                    a default dimension is computed using hidden size and number of heads.
        """
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(
            SelfAttention, n_heads, hidden_size, max_seq_len, dropout_pe, False, d_k
        )
        self.residual_add_norm_attn = ResidualAddNorm(n_features=hidden_size)
        self.residual_add_norm_ff = ResidualAddNorm(n_features=hidden_size)
        self.feed_forward = FeedForward(
            hidden_size, ff_hidden_size=ff_hidden_size, dropout_pe=dropout_pe
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        attn_output = self.multi_head_attn(x=x, pad_mask=pad_mask)
        res_norm = self.residual_add_norm_attn(x, attn_output)
        ff_output = self.feed_forward(res_norm)
        return self.residual_add_norm_ff(res_norm, ff_output)

    def _init_layer(self, initializer: Callable, init_bias: bool):
        if hasattr(self.multi_head_attn, "_init_layer"):
            self.multi_head_attn._init_layer(initializer, init_bias)
        if hasattr(self.feed_forward, "_init_layer"):
            self.feed_forward._init_layer(initializer, init_bias)
