import torch
import torch.nn as nn
from typing import Callable, Union
from arch.attentions.multihead import MultiHeadAttention
from arch.attentions.cross import CrossAttention
from arch.residual_add_norm import ResidualAddNorm
from arch.feed_forward import FeedForward
from arch.attentions.self import SelfAttention


class DecoderLayer(nn.Module):
    """
    Represents a single decoder layer for transformer-based models.

    This layer is designed to perform masked self-attention, cross-attention,
    and feed-forward transformations with residual connections and layer
    normalization applied to each stage. These components work together
    to facilitate effective learning and contextual understanding in the
    decoder part of the transformer architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        dropout_pe: float,
        n_heads: int,
        ff_hidden_size: int,
        d_k: Union[int, None] = None,
    ):
        """
        Initializes a decoder layer with masked self-attention, cross-attention, and feed-forward components.

        The decoder layer uses the provided hyperparameters to define each sublayer:
        masked multi-head attention for self-attention mechanism, cross-multi-head attention
        for integrating encoder information, and a feed-forward layer for additional processing.
        Each component is followed by residual connections and layer normalization.

        Args:
            hidden_size (int): The size of the input and output features of each
                sublayer within the decoder.
            max_seq_len (int): The maximum sequence length the layer is designed to handle.
            dropout_pe (float): Dropout probability to be applied to internal operations
                such as attention computations and feed-forward layers.
            n_heads (int): The number of attention heads in the multi-head attention
                mechanism.
            ff_hidden_size (int): The hidden layer size for the feed-forward network.
            d_k (int, optional): The dimensionality of each key/query vector in
                the attention mechanism. If not provided, it will default to a value
                based on the implementation. Defaults to None.
        """
        super().__init__()

        # Mask = True for decoder masked multihead attention layer
        self.masked_multi_head_attn = MultiHeadAttention(
            SelfAttention, n_heads, hidden_size, max_seq_len, dropout_pe, True, d_k
        )
        self.residual_add_norm_ma = ResidualAddNorm(n_features=hidden_size)

        # Mask = False for decoder cross-multihead attention layer
        self.cross_multi_head_attn = MultiHeadAttention(
            CrossAttention, n_heads, hidden_size, max_seq_len, dropout_pe, False, d_k
        )
        self.residual_add_norm_ca = ResidualAddNorm(n_features=hidden_size)
        self.feed_forward = FeedForward(
            hidden_size, ff_hidden_size=ff_hidden_size, dropout_pe=dropout_pe
        )

        self.residual_add_norm_ff = ResidualAddNorm(n_features=hidden_size)

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs forward pass through the decoder layer.

        Applies masked self-attention, cross-attention with encoder output, and feed-forward
        transformation, with residual connections and layer normalization at each step.

        Args:
            x (torch.Tensor): Input tensor from previous decoder layer or embedding layer.
            pad_mask (torch.Tensor): Padding mask for masking padding tokens in self-attention.
            encoder_output (torch.Tensor): Output from the encoder for cross-attention.

        Returns:
            torch.Tensor: Processed tensor ready for the next decoder layer or final output.
        """
        masked_attn_output = self.masked_multi_head_attn(x=x, pad_mask=pad_mask)
        res_norm_ma = self.residual_add_norm_ma(x, masked_attn_output)
        cross_attn_output = self.cross_multi_head_attn(
            x=res_norm_ma, pad_mask=None, encoder_output=encoder_output
        )
        res_norm_ca = self.residual_add_norm_ca(res_norm_ma, cross_attn_output)
        ff_output = self.feed_forward(res_norm_ca)
        return self.residual_add_norm_ff(res_norm_ca, ff_output)

    def _init_layer(self, initializer: Callable, init_bias: bool):
        if hasattr(self.masked_multi_head_attn, "_init_layer"):
            self.masked_multi_head_attn._init_layer(initializer, init_bias)

        if hasattr(self.cross_multi_head_attn, "_init_layer"):
            self.cross_multi_head_attn._init_layer(initializer, init_bias)

        if hasattr(self.feed_forward, "_init_layer"):
            self.feed_forward._init_layer(initializer, init_bias)
