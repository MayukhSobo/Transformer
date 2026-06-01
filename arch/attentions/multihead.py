from collections.abc import Callable

import torch
from torch import nn

from arch.attentions.cross import CrossAttention
from arch.attentions.self import SelfAttention


class MultiHeadAttention(nn.Module):
    """
    Implements a multi-head attention mechanism for neural networks.

    This allows the model to attend to different parts of the input sequence
    independently using separate attention heads. The outputs of all attention
    heads are concatenated and projected through a linear transformation.
    """

    def __init__(
        self,
        attention_type: type[SelfAttention | CrossAttention],
        n_heads: int,
        hidden_size: int,
        max_seq_len: int,
        dropout_pe: float,
        masking: bool,
        d_k: int | None = None,
    ):
        """
        Initializes the multi-head attention mechanism with specified parameters.

        Validates input parameters for compatibility, including ensuring the key dimension
        is not greater than the hidden size and that the number of heads divides the
        hidden size evenly.

        Args:
            attention_type (Type[Union[SelfAttention, CrossAttention]]): Type of attention
                mechanism to use (SelfAttention or CrossAttention).
            n_heads (int): The number of attention heads.
            hidden_size (int): The size of the input representation or embeddings.
            max_seq_len (int): Maximum sequence length for attention mechanism.
            dropout_pe (float): Dropout probability for attention outputs.
            masking (bool): Whether to apply causal masking (True for decoder, False for encoder).
            d_k (int, optional): The size of the query/key vectors. If not provided, it will be
                automatically calculated as `hidden_size // n_heads`. Defaults to None.

        Raises:
            ValueError: If the provided key dimension exceeds the hidden size.
            ValueError: If the number of attention heads does not evenly divide the hidden size.
        """
        super().__init__()
        if d_k and d_k > hidden_size:
            raise ValueError("Key dimension must be less than or equal to hidden size")
        if hidden_size % n_heads != 0:
            raise ValueError("Number of heads must divide hidden size evenly")
        if not d_k:
            d_k = hidden_size // n_heads

        self.d_k = d_k
        self.IsSelfAttention = attention_type is SelfAttention
        # Create `n_heads` number of self-attention heads
        self.attention_heads: nn.ModuleList = nn.ModuleList(
            [
                attention_type(
                    hidden_size=hidden_size,
                    max_seq_len=max_seq_len,
                    d_k=self.d_k,
                    dropout_pe=dropout_pe,
                    masking=masking,
                )
                for _ in range(n_heads)
            ]
        )

        # Create the W_o for Linear projection
        self.W_o = nn.Linear(d_k * n_heads, hidden_size)
        self.out_dropout = nn.Dropout(p=dropout_pe)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        encoder_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Performs forward pass of the multi-head attention mechanism.

        Processes input through multiple attention heads, concatenates their outputs,
        and applies a linear projection to produce the final output.

        Args:
            x (torch.Tensor): Input tensor of shape [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE].
            pad_mask (torch.Tensor): Padding mask for masking padding tokens.
            encoder_output (torch.Tensor, optional): Encoder output for cross-attention.
                Required for CrossAttention, ignored for SelfAttention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE].
        """
        head_outputs = []

        if self.IsSelfAttention:
            # Pass the inputs with all the attn heads
            head_outputs.extend(head(x, pad_mask) for head in self.attention_heads)
        else:
            # For Cross Attention, we give both input and encoder output
            head_outputs.extend(
                head(x, encoder_output) for head in self.attention_heads
            )

        # Concat all the head's output horizontally
        # dim = -1 indicate the last dimension wise which is hidden_size
        concat_outputs: torch.Tensor = torch.cat(
            head_outputs, dim=-1
        )  # [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]

        # Project the output with Linear layer
        # This ensures
        # # 1. Correct output dim [BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
        # # 2. Learns the combination of head outputs
        # # 3. Removes any redundant head's output
        projected_output: torch.Tensor = self.W_o(concat_outputs)

        # Apply the dropout
        return self.out_dropout(projected_output)

    def _init_layer(self, initializer: Callable, init_bias: bool):
        for i, head in enumerate(self.attention_heads):
            if hasattr(head, "_init_layer"):
                head._init_layer(initializer, init_bias)  # type: ignore
            else:
                raise NotImplementedError(f"Head {i} does not have _init_layer method")

        initializer(self.W_o.weight)
        if init_bias:
            nn.init.zeros_(self.W_o.bias)
