import math

import torch
from torch import nn


class Embeddings(nn.Module):
    """
    Token embedding layer with scaling for transformer models.

    This class implements the embedding layer used in transformer architectures,
    converting token indices into dense vector representations. It includes the
    scaling factor as described in the "Attention Is All You Need" paper (Section 3.4).

    Attributes:
        embedding (nn.Embedding): PyTorch embedding layer that maps token indices
            to dense vectors.
        factor (float): Scaling factor equal to sqrt(hidden_size) applied to
            embeddings as per the transformer paper.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        """
        Initialize the embedding layer with specified vocabulary and hidden dimensions.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            hidden_size (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.d_model = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.factor = math.sqrt(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input token indices into scaled embeddings.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape
                [batch_size, seq_len].

        Returns:
            torch.Tensor: Scaled embedding vectors with shape
                [batch_size, seq_len, hidden_size].

        Note:
            The embeddings are scaled by sqrt(hidden_size) as specified in
            the transformer paper (https://arxiv.org/pdf/1706.03762, Section 3.4).
        """
        # The paper https://arxiv.org/pdf/1706.03762 at section 3.4
        # Scales the embedding layer output with the `factor`
        return self.embedding(x) * self.factor  # [BATCH, SEQ_LEN, HIDDEN_SIZE]

    def _init_layer(self):
        nn.init.normal_(
            self.embedding.weight, mean=0.0, std=self.d_model**-0.5
        )  # 1/sqrt(d_model)
