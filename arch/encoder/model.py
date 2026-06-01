from collections.abc import Callable, Iterator

import torch
from torch import nn

from arch.embedding import Embeddings
from arch.encoder.layer import EncoderLayer
from arch.positional_encoding import PositionalEncoding
from config import Config


class Encoder(nn.Module):
    """
    Transformer Encoder module for processing input sequences into contextualized embeddings.

    This class implements a complete Transformer encoder consisting of token embeddings,
    positional encodings, and a stack of encoder layers. The encoder processes sequences
    efficiently using self-attention mechanisms, making it suitable for various NLP tasks.

    Attributes:
        token_embedding (Embeddings): Embedding layer to convert input tokens into dense
            vector representations.
        position_encoding (PositionalEncoding): Positional encoding module to incorporate
            sequence position information into the embeddings.
        encoder_layers (nn.ModuleList): Stack of Transformer encoder layers for sequential
            processing of the input sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        seq_len: int,
        dropout_pe: float,
        n_layers: int,
        n_heads: int,
        ff_size: int,
        d_k: int,
    ):
        """
        Initialize the Transformer encoder with specified architecture parameters.

        Args:
            vocab_size (int): Size of the vocabulary for the token embedding layer.
            hidden_size (int): Dimensionality of token embeddings, positional encodings,
                and hidden states throughout the model.
            seq_len (int): Maximum sequence length that the encoder can process.
            dropout_pe (float): Dropout probability applied to positional encodings.
            n_layers (int): Number of transformer encoder layers in the stack.
            n_heads (int): Number of attention heads in each encoder layer.
            ff_size (int): Hidden layer dimensionality of the feed-forward network
                in each encoder layer.
            d_k (int): Dimensionality of each attention head's key and query vectors.
        """
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size)
        self.positional_encoder = PositionalEncoding(seq_len, hidden_size, dropout_pe)
        self.encoder_layers: Iterator[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    max_seq_len=seq_len,
                    dropout_pe=dropout_pe,
                    n_heads=n_heads,
                    ff_hidden_size=ff_size,
                    d_k=d_k,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder stack.

        Args:
            x (torch.Tensor): Input token IDs tensor of shape [batch_size, seq_len].
            pad_mask (torch.Tensor): Padding mask tensor of shape [batch_size, seq_len]
                where True indicates padding tokens to be ignored.

        Returns:
            torch.Tensor: Encoded representations of shape [batch_size, seq_len, hidden_size].
        """
        inp_embd = self.embeddings(x)
        out = self.positional_encoder(inp_embd)
        for enc_layer in self.encoder_layers:
            out = enc_layer(out, pad_mask)
        return out

    def _init_layers(self, initializer: Callable, init_bias: bool):
        if hasattr(self.embeddings, "_init_layer"):
            self.embeddings._init_layer()

        for encoder_layer in self.encoder_layers:
            if hasattr(encoder_layer, "_init_layer"):
                encoder_layer._init_layer(initializer, init_bias)


def get_encoder(conf: Config) -> Encoder:
    """
    Create and configure a Transformer encoder based on the provided configuration.

    This function initializes an Encoder instance with parameters derived from the
    configuration object, including model architecture settings like hidden dimensions,
    attention heads, and layer counts.

    Args:
        conf (Config): Configuration object containing model parameters including:
            - model.hidden_size: Hidden dimension size
            - model.max_seq_len: Maximum sequence length
            - model.vocab_size: Vocabulary size
            - model.dropout_pe: Positional encoding dropout rate
            - model.n_heads: Number of attention heads
            - model.ff_hidden_size: Feed-forward hidden size
            - model.n_layers: Number of encoder layers
            - model.d_k: Key dimension (optional)

    Returns:
        Encoder: Initialized Transformer encoder ready for training or inference.
    """
    hidden_size: int = conf.model.hidden_size
    max_seq_len: int = conf.model.max_seq_len
    vocab_size: int = conf.model.vocab_size
    dropout_pe: float = conf.model.dropout_pe
    n_heads: int = conf.model.n_heads
    d_k: int = conf.model.get("d_k", None)
    ff_hidden_size: int = conf.model.ff_hidden_size
    n_layers: int = conf.model.n_layers

    return Encoder(
        vocab_size,
        hidden_size,
        max_seq_len,
        dropout_pe,
        n_layers,
        n_heads,
        ff_hidden_size,
        d_k,
    )
