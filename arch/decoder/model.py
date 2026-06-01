import logging
from collections.abc import Callable

import torch
from torch import nn

from arch.decoder.layer import DecoderLayer
from arch.embedding import Embeddings
from arch.positional_encoding import PositionalEncoding
from config import Config

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    """
    Represents a Transformer Decoder module which processes target sequences and encoder outputs
    to generate predictions. Commonly used in sequence-to-sequence tasks like machine translation.

    This class is composed of token embeddings, positional encodings, and a stack of
    Transformer decoder layers. Each decoder layer includes masked self-attention, cross-attention
    with encoder outputs, and feed-forward networks.

    :ivar token_embedding: Embedding layer to convert target tokens into dense vector representations.
    :type token_embedding: Embeddings
    :ivar position_encoding: Positional encoding module to incorporate sequence position information.
    :type position_encoding: PositionalEncoding
    :ivar decoder_layers: List of Transformer decoder layers stacked sequentially.
    :type decoder_layers: nn.ModuleList
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
        d_k: int | None,
    ):
        super().__init__()
        self.embeddings = Embeddings(vocab_size=vocab_size, hidden_size=hidden_size)
        self.target_position_encoding = PositionalEncoding(
            seq_len, hidden_size, dropout_pe
        )
        self.decoder_layers: nn.ModuleList = nn.ModuleList(
            [
                DecoderLayer(
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

    def forward(
        self, x: torch.Tensor, pad_mask: torch.Tensor, encoder_output: torch.Tensor
    ) -> torch.Tensor:
        out_embd = self.embeddings(x)
        out = self.target_position_encoding(out_embd)
        for dec_layer in self.decoder_layers:
            out = dec_layer(out, pad_mask, encoder_output)
        return out

    def _init_layers(self, intializer: Callable, init_bias: bool):
        if hasattr(self.embeddings, "_init_layer"):
            self.embeddings._init_layer()

        for decoder_layer in self.decoder_layers:
            if hasattr(decoder_layer, "_init_layer"):
                decoder_layer._init_layer(intializer, init_bias)  # type: ignore


def get_decoder(conf: Config) -> Decoder:
    """
    Creates and returns a Decoder instance based on the provided configuration.
    The decoder is configured using parameters including hidden size, vocabulary size,
    dropout rate, number of attention heads, feed-forward hidden size, and number of layers.

    :param conf: Configuration object containing model parameters such as hidden size,
        dropout probability, number of heads, feed-forward hidden size, and number of layers.
    :type conf: Config
    :return: An instance of Decoder initialized with parameters from the configuration.
    :rtype: Decoder
    """
    hidden_size: int = conf.model.hidden_size
    max_seq_len: int = conf.model.max_seq_len
    vocab_size: int = conf.model.vocab_size
    dropout_pe: float = conf.model.dropout_pe
    n_heads: int = conf.model.n_heads
    d_k: int | None = getattr(conf.model, "d_k", None)
    ff_hidden_size: int = conf.model.ff_hidden_size
    n_layers: int = conf.model.n_layers
    return Decoder(
        vocab_size,
        hidden_size,
        max_seq_len,
        dropout_pe,
        n_layers,
        n_heads,
        ff_hidden_size,
        d_k,
    )
