import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Tuple

import torch
from codetiming import Timer
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset
from torch.utils.data import IterableDataset

from tokenizer.sentencepiece import SentencePieceTokenizer
from tokenizer.word import WordTokenizer

logger = logging.getLogger(__name__)


@Timer(name="load_data", text="Loading data took {:.2f} seconds")
def get_dataset(
    data_path: Path, validation: object = True, streaming: bool = True
) -> tuple:
    """
    Load the WMT14 German-English dataset with optional streaming for memory efficiency.

    This function loads the WMT14 German-English translation dataset from HuggingFace
    datasets. It supports streaming mode to avoid loading the entire dataset into
    memory, which is crucial for preventing RAM overflow with large datasets.

    Args:
        data_path (Path): Path to the cache directory for dataset storage. If the
            dataset exists in cache, it will be loaded from there.
        validation (object): Whether to include the validation dataset in the
            returned tuple. Defaults to True.
        streaming (bool): Whether to use streaming mode to avoid loading the entire
            dataset into memory. Defaults to True for memory efficiency.

    Returns:
        tuple: If validation is True, returns (train_dataset, validation_dataset,
            test_dataset). If False, returns (train_dataset, None, test_dataset).

    Note:
        Streaming mode is recommended for large datasets to prevent memory issues.
        The function uses HuggingFace's datasets library with caching for efficiency.
    """
    dataset = load_dataset(
        "wmt/wmt14",
        "de-en",
        cache_dir=str(data_path),
        download_mode="reuse_dataset_if_exists",
        streaming=streaming,
    )
    if validation:
        return dataset["train"], dataset["validation"], dataset["test"]
    return dataset["train"], None, dataset["test"]


class TransformerDataset(IterableDataset):  # pylint: disable=abstract-method
    """
    An iterable dataset for tokenizing and processing language translation data efficiently.

    This dataset takes an existing dataset, tokenizes the input and target text using
    a specified tokenizer, and ensures the sequence length is restricted to a defined
    maximum. It supports lazy iteration over the dataset, converting text samples into
    tensors of tokenized data for direct use in translation models.

    Args:
        dataset (Dataset): Underlying dataset containing text samples with translations.
        tokenizer (Union[SentencePieceTokenizer, WordTokenizer]): Tokenizer used to
            process input and target texts.
        max_seq_len (int): Maximum allowed sequence length for tokenized data.

    Note:
        The dataset implements lazy iteration for memory efficiency and produces
        tokenized tensor pairs suitable for training translation models.
    """

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: SentencePieceTokenizer | WordTokenizer,
        max_seq_len: int,
    ):
        self.tokenizer: SentencePieceTokenizer | WordTokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset: Iterable[dict[str, Any]] = dataset  # type: ignore[assignment]

        # Get language keys
        first_sample = next(iter(dataset))
        assert isinstance(
            first_sample, dict
        ), f"Expected a dict sample from dataset, got {type(first_sample).__name__}"
        lang_keys = list(first_sample["translation"].keys())
        self.tokenizer.train(dataset, lang_keys)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for sample in self.dataset:
            src_text = sample["translation"]["en"]
            tgt_text = sample["translation"]["de"]

            src_tokens, tgt_tokens = self.tokenizer.encode(src_text, tgt_text)
            if src_tokens and tgt_tokens:
                yield torch.tensor(src_tokens), torch.tensor(tgt_tokens)


@dataclass
class Dataset:
    dataset: TransformerDataset
    tokenizer: SentencePieceTokenizer | WordTokenizer
