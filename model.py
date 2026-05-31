from pathlib import Path
from typing import Union
import torch
from config import Config
from dataset import TransformerDataset, get_dataset
import logging
from tokenizer.tokenizer import get_tokenizer
from arch.encoder.model import get_encoder, Encoder
from arch.decoder.model import get_decoder, Decoder
from arch.classifier import get_classifier, Classifier
from dataclasses import dataclass
from shutil import rmtree
import humanize
from utils import get_device
from dataset import Dataset
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class TransformerModel:
    encoder: Encoder
    decoder: Decoder
    classifier: Classifier
    device: torch.device

    def forward(
        self,
        src_batch: torch.Tensor,
        tgt_batch: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the complete transformer model.

        :param src_batch: Source sequences [batch_size, seq_len]
        :param tgt_batch: Target sequences [batch_size, seq_len]
        :return: Output logits [batch_size, seq_len, vocab_size]
        """
        # Encoder forward pass
        encoder_output = self.encoder(src_batch, src_pad_mask)

        # Decoder forward pass with cross-attention to encoder output
        decoder_output = self.decoder(tgt_batch, tgt_pad_mask, encoder_output)

        # Classifier forward pass to get vocabulary logits
        logits = self.classifier(decoder_output)

        return logits

    def to(self, device: torch.device):
        """
        Move all model components to the specified device.

        Args:
            device (torch.device): Target device for model components.

        Returns:
            TransformerModel: Self reference for method chaining.
        """
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.classifier = self.classifier.to(device)
        self.device = device
        return self

    def get_layer(self, layer_chain: str) -> Union[torch.nn.Module, None]:
        parent = self
        for layer in layer_chain.split("."):
            model = getattr(parent, layer, None)
            if model is None:
                break
            parent = model
        if model is None:
            logger.error(f"Layer chain {layer_chain} not found in model")
            return None
        else:
            return model
            # logger.info(f"Model for layer {layer}: {model}")

    @staticmethod
    def _eval_initializer(init_method: str):
        match init_method:
            case "xavier_uniform":
                return torch.nn.init.xavier_uniform_
            case "xavier_normal":
                return torch.nn.init.xavier_normal_
            case "kaiming_uniform":
                return torch.nn.init.kaiming_uniform_
            case "kaiming_normal":
                return torch.nn.init.kaiming_normal_
        logger.warning(
            f"Unknown init method {init_method}, defaulting to xavier_uniform"
        )
        return torch.nn.init.xavier_uniform_

    def initialize_weights_(self, init_method: str, init_bias: bool):
        initializer = TransformerModel._eval_initializer(init_method)

        # Initialize weights for encoder
        self.encoder._init_layers(initializer, init_bias)

        # Initialize weights for decoder
        self.decoder._init_layers(initializer, init_bias)

        logger.debug(f"📝 Initialized embedding layers")


def init_dataset(tokenizer, config: Config):
    """
    Initialize the training dataset with streaming support for memory efficiency.

    This function loads the WMT14 German-English dataset in streaming mode to avoid
    loading the entire dataset into memory, which can cause RAM overflow issues.

    Args:
        tokenizer: Tokenizer instance (SentencePiece or Word tokenizer) for text
            processing.
        config (Config): Configuration object containing dataset path and model
            parameters.

    Returns:
        TransformerDataset: Initialized dataset ready for training with the
            specified tokenizer and maximum sequence length.

    Note:
        Currently only loads the training split. Validation and test datasets
        are not yet implemented (marked as TODO).
    """
    dataset_path = Path(config.dataset.path)

    # // TODO: Cater for test and validation datasets
    # Use streaming=True to avoid loading entire dataset into memory
    train_split, _, _ = get_dataset(dataset_path, validation=False, streaming=True)
    train_ds = TransformerDataset(train_split, tokenizer, config.model.max_seq_len)
    return train_ds


def init_tokenizer(config: Config):
    """
    Initialize a tokenizer with configuration-specific model paths.

    This function creates a tokenizer with a unique directory structure based on
    the tokenizer configuration parameters (sample_size, algorithm, vocab_size).
    This ensures that different tokenizer configurations don't conflict with
    each other and allows for easy parameter experimentation.

    Args:
        config (Config): Configuration object containing tokenizer parameters:
            - tokenizer.kind: Type of tokenizer ('sentencepiece' or 'custom')
            - tokenizer.model: Base path for tokenizer model storage
            - tokenizer.sample_size: Number of samples for training
            - tokenizer.algorithm: Algorithm type ('bpe' or 'unigram')
            - tokenizer.recreate: Whether to recreate existing models
            - model.vocab_size: Target vocabulary size

    Returns:
        Tokenizer: Initialized tokenizer instance (SentencePiece or Word tokenizer)
            ready for training and encoding.

    Note:
        The tokenizer model path follows the pattern:
        base_path/sample_size-algorithm-vocab_size/tokenizer.model
        This allows multiple configurations to coexist without conflicts.
    """
    kind = config.tokenizer.kind
    vocab_size = config.model.vocab_size
    base_path = Path(config.tokenizer.model)

    # Create a configuration-specific path to avoid conflicts when parameters change
    config_dir = (
        f"{config.tokenizer.sample_size}-{config.tokenizer.algorithm}-{vocab_size}"
    )
    tk_model_path = base_path / config_dir / "tokenizer"

    logger.info(f"Using tokenizer path: {tk_model_path}")

    if config.tokenizer.recreate:
        # Delete the existing model and vocab files for this specific configuration
        rmtree(tk_model_path.parent, ignore_errors=True)
        logger.info(f"Removed existing tokenizer at {tk_model_path.parent}")

    return get_tokenizer(
        tokenizer_kind=kind,
        model_path=tk_model_path,
        vocab_size=vocab_size,
        algorithm=config.tokenizer.algorithm,
        sample_size=config.tokenizer.sample_size,
    )


def build_transformer(config: Config) -> tuple[TransformerModel, Dataset]:
    """
    Builds and initializes the main transformer model and its associated dataset.
    This function performs the following steps:
    - Determines the best available device (CUDA/CPU)
    - Initializes the tokenizer based on the provided configuration.
    - Prepares the dataset, ensuring the tokenizer and dataset align in terms of
      vocabulary size.
    - Initializes encoder, decoder, and classifier modules to construct the model.
    - Moves all model components to the selected device.

    :param config: Configuration object containing parameters for initializing
        the tokenizer, dataset, and model components.
    :type config: Config
    :return: A tuple containing the initialized `Model` and `Dataset` objects.
    :rtype: tuple[Model, Dataset]
    """

    # Log initial memory usage
    memory_before = psutil.virtual_memory()
    memory_used = humanize.naturalsize(memory_before.used)
    total_memory = humanize.naturalsize(memory_before.total)
    logger.info(
        f"Memory before model creation: {memory_before.percent:.1f}% "
        f"used ({memory_used}/{total_memory})"
    )

    # Determine the device to use
    device = get_device(config)

    # Get the tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = init_tokenizer(config=config)

    # Log memory after tokenizer
    memory_after_tokenizer = psutil.virtual_memory()
    logger.info(
        f"Memory after tokenizer: {memory_after_tokenizer.percent:.1f}% "
        f"used ({humanize.naturalsize(memory_after_tokenizer.used)}/"
        f"{humanize.naturalsize(memory_after_tokenizer.total)})"
    )

    # Get the dataset
    logger.info("Initializing dataset...")
    dataset = init_dataset(config=config, tokenizer=tokenizer)
    # This is needed because the tokenizer can have vocab size != desired vocab size
    config.model.vocab_size = dataset.tokenizer.n_vocab

    # Log memory after dataset
    memory_after_dataset = psutil.virtual_memory()
    logger.info(
        f"Memory after dataset: {memory_after_dataset.percent:.1f}% "
        f"used ({humanize.naturalsize(memory_after_dataset.used)}/"
        f"{humanize.naturalsize(memory_after_dataset.total)})"
    )

    # Get the encoder
    logger.info("Initializing encoder...")
    encoder_model = get_encoder(conf=config).to(device)
    logger.debug("Encoder initialized and moved to %s: %s", device, encoder_model)

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the decoder
    logger.info("Initializing decoder...")
    decoder_model = get_decoder(conf=config).to(device)
    logger.debug("Decoder initialized and moved to %s: %s", device, decoder_model)

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the Classifier Head
    logger.info("Initializing classifier...")
    classifier_model = get_classifier(conf=config).to(device)
    logger.debug("Classifier initialized and moved to %s", device)

    # Log total parameters
    total_params = (
        sum(p.numel() for p in encoder_model.parameters())
        + sum(p.numel() for p in decoder_model.parameters())
        + sum(p.numel() for p in classifier_model.parameters())
    )
    logger.info(f"Total model parameters: {humanize.intword(total_params)}")

    # Calculate approximate model memory usage
    param_memory = total_params * 4  # 4 bytes per float32 parameter
    logger.info(f"Approximate model memory usage: {humanize.naturalsize(param_memory)}")

    # Log final memory usage
    memory_final = psutil.virtual_memory()
    logger.info(
        f"Memory after model creation: {memory_final.percent:.1f}% "
        f"used ({humanize.naturalsize(memory_final.used)}/"
        f"{humanize.naturalsize(memory_final.total)})"
    )

    # Force final garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return TransformerModel(
        encoder=encoder_model,
        decoder=decoder_model,
        classifier=classifier_model,
        device=device,
    ), Dataset(dataset=dataset, tokenizer=tokenizer)
