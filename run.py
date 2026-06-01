"""
This module can be used for running various operations related to the transformer model.
For testing the model, please use the test suite in the tests/ directory.
"""

import logging
from pathlib import Path

from config import Config
from loss import get_loss_function
from model import build_transformer
from train import train_epoch_avg_CE
from utils import get_dataloader
from wnb import ExperimentTracker

logger = logging.getLogger(__name__)


def run_train(config_file: Path):
    """
    Execute the complete transformer training pipeline.

    This function orchestrates the entire training process including model
    initialization, dataset preparation, tokenizer setup, and training execution.
    It supports optional experiment tracking via Weights & Biases based on
    configuration settings.

    Args:
        config_file (Path): Path to the TOML configuration file containing all
            training parameters including model architecture, dataset settings,
            tokenizer configuration, and experiment tracking options.

    Note:
        The function automatically handles:
        - Model and dataset initialization with memory optimization
        - Device selection (CUDA/CPU) based on configuration
        - Experiment tracking activation based on config.experiment.active
        - Loss function setup with appropriate padding token handling
    """
    config = Config(config_file=config_file)

    transformer, dataset = build_transformer(config)

    train_data_loader = get_dataloader(dataset, config)

    loss_fn = get_loss_function(config, dataset.tokenizer.pad_id)

    # Wandb requires metrics to be in dict format
    wandb_config = {
        "batch_size": config.training.batch_size,
        "d_model": config.model.hidden_size,
        "num_heads": config.model.n_heads,
        "num_layers": config.model.n_layers,
        "max_seq_len": config.model.max_seq_len,
    }

    # We should initialize the model before training so that it converges faster
    initializer_kind: str = config.model.initializer
    transformer.initialize_weights_(init_method=initializer_kind, init_bias=True)

    # Experiment tracking can be off if running offline without internet
    if config.experiment.active:
        logger.info("Experiment tracking is enabled")
        with ExperimentTracker(
            config.experiment.name, wandb_config, config_file=config_file
        ) as run:
            avg_batch_loss = train_epoch_avg_CE(
                model=transformer,
                train_data_loader=train_data_loader,
                loss_fn=loss_fn,
                wandb_run=run,
            )
            run.log({"epoch_avg_loss": avg_batch_loss.item()})
            logger.info(
                f"Average training loss in one epoch: {avg_batch_loss.item(): .5f}"
            )
    else:
        logger.info("Experiment tracking is disabled")
        avg_batch_loss = train_epoch_avg_CE(
            model=transformer,
            train_data_loader=train_data_loader,
            loss_fn=loss_fn,
            wandb_run=None,
        )
        logger.info(f"Average training loss in one epoch: {avg_batch_loss.item(): .5f}")
