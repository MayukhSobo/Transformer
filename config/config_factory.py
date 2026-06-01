import logging
import tomllib
from pathlib import Path

from .schema import (
    DatasetConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file: Path):
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "rb") as f:
            data = tomllib.load(f)
        self.model = ModelConfig(**data["model"])
        self.tokenizer = TokenizerConfig(**data["tokenizer"])
        self.training = TrainingConfig(**data["training"])
        self.dataset = DatasetConfig(**data["dataset"])
        self.loss = LossConfig(**data["loss"])
        self.experiment = ExperimentConfig(**data["experiment"])
