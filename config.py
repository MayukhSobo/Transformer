import tomllib
from pathlib import Path
from box import Box
import logging

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_file: Path):
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "rb") as f:
            self._config = Box(tomllib.load(f))

    @property
    def model(self):
        return self._config.model

    @property
    def tokenizer(self):
        return self._config.tokenizer

    @property
    def training(self):
        return self._config.training

    @property
    def dataset(self):
        return self._config.dataset

    @property
    def loss(self):
        return self._config.loss

    @property
    def experiment(self):
        return self._config.experiment


if __name__ == "__main__":
    config = Config(config_file=Path("config.toml"))
    logger.info("Model.ffwfs = %s", config.model.get("ffwfs"))
