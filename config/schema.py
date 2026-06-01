from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    hidden_size: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    dropout_pe: float = Field(ge=0.0, le=1.0)
    n_heads: int = Field(gt=0)
    ff_hidden_size: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    initializer: Literal[
        "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
    ]


class TokenizerConfig(BaseModel):
    kind: Literal["sentencepiece", "custom"]
    model: Path
    sample_size: int = Field(gt=0)
    algorithm: Literal["bpe", "unigram"]
    recreate: bool


class TrainingConfig(BaseModel):
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    device: Literal["auto", "cuda", "cpu"]


class DatasetConfig(BaseModel):
    path: Path


class LossConfig(BaseModel):
    type: Literal["cross_entropy"]
    label_smoothing: float = Field(ge=0.0, le=1.0)


class ExperimentConfig(BaseModel):
    name: str
    active: bool
    backend: Literal["wandb"]
    tracking: Literal["cloud", "offline"]
    description: str
