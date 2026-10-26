"""Validated model and training configurations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.n_embd < 1 or self.n_head < 1 or self.n_layer < 1:
            raise ValueError("embedding size, heads, and layers must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_steps: int = 50
    eval_interval: int = 10
    eval_batches: int = 5
    seed: int = 1337
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("evaluation interval and batches must be positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.grad_clip is not None and (
            not math.isfinite(self.grad_clip) or self.grad_clip <= 0
        ):
            raise ValueError("grad_clip must be positive when provided")


MODEL_PROFILES: dict[str, dict[str, int | float]] = {
    "tiny": {"n_embd": 64, "n_head": 4, "n_layer": 2, "block_size": 128, "dropout": 0.1},
    "small": {"n_embd": 128, "n_head": 4, "n_layer": 4, "block_size": 256, "dropout": 0.1},
    "medium": {"n_embd": 256, "n_head": 8, "n_layer": 6, "block_size": 256, "dropout": 0.1},
}


def model_config_for_profile(
    profile: str, vocab_size: int, *, block_size: int | None = None
) -> ModelConfig:
    try:
        values = MODEL_PROFILES[profile].copy()
    except KeyError as error:
        raise ValueError(f"unknown model profile: {profile}") from error
    if block_size is not None:
        values["block_size"] = block_size
    return ModelConfig(vocab_size=vocab_size, **values)


def count_trainable_parameters(model: object) -> int:
    """Count trainable PyTorch parameters without counting buffers."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        raise TypeError("model must provide a parameters() method")
    return sum(parameter.numel() for parameter in parameters() if parameter.requires_grad)
