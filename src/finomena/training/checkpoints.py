"""Validated checkpoints loaded through PyTorch's restricted weights-only loader."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

from finomena.config import ModelConfig
from finomena.models.bigram import BigramLanguageModel
from finomena.models.gpt import GPTLanguageModel
from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def save_checkpoint(
    path: str | Path,
    *,
    model_type: str,
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    seed: int,
    step: int,
    best_validation_loss: float,
    metadata: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if model_type == "gpt" and isinstance(model, GPTLanguageModel):
        model_config = model.config.to_dict()
    elif model_type == "bigram" and isinstance(model, BigramLanguageModel):
        model_config = {
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
        }
    else:
        raise ValueError(f"model type {model_type!r} does not match the supplied model")
    if optimizer is not None and optimizer_state is not None:
        raise ValueError("provide optimizer or optimizer_state, not both")
    payload = {
        "format_version": 1,
        "model_type": model_type,
        "model_config": model_config,
        "model_state": model.state_dict(),
        "tokenizer_state": tokenizer.to_state(),
        "optimizer_state": (
            optimizer.state_dict() if optimizer is not None else optimizer_state
        ),
        "step": step,
        "seed": seed,
        "best_validation_loss": best_validation_loss,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "metadata": metadata,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, CharacterTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported or invalid checkpoint format")
    model_type = payload.get("model_type")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no valid model configuration")
    if model_type == "gpt":
        model = GPTLanguageModel(ModelConfig(**config))
    elif model_type == "bigram":
        model = BigramLanguageModel(**config)
    else:
        raise ValueError(f"unsupported checkpoint model type: {model_type!r}")
    tokenizer = CharacterTokenizer.from_state(payload.get("tokenizer_state", {}))
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match model configuration")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    return model, tokenizer, payload
