"""Inspectable optimization and validation loop for character language models."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

import torch
from torch import Tensor, nn

from finomena.config import TrainingConfig
from finomena.data.dataset import sample_batch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_loss(
    model: nn.Module,
    token_streams: dict[str, Tensor],
    block_size: int,
    batch_size: int,
    evaluation_batches: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}
    try:
        with torch.inference_mode():
            for split_name, tokens in token_streams.items():
                batch_losses: list[float] = []
                for _ in range(evaluation_batches):
                    inputs, targets = sample_batch(tokens, batch_size, block_size, device)
                    _, loss = model(inputs, targets)
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError(f"{split_name} loss is not finite")
                    batch_losses.append(loss.item())
                losses[split_name] = sum(batch_losses) / len(batch_losses)
    finally:
        model.train(was_training)
    return losses


@dataclass
class TrainingResult:
    status: str
    steps_completed: int
    duration_seconds: float
    train_loss: float
    validation_loss: float
    best_validation_loss: float
    best_step: int
    history: list[dict[str, float | int]]
    best_model_state: dict[str, Tensor]
    best_optimizer_state: dict[str, Any]
    final_model_state: dict[str, Tensor]
    final_optimizer_state: dict[str, Any]


def train_model(
    model: nn.Module,
    train_tokens: Tensor,
    validation_tokens: Tensor,
    config: TrainingConfig,
    device: torch.device,
    *,
    callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Event | None = None,
) -> TrainingResult:
    block_size = int(getattr(model, "block_size", 0))
    if block_size < 1:
        raise ValueError("model must expose a positive block_size")
    for split_name, tokens in (("train", train_tokens), ("validation", validation_tokens)):
        if tokens.numel() <= block_size:
            raise ValueError(f"{split_name} token stream must exceed context length {block_size}")

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_step = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    final_losses: dict[str, float] | None = None
    steps_completed = 0
    status = "completed"

    for step in range(1, config.max_steps + 1):
        if cancel_event is not None and cancel_event.is_set():
            status = "cancelled"
            break
        model.train()
        inputs, targets = sample_batch(
            train_tokens, config.batch_size, block_size, device
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"training loss is not finite at step {step}")
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        steps_completed = step

        if step % config.eval_interval == 0 or step == 1 or step == config.max_steps:
            final_losses = estimate_loss(
                model,
                {"train": train_tokens, "validation": validation_tokens},
                block_size,
                config.batch_size,
                config.eval_batches,
                device,
            )
            snapshot = {
                "step": step,
                "train_loss": final_losses["train"],
                "validation_loss": final_losses["validation"],
            }
            history.append(snapshot)
            if final_losses["validation"] < best_validation_loss:
                best_validation_loss = final_losses["validation"]
                best_step = step
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            if callback is not None:
                callback(
                    {
                        **snapshot,
                        "elapsed_seconds": time.perf_counter() - started,
                        "max_steps": config.max_steps,
                    }
                )

    if final_losses is None:
        final_losses = estimate_loss(
            model,
            {"train": train_tokens, "validation": validation_tokens},
            block_size,
            config.batch_size,
            config.eval_batches,
            device,
        )
    if best_step == 0:
        best_validation_loss = final_losses["validation"]
        best_step = steps_completed
        best_model_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    return TrainingResult(
        status=status,
        steps_completed=steps_completed,
        duration_seconds=time.perf_counter() - started,
        train_loss=final_losses["train"],
        validation_loss=final_losses["validation"],
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        history=history,
        best_model_state=best_model_state,
        best_optimizer_state=best_optimizer_state,
        final_model_state=copy.deepcopy(model.state_dict()),
        final_optimizer_state=copy.deepcopy(optimizer.state_dict()),
    )
