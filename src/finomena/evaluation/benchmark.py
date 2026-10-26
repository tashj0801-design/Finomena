"""Repeatable inference measurements for a single device and model."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from finomena.evaluation.metrics import characters_per_second, summarize_latencies
from finomena.generation.generate import generate_text
from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    *,
    batch_size: int = 1,
    sequence_length: int = 32,
    generated_characters: int = 16,
    warmup: int = 2,
    repeats: int = 20,
    seed: int = 1234,
) -> dict[str, Any]:
    if batch_size < 1 or sequence_length < 1 or generated_characters < 1:
        raise ValueError("batch size, sequence length, and generated length must be positive")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    block_size = int(getattr(model, "block_size", sequence_length))
    if sequence_length > block_size:
        raise ValueError("sequence_length cannot exceed model context length")
    model = model.to(device).eval()
    inputs = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        _synchronize(device)
        forward_times: list[float] = []
        for _ in range(repeats):
            _synchronize(device)
            started = time.perf_counter()
            model(inputs)
            _synchronize(device)
            forward_times.append(time.perf_counter() - started)

    prompt_ids = tokenizer.decode([0] * sequence_length)
    for index in range(warmup):
        generate_text(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_new_chars=generated_characters,
            seed=seed + index,
        )
    _synchronize(device)
    generation_times: list[float] = []
    for index in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        generate_text(
            model,
            tokenizer,
            prompt=prompt_ids,
            max_new_chars=generated_characters,
            seed=seed + index,
        )
        _synchronize(device)
        generation_times.append(time.perf_counter() - started)

    result: dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "generated_characters": generated_characters,
        "warmup_iterations": warmup,
        "repeats": repeats,
        "forward_pass": summarize_latencies(forward_times),
        "autoregressive_generation": summarize_latencies(generation_times),
        "generation_characters_per_second": characters_per_second(
            generated_characters, sum(generation_times) / len(generation_times)
        ),
        "forward_tokens_per_second": characters_per_second(
            batch_size * sequence_length, sum(forward_times) / len(forward_times)
        ),
        "peak_gpu_memory_bytes": None,
        "peak_gpu_memory_note": "Unavailable: benchmark device is not CUDA."
        if device.type != "cuda"
        else None,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            model(inputs)
        _synchronize(device)
        result["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_gpu_memory_note"] = None
    return result
