"""Small metric functions with explicit numerical edge-case handling."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from torch import nn


def character_perplexity(mean_loss: float) -> float | None:
    """Convert mean character-level cross-entropy to perplexity when finite."""
    if not math.isfinite(mean_loss):
        return None
    try:
        value = math.exp(mean_loss)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def summarize_latencies(seconds: Sequence[float]) -> dict[str, float | int | None]:
    if not seconds or any(value < 0 or not math.isfinite(value) for value in seconds):
        raise ValueError("latencies must be a non-empty sequence of finite non-negative values")
    ordered = sorted(seconds)
    p95 = (
        statistics.quantiles(ordered, n=100, method="inclusive")[94]
        if len(ordered) >= 20
        else None
    )
    return {
        "repeat_count": len(ordered),
        "mean_seconds": statistics.fmean(ordered),
        "median_seconds": statistics.median(ordered),
        "p95_seconds": p95,
    }


def characters_per_second(character_count: int, elapsed_seconds: float) -> float | None:
    if character_count < 0 or elapsed_seconds < 0 or not math.isfinite(elapsed_seconds):
        raise ValueError("character count and elapsed seconds must be non-negative")
    if elapsed_seconds == 0:
        return None
    return character_count / elapsed_seconds


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
