import math

import pytest
import torch

from finomena.evaluation.metrics import (
    character_perplexity,
    characters_per_second,
    parameter_count,
    summarize_latencies,
)


def test_quality_and_latency_metrics() -> None:
    assert character_perplexity(0) == 1
    assert character_perplexity(1) == pytest.approx(math.e)
    assert character_perplexity(float("inf")) is None
    assert characters_per_second(20, 2) == 10
    short = summarize_latencies([1.0, 2.0, 3.0])
    assert short["mean_seconds"] == 2
    assert short["median_seconds"] == 2
    assert short["p95_seconds"] is None
    long = summarize_latencies(list(range(1, 21)))
    assert long["p95_seconds"] is not None
    assert parameter_count(torch.nn.Linear(2, 3)) == 9
