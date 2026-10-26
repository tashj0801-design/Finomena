"""Reproducible document-level splits with chronological mode when possible."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime

from finomena.data.sources import Document


def split_documents(
    documents: list[Document],
    seed: int = 1337,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[dict[str, list[Document]], str]:
    if len(documents) < 3:
        raise ValueError("at least three unique documents are required for train/validation/test")
    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
    ):
        raise ValueError("train and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    dated: list[tuple[datetime, Document]] = []
    for document in documents:
        try:
            date = datetime.fromisoformat(document.published_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            dated = []
            break
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        dated.append((date, document))

    if dated:
        ordered = [document for _, document in sorted(dated, key=lambda pair: pair[0])]
        method = "chronological by publication timestamp"
    else:
        ordered = documents.copy()
        random.Random(seed).shuffle(ordered)
        method = f"seeded document-level random split (seed={seed}; dates missing or invalid)"

    count = len(ordered)
    train_end = max(1, min(count - 2, int(count * train_fraction)))
    validation_end = max(train_end + 1, min(count - 1, train_end + int(count * validation_fraction)))
    return (
        {
            "train": ordered[:train_end],
            "validation": ordered[train_end:validation_end],
            "test": ordered[validation_end:],
        },
        method,
    )
