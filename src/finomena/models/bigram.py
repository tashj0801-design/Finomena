"""A trainable previous-character baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predict the next character from the current character alone."""

    def __init__(self, vocab_size: int, block_size: int = 128) -> None:
        super().__init__()
        if vocab_size < 2 or block_size < 1:
            raise ValueError("vocab_size must be at least 2 and block_size positive")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.next_character = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [batch, sequence]")
        logits = self.next_character(inputs)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss
