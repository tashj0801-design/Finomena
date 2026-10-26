"""Transparent scaled dot-product causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        future_positions = torch.triu(
            torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("future_positions", future_positions, persistent=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, sequence, channels = hidden_states.shape
        if sequence > self.future_positions.size(0):
            raise ValueError("sequence length exceeds configured context length")
        query, key, value = self.qkv(hidden_states).chunk(3, dim=-1)
        # Each projection is split into heads: [B,T,C] -> [B,H,T,C/H].
        query = query.view(batch, sequence, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch, sequence, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch, sequence, self.n_head, self.head_size).transpose(1, 2)

        # Scores are [B,H,T,T]; mask future keys before the row-wise softmax.
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.future_positions[:sequence, :sequence], float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.residual_dropout(self.projection(attended))
