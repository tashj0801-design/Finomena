"""Pre-layer-normalised Transformer decoder block."""

from __future__ import annotations

from torch import Tensor, nn

from finomena.models.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self, n_embd: int, n_head: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.feed_forward_norm = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
