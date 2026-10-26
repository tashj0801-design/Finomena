"""A decoder-only GPT-style character language model built from PyTorch layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from finomena.config import ModelConfig, count_trainable_parameters
from finomena.models.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *(
                TransformerBlock(
                    config.n_embd, config.n_head, config.block_size, config.dropout
                )
                for _ in range(config.n_layer)
            )
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_model_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self, inputs: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [batch, sequence]")
        _, sequence = inputs.shape
        if sequence > self.block_size:
            raise ValueError("input sequence exceeds configured context length")
        positions = torch.arange(sequence, device=inputs.device)
        hidden_states = self.embedding_dropout(
            self.token_embedding(inputs) + self.position_embedding(positions)
        )
        logits = self.language_model_head(self.final_norm(self.blocks(hidden_states)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return logits, loss

    def parameter_count(self) -> int:
        return count_trainable_parameters(self)
