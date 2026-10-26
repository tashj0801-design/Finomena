"""Encoded next-character windows and random contiguous training batches."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from finomena.tokenizer.character_tokenizer import CharacterTokenizer


class TokenWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, text: str, tokenizer: CharacterTokenizer, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError("text must contain more tokens than block_size")

    def __len__(self) -> int:
        return len(self.tokens) - self.block_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        window = self.tokens[index : index + self.block_size + 1]
        return window[:-1], window[1:]


def encode_documents(documents: list[str], tokenizer: CharacterTokenizer) -> Tensor:
    """Join documents with an explicit newline boundary before encoding."""
    return torch.tensor(tokenizer.encode("\n\n".join(documents)), dtype=torch.long)


def sample_batch(
    tokens: Tensor, batch_size: int, block_size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    if batch_size < 1 or block_size < 1:
        raise ValueError("batch_size and block_size must be positive")
    if tokens.ndim != 1 or tokens.numel() <= block_size:
        raise ValueError("token stream must be one-dimensional and longer than block_size")
    starts = torch.randint(tokens.numel() - block_size, (batch_size,))
    inputs = torch.stack([tokens[start : start + block_size] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)
