"""Autoregressive next-character generation with temperature and top-k."""

from __future__ import annotations

import math
import torch
from torch import nn

from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def generate_text(
    model: nn.Module,
    tokenizer: CharacterTokenizer,
    prompt: str = "",
    max_new_chars: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    if max_new_chars < 0:
        raise ValueError("max_new_chars cannot be negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and not 1 <= top_k <= tokenizer.vocab_size:
        raise ValueError("top_k must be between 1 and vocabulary size")

    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error
    context_length = int(getattr(model, "block_size", 128))
    prompt_ids = tokenizer.encode(prompt) or [0]
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_chars):
                context = tokens[:, -context_length:]
                logits, _ = model(context)
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    top_indices = torch.topk(next_logits, top_k, dim=-1).indices
                    keep = torch.zeros_like(next_logits, dtype=torch.bool).scatter(
                        -1, top_indices, True
                    )
                    next_logits = next_logits.masked_fill(~keep, float("-inf"))
                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
                tokens = torch.cat((tokens, next_token), dim=1)
    finally:
        model.train(was_training)

    return tokenizer.decode(tokens[0, len(prompt_ids) :].tolist())
