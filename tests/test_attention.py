import torch

from finomena.models.attention import CausalSelfAttention
from finomena.models.gpt import GPTLanguageModel
from finomena.config import ModelConfig


def test_attention_output_shape() -> None:
    attention = CausalSelfAttention(n_embd=16, n_head=4, block_size=8, dropout=0)
    assert attention(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_gpt_earlier_logits_do_not_depend_on_future_tokens() -> None:
    torch.manual_seed(4)
    model = GPTLanguageModel(
        ModelConfig(vocab_size=9, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0)
    ).eval()
    original = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 7, 8]])
    with torch.inference_mode():
        original_logits, _ = model(original)
        changed_logits, _ = model(changed_future)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3], atol=1e-6, rtol=1e-6)
