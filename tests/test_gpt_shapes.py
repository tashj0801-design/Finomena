import torch

from finomena.config import ModelConfig, count_trainable_parameters
from finomena.models.gpt import GPTLanguageModel


def test_gpt_shapes_loss_and_finite_gradients() -> None:
    model = GPTLanguageModel(
        ModelConfig(vocab_size=11, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0)
    )
    inputs = torch.randint(0, 11, (2, 5))
    targets = torch.randint(0, 11, (2, 5))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 5, 11)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert count_trainable_parameters(model) == model.parameter_count()
