import torch

from finomena.config import ModelConfig, TrainingConfig
from finomena.models.gpt import GPTLanguageModel
from finomena.training.trainer import seed_everything, train_model


def test_short_cpu_training_updates_model_and_returns_finite_losses() -> None:
    seed_everything(10)
    model = GPTLanguageModel(
        ModelConfig(vocab_size=7, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0)
    )
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    tokens = torch.tensor(([0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5] * 4), dtype=torch.long)
    result = train_model(
        model,
        tokens,
        tokens,
        TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            max_steps=3,
            eval_interval=2,
            eval_batches=1,
        ),
        torch.device("cpu"),
    )
    assert result.status == "completed"
    assert result.steps_completed == 3
    assert torch.isfinite(torch.tensor(result.train_loss))
    assert torch.isfinite(torch.tensor(result.validation_loss))
    assert any(not torch.equal(initial[name], tensor) for name, tensor in model.state_dict().items())


def test_seed_controls_model_initialization() -> None:
    config = ModelConfig(vocab_size=7, block_size=8, n_embd=16, n_head=4, n_layer=1)
    seed_everything(31)
    first = GPTLanguageModel(config)
    seed_everything(31)
    second = GPTLanguageModel(config)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
