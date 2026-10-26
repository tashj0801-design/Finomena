import pytest

from finomena.config import ModelConfig, TrainingConfig, model_config_for_profile


def test_model_configuration_validates_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, n_embd=10, n_head=4)


def test_profiles_and_training_configuration_are_valid() -> None:
    assert model_config_for_profile("tiny", vocab_size=20).block_size == 128
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=float("nan"))
