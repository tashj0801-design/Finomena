import torch

from finomena.config import ModelConfig
from finomena.generation.generate import generate_text
from finomena.models.gpt import GPTLanguageModel
from finomena.tokenizer.character_tokenizer import CharacterTokenizer
from finomena.training.checkpoints import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_preserves_logits(tmp_path) -> None:
    torch.manual_seed(7)
    tokenizer = CharacterTokenizer.fit("abcde")
    model = GPTLanguageModel(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=8,
            n_embd=16,
            n_head=4,
            n_layer=1,
            dropout=0,
        )
    ).eval()
    inputs = torch.tensor([tokenizer.encode("abc")])
    with torch.inference_mode():
        expected, _ = model(inputs)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model_type="gpt",
        model=model,
        tokenizer=tokenizer,
        seed=7,
        step=3,
        best_validation_loss=1.2,
        metadata={"experiment_id": "unit-test"},
    )
    restored, restored_tokenizer, payload = load_checkpoint(path)
    restored.eval()
    with torch.inference_mode():
        actual, _ = restored(inputs)
    torch.testing.assert_close(actual, expected)
    assert restored_tokenizer.to_state() == tokenizer.to_state()
    assert payload["step"] == 3
    assert len(generate_text(restored, restored_tokenizer, "abc", max_new_chars=4, seed=3)) == 4
