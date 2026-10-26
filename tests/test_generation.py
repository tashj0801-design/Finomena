import pytest

from finomena.models.bigram import BigramLanguageModel
from finomena.generation.generate import generate_text
from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def test_generation_handles_prompt_empty_input_top_k_seed_and_length() -> None:
    tokenizer = CharacterTokenizer.fit("a b")
    model = BigramLanguageModel(tokenizer.vocab_size, block_size=2)
    assert len(generate_text(model, tokenizer, "a", 12, top_k=1, seed=7)) == 12
    assert len(generate_text(model, tokenizer, "", 3, seed=7)) == 3
    assert generate_text(model, tokenizer, "a", 12, top_k=1, seed=7) == generate_text(
        model, tokenizer, "a", 12, top_k=1, seed=7
    )


def test_generation_rejects_invalid_sampling_settings() -> None:
    tokenizer = CharacterTokenizer.fit("abc")
    model = BigramLanguageModel(tokenizer.vocab_size)
    with pytest.raises(ValueError, match="temperature"):
        generate_text(model, tokenizer, temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        generate_text(model, tokenizer, temperature=float("nan"))
    with pytest.raises(ValueError, match="top_k"):
        generate_text(model, tokenizer, top_k=tokenizer.vocab_size + 1)
