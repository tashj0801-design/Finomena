"""A deterministic character vocabulary with a dedicated unknown token."""

from __future__ import annotations

from dataclasses import dataclass

UNK_TOKEN = "<unk>"
UNK_RENDERING = "\ufffd"


@dataclass(frozen=True)
class CharacterTokenizer:
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or self.tokens[0] != UNK_TOKEN:
            raise ValueError("vocabulary must start with the reserved <unk> token")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary entries must be unique")
        if any(len(token) != 1 for token in self.tokens[1:]):
            raise ValueError("character vocabulary entries must be one character")

    @classmethod
    def fit(cls, training_text: str) -> CharacterTokenizer:
        return cls((UNK_TOKEN, *tuple(sorted(set(training_text)))))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def stoi(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str) -> list[int]:
        vocabulary = self.stoi
        unknown_id = vocabulary[UNK_TOKEN]
        return [vocabulary.get(character, unknown_id) for character in text]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        output: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            output.append(UNK_RENDERING if token_id == 0 else self.tokens[token_id])
        return "".join(output)

    def to_state(self) -> dict[str, list[str]]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> CharacterTokenizer:
        tokens = state.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("tokenizer state must contain a list of strings under 'tokens'")
        return cls(tuple(tokens))
