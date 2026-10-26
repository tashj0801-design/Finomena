from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def test_known_characters_round_trip_and_state_round_trip() -> None:
    tokenizer = CharacterTokenizer.fit("fin 2.5%")
    text = "fin 2.5%"
    restored = CharacterTokenizer.from_state(tokenizer.to_state())
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert restored.to_state() == tokenizer.to_state()


def test_unknown_characters_use_reserved_id_and_visible_replacement() -> None:
    tokenizer = CharacterTokenizer.fit("abc")
    assert tokenizer.encode("z") == [0]
    assert tokenizer.decode([0]) == "\ufffd"
