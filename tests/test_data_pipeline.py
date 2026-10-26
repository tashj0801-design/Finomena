from finomena.data.cleaning import clean_documents, clean_text
from finomena.data.dataset import TokenWindowDataset
from finomena.data.sources import Document
from finomena.data.splits import split_documents
from finomena.tokenizer.character_tokenizer import CharacterTokenizer


def test_cleaning_preserves_financial_notation_and_removes_empty_and_duplicate() -> None:
    assert clean_text("  $1,234.50\r\n-2.4%  ") == "$1,234.50\n-2.4%"
    documents = [
        Document("", " AAPL rose 2.4%. ", "", ""),
        Document("", "AAPL rose 2.4%.", "", ""),
        Document("", "\t \n", "", ""),
    ]
    cleaned, report = clean_documents(documents)
    assert len(cleaned) == 1
    assert report.empty_removed == 1
    assert report.duplicates_removed == 1


def test_splits_are_reproducible_and_document_level() -> None:
    documents = [Document(str(i), f"unique-{i}", "", "") for i in range(20)]
    left, method = split_documents(documents, seed=17)
    right, _ = split_documents(documents, seed=17)
    assert method.startswith("seeded document-level")
    assert left == right
    sets = [{document.text for document in left[name]} for name in ("train", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_dated_documents_split_chronologically() -> None:
    documents = [
        Document(str(day), str(day), "", f"2025-01-{day:02}T00:00:00+00:00")
        for day in range(1, 11)
    ]
    splits, method = split_documents(documents)
    assert method == "chronological by publication timestamp"
    assert splits["train"][-1].published_at < splits["validation"][0].published_at
    assert splits["validation"][-1].published_at < splits["test"][0].published_at


def test_windows_return_shifted_input_and_target() -> None:
    tokenizer = CharacterTokenizer.fit("abcde")
    dataset = TokenWindowDataset("abcde", tokenizer, block_size=3)
    inputs, targets = dataset[0]
    assert inputs.tolist() == tokenizer.encode("abc")
    assert targets.tolist() == tokenizer.encode("bcd")
