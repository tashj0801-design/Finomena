from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finomena.data.cleaning import clean_documents
from finomena.data.sources import FEED_URL, fetch_press_releases, read_jsonl, write_jsonl
from finomena.data.splits import split_documents
from finomena.tokenizer.character_tokenizer import CharacterTokenizer

def prepare(source_path: Path | None, max_documents: int, seed: int) -> dict[str, object]:
    raw_path = ROOT / "data" / "raw" / "ecb_press_releases.jsonl"
    documents = read_jsonl(source_path) if source_path else fetch_press_releases(max_documents)
    if not source_path:
        write_jsonl(documents, raw_path)

    clean, cleaning_report = clean_documents(documents)
    splits, split_method = split_documents(clean, seed=seed)
    tokenizer = CharacterTokenizer.fit("\n\n".join(document.text for document in splits["train"]))

    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_documents_list in splits.items():
        write_jsonl(split_documents_list, processed_dir / f"{split_name}.jsonl")
    fingerprint_input = "\n".join(document.text for document in clean).encode("utf-8")
    report: dict[str, object] = {
        "source": (
            "synthetic local test fixture"
            if source_path and source_path.name == "synthetic.jsonl"
            else "local JSONL supplied by user"
            if source_path
            else "ECB English press releases"
        ),
        "synthetic": bool(source_path and source_path.name == "synthetic.jsonl"),
        "feed_url": None if source_path else FEED_URL,
        "seed": seed,
        "split_method": split_method,
        "fingerprint_sha256": hashlib.sha256(fingerprint_input).hexdigest(),
        "cleaning": asdict(cleaning_report),
        "split_documents": {name: len(rows) for name, rows in splits.items()},
        "split_characters": {
            name: sum(len(document.text) for document in rows)
            for name, rows in splits.items()
        },
        "training_vocabulary_size_including_unk": tokenizer.vocab_size,
    }
    (processed_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an attributable financial-text corpus.")
    parser.add_argument("--source-jsonl", type=Path, help="Use a local JSONL source instead of downloading.")
    parser.add_argument("--max-documents", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    report = prepare(args.source_jsonl, args.max_documents, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
