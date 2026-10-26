# Data directories

- `raw/` stores source documents and their attribution metadata after an
  explicit acquisition request. Do not commit source text.
- `processed/` stores cleaned document-level train, validation and test JSONL
  splits plus the dataset manifest. These are reproducibly generated and
  Git-ignored.
- `sample/synthetic.jsonl` is fabricated finance-like text for offline
  verification only. It is labelled synthetic and is not financial news.

The ECB source, reuse terms, cleaning rules and measured first download are
documented in the repository's `DATASET_REPORT.md`.
