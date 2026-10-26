# Data pipeline

The downloader reads `https://www.ecb.europa.eu/rss/press.html`, selects
unique English HTML links whose path contains `/press/pr/`, and sequentially
fetches at most 40 qualifying release pages by default with a one-second delay.
It saves release title,
URL, publication timestamp and text to ignored `data/raw/ecb_press_releases.jsonl`.
It does not follow the feed's speech/interview links or fetch PDFs. See
`DATASET_REPORT.md` for terms, attribution and limitations.

`clean_text` applies Unicode NFC, standardises line endings, compresses
horizontal whitespace and trims excessive blank lines. It intentionally does
not strip financial signs, numeric notation, punctuation or ticker-like
strings. Empty bodies and exact duplicate cleaned bodies are removed before
splitting. The manifest records counts, character totals, split method, seed,
vocabulary size and SHA-256 over ordered cleaned text.

Documents with valid timestamps are sorted oldest to newest and divided into
oldest 80% training, next 10% validation, newest 10% test. If timestamps are
missing or invalid, a seeded document-level shuffle is used and disclosed.
Deduplication happens first, so the same cleaned body cannot cross a split.
The test documents are not sampled by the training loop.

Only training text creates the sorted tokenizer vocabulary. Each Unicode
character becomes one integer; an unknown held-out character maps to reserved
ID 0. Encoded documents are separated by two newline characters. A context
window `[t0,t1,t2,t3]` produces input `[t0,t1,t2]` and target
`[t1,t2,t3]`, so every position learns to predict the following character.
`TokenWindowDataset` exposes deterministic index windows; the trainer samples
random contiguous windows from each already-separated split.

For offline tests, `data/sample/synthetic.jsonl` contains fabricated
finance-like strings clearly labelled synthetic. Pass it with
`--source-jsonl` to prepare or `--data` to train. Do not report those runs as
evidence about financial news.
