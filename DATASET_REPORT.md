# Dataset report

## Selected source and terms

- **Source:** European Central Bank (ECB), official English-language press
  releases only (`/press/pr/`); speeches and interviews in the shared press
  feed are excluded.
- **Provider:** European Central Bank.
- **Feed:** <https://www.ecb.europa.eu/rss/press.html>
- **Access method:** the standard RSS feed and linked HTML press-release pages.
  `scripts/prepare_data.py` fetches on request, follows the feed’s release
  links, and waits one second between page requests. It does not bypass access
  controls. Network failures are shown to the caller.
- **Terms checked:** ECB website disclaimer and copyright terms,
  <https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html>,
  accessed 2026-10-27. The ECB permits free use of information obtained
  directly from its site provided the information remains accurate and the ECB
  is cited as source; if information is modified, the modification must be
  stated explicitly. Finomena documents its deterministic whitespace/Unicode
  cleanup and does not redistribute the acquired articles. The explicit
  prior-authorisation exception for
  author-named working/occasional papers is why this pipeline selects
  institutional press releases, not speeches, interviews, or named research
  papers. This is a summary, not legal advice; users should recheck terms
  before reuse.
- **Fields used:** release title, publication timestamp, canonical page URL,
  and visible page text. Attribution is retained in raw JSONL records.
- **Access date:** the feed and terms page were reachable and inspected on
  2026-10-27. Exact downloaded records and counts are written to
  `data/processed/manifest.json` when the preparation script succeeds.
- **Measured initial corpus (2026-10-27):** requested up to 40 qualifying
  press releases from the current RSS feed; 3 release documents were found
  and retained, with 0
  empty and 0 duplicate documents. The cleaned corpus contains 19,427
  characters. The chronological train/validation/test split contains 1
  document each (5,311 / 5,936 / 8,180 characters). The train-only vocabulary
  contains 69 IDs including `<unk>`. These counts describe this one rolling
  feed download, not a stable dataset release.
- **Redistribution:** raw and processed text are not committed. The repository
  includes only a clearly labelled synthetic fixture for offline tests.

The ECB publishes institutional monetary/economic news and official
commentary, rather than third-party financial journalism. The feed has a
rolling, finite set of recent items; a small sample is not representative of
markets or all financial writing. It can contain charts, tables, and contact
details; HTML text extraction is deliberately simple and is not semantic
article parsing.

## Processing and leakage controls

The preparation workflow normalises Unicode to NFC, line endings and
horizontal whitespace; preserves financial punctuation, currency signs,
percentages, ticker-like text, dates, decimals, commas, and signs; removes
empty records and exact duplicate cleaned bodies; and splits complete
documents, never overlapping windows. When every document has a valid feed
timestamp, the oldest 80% are training, the next 10% validation, and newest
10% test (with nonempty splits); otherwise it uses a seeded document shuffle.
The tokenizer is fitted only on training text. Its `<unk>` ID safely handles
new validation/test characters. SHA-256 fingerprint, counts, split sizes,
split method, seed, and training vocabulary size are recorded in the generated
manifest. Test loss is computed by the explicit evaluation command, not
routine training selection.

`data/sample/synthetic.jsonl` is fabricated for offline smoke runs. It is not
part of the ECB dataset, and its metrics must not be described as financial
language quality results.

The exact measured starter-model runs and benchmark methodology are in
[`docs/MEASURED_RESULTS.md`](docs/MEASURED_RESULTS.md).
