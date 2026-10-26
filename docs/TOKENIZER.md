# Character tokenizer

`CharacterTokenizer.fit(training_text)` sorts the unique Unicode characters
seen in training text and places a reserved `<unk>` token at ID 0. `encode`
maps each character to its integer; `decode` maps IDs back. Unseen validation
or test characters map to ID 0 and render as `�`. State serializes as an
ordered token list, so a checkpoint uses the exact vocabulary it was trained
with.

For example, fitting on `cash` gives a deterministic vocabulary containing
`<unk>`, `a`, `c`, `h`, `s`; encoding `cash` gives IDs in that order's
corresponding positions, and decoding recovers `cash`. Sorting makes the
vocabulary independent of first occurrence. The unknown character prevents
held-out text from crashing or leaking into vocabulary construction.

The character approach is intentionally simple and transparent. It avoids a
pretrained tokenizer, but words take many model positions, context length is
measured in characters, and spelling/whitespace errors are common. Unicode
code points are not the same thing as grapheme clusters, so some visual
characters may be represented by several code points.
