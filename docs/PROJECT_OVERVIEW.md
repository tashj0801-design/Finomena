# Project overview

Finomena is a small, inspectable experiment for learning how character-level
language models turn a text corpus into next-character predictions. It has
three boundaries: an attributed source-data pipeline, trainable PyTorch models
(a bigram baseline and a decoder-only Transformer), and interfaces for
experiments, generation, evaluation, and benchmarking.

The lifecycle is: fetch or provide text → clean and deduplicate whole
documents → split documents → fit a vocabulary only on training text → encode
characters → sample shifted contiguous windows → train and validate →
checkpoint → optionally score held-out test data → generate and benchmark.
The Streamlit dashboard calls the same package and scripts instead of
implementing model operations itself.

The main source is the European Central Bank's official English press-release
feed. Its public-use terms require accurate reproduction and ECB attribution.
Only `/press/pr/` feed items are acquired, not the speeches or interviews that
share the feed. See `DATASET_REPORT.md` for source and terms. A synthetic local
fixture supports deterministic offline tests and is not financial news.

The Transformer is built from ordinary PyTorch layers and tensor operations;
no pretrained weights or tokenizer are used. `Finomena-Tiny` is a
configuration, not a pretrained foundation model. This repository is for
education, not verified news, financial advice, or investment decisions.
