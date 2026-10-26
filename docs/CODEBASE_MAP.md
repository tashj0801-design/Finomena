# Codebase map

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit presentation, controls, progress and error display |
| `src/finomena/config.py` | Validated profiles, model/training settings, parameter counts |
| `src/finomena/environment.py` | CPU/CUDA detection and safe device selection |
| `src/finomena/data/sources.py` | ECB RSS acquisition and JSONL metadata |
| `src/finomena/data/cleaning.py` | Unicode/whitespace cleanup, empty and exact duplicate removal |
| `src/finomena/data/splits.py` | Reproducible whole-document temporal or seeded splits |
| `src/finomena/data/dataset.py` | Character-stream encoding, shifted windows, random batches |
| `src/finomena/tokenizer/character_tokenizer.py` | Train-only character vocabulary and serializable state |
| `src/finomena/models/bigram.py` | Trainable one-character context baseline |
| `src/finomena/models/attention.py` | Manual Q/K/V projections, causal mask and multi-head attention |
| `src/finomena/models/transformer_block.py` | Pre-norm attention/MLP residual block |
| `src/finomena/models/gpt.py` | Embeddings, decoder stack, vocabulary logits and cross-entropy |
| `src/finomena/training/trainer.py` | Seeded AdamW updates, validation, gradients and progress callback |
| `src/finomena/training/checkpoints.py` | Versioned checkpoints and restricted safe loading |
| `src/finomena/training/experiment.py` | Atomic JSONL experiment history |
| `src/finomena/evaluation/metrics.py` | Character perplexity, latency summaries and throughput |
| `src/finomena/evaluation/benchmark.py` | Warmed inference measurements and CUDA synchronization |
| `src/finomena/generation/generate.py` | Autoregressive temperature/top-k sampling |
| `scripts/prepare_data.py` | Fetch or ingest JSONL, clean, split, fingerprint and manifest |
| `scripts/train.py` | CLI entry to real training and checkpoint/experiment output |
| `scripts/evaluate.py` | Explicit split evaluation (test is not used in model selection) |
| `scripts/benchmark.py` | Load and benchmark a saved model |
| `scripts/inspect_environment.py` | Print observed Python/PyTorch/device details |
| `tests/` | Offline unit, correctness, smoke and import tests |

## Call flow

`app.py` and `scripts/train.py` read the processed training split, fit the
tokenizer there only, create a model, encode training/validation documents, and
call `train_model`. Training writes best and latest weights plus optimizer
state, tokenizer, seed, data fingerprint, config and environment. The evaluate
CLI reloads that checkpoint and evaluates one requested split. The generation
module and benchmark runner consume the same checkpoint-bound vocabulary.
Dataset download is a user-triggered button or CLI action; importing the
dashboard does not download data or train a model.
