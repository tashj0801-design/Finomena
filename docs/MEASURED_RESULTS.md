# Measured starter experiment (2026-10-27)

These are actual CPU measurements from this implementation session, not
placeholders. They describe a very small starter run; they do not establish
useful financial-language generation or generalization.

## Corpus and split

The ECB RSS downloader requested up to 40 qualifying release pages and found **3**
qualifying English press-release pages. Cleaning retained all three documents:
0 empty and 0 duplicate bodies; 19,427 characters total. Every publication
timestamp parsed, so the split was chronological: one oldest document for
training (5,311 characters), one for validation (5,936 characters), and one
newest for test (8,180 characters). The train-only vocabulary has 69 IDs,
including `<unk>`. Dataset SHA-256:
`a0f1a47a2a7ed91e83d05232ebbbdc2ba92b6e938100708dccbb5ea33eb14c09`.

Three documents are a very small sample, and each split contains only one
document. No claims about representativeness or market knowledge follow from
these metrics. The raw and processed articles are excluded from Git.

## Model quality and training

Both runs used CPU, seed 1337, batch size 4, context length 64, AdamW learning
rate 0.0003, weight decay 0.01, gradient clip 1.0, 10 optimizer steps,
validation every 5 steps, and two random evaluation batches per report. Loss
values are nats per character; perplexity is character-level.

| Model | Trainable parameters | Training duration | Final train loss / perplexity | Final validation loss / perplexity | Best validation (step) | Final test loss / perplexity |
|---|---:|---:|---:|---:|---:|---:|
| Bigram | 4,761 | 0.059 s | 4.6840 / 108.20 | 4.7021 / 110.18 | 4.7007 (1) | 4.7597 / 116.71 |
| Finomena-Tiny | 113,093 | 0.554 s | 4.0544 / 57.65 | 4.1016 / 60.44 | 4.1016 (10) | 4.0715 / 58.64 |

The Tiny configuration used embedding 64, four heads, two blocks, and 0.10
dropout. Its context was overridden from the profile's 128 to 64 for this
starter run. The test set was evaluated only after both models and best
validation checkpoints were selected. These very short runs and tiny split
cannot support a quality comparison beyond confirming measurable execution.

Separate offline synthetic-fixture smoke runs also completed for both models
(5 steps each). Their metrics are not mixed into the ECB results above and are
not financial-text evidence.

## Inference benchmark

Both best checkpoints were measured on the same CPU with batch 1, sequence
length 32, 16 generated characters, 2 warm-ups, and 20 repeats. Forward and
generation timing used inference mode; CUDA synchronization was not relevant.
The p95 is available because there were 20 observations. Throughput is input
tokens per forward duration or generated characters per end-to-end
autoregressive duration.

| Model | Load time | Forward mean / median / p95 | Forward tokens/s | Generate mean / median / p95 | Generated chars/s | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---|
| Bigram | 16.1 ms | 0.060 / 0.061 / 0.078 ms | 532,003 | 5.613 / 5.737 / 6.667 ms | 2,851 | Unavailable (CPU) |
| Finomena-Tiny | 78.8 ms | 3.268 / 2.811 / 6.013 ms | 9,792 | 51.488 / 50.294 / 59.130 ms | 311 | Unavailable (CPU) |

The measured best-checkpoint experiment IDs are `20261026T202131Z-274d8a24`
(Bigram) and `20261026T202205Z-13fff4e2` (Tiny).

These short CPU timings are environment-specific and subject to scheduling
noise. Loading includes checkpoint deserialization/model construction. They
should not be compared to results on other devices.

## Reproduction commands

```powershell
python scripts\prepare_data.py --max-documents 40 --seed 1337
python scripts\train.py --model bigram --steps 10 --context-length 64 --batch-size 4 --learning-rate 0.0003 --eval-interval 5 --eval-batches 2 --seed 1337
python scripts\train.py --model gpt --profile tiny --steps 10 --context-length 64 --batch-size 4 --learning-rate 0.0003 --eval-interval 5 --eval-batches 2 --seed 1337
python scripts\evaluate.py --checkpoint checkpoints\bigram_latest.pt --split test --batches 20 --batch-size 4 --device cpu
python scripts\evaluate.py --checkpoint checkpoints\gpt_latest.pt --split test --batches 20 --batch-size 4 --device cpu
python scripts\benchmark.py --checkpoint checkpoints\bigram_latest.pt --batch-size 1 --sequence-length 32 --generated-characters 16 --warmup 2 --repeats 20 --device cpu
python scripts\benchmark.py --checkpoint checkpoints\gpt_latest.pt --batch-size 1 --sequence-length 32 --generated-characters 16 --warmup 2 --repeats 20 --device cpu
```

The four final commands reproduce the measurement workflow using stable
latest-checkpoint names after the preceding training commands; the reported
numbers above came from the best-validation checkpoints. The experiment
JSONL, benchmark JSON files, checkpoint weights, and downloaded records are
local ignored artifacts. The feed is rolling, so later downloads need not
match this exact fingerprint or counts.
