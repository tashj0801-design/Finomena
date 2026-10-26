# Finomena educational model card

## Intended use

Teach the mechanics of a character tokenizer, trainable bigram baseline,
causal self-attention, a small decoder-only Transformer, training and
evaluation. The sample corpus is the ECB's official English press releases;
download and reuse terms are summarized in `DATASET_REPORT.md`. The source
license/terms require accurate content and ECB attribution.

## Out-of-scope use

Financial advice, investment or credit decisions, automated trading, risk
scoring, verified news, legal/medical advice, or any production use where
accuracy matters. Do not publish generated output as verified financial
analysis.

## Architecture and training

The main model is a from-scratch PyTorch decoder-only character Transformer
with learned token and position embeddings, explicit masked multi-head
attention, residual pre-norm blocks, GELU feed-forward layers, dropout, and a
next-character linear head. A trainable previous-character table provides a
baseline. There are no pretrained weights or pretrained tokenizer.
Configuration names (`tiny`, `small`, `medium`) are shapes, not trained
foundation models. Actual counts and metrics are created by experiment runs,
not embedded estimates.

## Evaluation

Training and validation loss, character-level perplexity, parameters,
duration, and optional held-out test loss are recorded. Inference benchmarks
report methodology and host metadata. Character perplexity is not comparable
directly to word/subword perplexity. The synthetic fixture supports software
smoke tests only and is not a financial corpus.

## Risks and limitations

The ECB feed is a small, rolling sample of institutional commentary, not
independent or representative market news. The model may reproduce source
biases, confuse dates and amounts, hallucinate facts, and produce malformed
text. It has no guarantee of current information or factual grounding. Users
must not treat output as verified financial information or advice.
