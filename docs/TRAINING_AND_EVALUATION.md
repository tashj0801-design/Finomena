# Training and evaluation

The trainer samples a batch of contiguous character windows from training
tokens. Inputs and targets differ by one character: input positions predict
the next ID. Flattened cross-entropy is the mean negative log likelihood.
Autograd computes gradients; AdamW updates trainable parameters. Optional
gradient-norm clipping is on by default. Python, optional NumPy and PyTorch
random seeds are set before initialization and sampling.

At configured intervals, `estimate_loss` enters evaluation mode and inference
mode, samples batches from training and validation only, then restores the
model's former training state. The lowest observed validation loss chooses
the best state. Training reports final train/validation loss, best validation
step, actual wall duration and steps; test metrics remain unset until the
explicit evaluation command is used.

Test loss is computed from the test split only after a checkpoint has been
selected. Repeatedly adjusting a model based on test loss would leak test
information and invalidate that final evaluation. Cross-entropy is reported
in nats per character. Character-level perplexity is `exp(mean loss)` and is
not directly comparable to word or subword-token perplexity; overflow or
non-finite values are marked unavailable.

Checkpoint files preserve architecture, state dictionary, tokenizer,
optimizer state, step, seed, PyTorch RNG state, validation result, experiment
metadata and environment. Loading uses `torch.load(weights_only=True)` and
validates model type, configuration and vocabulary. Optimizer/RNG state is
saved for inspection, but a CLI resume-training workflow is not implemented;
checkpoints are for evaluation and generation, not exact training resumption.
Only checkpoints created by this project should be loaded.

Experiment records are kept in ignored `outputs/experiments.jsonl`. A run is
marked running before optimization and updated with completed/cancelled/failed
status. Test evaluation and benchmark results update that same experiment
record. A smoke run verifies pipeline mechanics, not meaningful learning.
