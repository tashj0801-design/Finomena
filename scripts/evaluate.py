from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finomena.data.dataset import encode_documents  # noqa: E402
from finomena.data.sources import read_jsonl  # noqa: E402
from finomena.environment import select_device  # noqa: E402
from finomena.evaluation.metrics import character_perplexity, parameter_count  # noqa: E402
from finomena.training.checkpoints import load_checkpoint  # noqa: E402
from finomena.training.experiment import load_experiments, upsert_experiment  # noqa: E402
from finomena.training.trainer import estimate_loss, seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on one held-out split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        parser.error("batches and batch-size must be positive")

    device = select_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    seed_everything(int(checkpoint["seed"]))
    split_path = ROOT / "data" / "processed" / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"prepared {args.split} split not found: {split_path}")
    documents = read_jsonl(split_path)
    tokens = encode_documents([document.text for document in documents], tokenizer)
    block_size = int(getattr(model, "block_size", 0))
    losses = estimate_loss(
        model, {args.split: tokens}, block_size, args.batch_size, args.batches, device
    )
    loss = losses[args.split]
    result: dict[str, Any] = {
        "split": args.split,
        "loss": loss,
        "character_perplexity": character_perplexity(loss),
        "documents": len(documents),
        "evaluation_batches": args.batches,
        "batch_size": args.batch_size,
        "device": str(device),
        "trainable_parameters": parameter_count(model),
        "checkpoint": str(args.checkpoint),
    }
    if args.split == "test":
        metadata = checkpoint.get("metadata", {})
        experiment_id = metadata.get("experiment_id") if isinstance(metadata, dict) else None
        records = load_experiments(ROOT / "outputs" / "experiments.jsonl")
        record = next((item for item in records if item["experiment_id"] == experiment_id), None)
        if record is not None:
            record["metrics"]["final_test_loss"] = loss
            record["metrics"]["final_test_character_perplexity"] = character_perplexity(loss)
            upsert_experiment(record, ROOT / "outputs" / "experiments.jsonl")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
