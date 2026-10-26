from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finomena.config import (  # noqa: E402
    MODEL_PROFILES,
    ModelConfig,
    TrainingConfig,
    count_trainable_parameters,
)
from finomena.data.dataset import encode_documents  # noqa: E402
from finomena.data.sources import read_jsonl  # noqa: E402
from finomena.environment import collect_environment, select_device  # noqa: E402
from finomena.models.bigram import BigramLanguageModel  # noqa: E402
from finomena.models.gpt import GPTLanguageModel  # noqa: E402
from finomena.tokenizer.character_tokenizer import CharacterTokenizer  # noqa: E402
from finomena.training.checkpoints import save_checkpoint  # noqa: E402
from finomena.training.experiment import new_experiment_id, upsert_experiment  # noqa: E402
from finomena.training.trainer import seed_everything, train_model  # noqa: E402
from finomena.evaluation.metrics import character_perplexity  # noqa: E402


def _prepare_local_data(source: Path, seed: int) -> dict[str, Any]:
    from scripts.prepare_data import prepare

    report = prepare(source, max_documents=40, seed=seed)
    return report


def _load_split(name: str, tokenizer: CharacterTokenizer) -> torch.Tensor:
    rows = read_jsonl(ROOT / "data" / "processed" / f"{name}.jsonl")
    return encode_documents([row.text for row in rows], tokenizer)


def run_training(
    args: argparse.Namespace,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    if args.data:
        data_report = _prepare_local_data(args.data.resolve(), args.seed)
    else:
        manifest_path = ROOT / "data" / "processed" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Prepared dataset not found. Run scripts\\prepare_data.py or pass --data."
            )
        data_report = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_documents = read_jsonl(ROOT / "data" / "processed" / "train.jsonl")
    tokenizer = CharacterTokenizer.fit("\n\n".join(row.text for row in train_documents))
    seed_everything(args.seed)
    if args.model == "bigram":
        block_size = args.context_length or 64
        model = BigramLanguageModel(tokenizer.vocab_size, block_size)
    else:
        profile = args.profile or "tiny"
        profile_values = MODEL_PROFILES[profile].copy()
        overrides = {
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        }
        profile_values.update({key: value for key, value in overrides.items() if value is not None})
        if args.context_length is not None:
            profile_values["block_size"] = args.context_length
        model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **profile_values)
        model = GPTLanguageModel(model_config)

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        weight_decay=args.weight_decay,
        grad_clip=None if args.grad_clip == 0 else args.grad_clip,
    )
    seed_everything(training_config.seed)
    device = select_device(args.device)
    train_tokens = _load_split("train", tokenizer)
    validation_tokens = _load_split("validation", tokenizer)
    experiment_id = new_experiment_id()
    model_name = args.model if args.model == "bigram" else f"gpt-{args.profile or 'tiny'}"
    experiment_path = ROOT / "outputs" / "experiments.jsonl"
    checkpoint_path = ROOT / "checkpoints" / f"{experiment_id}_best.pt"
    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_type": args.model,
        "model_config": (
            model.config.to_dict() if isinstance(model, GPTLanguageModel)
            else {"vocab_size": model.vocab_size, "block_size": model.block_size}
        ),
        "training_config": {
            "optimizer": "AdamW",
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "max_steps": training_config.max_steps,
            "eval_interval": training_config.eval_interval,
            "eval_batches": training_config.eval_batches,
            "seed": training_config.seed,
            "weight_decay": training_config.weight_decay,
            "grad_clip": training_config.grad_clip,
        },
        "dataset_report": data_report,
        "dataset_fingerprint": data_report["fingerprint_sha256"],
        "dataset_source": data_report["source"],
        "synthetic_dataset": data_report["synthetic"],
        "split_method": data_report["split_method"],
        "preprocessing": {
            "unicode_normalization": "NFC",
            "line_endings": "normalized to LF",
            "whitespace": "horizontal whitespace collapsed; excess blank lines reduced",
            "preserve_financial_notation": True,
            "empty_documents": "removed",
            "duplicates": "exact cleaned body duplicates removed before splitting",
            "tokenizer_vocabulary": "fit from training documents only",
        },
        "environment": collect_environment(),
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "status": "running",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "metrics": {},
        "errors": [],
        "warnings": (
            ["The dataset contains fewer than 30 documents; metrics are exploratory only."]
            if data_report["cleaning"]["documents_retained"] < 30
            else []
        ),
    }
    upsert_experiment(record, experiment_path)

    def report_progress(update: dict[str, Any]) -> None:
        if progress_callback is None:
            print(json.dumps(update), flush=True)
        else:
            progress_callback(update)

    try:
        result = train_model(
            model,
            train_tokens,
            validation_tokens,
            training_config,
            device,
            callback=report_progress,
        )
        latest_path = ROOT / "checkpoints" / f"{args.model}_latest.pt"
        record["status"] = result.status
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        record["steps_completed"] = result.steps_completed
        record["duration_seconds"] = result.duration_seconds
        record["best_validation_step"] = result.best_step
        record["history"] = result.history
        record["metrics"] = {
            "final_train_loss": result.train_loss,
            "final_train_character_perplexity": character_perplexity(result.train_loss),
            "final_validation_loss": result.validation_loss,
            "final_validation_character_perplexity": character_perplexity(
                result.validation_loss
            ),
            "best_validation_loss": result.best_validation_loss,
            "final_test_loss": None,
            "final_test_character_perplexity": None,
        }
        record["latest_checkpoint"] = str(latest_path.relative_to(ROOT))
        model.load_state_dict(result.best_model_state)
        save_checkpoint(
            checkpoint_path,
            model_type=args.model,
            model=model,
            tokenizer=tokenizer,
            seed=training_config.seed,
            step=result.best_step,
            best_validation_loss=result.best_validation_loss,
            metadata=record.copy(),
            optimizer_state=result.best_optimizer_state,
        )
        model.load_state_dict(result.final_model_state)
        save_checkpoint(
            latest_path,
            model_type=args.model,
            model=model,
            tokenizer=tokenizer,
            seed=training_config.seed,
            step=result.steps_completed,
            best_validation_loss=result.best_validation_loss,
            metadata=record.copy(),
            optimizer_state=result.final_optimizer_state,
        )
    except Exception as error:
        record["status"] = "failed"
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        record["errors"] = [f"{type(error).__name__}: {error}"]
        upsert_experiment(record, experiment_path)
        raise
    upsert_experiment(record, experiment_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Finomena character language model.")
    parser.add_argument("--model", choices=("bigram", "gpt"), default="gpt")
    parser.add_argument("--profile", choices=tuple(MODEL_PROFILES), default="tiny")
    parser.add_argument("--data", type=Path, help="Optional local JSONL input (including synthetic fixture).")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Use 0 to disable clipping.")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_training(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wall_clock_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
