from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finomena.environment import collect_environment, select_device  # noqa: E402
from finomena.evaluation.benchmark import benchmark_model  # noqa: E402
from finomena.training.checkpoints import load_checkpoint  # noqa: E402
from finomena.training.experiment import load_experiments, upsert_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference from a real checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--generated-characters", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    started = time.perf_counter()
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started
    result = benchmark_model(
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        generated_characters=args.generated_characters,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result["model_load_seconds"] = load_seconds
    result["environment"] = collect_environment()
    result["model_type"] = checkpoint["model_type"]
    result["model_config"] = checkpoint["model_config"]
    result["checkpoint"] = str(args.checkpoint)
    output_dir = ROOT / "outputs" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_metadata = checkpoint.get("metadata", {})
    experiment_id = (
        experiment_metadata.get("experiment_id")
        if isinstance(experiment_metadata, dict)
        else None
    )
    result["experiment_id"] = experiment_id
    output_path = output_dir / (
        f"{experiment_id}.json" if experiment_id else f"benchmark-{time.time_ns()}.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if experiment_id:
        records = load_experiments(ROOT / "outputs" / "experiments.jsonl")
        record = next((item for item in records if item["experiment_id"] == experiment_id), None)
        if record is not None:
            record["benchmark"] = result
            upsert_experiment(record, ROOT / "outputs" / "experiments.jsonl")
    result["result_file"] = str(output_path.relative_to(ROOT))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
