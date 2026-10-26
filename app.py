from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from finomena.config import MODEL_PROFILES, TrainingConfig, count_trainable_parameters, model_config_for_profile  # noqa: E402
from finomena.data.dataset import encode_documents  # noqa: E402
from finomena.data.sources import read_jsonl  # noqa: E402
from finomena.environment import collect_environment, select_device  # noqa: E402
from finomena.evaluation.benchmark import benchmark_model  # noqa: E402
from finomena.evaluation.metrics import character_perplexity  # noqa: E402
from finomena.generation.generate import generate_text  # noqa: E402
from finomena.models.bigram import BigramLanguageModel  # noqa: E402
from finomena.models.gpt import GPTLanguageModel  # noqa: E402
from finomena.tokenizer.character_tokenizer import CharacterTokenizer  # noqa: E402
from finomena.training.checkpoints import load_checkpoint  # noqa: E402
from finomena.training.experiment import load_experiments, upsert_experiment  # noqa: E402
from scripts.prepare_data import prepare  # noqa: E402
from scripts.train import run_training  # noqa: E402


DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
EXPERIMENTS_FILE = ROOT / "outputs" / "experiments.jsonl"


def _experiments() -> list[dict[str, Any]]:
    return load_experiments(EXPERIMENTS_FILE)


def _dataset_manifest() -> dict[str, Any] | None:
    path = DATA_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _checkpoint_paths() -> list[Path]:
    return sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)


def _show_overview() -> None:
    st.header("Finomena | Neural Financial Text Lab")
    st.caption("Where Finance Meets Neural Intelligence")
    st.warning(
        "Educational experiment only. Generated text is not verified news, "
        "market analysis, or financial advice."
    )
    environment = collect_environment()
    experiments = _experiments()
    manifest = _dataset_manifest()
    checkpoints = _checkpoint_paths()
    left, middle, right = st.columns(3)
    left.metric("Dataset", "Prepared" if manifest else "Not prepared")
    middle.metric("Checkpoints", len(checkpoints))
    right.metric("Last experiment", experiments[-1]["status"] if experiments else "Not started")
    st.subheader("Environment")
    st.json(environment)
    st.caption(
        "This character model is trained from random initialization. A successful "
        "smoke run demonstrates code flow, not reliable financial-language quality."
    )


def _show_dataset() -> None:
    st.header("Dataset")
    st.markdown(
        "The acquisition workflow uses the ECB English press-release RSS feed. "
        "It filters out speeches and interviews, keeps source attribution, and "
        "downloads only when you request preparation."
    )
    st.markdown(
        "[ECB feed](https://www.ecb.europa.eu/rss/press.html) · "
        "[ECB reuse terms](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)"
    )
    if st.button("Prepare / refresh dataset", key="prepare_data"):
        try:
            with st.spinner("Fetching feed and preparing document splits..."):
                report = prepare(None, max_documents=40, seed=1337)
            st.success("Dataset preparation completed.")
            st.json(report)
        except Exception as error:
            st.error(f"Dataset preparation failed: {type(error).__name__}: {error}")
    manifest = _dataset_manifest()
    if manifest:
        cleaning = manifest["cleaning"]
        st.subheader("Latest cleaning and split report")
        st.write(
            f"{cleaning['documents_retained']} documents retained from "
            f"{cleaning['documents_read']} read; {cleaning['empty_removed']} empty "
            f"and {cleaning['duplicates_removed']} duplicate documents removed."
        )
        st.write(f"Split method: {manifest['split_method']}")
        st.write(f"Training vocabulary size (including unknown): {manifest['training_vocabulary_size_including_unk']}")
        st.json({"documents": manifest["split_documents"], "characters": manifest["split_characters"]})
    else:
        st.info("No prepared dataset yet. Prepare it here or use the CLI instructions in the README.")


def _show_model_configuration() -> None:
    st.header("Model configuration")
    manifest = _dataset_manifest()
    if not manifest:
        st.info("Prepare a dataset to estimate a model's vocabulary-dependent parameter count.")
        return
    vocabulary_size = int(manifest["training_vocabulary_size_including_unk"])
    profile = st.selectbox("Transformer profile", tuple(MODEL_PROFILES), index=0)
    config = model_config_for_profile(profile, vocabulary_size)
    model = GPTLanguageModel(config)
    st.json(
        {
            "profile": profile,
            **config.to_dict(),
            "estimated_trainable_parameters": count_trainable_parameters(model),
        }
    )
    environment = collect_environment()
    if not environment["cuda_available"] and profile != "tiny":
        st.warning("This machine has no CUDA GPU. Larger profiles may train slowly on CPU.")
    if count_trainable_parameters(model) >= 5_000_000:
        st.warning("This profile has millions of parameters; memory and training time may be substantial.")
    st.caption("Profiles describe model shapes, not separate pretrained models.")


def _show_training() -> None:
    st.header("Training")
    if not _dataset_manifest():
        st.info("Prepare the ECB data first, or use the offline synthetic fixture through the CLI.")
        return
    model_type = st.selectbox("Model", ("bigram", "gpt"))
    profile = st.selectbox("Transformer profile", tuple(MODEL_PROFILES), index=0, disabled=model_type == "bigram")
    custom_dimensions = model_type == "gpt" and st.checkbox("Override Transformer dimensions")
    n_embd = n_head = n_layer = dropout = None
    if custom_dimensions:
        defaults = MODEL_PROFILES[profile]
        dimension_columns = st.columns(4)
        n_embd = dimension_columns[0].number_input(
            "Embedding dimension", min_value=8, max_value=512,
            value=int(defaults["n_embd"]), step=8,
        )
        n_head = dimension_columns[1].number_input(
            "Attention heads", min_value=1, max_value=16,
            value=int(defaults["n_head"]),
        )
        n_layer = dimension_columns[2].number_input(
            "Transformer blocks", min_value=1, max_value=12,
            value=int(defaults["n_layer"]),
        )
        dropout = dimension_columns[3].number_input(
            "Dropout", min_value=0.0, max_value=0.95,
            value=float(defaults["dropout"]),
        )
    columns = st.columns(3)
    steps = columns[0].number_input("Training steps", min_value=1, max_value=100_000, value=20)
    batch_size = columns[1].number_input("Batch size", min_value=1, max_value=512, value=4)
    context = columns[2].number_input("Context length", min_value=8, max_value=512, value=64)
    columns = st.columns(3)
    learning_rate = columns[0].number_input("Learning rate", min_value=0.000001, max_value=1.0, value=0.0003, format="%.6f")
    weight_decay = columns[1].number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.01)
    grad_clip = columns[2].number_input("Gradient clipping (0 disables)", min_value=0.0, max_value=10.0, value=1.0)
    columns = st.columns(3)
    eval_interval = columns[0].number_input("Evaluation interval", min_value=1, max_value=10_000, value=10)
    eval_batches = columns[1].number_input("Evaluation batches", min_value=1, max_value=100, value=5)
    seed = int(columns[2].number_input("Random seed", min_value=0, max_value=2_147_483_647, value=1337))
    available_devices = ("auto", "cpu", "cuda") if collect_environment()["cuda_available"] else ("auto", "cpu")
    device = st.selectbox("Device", available_devices, index=0)
    if model_type == "gpt" and not collect_environment()["cuda_available"]:
        configured_dimensions = (
            int(n_embd) if n_embd is not None else int(MODEL_PROFILES[profile]["n_embd"]),
            int(n_layer) if n_layer is not None else int(MODEL_PROFILES[profile]["n_layer"]),
        )
        if (
            profile != "tiny"
            or configured_dimensions[0] >= 256
            or configured_dimensions[1] >= 6
            or int(context) >= 256
            or int(batch_size) * int(context) >= 8192
        ):
            st.warning(
                "This selected configuration may be slow or memory-intensive on CPU. "
                "Start with a smaller context and batch."
            )

    if st.session_state.get("training_status") == "running":
        st.warning("A training run is already active in this session.")
    if st.button(
        "Start training",
        type="primary",
        disabled=st.session_state.get("training_status") == "running",
    ):
        st.session_state.training_status = "running"
        progress = st.progress(0.0)
        status = st.empty()

        def update_progress(values: dict[str, Any]) -> None:
            progress.progress(min(1.0, values["step"] / values["max_steps"]))
            status.write(
                f"Step {values['step']}/{values['max_steps']} · "
                f"train loss {values['train_loss']:.4f} · "
                f"validation loss {values['validation_loss']:.4f} · "
                f"{values['elapsed_seconds']:.1f}s"
            )

        args = argparse.Namespace(
            data=None,
            model=model_type,
            profile=profile,
            context_length=int(context),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            grad_clip=float(grad_clip),
            steps=int(steps),
            eval_interval=int(eval_interval),
            eval_batches=int(eval_batches),
            seed=seed,
            device=device,
            n_embd=None if n_embd is None else int(n_embd),
            n_head=None if n_head is None else int(n_head),
            n_layer=None if n_layer is None else int(n_layer),
            dropout=None if dropout is None else float(dropout),
        )
        try:
            with st.spinner("Training synchronously; do not close this page."):
                result = run_training(args, progress_callback=update_progress)
            st.session_state.training_status = result["status"]
            st.success(f"Run {result['status']}: {result['experiment_id']}")
            st.json(result["metrics"])
        except Exception as error:
            st.session_state.training_status = "failed"
            st.error(f"Training failed: {type(error).__name__}: {error}")
        finally:
            st.session_state.pop("training_status", None)


def _show_generation() -> None:
    st.header("Text generation")
    paths = _checkpoint_paths()
    if not paths:
        st.info("No trained checkpoint exists. Complete a training run first.")
        return
    selected_name = st.selectbox("Trained checkpoint", [path.name for path in paths])
    prompt = st.text_area("Prompt (may be empty)", value="The financial outlook")
    available_devices = ("auto", "cpu", "cuda") if collect_environment()["cuda_available"] else ("auto", "cpu")
    device = st.selectbox("Device", available_devices, index=0, key="generation_device")
    columns = st.columns(4)
    length = columns[0].number_input("New characters", min_value=0, max_value=1000, value=120)
    temperature = columns[1].number_input("Temperature", min_value=0.01, max_value=5.0, value=0.8)
    top_k_value = columns[2].number_input("Top-k (0 disables)", min_value=0, max_value=512, value=20)
    seed_value = columns[3].number_input("Seed (-1 is random)", min_value=-1, max_value=2_147_483_647, value=123)
    generate_column, clear_column = st.columns(2)
    should_generate = generate_column.button("Generate", type="primary")
    if clear_column.button("Clear output"):
        st.session_state.generated_text = ""
    if should_generate:
        try:
            device = select_device(device)
            model, tokenizer, checkpoint = load_checkpoint(CHECKPOINT_DIR / selected_name, device)
            generated = generate_text(
                model,
                tokenizer,
                prompt=prompt,
                max_new_chars=int(length),
                temperature=float(temperature),
                top_k=None if top_k_value == 0 else int(top_k_value),
                seed=None if seed_value < 0 else int(seed_value),
            )
            st.session_state.generated_text = prompt + generated
            st.session_state.generation_label = (
                f"{checkpoint['model_type']} · {checkpoint['step']} training steps · "
                "May be inaccurate, nonsensical, or contain misspellings. Not financial advice."
            )
        except Exception as error:
            st.error(f"Generation failed: {type(error).__name__}: {error}")
    if st.session_state.get("generated_text"):
        st.subheader("Model-generated experimental text")
        st.code(st.session_state.generated_text)
        st.caption(st.session_state.get("generation_label", "Experimental output; not financial advice."))


def _show_comparison() -> None:
    st.header("Model comparison")
    checkpoint_paths = _checkpoint_paths()
    if checkpoint_paths:
        selected_checkpoint = st.selectbox(
            "Checkpoint to benchmark",
            [path.name for path in checkpoint_paths],
            key="benchmark_checkpoint",
        )
        if st.button("Measure inference performance"):
            checkpoint_path = CHECKPOINT_DIR / selected_checkpoint
            try:
                started = time.perf_counter()
                device = select_device("auto")
                model, tokenizer, payload = load_checkpoint(checkpoint_path, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                load_seconds = time.perf_counter() - started
                measured = benchmark_model(model, tokenizer, device)
                measured["model_load_seconds"] = load_seconds
                measured["environment"] = collect_environment()
                measured["model_type"] = payload["model_type"]
                measured["model_config"] = payload["model_config"]
                metadata = payload.get("metadata", {})
                experiment_id = (
                    metadata.get("experiment_id") if isinstance(metadata, dict) else None
                )
                measured["experiment_id"] = experiment_id
                if experiment_id:
                    record = next(
                        (item for item in _experiments() if item["experiment_id"] == experiment_id),
                        None,
                    )
                    if record:
                        record["benchmark"] = measured
                        upsert_experiment(record, EXPERIMENTS_FILE)
                st.success("Benchmark completed; raw methodology and results saved to experiment history.")
                st.json(measured)
            except Exception as error:
                st.error(f"Benchmark failed: {type(error).__name__}: {error}")
    records = [record for record in _experiments() if record.get("status") == "completed"]
    if not records:
        st.info("No completed experiments yet; comparison values will appear after real training.")
        return
    rows = []
    for record in records:
        metrics = record.get("metrics", {})
        benchmark = record.get("benchmark", {})
        forward = benchmark.get("forward_pass", {})
        generation = benchmark.get("autoregressive_generation", {})
        rows.append(
            {
                "experiment": record["experiment_id"],
                "model": record["model_name"],
                "dataset": record.get("dataset_source"),
                "synthetic": record.get("synthetic_dataset", False),
                "warnings": "; ".join(record.get("warnings", [])),
                "train loss": metrics.get("final_train_loss"),
                "train char perplexity": metrics.get("final_train_character_perplexity"),
                "validation loss": metrics.get("final_validation_loss"),
                "validation char perplexity": metrics.get("final_validation_character_perplexity"),
                "test loss": metrics.get("final_test_loss"),
                "test char perplexity": metrics.get("final_test_character_perplexity"),
                "trainable parameters": record.get("trainable_parameters"),
                "training seconds": record.get("duration_seconds"),
                "forward mean ms": (
                    1000 * forward["mean_seconds"] if forward.get("mean_seconds") is not None else None
                ),
                "generation mean ms": (
                    1000 * generation["mean_seconds"] if generation.get("mean_seconds") is not None else None
                ),
                "peak GPU memory bytes": benchmark.get("peak_gpu_memory_bytes"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Only available measurements are shown. Short CPU runs on small samples "
        "do not establish generalization; compare runs only under matching conditions."
    )


def _show_history() -> None:
    st.header("Experiment history")
    records = _experiments()
    if not records:
        st.info("No experiment records exist yet.")
        return
    st.dataframe(
        [
            {
                "id": row["experiment_id"],
                "model": row.get("model_name"),
                "dataset": row.get("dataset_source"),
                "synthetic": row.get("synthetic_dataset", False),
                "warnings": "; ".join(row.get("warnings", [])),
                "status": row.get("status"),
                "steps": row.get("steps_completed", 0),
                "train loss": row.get("metrics", {}).get("final_train_loss"),
                "validation loss": row.get("metrics", {}).get("final_validation_loss"),
                "test loss": row.get("metrics", {}).get("final_test_loss"),
            }
            for row in records
        ],
        width="stretch",
        hide_index=True,
    )
    for record in reversed(records):
        with st.expander(f"{record.get('model_name')} · {record['experiment_id']}"):
            st.json(record)
    st.download_button(
        "Export experiment history as JSON",
        data=json.dumps(records, indent=2, ensure_ascii=False),
        file_name="finomena-experiments.json",
        mime="application/json",
    )
    buffer = io.StringIO()
    if records:
        keys = sorted({key for row in records for key in row})
        writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            flattened = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)
        st.download_button(
            "Export experiment history as CSV",
            data=buffer.getvalue(),
            file_name="finomena-experiments.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="Finomena | Neural Financial Text Lab", page_icon="📈", layout="wide")
    st.markdown(
        """<style>
        .stApp { background: #f5f7fb; }
        [data-testid="stMetric"] { background: white; padding: 1rem; border-radius: .7rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    section = st.sidebar.radio(
        "Finomena lab",
        ("Overview", "Dataset", "Model configuration", "Training", "Text generation", "Model comparison", "Experiment history"),
    )
    {
        "Overview": _show_overview,
        "Dataset": _show_dataset,
        "Model configuration": _show_model_configuration,
        "Training": _show_training,
        "Text generation": _show_generation,
        "Model comparison": _show_comparison,
        "Experiment history": _show_history,
    }[section]()


if __name__ == "__main__":
    main()
