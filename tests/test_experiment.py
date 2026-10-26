from finomena.training.experiment import load_experiments, upsert_experiment


def test_experiment_history_upserts_durable_jsonl_records(tmp_path) -> None:
    path = tmp_path / "experiments.jsonl"
    first = {"experiment_id": "run-1", "status": "running"}
    finished = {"experiment_id": "run-1", "status": "completed", "steps_completed": 3}
    other = {"experiment_id": "run-2", "status": "failed"}
    upsert_experiment(first, path)
    upsert_experiment(finished, path)
    upsert_experiment(other, path)
    records = load_experiments(path)
    assert records == [finished, other]
