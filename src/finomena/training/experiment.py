"""Durable JSONL experiment history."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_experiment_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def load_experiments(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: dict[str, dict[str, Any]] = {}
    with file_path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[record["experiment_id"]] = record
    return list(records.values())


def upsert_experiment(record: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    records = {
        item["experiment_id"]: item for item in load_experiments(file_path)
    }
    records[record["experiment_id"]] = record
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for item in records.values():
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, file_path)
