from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training_agent.state import TrainState, event, now_iso


def load_dataset(state: TrainState, *_args: object, **_kwargs: object) -> dict[str, Any]:
    if state["dataset_summary"] and state["train_dataset_uri"] and state["validation_dataset_uri"]:
        return {
            "updated_at": now_iso(),
            "events": [event("load_dataset", "skipped", "dataset already prepared")],
        }
    path = Path(state["dataset_uri"])
    if not path.exists():
        return _fatal("load_dataset", "dataset file does not exist")
    count = 0
    fields: set[str] = set()
    empty = 0
    labels: set[str] = set()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as raw:
            for line_number, raw_line in enumerate(raw, start=1):
                digest.update(raw_line)
                line = raw_line.decode("utf-8")
                if not line.strip():
                    empty += 1
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                count += 1
                fields.update(row.keys())
                if "label" in row:
                    labels.add(str(row["label"]))
    except Exception as exc:
        return _fatal("load_dataset", f"invalid JSONL dataset: {exc}")
    summary = {
        "sample_count": count,
        "fields": sorted(fields),
        "empty_lines": empty,
        "label_candidates": sorted(labels)[:50],
    }
    return {
        "updated_at": now_iso(),
        "dataset_version": digest.hexdigest(),
        "dataset_summary": summary,
        "events": [event("load_dataset", "loaded", f"loaded {count} samples")],
    }


def classify_task(state: TrainState, *_args: object, **_kwargs: object) -> dict[str, Any]:
    if state["task_type"]:
        task_type = state["task_type"]
    elif "label" in state["dataset_summary"].get("fields", []):
        task_type = "classification"
    else:
        return _fatal("classify_task", "unable to infer task type")
    return {
        "updated_at": now_iso(),
        "task_type": task_type,
        "events": [event("classify_task", "classified", str(task_type))],
    }


def validate_inputs(state: TrainState, *_args: object, **_kwargs: object) -> dict[str, Any]:
    fields = set(state["dataset_summary"].get("fields", []))
    issues = []
    if state["task_type"] == "classification" and not {"text", "label"}.issubset(fields):
        issues.append("classification data requires text and label fields")
    if int(state["dataset_summary"].get("sample_count", 0)) < 3:
        issues.append("dataset requires at least 3 non-empty records")
    if state["objective_name"] not in {"val_f1", "val_loss"}:
        issues.append("mock evaluator supports val_f1 and val_loss objectives")
    if issues:
        return _fatal("validate_inputs", "; ".join(issues))
    return {
        "updated_at": now_iso(),
        "input_validation_passed": True,
        "events": [event("validate_inputs", "passed", "input validation passed")],
    }


def _fatal(node: str, message: str) -> dict[str, Any]:
    return {
        "updated_at": now_iso(),
        "workflow_status": "failed",
        "execution_status": "fatal_error",
        "error_type": "invalid_data",
        "error_message": message,
        "error_node": node,
        "retryable": False,
        "stop_reason": "fatal_error",
        "should_stop": True,
        "events": [event(node, "failed", message)],
    }
