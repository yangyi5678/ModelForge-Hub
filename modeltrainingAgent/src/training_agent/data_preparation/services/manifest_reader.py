from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training_agent.data_preparation.models import ManifestStats


def inspect_jsonl(path: str | Path) -> ManifestStats:
    fields: set[str] = set()
    sample_count = 0
    empty_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                empty_lines += 1
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            sample_count += 1
            fields.update(row.keys())
    return ManifestStats(
        path=str(path),
        sample_count=sample_count,
        fields=sorted(fields),
        empty_lines=empty_lines,
    )


def iter_jsonl(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield row
