from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training_agent.data_preparation.services.uri import parse_uri


def local_asset_path(
    *,
    workspace: str | Path,
    split: str,
    asset_kind: str,
    source_uri: str,
    sample_id: str,
) -> Path:
    parsed = parse_uri(source_uri)
    suffix = Path(parsed.key).suffix
    if not suffix:
        suffix = ".bin"
    safe_id = _safe_filename(sample_id)
    return Path(workspace) / "files" / split / asset_kind / f"{safe_id}{suffix}"


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return destination


def _safe_filename(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "sample"

