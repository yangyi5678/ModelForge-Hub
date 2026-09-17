from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        path = self.root / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def trial_dir(self, task_id: str, trial_number: int) -> Path:
        path = self.task_dir(task_id) / "trials" / f"trial_{trial_number:04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def attempt_dir(self, task_id: str, trial_number: int, attempt: int) -> Path:
        path = self.trial_dir(task_id, trial_number) / "attempts" / f"attempt_{attempt:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, payload: dict[str, Any]) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
        return str(path)

    def read_json(self, uri: str | Path) -> dict[str, Any]:
        with Path(uri).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("expected JSON object")
        return data

    def write_text(self, relative: str | Path, text: str) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return str(path)

    def trial_config_path(self, task_id: str, trial_number: int) -> Path:
        return self.trial_dir(task_id, trial_number) / "trial_config.json"

    def final_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "final"
        path.mkdir(parents=True, exist_ok=True)
        return path
