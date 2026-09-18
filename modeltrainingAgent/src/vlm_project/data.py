from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_QUESTION = "请根据这张前视相机图像，分析驾驶场景、关键目标、自车意图，并给出驾驶决策建议。"


class VLMDataset(Dataset):
    def __init__(self, cfg: dict[str, Any], processor: Any) -> None:
        self.cfg = cfg
        self.processor = processor

        data_cfg = cfg["data"]
        manifest_path = data_cfg.get("train_jsonl_path")
        if not manifest_path:
            raise ValueError("data.train_jsonl_path is required")
        self.question = data_cfg.get("question", DEFAULT_QUESTION)
        self.samples = _read_jsonl(manifest_path)
        if not self.samples:
            raise ValueError(f"empty VLM dataset manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        image_path = _required_string(row, "local_image_path")
        annotation_path = _required_string(row, "local_annotation_path")

        image = Image.open(image_path).convert("RGB")
        annotation = _read_json(annotation_path)
        answer = _annotation_to_answer(annotation)
        return {
            "image": image,
            "question": self.question,
            "answer": answer,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} in {path} is not a JSON object")
            rows.append(row)
    return rows


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected at {path}")
    return value


def _annotation_to_answer(annotation: dict[str, Any]) -> str:
    annotations = annotation.get("annotations", {})
    if not isinstance(annotations, dict):
        raise ValueError("annotation missing object field: annotations")

    description = annotations.get("描述")
    if isinstance(description, str) and description.strip():
        return description.strip()

    decision = annotations.get("决策", {})
    if isinstance(decision, dict):
        actions = decision.get("元动作")
        if isinstance(actions, list) and actions:
            return "建议动作：" + "、".join(str(action) for action in actions)

    raise ValueError("annotation missing usable answer field: annotations.描述")


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"localized manifest row missing required string field: {key}")
    return value
