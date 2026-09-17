from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_agent.data_preparation.graph import build_data_preparation_graph


def test_data_preparation_pipeline_resolves_local_snapshot(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(
        "\n".join(
            [
                '{"sample_id":"1","image_oss_path":"oss://bucket/a.jpg","annotation_oss_path":"oss://bucket/a.json"}',
                '{"sample_id":"2","image_oss_path":"oss://bucket/b.jpg","annotation_oss_path":"oss://bucket/b.json"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val.write_text(
        '{"sample_id":"3","image_oss_path":"oss://bucket/c.jpg","annotation_oss_path":"oss://bucket/c.json"}\n',
        encoding="utf-8",
    )
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "dataset_name": "vlm_dataset",
                "version": "v1",
                "schema_version": 1,
                "snapshot_id": "snap-001",
                "status": "completed",
                "result_uri": str(result),
                "train_manifest_uri": str(train),
                "val_manifest_uri": str(val),
                "train_sha256": _sha256(train),
                "val_sha256": _sha256(val),
                "train_count": 2,
                "val_count": 1,
                "summary": {"valid": 3},
            }
        ),
        encoding="utf-8",
    )

    final = build_data_preparation_graph().invoke(
        {
            "result_uri": str(result),
            "work_dir": str(tmp_path / "workspace"),
            "model_family": "qwen_vl",
            "events": [],
        }
    )

    assert not final.get("should_stop")
    assert final["dataset_name"] == "vlm_dataset"
    assert final["train_dataset_uri"].endswith("train_manifest.jsonl")
    assert final["validation_dataset_uri"].endswith("val_manifest.jsonl")
    assert final["dataset_summary"]["sample_count"] == 3
    assert Path(final["runtime_dataset_uri"]).exists()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
