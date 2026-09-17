from __future__ import annotations

from pathlib import Path

import pytest

from training_agent.config import AppConfig


@pytest.fixture()
def config_factory(tmp_path: Path):
    def make(
        *,
        task_id: str = "task-001",
        mode: str = "local",
        behavior: str = "completed",
        max_trials: int = 3,
        target_value: float | None = 0.99,
    ) -> AppConfig:
        data = tmp_path / "data.jsonl"
        data.write_text(
            "\n".join(
                [
                    '{"id":"1","text":"a","label":"x"}',
                    '{"id":"2","text":"b","label":"y"}',
                    '{"id":"3","text":"c","label":"x"}',
                    '{"id":"4","text":"d","label":"y"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return AppConfig.model_validate(
            {
                "task": {
                    "task_id": task_id,
                    "task_type": "classification",
                    "dataset_uri": str(data),
                    "base_checkpoint_uri": str(tmp_path / "base"),
                    "random_seed": 7,
                },
                "execution": {"mode": mode, "max_retries": 3, "mock_behavior": behavior},
                "optimization": {
                    "study_name": task_id,
                    "storage_key": "OPTUNA_STORAGE_MAIN",
                    "sampler": "random",
                    "direction": "maximize",
                    "objective": "val_f1",
                    "target_value": target_value,
                    "max_trials": max_trials,
                    "max_duration_seconds": 3600,
                    "patience": 10,
                    "min_delta": 0.001,
                    "search_space": {
                        "learning_rate": {
                            "type": "float",
                            "low": 1.0e-5,
                            "high": 1.0e-3,
                            "log": True,
                        },
                        "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
                        "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
                    },
                },
                "training": {"epochs": 4, "precision": "fp32"},
                "artifacts_dir": str(tmp_path / "artifacts"),
                "repository_uri": str(tmp_path / "repo.sqlite"),
            }
        )

    return make
