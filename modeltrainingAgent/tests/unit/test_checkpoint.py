from __future__ import annotations

import pytest

from training_agent.errors import TrainingInterruptedError
from training_agent.models import TrainingContext
from training_agent.trainers.base import stable_params_hash
from training_agent.trainers.mock import DeterministicMockTrainerAdapter


def test_checkpoint_hash_mismatch_rejects_resume(tmp_path) -> None:
    trainer = DeterministicMockTrainerAdapter()
    params = {"learning_rate": 1e-4, "batch_size": 16}
    context = TrainingContext(
        task_id="t",
        trial_number=1,
        attempt=1,
        params=params,
        runtime_params={"micro_batch_size": 16},
        params_hash=stable_params_hash(params),
        base_checkpoint_uri="base",
        resume_checkpoint_uri=None,
        train_dataset_uri="data",
        validation_dataset_uri="data",
        total_epochs=2,
        artifact_dir=str(tmp_path),
    )
    result = trainer.train(context)
    bad = context.model_copy(
        update={"params_hash": "bad", "resume_checkpoint_uri": result.checkpoint_uri}
    )
    with pytest.raises(TrainingInterruptedError):
        trainer.resume(bad)
