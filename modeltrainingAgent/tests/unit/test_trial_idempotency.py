from __future__ import annotations

import pytest

from training_agent.graph import build_services
from training_agent.models import TrialRecord
from training_agent.nodes.trials import complete_optuna_trial, suggest_trial
from training_agent.state import build_initial_state
from training_agent.trainers.base import stable_params_hash


def test_suggest_replay_does_not_create_duplicate_trial(config_factory) -> None:
    cfg = config_factory()
    services = build_services(cfg)
    state = build_initial_state(cfg)
    services.optuna.ensure_study()
    update1 = suggest_trial(state, services)
    state.update(update1)
    update2 = suggest_trial(state, services)
    assert update2 == {}


def test_repository_rejects_param_overwrite(config_factory) -> None:
    services = build_services(config_factory())
    record = TrialRecord(
        task_id="t",
        trial_number=1,
        request_id="r",
        params={"x": 1},
        runtime_params={},
        params_hash=stable_params_hash({"x": 1}),
    )
    services.repository.create_or_get_trial(record)
    with pytest.raises(ValueError):
        services.repository.create_or_get_trial(record.model_copy(update={"params_hash": "bad"}))


def test_complete_optuna_replay_is_idempotent(config_factory) -> None:
    cfg = config_factory()
    services = build_services(cfg)
    state = build_initial_state(cfg)
    state.update(suggest_trial(state, services))
    assert state["trial_number"] is not None
    state["objective_value"] = 0.8
    state["trial_best_metrics"] = {"val_f1": 0.8}
    state["trial_best_checkpoint_uri"] = "ckpt"
    complete_optuna_trial(state, services)
    complete_optuna_trial(state, services)
