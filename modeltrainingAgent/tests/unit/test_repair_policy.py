from __future__ import annotations

from training_agent.graph import build_services
from training_agent.nodes.trials import repair_trial
from training_agent.state import build_initial_state
from training_agent.trainers.base import effective_batch_size


def test_oom_repair_does_not_change_trial_params(config_factory) -> None:
    cfg = config_factory()
    services = build_services(cfg)
    state = build_initial_state(cfg)
    state.update(
        {
            "trial_number": 1,
            "trial_params": {"learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 32},
            "runtime_params": {"micro_batch_size": 16, "gradient_accumulation_steps": 2},
            "trial_attempt": 1,
            "error_type": "oom",
        }
    )
    before_params = dict(state["trial_params"])
    before_effective = effective_batch_size(state["trial_params"], state["runtime_params"])
    update = repair_trial(state, services)
    assert state["trial_params"] == before_params
    assert effective_batch_size(before_params, update["runtime_params"]) == before_effective
    assert update["trial_attempt"] == 2
