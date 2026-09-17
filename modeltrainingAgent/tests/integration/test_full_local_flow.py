from __future__ import annotations

from pathlib import Path

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_full_local_flow(config_factory) -> None:
    cfg = config_factory(max_trials=3, target_value=0.999)
    services = build_services(cfg)
    final = build_graph(services).invoke(build_initial_state(cfg), config={"recursion_limit": 200})
    assert final["workflow_status"] == "completed"
    assert final["completed_trial_count"] == 3
    assert final["best_trial_number"] is not None
    assert Path(final["final_report_uri"]).exists()
