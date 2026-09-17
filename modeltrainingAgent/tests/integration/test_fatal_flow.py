from __future__ import annotations

from pathlib import Path

from training_agent.graph import build_graph, build_services
from training_agent.state import build_initial_state


def test_bad_jsonl_writes_failure_report(config_factory, tmp_path: Path) -> None:
    cfg = config_factory()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": 1\n', encoding="utf-8")
    cfg.task.dataset_uri = str(bad)
    services = build_services(cfg)
    final = build_graph(services).invoke(build_initial_state(cfg), config={"recursion_limit": 100})
    assert final["workflow_status"] == "failed"
    assert Path(final["final_report_uri"]).exists()
    assert services.repository.list_trials(cfg.task.task_id) == []
