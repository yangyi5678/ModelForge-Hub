from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.checkpoint.sqlite import SqliteSaver

from training_agent.config import AppConfig
from training_agent.graph import build_graph, build_services
from training_agent.services.trial_repository import SQLiteTrialRepository
from training_agent.state import build_initial_state

app = typer.Typer(no_args_is_help=True)


def _thread(task_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": task_id}, "recursion_limit": 200}


@app.command()
def run(config: Annotated[Path, typer.Option("--config", exists=True, readable=True)]) -> None:
    cfg = AppConfig.from_yaml(config)
    services = build_services(cfg, str(config))
    with _sqlite_checkpointer(cfg) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        state = build_initial_state(cfg)
        final = graph.invoke(state, config=_thread(cfg.task.task_id))
    typer.echo(json.dumps(_summary(final), indent=2, sort_keys=True))
    if final["workflow_status"] == "failed":
        raise typer.Exit(1)


@app.command()
def resume(
    task_id: Annotated[str, typer.Option("--task-id")],
    repository_uri: Path = Path("artifacts/trials.sqlite"),
) -> None:
    repo = SQLiteTrialRepository(repository_uri)
    config_path = repo.get_task_config_path(task_id)
    if config_path is None:
        typer.echo("No registered config path for task.", err=True)
        raise typer.Exit(1)
    cfg = AppConfig.from_yaml(config_path)
    services = build_services(cfg, config_path)
    with _sqlite_checkpointer(cfg) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        state = build_initial_state(cfg)
        final = graph.invoke(state, config=_thread(task_id))
    typer.echo(json.dumps(_summary(final), indent=2, sort_keys=True))
    if final["workflow_status"] == "failed":
        raise typer.Exit(1)


@app.command()
def status(
    task_id: Annotated[str, typer.Option("--task-id")],
    repository_uri: Path = Path("artifacts/trials.sqlite"),
) -> None:
    repo = SQLiteTrialRepository(repository_uri)
    trials = repo.list_trials(task_id)
    typer.echo(
        json.dumps({"task_id": task_id, "trial_count": len(trials), "trials": trials}, indent=2)
    )


@app.command()
def trials(
    task_id: Annotated[str, typer.Option("--task-id")],
    repository_uri: Path = Path("artifacts/trials.sqlite"),
) -> None:
    repo = SQLiteTrialRepository(repository_uri)
    typer.echo(json.dumps(repo.list_trials(task_id), indent=2, sort_keys=True))


@app.command()
def report(
    task_id: Annotated[str, typer.Option("--task-id")],
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    candidates = [
        artifacts_dir / task_id / "final" / "report.json",
        artifacts_dir / task_id / "final" / "failure_report.json",
    ]
    for path in candidates:
        if path.exists():
            typer.echo(path.read_text(encoding="utf-8"))
            return
    typer.echo("No report found.", err=True)
    raise typer.Exit(1)


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": state["task_id"],
        "workflow_status": state["workflow_status"],
        "stop_reason": state["stop_reason"],
        "completed_trial_count": state["completed_trial_count"],
        "best_trial_number": state["best_trial_number"],
        "best_objective_value": state["best_objective_value"],
        "final_report_uri": state["final_report_uri"],
    }


def _sqlite_checkpointer(cfg: AppConfig) -> Any:
    uri = cfg.artifacts_dir + "/langgraph.db"
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(uri)


if __name__ == "__main__":
    app()
