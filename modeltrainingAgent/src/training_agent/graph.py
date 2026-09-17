from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from training_agent.config import AppConfig
from training_agent.executors.remote_fake import FakeRemoteExecutor
from training_agent.nodes.config_import import import_model_project_config_node
from training_agent.nodes.data import classify_task, load_dataset, validate_inputs
from training_agent.nodes.data_preparation import prepare_dataset_snapshot
from training_agent.nodes.diagnosis import diagnose_trial
from training_agent.nodes.evaluation import evaluate_trial
from training_agent.nodes.execution import monitor_job, run_training, submit_job
from training_agent.nodes.finalization import failure_report, finalize
from training_agent.nodes.preparation import prepare_runtime
from training_agent.nodes.trials import (
    complete_optuna_trial,
    persist_trial,
    repair_trial,
    resume_trial,
    suggest_trial,
    validate_trial,
)
from training_agent.routes import route_after_execution, route_after_trial, route_execution_mode
from training_agent.services.artifact_store import ArtifactStore
from training_agent.services.diagnosis_service import NoOpDiagnosisService
from training_agent.services.optuna_service import RDBOptunaService
from training_agent.services.trial_repository import SQLiteTrialRepository
from training_agent.state import TrainState
from training_agent.trainers.auto import AutoTrainerAdapter


def build_services(config: AppConfig, config_path: str | None = None) -> Any:
    artifacts = ArtifactStore(config.artifacts_dir)
    repository = SQLiteTrialRepository(config.repository_uri)
    if config_path is not None:
        repository.register_task(config.task.task_id, str(Path(config_path).resolve()))
    trainer = AutoTrainerAdapter()
    return __import__("training_agent.models").models.Services(
        optuna=RDBOptunaService(
            study_name=config.optimization.study_name,
            storage_url=config.optuna_storage_url(),
            direction=config.optimization.direction,
            sampler=config.optimization.sampler,
            seed=config.task.random_seed,
        ),
        repository=repository,
        artifacts=artifacts,
        trainer=trainer,
        remote=FakeRemoteExecutor(trainer),
        diagnosis=NoOpDiagnosisService(),
        config=config,
    )


def build_graph(services: Any, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(TrainState)
    graph.add_node("import_model_project_config", import_model_project_config_node)
    graph.add_node("prepare_dataset", partial(prepare_dataset_snapshot, services=services))
    graph.add_node("load_dataset", load_dataset)
    graph.add_node("classify_task", classify_task)
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("prepare_runtime", partial(prepare_runtime, services=services))
    graph.add_node("suggest_trial", partial(suggest_trial, services=services))
    graph.add_node("validate_trial", partial(validate_trial, services=services))
    graph.add_node("persist_trial", partial(persist_trial, services=services))
    graph.add_node("run_training", partial(run_training, services=services))
    graph.add_node("submit_job", partial(submit_job, services=services))
    graph.add_node("monitor_job", partial(monitor_job, services=services))
    graph.add_node("resume_trial", partial(resume_trial, services=services))
    graph.add_node("repair_trial", partial(repair_trial, services=services))
    graph.add_node("evaluate_trial", partial(evaluate_trial, services=services))
    graph.add_node("complete_optuna_trial", partial(complete_optuna_trial, services=services))
    graph.add_node("diagnose_trial", partial(diagnose_trial, services=services))
    graph.add_node("finalize", partial(finalize, services=services))
    graph.add_node("failure_report", partial(failure_report, services=services))

    graph.set_entry_point("import_model_project_config")
    graph.add_edge("import_model_project_config", "prepare_dataset")
    graph.add_conditional_edges("prepare_dataset", _fatal_or_next("load_dataset"))
    graph.add_conditional_edges("load_dataset", _fatal_or_next("classify_task"))
    graph.add_conditional_edges("classify_task", _fatal_or_next("validate_inputs"))
    graph.add_conditional_edges("validate_inputs", _fatal_or_next("prepare_runtime"))
    graph.add_conditional_edges("prepare_runtime", _fatal_or_next("suggest_trial"))
    graph.add_edge("suggest_trial", "validate_trial")
    graph.add_conditional_edges("validate_trial", _fatal_or_next("persist_trial"))
    graph.add_conditional_edges(
        "persist_trial",
        route_execution_mode,
        {"local": "run_training", "remote": "submit_job"},
    )
    graph.add_edge("submit_job", "monitor_job")
    graph.add_conditional_edges(
        "run_training",
        route_after_execution,
        {
            "completed": "evaluate_trial",
            "interrupted": "resume_trial",
            "recoverable": "repair_trial",
            "fatal": "failure_report",
        },
    )
    graph.add_conditional_edges(
        "monitor_job",
        route_after_execution,
        {
            "completed": "evaluate_trial",
            "interrupted": "resume_trial",
            "recoverable": "repair_trial",
            "fatal": "failure_report",
        },
    )
    graph.add_edge("resume_trial", "run_training")
    graph.add_edge("repair_trial", "validate_trial")
    graph.add_conditional_edges("evaluate_trial", _fatal_or_next("complete_optuna_trial"))
    graph.add_edge("complete_optuna_trial", "diagnose_trial")
    graph.add_conditional_edges(
        "diagnose_trial",
        route_after_trial,
        {
            "continue": "suggest_trial",
            "target_or_budget_reached": "finalize",
            "fatal": "failure_report",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("failure_report", END)
    return graph.compile(checkpointer=checkpointer)


def _fatal_or_next(next_node: str) -> Any:
    def route(state: TrainState) -> str:
        return "failure_report" if state["workflow_status"] == "failed" else next_node

    return route
