from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from training_agent.data_preparation.nodes.build_model_dataset import build_model_dataset_node
from training_agent.data_preparation.nodes.emit_runtime_dataset import emit_runtime_dataset_node
from training_agent.data_preparation.nodes.fetch_manifests import fetch_manifests_node
from training_agent.data_preparation.nodes.inspect_manifests import inspect_manifests_node
from training_agent.data_preparation.nodes.localize_assets import localize_assets_node
from training_agent.data_preparation.nodes.resolve_snapshot import resolve_snapshot_node
from training_agent.data_preparation.state import DataPreparationState


def build_data_preparation_graph(checkpointer: Any | None = None) -> Any:
    graph = StateGraph(DataPreparationState)
    graph.add_node("resolve_snapshot", resolve_snapshot_node)
    graph.add_node("fetch_manifests", fetch_manifests_node)
    graph.add_node("inspect_manifests", inspect_manifests_node)
    graph.add_node("localize_assets", localize_assets_node)
    graph.add_node("build_model_dataset", build_model_dataset_node)
    graph.add_node("emit_runtime_dataset", emit_runtime_dataset_node)

    graph.set_entry_point("resolve_snapshot")
    graph.add_conditional_edges("resolve_snapshot", _fatal_or_next("fetch_manifests"))
    graph.add_conditional_edges("fetch_manifests", _fatal_or_next("inspect_manifests"))
    graph.add_conditional_edges("inspect_manifests", _fatal_or_next("localize_assets"))
    graph.add_conditional_edges("localize_assets", _fatal_or_next("build_model_dataset"))
    graph.add_conditional_edges("build_model_dataset", _fatal_or_next("emit_runtime_dataset"))
    graph.add_edge("emit_runtime_dataset", END)
    return graph.compile(checkpointer=checkpointer)


def _fatal_or_next(next_node: str) -> Any:
    def route(state: DataPreparationState) -> str:
        return END if state.get("should_stop") else next_node

    return route

