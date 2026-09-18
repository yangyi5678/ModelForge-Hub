"""Backward-compatible entrypoints for external training orchestrators.

New integrations should prefer
``python://vlm_project.adapters.training_agent_entrypoints:<function>``.
"""
from __future__ import annotations

from typing import Any

from vlm_project.adapters.training_agent_entrypoints import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)

__all__ = ["build_model", "build_loss", "build_optimizer", "build_scheduler"]
