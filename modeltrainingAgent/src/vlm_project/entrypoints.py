"""
Stable VLM training entrypoints.

External orchestrators should use this module.
"""

from __future__ import annotations

from vlm_project.adapters.training_agent_entrypoints import (
    build_loss,
    build_model,
    build_optimizer,
    build_scheduler,
)


__all__ = [
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
]
