"""Standard entrypoints for external training orchestrators.

This module is a thin facade over the VLM project's internal builders. It is
safe to import without importing torch/transformers; heavy dependencies are
loaded only when a build function is called.
"""
from __future__ import annotations

from typing import Any


def build_model(cfg: dict[str, Any]):
    from vlm_project.model import build_model as _build_model

    return _build_model(cfg)


def build_loss(cfg: dict[str, Any]):
    from vlm_project.loss import build_loss as _build_loss

    return _build_loss(cfg)


def build_optimizer(model, cfg: dict[str, Any]):
    from vlm_project.optimizer import build_optimizer as _build_optimizer

    return _build_optimizer(model, cfg)


def build_scheduler(optimizer, cfg: dict[str, Any], num_training_steps: int):
    from vlm_project.scheduler import build_scheduler as _build_scheduler

    return _build_scheduler(optimizer, cfg, num_training_steps)


__all__ = ["build_model", "build_loss", "build_optimizer", "build_scheduler"]
