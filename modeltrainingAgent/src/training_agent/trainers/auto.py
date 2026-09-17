from __future__ import annotations

from training_agent.models import (
    EvaluationContext,
    EvaluationResult,
    TrainingContext,
    TrainingResult,
    ValidationIssue,
)
from training_agent.trainers.mock import DeterministicMockTrainerAdapter
from training_agent.trainers.vlm import VLMTrainerAdapter


class AutoTrainerAdapter:
    def __init__(self) -> None:
        self.mock = DeterministicMockTrainerAdapter()
        self.vlm = VLMTrainerAdapter()

    def validate(self, context: TrainingContext) -> list[ValidationIssue]:
        return self._select(context).validate(context)

    def train(self, context: TrainingContext) -> TrainingResult:
        return self._select(context).train(context)

    def resume(self, context: TrainingContext) -> TrainingResult:
        return self._select(context).resume(context)

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        adapter = context.fixed_training_config.get("trainer_adapter", "mock")
        if adapter == "vlm":
            return self.vlm.evaluate(context)
        return self.mock.evaluate(context)

    def _select(self, context: TrainingContext):
        adapter = context.fixed_training_config.get("trainer_adapter", "mock")
        if adapter == "vlm":
            return self.vlm
        return self.mock
