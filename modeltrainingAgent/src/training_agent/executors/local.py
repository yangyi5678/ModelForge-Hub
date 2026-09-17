from __future__ import annotations

from training_agent.models import TrainerAdapter, TrainingContext, TrainingResult


class LocalExecutor:
    def __init__(self, trainer: TrainerAdapter) -> None:
        self.trainer = trainer

    def run(self, context: TrainingContext) -> TrainingResult:
        if context.resume_checkpoint_uri:
            return self.trainer.resume(context)
        return self.trainer.train(context)
