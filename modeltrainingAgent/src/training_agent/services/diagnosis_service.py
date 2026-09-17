from __future__ import annotations

from training_agent.models import StructuredDiagnosis


class NoOpDiagnosisService:
    def diagnose(self, *_args: object, **_kwargs: object) -> StructuredDiagnosis:
        return StructuredDiagnosis()
