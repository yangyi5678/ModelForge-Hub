from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState

from training_agent.config import ObjectiveDirection
from training_agent.models import BestTrial, TrialSuggestion


class RDBOptunaService:
    def __init__(
        self,
        study_name: str,
        storage_url: str,
        direction: ObjectiveDirection,
        sampler: str,
        seed: int,
    ) -> None:
        self.study_name = study_name
        self.storage_url = storage_url
        self.direction = direction
        self.sampler_name = sampler
        self.seed = seed
        self._study: optuna.Study | None = None

    def ensure_study(self) -> None:
        if self._study is not None:
            return
        self._ensure_sqlite_parent()
        sampler = (
            optuna.samplers.RandomSampler(seed=self.seed)
            if self.sampler_name == "random"
            else optuna.samplers.TPESampler(seed=self.seed)
        )
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=self.direction,
            sampler=sampler,
            load_if_exists=True,
        )

    @property
    def study(self) -> optuna.Study:
        self.ensure_study()
        assert self._study is not None
        return self._study

    def ask(self, request_id: str, search_space: dict[str, Any]) -> TrialSuggestion:
        reconciled = self.reconcile_running_trial(request_id)
        if reconciled is not None:
            return reconciled
        trial = self.study.ask()
        params: dict[str, Any] = {}
        for name, spec in search_space.items():
            if spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                    step=spec.get("step"),
                )
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(spec["low"]),
                    int(spec["high"]),
                    log=bool(spec.get("log", False)),
                    step=int(spec.get("step") or 1),
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, list(spec["choices"]))
            else:
                raise ValueError(f"unsupported search parameter type: {spec['type']}")
        trial.set_user_attr("request_id", request_id)
        return TrialSuggestion(request_id=request_id, trial_number=trial.number, params=params)

    def complete(self, trial_number: int, value: float) -> None:
        trial = self._frozen(trial_number)
        if trial.state == TrialState.COMPLETE:
            if trial.value != value:
                raise ValueError("completed Optuna trial value conflicts with replayed value")
            return
        if trial.state.is_finished():
            raise ValueError(f"cannot complete finished trial in state {trial.state}")
        self.study.tell(trial_number, value)

    def prune(self, trial_number: int, reason: str) -> None:
        trial = self._frozen(trial_number)
        if trial.state.is_finished():
            return
        self.study.tell(trial_number, state=TrialState.PRUNED)
        self.study._storage.set_trial_user_attr(trial._trial_id, "prune_reason", reason)

    def fail(self, trial_number: int, reason: str) -> None:
        trial = self._frozen(trial_number)
        if trial.state.is_finished():
            return
        self.study.tell(trial_number, state=TrialState.FAIL)
        self.study._storage.set_trial_user_attr(trial._trial_id, "fail_reason", reason)

    def best_trial(self) -> BestTrial | None:
        try:
            trial = self.study.best_trial
        except ValueError:
            return None
        if trial.value is None:
            return None
        return BestTrial(
            trial_number=trial.number, value=float(trial.value), params=dict(trial.params)
        )

    def reconcile_running_trial(self, request_id: str) -> TrialSuggestion | None:
        for trial in self.study.get_trials(deepcopy=False):
            if trial.user_attrs.get("request_id") == request_id and not trial.state.is_finished():
                return TrialSuggestion(
                    request_id=request_id,
                    trial_number=trial.number,
                    params=dict(trial.params),
                )
        return None

    def counts(self) -> dict[str, int]:
        counts = {"completed": 0, "failed": 0, "pruned": 0}
        for trial in self.study.get_trials(deepcopy=False):
            if trial.state == TrialState.COMPLETE:
                counts["completed"] += 1
            elif trial.state == TrialState.FAIL:
                counts["failed"] += 1
            elif trial.state == TrialState.PRUNED:
                counts["pruned"] += 1
        return counts

    def summaries(self) -> list[dict[str, Any]]:
        result = []
        for trial in self.study.get_trials(deepcopy=False):
            result.append(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": dict(trial.params),
                    "user_attrs": dict(trial.user_attrs),
                }
            )
        return result

    def _frozen(self, trial_number: int) -> optuna.trial.FrozenTrial:
        trials = [t for t in self.study.get_trials(deepcopy=False) if t.number == trial_number]
        if not trials:
            raise ValueError(f"unknown trial number {trial_number}")
        return trials[0]

    def _ensure_sqlite_parent(self) -> None:
        if not self.storage_url.startswith("sqlite:///"):
            return
        raw = self.storage_url.removeprefix("sqlite:///")
        if raw and raw != ":memory:":
            Path(raw).parent.mkdir(parents=True, exist_ok=True)
