from __future__ import annotations

from training_agent.executors.local import LocalExecutor
from training_agent.models import RemoteJobStatus, RemoteTrainingRequest, TrainerAdapter


class FakeRemoteExecutor:
    def __init__(self, trainer: TrainerAdapter) -> None:
        self.local = LocalExecutor(trainer)
        self.jobs: dict[str, RemoteJobStatus] = {}
        self.submissions: dict[str, str] = {}

    def submit(self, request: RemoteTrainingRequest) -> str:
        if request.request_id in self.submissions:
            return self.submissions[request.request_id]
        job_id = f"fake-job-{len(self.submissions) + 1:04d}"
        self.submissions[request.request_id] = job_id
        result = self.local.run(request.context)
        self.jobs[job_id] = RemoteJobStatus(job_id=job_id, status=result.status, result=result)
        return job_id

    def status(self, job_id: str) -> RemoteJobStatus:
        return self.jobs.get(job_id, RemoteJobStatus(job_id=job_id, status="running"))

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id] = RemoteJobStatus(job_id=job_id, status="fatal_error")
