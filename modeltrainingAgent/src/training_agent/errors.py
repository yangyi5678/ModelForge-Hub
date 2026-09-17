class RecoverableTrainingError(Exception):
    """Base class for errors that may be retried or repaired."""


class TrainingOOMError(RecoverableTrainingError):
    """Training ran out of memory."""


class TrainingInterruptedError(RecoverableTrainingError):
    """Training was interrupted after writing a checkpoint."""


class InvalidTrainingDataError(Exception):
    """Dataset failed validation."""


class InvalidTrialConfigError(Exception):
    """Trial parameters or runtime parameters are invalid."""


class FatalTrainingError(Exception):
    """Non-recoverable training failure."""
