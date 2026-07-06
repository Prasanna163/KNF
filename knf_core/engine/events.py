from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class EventKind(str, Enum):
    BATCH_STARTED = "batch_started"
    FILE_STARTED = "file_started"
    FILE_SUCCEEDED = "file_succeeded"
    FILE_FAILED = "file_failed"
    FILE_SKIPPED = "file_skipped"
    FILE_STOPPED = "file_stopped"
    BATCH_FINISHED = "batch_finished"


@dataclass(frozen=True)
class JobEvent:
    kind: EventKind
    input_file: str | None = None
    message: str | None = None
    completed: int | None = None
    total: int | None = None
    elapsed_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class OnEvent(Protocol):
    def __call__(self, event: JobEvent) -> None:
        ...
