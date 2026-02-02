from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class TraceStep:
    state: Dict[str, Any]
    action: Dict[str, Any]

class EpisodeTrace:
    def __init__(self, maxlen: int = 8):
        self.steps = deque(maxlen=maxlen)

    def record(self, state_snapshot: Dict[str, Any], action: Dict[str, Any]):
        self.steps.append(
            TraceStep(state=state_snapshot, action=action)
        )

    def last(self) -> TraceStep | None:
        return self.steps[-1] if self.steps else None
