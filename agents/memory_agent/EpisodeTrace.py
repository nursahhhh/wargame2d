from collections import deque
from dataclasses import dataclass
from typing import Any, Dict
from enum import Enum
import uuid
import time
import json
from pathlib import Path


@dataclass
class TraceStep:
    state: Dict[str, Any]
    action: Dict[str, Any]


class EpisodeTrace:
    def __init__(self, maxlen: int = 8):
        self.steps = deque(maxlen=maxlen)

    def record(self, state_snapshot: Dict[str, Any], action: Dict[str, Any]):
        self.steps.append(TraceStep(state=state_snapshot, action=action))

    def last(self) -> TraceStep | None:
        return self.steps[-1] if self.steps else None


class FailureType(str, Enum):
    ALLY_LOSS = "ALLY_LOSS"
    DETECTED = "DETECTED"
    MISSION_FAILURE = "MISSION_FAILURE"
    STRATEGIC_DEADEND = "STRATEGIC_DEADEND"


def build_failure_record(
    failure_type: FailureType,
    trace_step: TraceStep,
    outcome: str,
    confidence: float,
):
    return {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "failure_type": failure_type.value,
        "confidence": round(confidence, 2),
        "context": trace_step.state,
        "action_taken": trace_step.action,
        "observed_outcome": outcome,
        "avoid_rule": None
    }


class MemoryStore:
    def __init__(self, path: str = "failure_memory.jsonl"):
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def write(self, record: Dict[str, Any]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load_all(self):
        with self.path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
