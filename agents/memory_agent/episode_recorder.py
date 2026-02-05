from collections import deque
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EpisodeStep:
    step: int
    world_before: Any
    actions: Dict[str, Any]
    action_metadata: Dict[str, Any]
    step_info: Any


class EpisodeRecorder:
    def __init__(self, max_window: int = 12):
        self.max_window = max_window
        self._steps: list[EpisodeStep] = []

    def record(
        self,
        step: int,
        world_before,
        actions,
        action_metadata,
        step_info,
    ) -> None:
        self._steps.append(
            EpisodeStep(
                step=step,
                world_before=world_before.to_dict(),
                actions=actions,
                action_metadata=action_metadata,
                events=getattr(step_info, "events", []),
            )
        )

        if len(self._steps) > self.max_window:
            self._steps.pop(0)

import json
from pathlib import Path
from typing import List

class EpisodeSegmentStore:
    def __init__(self, path="episode_segments.jsonl"):
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def write_segment(self, segment: List[dict]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(segment, default=str) + "\n")
