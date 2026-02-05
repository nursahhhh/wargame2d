# memory/memory_store.py

import json
import os
from typing import Dict, List, Any
from datetime import datetime
from env.core.actions import Action
from typing import Any

from enum import Enum


class MemoryStore:
    def __init__(self, episode_id: int, base_dir: str = "memory/raw"):
        self.episode_id = episode_id
        self.base_dir = base_dir

        self.step_buffer: List[Dict[str, Any]] = []
        self.segment_counter = 0

        self.base_dir = base_dir
        self.episode_id, self.episode_dir = \
            self._resolve_episode_dir(base_dir)
        print("episode id directory is ",self.episode_dir)
    


    def record_step(self, step_data: Dict[str, Any]):
        """Called every game step"""
        self.step_buffer.append(step_data)

    def _resolve_episode_dir(self, base_dir: str) -> tuple[int, str]:
        os.makedirs(base_dir, exist_ok=True)

        existing = [
            d for d in os.listdir(base_dir)
            if d.startswith("episode_") and d[8:].isdigit()
        ]

        if not existing:
            episode_id = 0
        else:
            episode_id = max(int(d[8:]) for d in existing) + 1

        episode_dir = os.path.join(base_dir, f"episode_{episode_id:03d}")
        os.makedirs(episode_dir, exist_ok=False)

        return episode_id, episode_dir

    
    def serialize_for_json(self,obj: Any) -> Any:
        """
        Recursively convert domain objects into JSON-serializable structures.
        """
        if isinstance(obj, Action):
            return obj.to_dict()

        if isinstance(obj, Enum):
            return obj.name

        if isinstance(obj, dict):
            return {k: self.serialize_for_json(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self.serialize_for_json(v) for v in obj]

        return obj


    def flush_segment(self, trigger_event: str, outcome: Dict[str, Any]):
        """Called on critical events"""
        if not self.step_buffer:
            print("step buffer is None !")
            return 
        
        segment = {
            "episode_id": self.episode_id,
            "segment_id": self.segment_counter,
            "trigger_event": trigger_event,
            "time_range": {
                "start_step": self.step_buffer[0]["step"],
                "end_step": self.step_buffer[-1]["step"],
            },
            "step_trace": self.step_buffer,
            "outcome": outcome,
            "created_at": datetime.utcnow().isoformat()
        }
        

        path = os.path.join(
            self.episode_dir,
            f"segment_{self.segment_counter:04d}.json"
        )

        with open(path, "w") as f:
            segment = self.serialize_for_json(segment)
            json.dump(segment, f, indent=2)

        # reset RAM
        self.step_buffer.clear()
        self.segment_counter += 1
