import json
from pathlib import Path
from typing import Dict, Any, List

class MemoryAgent:
    def __init__(
        self,
        llm_complete,  # callable(prompt:str) -> str
        episodes_dir="memory/raw",
        reflections_dir="memory/reflections",
        distilled_path="memory/distilled/experience_guidance.json"
    ):
        self.llm = llm_complete
        self.episodes_dir = Path(episodes_dir)
        self.reflections_dir = Path(reflections_dir)
        self.distilled_path = Path(distilled_path)

    # ----------------------------
    # Stage 1: Reflection Extraction
    # ----------------------------
    def extract_reflections(self):
        """Process all segment files and save reflections (stage 1)"""
        self.reflections_dir.mkdir(exist_ok=True, parents=True)
        for episode_dir in self.episodes_dir.iterdir():
            if episode_dir.is_dir():
                for segment_file in episode_dir.glob("segment_*.json"):
                    self._process_segment(segment_file)
        import logging
        logging.info("Reflection extraction completed.")

    def _process_segment(self, segment_path: Path):
        segment = json.loads(segment_path.read_text(encoding="utf-8"))
        prompt = self.build_segment_reflection_prompt(segment)
        response = self.llm(prompt)  # call sync-safe LLMClient
        reflection = self._safe_parse(response)
        self._write_reflection(segment, reflection)

    @staticmethod
    def build_segment_reflection_prompt(segment: dict) -> str:
        return f"""
You are a failure analysis memory agent.

You are given a gameplay segment that ended with a failure event.
The context information may be incomplete or null.
Your task is to analyze the ACTION SEQUENCE and infer why the failure occurred.

Segment data:
{segment}

Analyze and answer:

1. What happened in this segment?
2. Which action or repeated action pattern most likely caused the failure?
3. Why was this behavior risky or incorrect?
4. What action(s) should NOT have been taken?
5. Provide a general rule to avoid this failure in the future.

Return STRICT JSON in the following format:
{{
"summary": "...",
"root_cause": "...",
"bad_action_pattern": "...",
"avoid_rule": "...",
"confidence": 0.0
}}

Be concise, abstract, and generalize beyond this single episode.
"""

    def _safe_parse(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "summary": "Failed to analyze segment",
                "root_cause": "Unknown",
                "bad_action_pattern": "Unknown",
                "avoid_rule": "Avoid repeating this behavior in similar situations.",
                "confidence": 0.1
            }

    def _write_reflection(self, segment: dict, reflection: dict):
        filename = f"ep{segment['episode_id']}_seg{segment['segment_id']}.json"
        payload = {
            "episode_id": segment["episode_id"],
            "segment_id": segment["segment_id"],
            "trigger_event": segment["trigger_event"],
            "failure_analysis": reflection
        }
        (self.reflections_dir / filename).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )

    # ----------------------------
    # Stage 2: Experience Distillation
    # ----------------------------
    def distill_experience(self) -> Dict[str, Any]:
        """Read all reflections and produce distilled guidance (stage 2)"""
        reflections = self._load_all_reflections()
        if not reflections:
            raise RuntimeError("No reflections found for distillation")

        prompt = self._build_experience_distillation_prompt(reflections)
        response = self.llm(prompt)
        guidance = self._safe_parse_distillation(response)

        self.distilled_path.parent.mkdir(exist_ok=True, parents=True)
        self.distilled_path.write_text(
            json.dumps(guidance, indent=2),
            encoding="utf-8"
        )
        return guidance

    def _load_all_reflections(self) -> List[Dict[str, Any]]:
        reflections = []
        for path in self.reflections_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "failure_analysis" in data:
                reflections.append(data["failure_analysis"])
        return reflections

    def _build_experience_distillation_prompt(self, reflections: List[Dict[str, Any]]) -> str:
        return f"""
You are an experience distillation agent.

You are given a collection of FAILURE REFLECTIONS extracted from past gameplay.
These reflections may be noisy, conflicting, or low quality.

Your task is to extract a SMALL SET of high-quality NEGATIVE experience rules
that should guide a future game-playing agent.

Guidelines:
- Treat reflections as evidence, not truth
- Merge similar ideas
- Discard weak or overly specific rules
- Prefer general, safety-oriented guidance
- Focus ONLY on behaviors that should be AVOIDED

Input reflections:
{reflections}

Return STRICT JSON:

{{
  "experience_guidance": [
    {{"rule": "...", "rationale": "...", "confidence": 0.0}}
  ]
}}

Constraints:
- Max 5 rules
- Confidence in range [0.0, 1.0]
- Be concise and generalize
"""

    def _safe_parse_distillation(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            assert "experience_guidance" in data
            return data
        except Exception:
            return {
                "experience_guidance": [
                    {
                        "rule": "Avoid repeating behaviors that previously led to irreversible failure.",
                        "rationale": "Past failures indicate repeated unexamined actions increase risk.",
                        "confidence": 0.3
                    }
                ]
            }
