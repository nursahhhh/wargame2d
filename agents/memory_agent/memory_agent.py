import json

import logging
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
        """Process all episodes and extract segment + global reflections"""
        self.reflections_dir.mkdir(exist_ok=True, parents=True)

        for episode_dir in self.episodes_dir.iterdir():
            if episode_dir.is_dir():
                self._process_episode(episode_dir)

        logging.info("Reflection extraction completed.")


    def _process_segment(self, segment_path: Path):
        segment = json.loads(segment_path.read_text(encoding="utf-8"))
        prompt = self.build_segment_reflection_prompt(segment)
        response = self.llm(prompt)  # call sync-safe LLMClient
        reflection = self._safe_parse(response)
        self._write_reflection(segment, reflection)

    def _process_episode(self, episode_dir: Path):

        segments = sorted(episode_dir.glob("segment_*.json"))
        all_segments = []
        early_termination_detected = False

        for segment_file in segments:
            segment = json.loads(segment_file.read_text(encoding="utf-8"))
            all_segments.append(segment)

            # Segment-level reflection
            self._process_segment(segment_file)

            # Check early termination trigger
            if segment.get("trigger_event") == "EARLY_TERMINATION":
                early_termination_detected = True

        # If early termination occurred → global analysis
        if early_termination_detected:
            episode_id = all_segments[0]["episode_id"]
            self._analyze_episode_globally(episode_id, all_segments)

    def _analyze_episode_globally(self, episode_id: int, segments: List[Dict[str, Any]]):

        prompt = self._build_episode_analysis_prompt(segments)
        response = self.llm(prompt)
        analysis = self._safe_parse(response)

        episode_path = self.reflections_dir / f"episode_{episode_id}.json"

        if episode_path.exists():
            episode_data = json.loads(episode_path.read_text(encoding="utf-8"))
        else:
            episode_data = {
                "episode_id": episode_id,
                "segment_reflections": [],
                "episode_analysis": None
            }

        episode_data["episode_analysis"] = analysis

        episode_path.write_text(
            json.dumps(episode_data, indent=2),
            encoding="utf-8"
        )

    def _build_episode_analysis_prompt(self, segments: List[Dict[str, Any]]) -> str:
        return f"""
            You are a strategic tactical analysis agent.

            The following segments belong to a single episode that ended with EARLY_TERMINATION.

            Together they represent the full trajectory of the episode.

            Your task is to analyze the episode SYSTEMICALLY.

            Focus on:

            1. Why offensive capability collapsed
            2. Asset misallocation patterns
            3. Radar layering failures
            4. Coordination breakdowns
            5. Information dominance loss
            6. Extract 2-5 GLOBAL avoidance principles

            Do NOT repeat segment-level tactical mistakes.
            Focus only on team-level patterns.

            Segments:
            {segments}

            Return STRICT JSON:

            {{
            "systemic_failures": "...",
            "offensive_collapse_cause": "...",
            "coordination_breakdowns": "...",
            "global_avoid_principles": ["...", "..."],
            "confidence": 0.0
            }}
            """


    @staticmethod
    def build_segment_reflection_prompt(segment: dict) -> str:

        return f"""
                You are a tactical failure analysis agent.

                You are given a gameplay segment that ended with a failure event.
                The context information may be incomplete or null.

                DO NOT analyze the model's behavior, repetition, or decision style.
                DO NOT comment on adaptivity or action diversity.

                Instead, analyze the battlefield causality.

                Segment data:
                {segment}

                Your task:

                1. Identify which FRIENDLY ENTITY was destroyed or caused the failure.
                2. Describe the tactical situation at the time of failure:
                - Radar overlap
                - SAM coverage
                - Support proximity
                - Detection order
                3. Explain the specific tactical mistake made by that entity.
                4. Identify the violated combat principle.
                5. Generate a CONDITIONAL avoidance rule tied to entity type and spatial context.

                The rule must:
                - Be entity-specific (FIGHTER / AWACS / SAM / DECOY)
                - Reference spatial or radar conditions
                - Be reusable in future similar geometries

                Return STRICT JSON in the following format:

                {{
                "entity_type": "...",
                "tactical_error": "...",
                "combat_principle_violated": "...",
                "engagement_context": {{
                    "enemy_radar_overlap": true/false,
                    "friendly_support_overlap": true/false,
                    "within_enemy_sam_range": true/false
                }},
                "avoid_rule": "...",
                "confidence": 0.0
                }}

                Be precise, geometric, and operational.
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
        """
        Store reflections grouped by episode.
        One file per episode.
        """

        episode_id = segment["episode_id"]
        episode_path = self.reflections_dir / f"episode_{episode_id}.json"

        # --- Load existing episode file if present ---
        if episode_path.exists():
            episode_data = json.loads(episode_path.read_text(encoding="utf-8"))
        else:
            episode_data = {
                "episode_id": episode_id,
                "reflections": []
            }

        # --- Build reflection payload ---
        payload = {
            "segment_id": segment["segment_id"],
            "trigger_event": segment.get("trigger_event"),
            "failure_analysis": reflection
        }

        # --- Append ---
        episode_data["reflections"].append(payload)

        # --- Save back ---
        episode_path.write_text(
            json.dumps(episode_data, indent=2),
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

        segment_reflections = []
        episode_analyses = []

        for path in self.reflections_dir.glob("episode_*.json"):
            episode_data = json.loads(path.read_text(encoding="utf-8"))

            for item in episode_data.get("segment_reflections", []):
                segment_reflections.append(item["failure_analysis"])

            if episode_data.get("episode_analysis"):
                episode_analyses.append(episode_data["episode_analysis"])

        return {
            "segment_reflections": segment_reflections,
            "episode_analyses": episode_analyses
        }


    def _build_experience_distillation_prompt(self, data: Dict[str, Any]) -> str:

        return f"""
                You are an experience distillation agent.

                You are given TWO TYPES of reflective evidence:

                1) Segment-level reflections (local tactical mistakes)
                2) Episode-level analyses (systemic collapse reasoning)

                IMPORTANT:
                Episode-level analyses represent higher-level structural failures.
                They should be weighted MORE heavily than individual segment reflections.

                Your task:

                - Merge consistent ideas
                - Prioritize systemic patterns
                - Discard weak or overly specific insights
                - Produce 3-5 NEGATIVE doctrine rules

                Segment-level reflections:
                {data["segment_reflections"]}

                Episode-level analyses:
                {data["episode_analyses"]}

                Return STRICT JSON:

                {{
                "experience_guidance": [
                    {{"rule": "...", "rationale": "...", "confidence": 0.0}}
                ]
                }}

                Constraints:
                - Max 5 rules
                - Confidence in [0.0, 1.0]
                - Prefer structural principles over local mistakes
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
