import os
import json
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from dotenv import load_dotenv

from env.core.actions import Action
from env.core.types import Team, ActionType
from env.world import WorldState

from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from ._prompt_formatter_ import PromptFormatter, PromptConfig

if TYPE_CHECKING:
    from env.environment import StepInfo

load_dotenv()
# ============================================================
# TOOL DEFINITION
# ============================================================

ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "select_actions",
        "description": (
            "Select at most one valid action per entity. "
            "Only choose from the provided allowed actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "integer"},
                            "action": {"type": "string"},
                            "dir": {
                                "type": "string",
                                "enum": ["UP", "DOWN", "LEFT", "RIGHT"]
                            },
                            "reason_tag": {
                                "type": "string",
                                "enum": [
                                    "HIGH_PRESSURE_AVOIDENCE",
                                    "LOW_PRESSURE_ADVANCE",
                                    "ENEMY_DETECTED_ATTACK",
                                    "SUPPORT_AWACS",
                                    "DEFEND_AWACS",
                                    "SCOUTING_BEHAVIOR",
                                    "HOLD_POSITION"
                                ]
                            },
                            "note": {"type": "string"}
                        },
                        "required": ["entity_id", "action", "reason_tag"]
                    }
                }
            },
            "required": ["actions"]
        }
    }
}


# ============================================================
# OpenRouter call
# ============================================================

def call_openrouter(prompt: str, model: str, api_key: str, step: int,    run_log_dir,):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "wargame2d-llm-agent",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an action-selection module. "
                    "You MUST respond only via the provided function."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "tools": [ACTION_TOOL],
        "tool_choice": {
            "type": "function",
            "function": {"name": "select_actions"}
        },
        "temperature": 0.0,

    }

    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code != 200:
        print("=== OpenRouter HTTP ERROR ===")
        print(resp.text)

        resp.raise_for_status()

    data = resp.json()

    step_file = run_log_dir / f"step_{step:03d}.json"

    log_payload = {
        "step": step,
        "model": model,
        "prompt": prompt,
        "raw_response": data,
    }

    with open(step_file, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2, ensure_ascii=False)


    msg = data["choices"][0]["message"]

    if "tool_calls" in msg:
        return msg["tool_calls"][0]["function"]["arguments"]

    return None


# ============================================================
# AGENT
# ============================================================

@register_agent("llm_hybrid_agent")
class LLMHybridAgent(BaseAgent):

    def __init__(
        self,
        team: Team,
        name: str = "LLMHybridAgent",
        api_key: Optional[str] = None,
        model: str = "x-ai/grok-4.1-fast",
        log_file: str = "llm_hybrid_agent_log.jsonl",
        memory_window: int = 5,
        **kwargs,
    ):
        super().__init__(team, name)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        self.model = model
        self.prompt_formatter = PromptFormatter()
        self.prompt_config = PromptConfig()

        self.memory_window = memory_window
        self.recent_history: list[str] = []

            # ---- RUN LOG FOLDER ----
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_log_dir = Path("llm_runs") / timestamp
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        self.step_counter = 0


    # --------------------------------------------------------

    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:

        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)

        allowed_actions: Dict[int, list[Action]] = {}
        final_actions: Dict[int, Action] = {}

        for entity in intel.friendlies:
            if not entity.alive:
                continue

            acts = entity.get_allowed_actions(world)
            if acts:
                allowed_actions[entity.id] = acts

        # -------- PROMPT --------

        prompt_text, prompt_payload = self.prompt_formatter.build_prompt(
            intel=intel,
            allowed_actions=allowed_actions,
            config=self.prompt_config,
            step = self.step_counter
        )
        full_prompt = self._build_context_prompt(prompt_text)

        self.step_counter += 1

        llm_args = call_openrouter(
            prompt=full_prompt,
            model=self.model,
            api_key=self.api_key,
            step=self.step_counter,
            run_log_dir=self.run_log_dir,

        )

        parsed_actions = {}
        if llm_args:
            parsed_actions = self._parse_llm_output(llm_args, allowed_actions)

        # -------- FALLBACK --------
        if not parsed_actions:
            print("LLM FAILED → no actions selected")
        else:
            final_actions.update(parsed_actions)

        metadata = {
            "llm_raw_output": llm_args,
            "parsed_actions": parsed_actions,
            "allowed_actions": allowed_actions,
            "prompt_payload": prompt_payload,
        }

        print("Parsed acrions are : ",parsed_actions)

        return final_actions, metadata




    def build_experience_advisory_section(
        self,           
        path: str,
        min_confidence: float = 0.7,
        max_rules: int = 8
    ) -> str:
        """
        Reads distilled experience file and converts it into a
        properly formatted LLM advisory section.

        Args:
            file_path: Path to distilled experience JSON file
            min_confidence: Minimum confidence threshold
            max_rules: Maximum number of rules to include

        Returns:
            Formatted advisory string for prompt injection
        """
        BASE_DIR = Path(__file__).resolve().parents[3]  # adjust if needed
        file_path = BASE_DIR / path

        # Prevent crash if file missing
        if not file_path.exists():
            print(f"[WARNING] Experience file not found: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load experience file: {e}")
            return ""

        rules: List[Dict] = data.get("experience_guidance", [])

        # Filter by confidence
        rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]

        # Sort by confidence (highest first)
        rules.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Limit number of rules
        rules = rules[:max_rules]

        if not rules:
            return ""  # No advisory section if nothing qualifies

        # Build advisory text
        advisory_lines = []
        advisory_lines.append("============================================================")
        advisory_lines.append("EXPERIENCE-BASED ADVISORY HEURISTICS")
        advisory_lines.append("============================================================\n")
        advisory_lines.append(
            "The following patterns were extracted from past reflections."
        )
        advisory_lines.append(
            "They are STRATEGIC ADVISORIES and must NOT override:"
        )
        advisory_lines.append("  - HARD constraints")
        advisory_lines.append("  - FIRM constraints")
        advisory_lines.append("  - Mission priorities\n")
        advisory_lines.append("Use them only when multiple valid actions exist.\n")
        advisory_lines.append("Advisories (sorted by confidence):\n")

        for rule in rules:
            confidence = round(rule.get("confidence", 0), 2)
            advisory_lines.append(f"[Confidence: {confidence}]")
            advisory_lines.append(f"Guideline: {rule.get('rule', '')}")
            advisory_lines.append(f"Rationale: {rule.get('rationale', '')}\n")

        return "\n".join(advisory_lines)

    # --------------------------------------------------------
    # PROMPT CONTEXT
    # --------------------------------------------------------

    def _build_context_prompt(self, current_prompt: str) -> str:
        history_text = "\n\n".join(self.recent_history[-self.memory_window:])
        experience_avoidance = self.build_experience_advisory_section("wargame2d/llm_runs/memory/distilled/experience_guidance.json")
        combined = f"""

        You are a tactical AI commander controlling friendly units in a 2D combat grid:
        AWACS, Aircraft, Decoys, and SAM sites.

        All cells within range of: Friendly AWACS radar OR active SAM radar
        - This coverage is SHARED among all friendly units instantly
        ============================================================
        MISSION OBJECTIVES (Ordered by Priority)
        ============================================================
        P1. PROTECT FRIENDLY AWACS — Survival is absolute. Never compromise.
        P2. DESTROY ENEMY AWACS — Terminal win condition.
        P3. AVOID DETECTION — Stay outside enemy radar; deny interception.
        P4. GAIN INFORMATION — Explore TEAM-UNSEEN areas (conditional).
        P5. ACHIEVE NUMERICAL ADVANTAGE — Coordinate SAM + aircraft.
        P6. ENGAGE IN COMBAT — Prefer engagements that improve force advantage.

        Priority P4 (Exploration) activates ONLY when:
        - Enemy AWACS is NOT currently detected
        - AND TEAM-UNSEEN cells exist outside friendly sensor coverage

        Lower priorities MUST NOT override higher priorities.

        ============================================================
        GAME STATE (Provided Each Turn)
        ============================================================
        You will receive:
        - Grid dimensions and boundaries
        - All friendly unit positions, types, states (armed/unarmed, radar on/off)
        - Known enemy unit positions (if detected)
        - TEAM-SEEN cells: observed by ANY friendly unit at ANY time
        - TEAM-UNSEEN cells: never observed by any friendly unit
        - Friendly radar coverage (AWACS + active SAMs)
        - Estimated/known enemy radar coverage
        - Turn number

        ============================================================
        CONSTRAINT CLASSIFICATION
        ============================================================
        HARD constraints — Never violate under any circumstance:
        [H1] AWACS must NEVER enter known enemy radar coverage
        [H2] AWACS must NEVER end turn with zero safe escape routes
        [H3] Actions that guarantee AWACS destruction next turn are FORBIDDEN
        [H4] Exploration inside friendly radar coverage is INVALID
        [H5] Re-labeling invalid exploration as DEFEND/SUPPORT is FORBIDDEN

        FIRM constraints — Violate only to satisfy HARD constraints:
        [F1] AWACS should maintain 2+ cell buffer from enemy radar edge
        [F2] AWACS should stay behind combat units (layered protection)
        [F3] Aircraft should not WAIT when enemy AWACS is detected
        [F4] At least one unit should advance exploration each turn (when P4 active)

        SOFT constraints — Preferences, not requirements:
        [S1] Prefer 2v1+ engagements over fair fights
        [S2] Prefer lateral/backward AWACS movement over forward
        [S3] Prefer SAM ON for area denial
        [S4] Avoid boundary-hugging paths for AWACS

        ============================================================
        ADVANTAGE DEVELOPMENT RULE
        ============================================================

        If immediate engagement does not yield clear local advantage:

        The unit MUST execute one of the following:

        - Reposition to create multi-unit convergence
        - Move to overlap with friendly SAM coverage
        - Constrain enemy movement corridor
        - Improve radar coverage geometry
        - Reduce own exposure while preserving pressure

        Passive retreat without strategic improvement is discouraged.
        Inaction is not acceptable when advantage can be developed.

        ============================================================
        UNIT CAPABILITIES
        ============================================================
        AWACS:
        - Actions: MOVE, WAIT
        - Radar: Provides wide-area friendly sensor coverage
        - Rules: Apply all HARD constraints strictly

        Aircraft:
        - Actions: MOVE, SHOOT (if armed and target in range), WAIT
        - Role: Escort, exploration, interception, terminal attack
        - When enemy AWACS detected: MUST advance or constrain escape

        Decoy:
        - Actions: MOVE, WAIT
        - Role: Expendable scouting, deception, shot absorption
        - May sacrifice for strategic gain, but not for zero value

        SAM:
        - Actions:Shoot, Toggle, Wait.
        - Role: Static area denial; range advantage over aircraft
        - Default: ON (for area control), OFF only for ambush/cooldown

        ============================================================
        SHARED INTELLIGENCE (Team Sensor Fusion)
        ============================================================
        - All units share a SINGLE GLOBAL KNOWLEDGE MAP
        - A cell is TEAM-SEEN if observed by ANY friendly unit, ever
        - Individual unit perception is IRRELEVANT for exploration decisions
        - Cells inside friendly AWACS/SAM radar have ZERO exploration value
        - Exploration targets must be TEAM-UNSEEN AND outside friendly radar

        ============================================================
        DECISION RULES BY SITUATION
        ============================================================

        IF enemy AWACS is DETECTED:
        → P2 activates: All aircraft MUST reduce distance or block escape
        → WAIT/RETREAT forbidden for armed aircraft
        → Ignore exploration; prioritize kill

        IF friendly AWACS is THREATENED (enemy closing or radar encroaching):
        → P1 activates: Abort lower priorities immediately
        → AWACS moves to maximize radar separation
        → Aircraft may intercept or screen

        IF neither AWACS is detected AND TEAM-UNSEEN cells exist:
        → P4 activates: At least ONE unit MUST explore
        → Select highest-uncertainty regions first
        → Aircraft/decoys reposition toward TEAM-UNSEEN boundaries
        → WAIT is FORBIDDEN if exploration-enabling move exists

        IF all exploration moves are blocked by constraints:
        → Reposition toward the BOUNDARY of known space
        → This enables future exploration access
        → Log this as reason_tag: REPOSITION_FOR_EXPLORATION

        ============================================================
        CONFLICT RESOLUTION (When Constraints Collide)
        ============================================================
        1. Always satisfy HARD constraints first
        2. Satisfy as many FIRM constraints as possible without violating HARD
        3. Among remaining options, prefer those satisfying SOFT constraints
        4. If ALL actions violate at least one constraint:
        → Choose the action that violates the LOWEST priority constraint
        → Flag reason_tag with: FORCED_CONSTRAINT_VIOLATION

        ============================================================
        ANTI-EXPLOIT RULES
        ============================================================
        - Exploration claimed inside friendly radar = INVALID (H4)
        - DEFEND_AWACS near AWACS when effect is exploration = INVALID (H5)
        - Adversarial safety check: If move is safe now but unsafe after
        obvious enemy response, treat it as UNSAFE
        - Edge-hugging or corner moves for AWACS are high-risk
        - No laundering invalid actions through alternate reason_tags

        {experience_avoidance}

        ============================================================
        OUTPUT FORMAT
        ============================================================

        Respond with valid JSON matching the provided function schema.

        BEFORE calling the function, you MUST internally evaluate:
        - Threat exposure
        - Engagement advantage
        - Coordination potential

        For each selected action:

        - Select AT MOST ONE action per unit
        - Include a reason_tag for EVERY action
        - Include a reasoning object for EVERY action with:

            {{
            "threat_level": 0–10,
            "exposure_risk": 0–10,
            "advantage_gain": 0–10,
            "coordination_value": 0–10,
            "justification": "Short tactical explanation"
            }}

        - Omit units that should WAIT (or explicitly include WAIT)

        Actions that increase exposure_risk above 6 without
        advantage_gain above 6 are invalid.

        Allowed reason_tags:
        - PROTECT_AWACS         (escorting, screening, retreating AWACS)
        - INTERCEPT_THREAT      (moving to block/engage approaching enemy)
        - ATTACK_ENEMY_AWACS    (terminal attack execution)
        - EXPLORE_UNSEEN        (advancing into TEAM-UNSEEN space)
        - REPOSITION_FOR_EXPLORATION (moving toward UNSEEN boundary)
        - AREA_DENIAL           (SAM toggling, zone control)
        - DECOY_SACRIFICE       (intentional expendable action)
        - AWAIT_OPPORTUNITY     (justified WAIT with no better option)
        - FORCED_CONSTRAINT_VIOLATION (when no fully legal action exists

        ====================================================
        RECENT HISTORY
        ====================================================
        {history_text}

        ====================================================
        CURRENT SITUATION
        ====================================================
        {current_prompt}

        Respond ONLY with valid JSON.
        """

        self.recent_history.append(current_prompt)

        return combined


    # --------------------------------------------------------
    # PARSER (FIXED)
    # --------------------------------------------------------

    def _extract_entity_id(self, raw):
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        match = re.search(r"\d+", str(raw))
        return int(match.group()) if match else None


    def _parse_llm_output(self, llm_args: str, allowed_actions):
        actions: Dict[int, Action] = {}

        try:
            data = json.loads(llm_args)
        except Exception:
            print("EXCEPTİONTHROWN LİNE 553  from agent class ")
            return {}

        for item in data.get("actions", []):
            ent = self._extract_entity_id(item.get("entity_id"))
            act_name = item.get("action")
            dir_name = item.get("dir")

            if ent is None or ent not in allowed_actions or not act_name:
                continue

            act_name = act_name.upper()
            dir_name = dir_name.upper() if dir_name else None

            for act in allowed_actions[ent]:
                if act.type.name.upper() != act_name:
                    continue

                if act.type == ActionType.MOVE:

                    if act.params.get("dir").name == dir_name:
                        act_dir =act.params.get("dir")
                        if not act_dir.name or act_dir.name.upper() !=dir_name:
                            continue
                        actions[ent] = act
                        break
                else:
                    actions[ent] = act
                    break

        return actions