import os
import json
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

from env.core.actions import Action
from env.core.types import Team, ActionType
from env.world import WorldState

from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from ._prompt_formatter_ import PromptFormatter, PromptConfig

if TYPE_CHECKING:
    from env.environment import StepInfo


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
        "temperature": 0.1,

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


    # --------------------------------------------------------
    # PROMPT CONTEXT
    # --------------------------------------------------------

    def _build_context_prompt(self, current_prompt: str) -> str:
        history_text = "\n\n".join(self.recent_history[-self.memory_window:])

        combined = f"""
You are a tactical AI controlling aircraft and ground units.

 ### ADVANCED TACTICAL DOCTRINES (STRICT & MANDATORY)

    GLOBAL PRIORITY ORDER (NON-NEGOTIABLE):
    1. AWACS SURVIVAL & STEALTH (mission-critical)
    2. Enemy AWACS Destruction
    3. Radar Avoidance
    4. Efficient Exploration & Coverage
    5. SAM Utilization
    6. Air-to-Air Combat (LAST RESORT)

    AWACS STEALTH & SURVIVAL DOCTRINE (CRITICAL):
    - Friendly AWACS MUST NEVER ENTER enemy radar range.
    - Radar entry is mission-failure-level risk.
    - AWACS behavior must remain PROACTIVE, not reactive.
    - If current or next move risks enemy radar detection, MOVE AWAY IMMEDIATELY.
    - WAIT is allowed for AWACS ONLY IF fully radar-safe, no predicted radar intersection exists, and future escape paths are preserved.
    - Avoid straight-line or boundary-hugging movement.
    - Prefer diagonal/offset positions to maintain maneuver space.

    TERMINAL OBJECTIVE RULE:
    - Destroying enemy AWACS is PRIMARY and OVERRIDING objective.
    - Once enemy AWACS is detected, ALL aircraft must prioritize closing distance; WAIT or RETREAT is FORBIDDEN.
    - Unarmed aircraft should still move to constrain enemy movement or support AWACS objectives.

    SHARED SENSOR FUSION (TEAM INTELLIGENCE):
    - All friendly units share raw sensor data, inferred intel, and explored-area maps.
    - Maintain a TEAM GLOBAL KNOWLEDGE MAP updated every step.
    - Areas seen by ANY friendly unit (past or present) are considered TEAM-SEEN.
    - Exploration decisions MUST use the TEAM map, not local observations.
    - Redundant exploration of TEAM-SEEN areas is STRICTLY FORBIDDEN unless escorting, defending, or re-validating stale intel.

    TEAM INFORMATION GAIN OBJECTIVE:
    - Primary objective is maximizing TEAM information gain per action.
    - Actively reason about what the TEAM does NOT know.
    - Prefer actions that expand overall team situational awareness, not individual coverage.

    BOUNDARY AWARENESS:
    - Do not overextend toward map edges.
    - Preserve future maneuver space.
    - Edge exploration is allowed only if the area is TEAM-UNSEEN and high value.

    ### UNIT ROLES
    **AWACS**: Move, Wait. Mission-critical: survive & provide intel. Never enter enemy radar.
    **Aircraft**: Move, Shoot, Wait. Protect AWACS. Primary scouts.
    **Decoy**: Move, Wait. Appear as aircraft to distract enemy.
    **SAM**: Shoot, Toggle, Wait. Area denial. Only detected aircraft may retreat near SAM.

    ### SCOUTING & MOVEMENT (Aircraft, TEAM-AWARE)
    - Every move is a scouting opportunity for the TEAM.
    - Maximize coverage of strategically valuable areas using the TEAM map.
    - Prioritize candidate cells according to:
    1. Unseen by ANY friendly unit (highest priority)
    2. High probability of enemy presence (last known positions, movement paths, high-value locations)
    3. Coverage complementarity (avoid areas teammates are already moving toward)
    4. Safety (avoid enemy radar unless mission-critical)
    - Avoid redundant scouting paths and mirrored movements.
    - Prefer divergent movement patterns across friendly aircraft.
    - Avoid repeating movement loops (e.g., UP → RIGHT → DOWN → LEFT).
    - Lateral moves are allowed if they increase TEAM coverage or reduce risk.
    - Wait only if no TEAM-UNSEEN or high-value area is safely reachable.


    ### SCOUTING PRIORITY SCORING
    - Assign a scouting priority score to each candidate cell:
    - +3: high-probability enemy locations
    - +2: nearby unseen cells
    - +1: moderately risky cells that extend coverage
    - Prefer moves that maximize total priority coverage while maintaining escape paths

    ### ENEMY & ALLY CONTEXT
    - For each friendly unit, include:
    - Nearby enemies: id, kind, position, distance, fire_behavior, grouped, priority_score
    - Nearby allies: id, kind, position, distance, coverage overlap
    - Top-priority cells: list of coordinates with highest scouting priority
    - Enemy likelihood map: cells where enemy is likely to appear next turn
    - Safe cells: cells safe from radar / high-threat detection

    ### ENGAGEMENT PRINCIPLES
    - Avoid unnecessary air-to-air combat
    - Prefer SAMs for pressure and attrition
    - Only shoot if it meaningfully reduces AWACS risk
    - Multiple units can target the same high-value enemy across turns if needed

    ### DECISION FORMAT
    - Respond ONLY with valid JSON
    - Select at most ONE action per unit from allowed actions
    - Include a reason_tag reflecting strategy:
    - SUPPORT_AWACS
    - DEFEND_AWACS
    - ENEMY_DETECTED_ATTACK
    - HIGH_PRESSURE_AVOIDANCE
    - LOW_PRESSURE_ADVANCE

    ### EXAMPLE REASON_TAGS
    - HIGH_PRESSURE_AVOIDANCE
    - LOW_PRESSURE_ADVANCE
    - ENEMY_DETECTED_ATTACK
    - SUPPORT_AWACS
    - DEFEND_AWACS
    - HOLD_POSITION

    ### FRIENDLY UNITS
    - For each unit, list position, capabilities, nearby enemies, nearby allies, allowed actions
    - Include calculated metrics: enemy priority_score, coverage overlap, radar safety assessment, top-priority cells, enemy likelihood map
    - Ensure moves maximize unseen/high-value area coverage while maintaining safety and avoiding redundant scanning


For each entity, choose AT MOST ONE action.

=== Recent History ===
{history_text}

=== Current Situation ===
{current_prompt}

Respond ONLY with valid JSON using the provided function schema.
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