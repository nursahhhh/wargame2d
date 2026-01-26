import os
import json
import requests
import re
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

def call_openrouter(prompt: str, model: str, api_key: str, step: int):
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

    with open("LLM_response.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- STEP {step} ---\n")
        f.write(json.dumps(data.get("choices", [{}])[0].get("message", {}),  indent=2))

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

        self.log_file = log_file
        open(self.log_file, "w").close()

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
        )

        full_prompt = self._build_context_prompt(prompt_text)

        self.step_counter += 1

        llm_args = call_openrouter(
            prompt=full_prompt,
            model=self.model,
            api_key=self.api_key,
            step=self.step_counter
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

CRITICAL MOVEMENT RULE:
If enemy positions are known, horizontal movement (LEFT / RIGHT)
is often safer than vertical movement (UP / DOWN).
Do NOT choose UP/DOWN unless it clearly increases distance from enemy radar.

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