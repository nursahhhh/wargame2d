import os
import json
import requests
from dotenv import load_dotenv


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

STRATEGY_UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "request_strategy_update",
        "description": "Request a new high-level strategy from Commander when the current one is invalidated.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Concise explanation of why the current strategy is no longer valid."
                },
                "observed_shift": {
                    "type": "string",
                    "description": "Describe the macro-level change in battlefield state (enemy posture shift, phase transition, loss of asset, etc.)."
                },
                "urgency_level": {
                    "type": "integer",
                    "description": "1-5 scale. 5 = immediate strategic danger.",
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["reason", "observed_shift", "urgency_level"]
        }
    }
}

COMMANDER_TOOL = {
    "type": "function",
    "function": {
        "name": "set_commander_intent",
        "description": "Set high-level strategic intent for the team.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "Short identifier of overall strategy"
                },
                "justification": {
                    "type": "string",
                    "description": "Reasoning behind the strategy"
                },
                "suggested_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "High-level tactical patterns to follow"
                }
            },
            "required": ["intent", "justification", "suggested_patterns"]
        }
    }
}


# ============================================================
# OpenRouter call
# ============================================================

def call_openrouter(prompt: str, model: str, api_key: str, step: int, agent_type : str,run_log_dir):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "wargame2d-llm-agent",
        "Content-Type": "application/json",
    }
    if agent_type == 'executor': 
       payload = {
    "model": model,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are an action-selection module. "
                "You MUST respond only via one of the provided functions."
            ),
        },
        {"role": "user", "content": prompt},
    ],
    "tools": [ACTION_TOOL, STRATEGY_UPDATE_TOOL],
    "tool_choice": "auto",
    "temperature": 0.0,
}
    elif agent_type == 'commander': 

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a high-level strategic commander AI. "
                        "You must infer the optimal strategic pattern dynamically "
                        "based on the evolving battlefield state. "
                        "You do NOT select low-level actions. "
                        "You must respond ONLY via the provided function."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            "tools": [COMMANDER_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "set_commander_intent"}
            },
            "temperature": 0.2,
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



    return msg