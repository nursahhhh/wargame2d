import json
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO
from agents.llm_agent.prompts.tactics import TACTICAL_GUIDE

load_dotenv()



class AnalystCompactOutput(BaseModel):
    analysis: str = Field(
        description="Detailed analysis and narration of the current state, consisting of the "
                    "\nGame Overview: Provide a brief overview of the current situation"
                    "\nCurrent Game Stage: Determine the stage of the game based on the information. Is it the early game, mid-game, or late game?"
                    "\nOur Situation: Describe our current status"
                    "\nOur Strategy:  Examine our current strategy and its effectiveness."
                    "\nEnemy's Strategy: Infer the enemy’s potential strategy, based on the available information"
                    "\nEnemy Targeting Analysis: Identify the most likely target for each visible armed enemy unit based on distance. Specify which of our units are in immediate danger and from which enemies, enabling strategic repositioning to control enemy aggro patterns."
                    "\nKey Information: Highlight any critical information that could impact decision-making, such as threats, opportunities, or resource status."
                    "\nDo not recommend direct actions."
    )
    action_inferences: Dict[int, str] = Field(
        description="For each alive visible enemy entity (integer key as id, like 3, 4 not like #3, #4), infer their next action using the history/logs/trends. ",
        default_factory=dict,
    )
    action_implications: Dict[int,str]  = Field(
        description="For each entity (integer key as id, like 3, 4 not like #3, #4), value is the analysis of each valid action (each move or shoot action separately) for this entity and its short-term implications also considering the potential enemy actions and other potential ally actions. "
                    "Consider nuances as well, example: if a decoy should screen another entity, they should act together (decoy through screening direction, other entity toward reverse direction) not separately. ",
        default_factory=dict,
    )

    key_points_for_executor: List[str] = Field(
        description="Bullet highlights matters for the executor should notice this turn (risks, blocked moves, threats, opportunities), executer doesn't see the history as you do, distill key information for him.",
        default_factory=list,
    )
    key_facts: List[str] = Field(
        description="Key facts or events for yourself to remember for future turns. (1-3 concise bullet points)",
        default_factory=list,
    )
    needs_replan: bool = Field(
        description="Whether the strategist should be called again to re-plan. Use True only for material shifts."
    )
    replan_reason: str = Field(
        description="Short explanation of what is going on currently and why re-planning is needed; empty if no re-plan.",
        default="",
    )


def _strip_turn_prefix(text: str, turn: int) -> str:
    """
    Remove a leading "Turn X" or "TX" label from a fact to avoid duplicated turn labels.
    """
    pattern = re.compile(rf"^(?:turn\s+{turn}|t{turn})\s*[:\-]?\s*", flags=re.IGNORECASE)
    cleaned = pattern.sub("", text.strip())
    return cleaned.strip(":- ").strip() or text.strip()


def _collect_step_logs(
    history: dict[int, dict],
    max_turns: int,
    team_name: Optional[str],
) -> Dict[int, tuple[list[str], list[str]]]:
    if not history:
        return {}
    logs: Dict[int, tuple[list[str], list[str]]] = {}
    turns = sorted(history.keys())[-max_turns:]
    for turn in turns:
        turn_log = history[turn] or {}
        our_lines: List[str] = []
        enemy_lines: List[str] = []

        for move in turn_log.get("movement", []) or []:
            if move.get("team") == team_name:
                our_lines.append(_describe_movement(move))
            else:
                enemy_lines.append(_describe_movement(move))

        for combat in turn_log.get("combat", []) or []:
            attacker_team = combat.get("attacker", {}).get("team")
            if attacker_team == team_name:
                our_lines.append(_describe_combat(combat))
            else:
                enemy_lines.append(_describe_combat(combat))

        if our_lines or enemy_lines:
            logs[turn] = (our_lines, enemy_lines)
    return logs


def _format_history(
    analyst_history: Dict[int, "AnalystCompactOutput"],
    visible_history: Dict[int, Dict[str, Any]],
    max_turns: int,
    team_name: Optional[str],
) -> str:
    step_logs = _collect_step_logs(visible_history, max_turns, team_name)
    key_facts: Dict[int, list[str]] = {}
    for turn in sorted(analyst_history.keys()):
        facts = [f for f in (analyst_history[turn].key_facts or []) if str(f).strip()]
        if facts:
            key_facts[turn] = [str(f) for f in facts]

    all_turns = sorted(set(step_logs.keys()) | set(key_facts.keys()))
    if not all_turns:
        return "- No history yet."

    lines: List[str] = []
    for turn in all_turns:
        lines.append(f"### Turn {turn}")

        if turn in key_facts:
            lines.append("  **Key Notes by You (Taken at Turn Start):**")
            for fact in key_facts[turn]:
                cleaned = _strip_turn_prefix(str(fact), turn)
                lines.append(f"    - {cleaned}")

        if turn in step_logs:
            our_lines, enemy_lines = step_logs[turn]
            lines.append("  **Observable Logs (At Turn End):**")
            lines.append("    Ally actions:")
            if our_lines:
                lines.extend([f"      - {l}" for l in our_lines])
            else:
                lines.append("      - None observed.")
            lines.append("    Enemy actions (observed):")
            if enemy_lines:
                lines.extend([f"      - {l}" for l in enemy_lines])
            else:
                lines.append("      - None observed.")

        lines.append("")

    return "\n".join(lines).strip()


def _describe_movement(entry: Dict[str, Any]) -> str:
    ent_type = entry.get("type") or "Unit"
    ent_team = entry.get("team") or "UNKNOWN"
    ent_id = entry.get("entity_id")
    direction = entry.get("direction")
    dest = entry.get("to") or {}
    base = f"{ent_type}#{ent_id}({ent_team})"
    target_pos = f"({dest.get('x')}, {dest.get('y')})"
    action = f"moves {direction.lower()}" if direction else "moves"
    line = f"{base} {action} to {target_pos}"
    if not entry.get("success", True):
        reason = entry.get("failure_reason") or "unknown"
        line += f" but fails ({reason})."
    else:
        line += "."
    return line


def _describe_combat(entry: Dict[str, Any]) -> str:
    attacker = entry.get("attacker", {})
    target = entry.get("target", {})
    fired = entry.get("fired", False)
    hit = entry.get("hit")
    killed = entry.get("target_killed", False)

    def _label(unit: Dict[str, Any]) -> str:
        uid = unit.get("id")
        uteam = unit.get("team") or "UNKNOWN"
        utype = unit.get("type") or "UNKNOWN"
        return f"{utype}#{uid}({uteam})" if uid is not None else f"{utype}({uteam})"

    line = f"{_label(attacker)}"
    if fired:
        line += f" fires at {_label(target)}"
        if hit is True:
            line += " -> HIT"
        elif hit is False:
            line += " -> MISS"
        if killed:
            line += " (KILL)"
        line += "."
    else:
        line += " engaged but did not fire."
    return line


def _format_step_logs(history: dict[int, dict], max_turns: int, current_turn: int, team_name: Optional[str]) -> str:
    del current_turn
    logs = _collect_step_logs(history, max_turns, team_name)
    if not logs:
        return ""
    lines: List[str] = []
    for turn in sorted(logs.keys()):
        our_lines, enemy_lines = logs[turn]
        lines.append(f"Turn {turn}:")
        lines.append("  Ally actions:")
        if our_lines:
            lines.extend([f"    - {l}" for l in our_lines])
        else:
            lines.append("    - None observed.")
        lines.append("  Enemy actions:")
        if enemy_lines:
            lines.extend([f"    - {l}" for l in enemy_lines])
        else:
            lines.append("    - None observed.")
        lines.append("")
    return "\n".join(lines).strip()


analyst_compact_agent = Agent[GameDeps, AnalystCompactOutput](
    "openrouter:x-ai/grok-4.1-fast",#"openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    deps_type=GameDeps,
    output_type=AnalystCompactOutput,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 32,
        openrouter_reasoning={"effort": "medium", "enabled":True},
    ),
    output_retries=3,
)


ANALYST_TASK = """
Your job is to read, carefully and objectively analyse the current game status along with history of events, logs, and the current game strategy (created by the strategist agent), 
and convert it to a well explained clear, concise analysis telling what is going on the game board verbally for the 'executer agent' who is responsible to take concrete actions.

Field executer will read your analysis after each turn to before taking actions. You can highlight key-points 
inside your analysis to the 'executer agent' to make winning easier for him.

- After each turn along with your analysis you can optionally record some key events/facts for future-self (they are only seen by you), like killed entities, fired missiles, anything you seem could be relevant for your future-self to better understand the history. Think of them like history notes.
- You will be given the current strategy along with some re-strategize conditions by the 'strategist    ' specifying you when it is the time to re-plan. Other than that, you are free to decide when to re-plan if you think the current strategy is invalidated or obsolete (a good reason is inferred enemy strategy potentially disrupting ours). Or it has been 10 turns since last re-plan.
- Thus you are responsible to take a 're-strategize' decision based on your analysis. It might mean current strategy phase is over either because it was successful or it was a failure and we need a new plan for the next phase.
- Keep it clear and concise.
"""


@analyst_compact_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    deps = ctx.deps
    team_label = deps.team_name

    strategy_text = (
        deps.strategy_plan.to_text(include_analysis=False)
        if getattr(deps, "strategy_plan", None)
        else "No strategy provided yet."
    )
    history = getattr(deps, "analyst_history", {}) or {}
    history_text = _format_history(
        history,
        getattr(deps, "visible_history", {}) or {},
        getattr(deps, "max_history_turns", 3),
        getattr(deps, "team_name", None),
    )
    prev_turns = [t for t in history.keys() if t < getattr(deps, "current_turn_number", 0)]
    prev_turn = max(prev_turns) if prev_turns else None
    previous_analysis = history[prev_turn].analysis if prev_turn is not None else "None yet."
    prev_heading_suffix = f" (Turn {prev_turn})" if prev_turn is not None else ""
    current_state = deps.current_state or "No current state available."

    return f"""
# YOUR ROLE
You are the analyst supporting the strategist and executer agents for {team_label} Team to win a 2D grid combat game.

---

# YOUR TASK
{ANALYST_TASK}

---

# GAME INFO
{GAME_INFO}

---

# TACTICAL GUIDE
{TACTICAL_GUIDE}

---

# STRATEGIST TELLS YOU
{strategy_text}

---

# OBSERVABLE AVAILABLE HISTORY TO YOU
{history_text}

## Previous Turn Analysis{prev_heading_suffix}
{previous_analysis}

---

# CURRENT GAME STATE
{current_state}

---

# OUTPUT
Use the AnalystCompactOutput schema with:
- analysis: clear narrative for executor with embedded highlights where helpful. Do not recommend actions.
- action_implications: for each entity (key as id, implications as value), analyse each valid action and its short-term implications considering potential enemy moves and other entity moves.
- key_points_for_executor: bullet observations or risks; no action recommendations.
- key_facts: facts for future-self (concise).
- needs_replan: True only if conditions match strategist callbacks or the plan is invalidated.
- replan_reason: short reason if needs_replan is True.

"""

# DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".
### RESPONSE FORMAT
#Respond with tool call 'final_result' using the AnalystCompactOutput schema.