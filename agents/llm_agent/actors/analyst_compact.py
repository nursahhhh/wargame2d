import json
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO

load_dotenv()


ANALYST_TASK = """
Your job is to read, carefully and objectively analyse the current game status along  with history of events, actions and the current game strategy (created by the director), and convert it to a well explained clear, concise analysis telling what is going on the game board verbally for the 'field executer'.
Field executer will read your analysis after each turn to take actions. You can highlight/suggest some key-points inside your analysis to the 'field executer' to make things easier for him.

- After each turn along with your analysis you can optionally record some key events/facts for future-self (they are only seen by you), like killed entities, fired missiles, anything you seem could be relevant for your future-self to better understand the history.
- You will given the current strategy along with some re-strategize conditions by the 'strategy directory' specifying you when it is the time to re-plan.
- Thus you are responsible to take a 're-strategize' decision based on your analysis. It might mean current strategy phase is over either because it was successful or it was a failure and we need a new plan for the next phase.
- Keep it clear and concise.
"""


class AnalystCompactOutput(BaseModel):
    analysis: str = Field(
        description="Detailed analysis and narration of the board state for field execution. Cite specific board facts and explain your reasoning for each observation."
    )
    key_points_for_executor: List[str] = Field(
        description="Optional bullet highlights or reminders for the executor to pay attention to this turn.",
        default_factory=list,
    )
    key_facts: List[str] = Field(
        description="Facts or events the analyst wants to remember for future turns.",
        default_factory=list,
    )
    needs_replan: bool = Field(
        description="Whether the strategist should be called again to re-plan. Use True only for material shifts."
    )
    replan_reason: str = Field(
        description="Short explanation of why re-planning is needed; empty if no re-plan.",
        default="",
    )


def _strip_turn_prefix(text: str, turn: int) -> str:
    """
    Remove a leading "Turn X" or "TX" label from a fact to avoid duplicated turn labels.
    """
    pattern = re.compile(rf"^(?:turn\s+{turn}|t{turn})\s*[:\-]?\s*", flags=re.IGNORECASE)
    cleaned = pattern.sub("", text.strip())
    return cleaned.strip(":- ").strip() or text.strip()


def _format_key_facts(analyst_history: Dict[int, "AnalystCompactOutput"]) -> str:
    if not analyst_history:
        return "- None recorded yet."
    lines: List[str] = []
    for turn in sorted(analyst_history.keys()):
        key_facts = [f for f in (analyst_history[turn].key_facts or []) if str(f).strip()]
        if not key_facts:
            continue
        lines.append(f"- Turn {turn}:")
        for fact in key_facts:
            cleaned = _strip_turn_prefix(str(fact), turn)
            lines.append(f"  - {cleaned}")
    return "\n".join(lines) if lines else "- None recorded yet."


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
    if not history:
        return "- No visible logs captured yet."
    lines: List[str] = []
    turns = sorted(history.keys())[-max_turns:]
    for turn in turns:
        delta = current_turn - turn
        if delta == 1:
            header = f"**Last turn (Turn {turn}):**"
        elif delta > 1:
            header = f"**{delta} turns ago (Turn {turn}):**"
        else:
            header = f"**Turn {turn}:**"
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

        lines.append(f"{header}")
        lines.append("OUR ACTIONS")
        if our_lines:
            lines.extend([f"  - {l}" for l in our_lines])
        else:
            lines.append("  - None observed.")

        lines.append("")  # spacing between our and enemy sections
        lines.append("ENEMY ACTIONS (Observed)")
        if enemy_lines:
            lines.extend([f"  - {l}" for l in enemy_lines])
        else:
            lines.append("  - None observed.")

        lines.append("")  # blank line between turns for clarity

    return "\n".join(lines).strip()


analyst_compact_agent = Agent[GameDeps, AnalystCompactOutput](
    "openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    deps_type=GameDeps,
    output_type=AnalystCompactOutput,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 32,
        openrouter_reasoning={"effort": "low"},
    ),
    output_retries=3,
)


@analyst_compact_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    deps = ctx.deps
    team_label = deps.team_name

    strategy_text = (
        deps.strategy_plan.to_text(include_analysis=True)
        if getattr(deps, "strategy_plan", None)
        else "No strategy provided yet."
    )
    history = getattr(deps, "analyst_history", {}) or {}
    key_facts = _format_key_facts(history)
    step_logs = _format_step_logs(
        getattr(deps, "visible_history", {}),
        getattr(deps, "max_history_turns", 3),
        getattr(deps, "current_turn_number", 0),
        getattr(deps, "team_name", None),
    )
    prev_turns = [t for t in history.keys() if t < getattr(deps, "current_turn_number", 0)]
    prev_turn = max(prev_turns) if prev_turns else None
    previous_analysis = history[prev_turn].analysis if prev_turn is not None else "None yet."
    prev_heading_suffix = f" (Turn {prev_turn})" if prev_turn is not None else ""
    current_state = deps.current_state or "No current state available."

    return f"""
# ROLE
You are the analyst supporting the strategist and executer agents for {team_label}.

---

# TASK
{ANALYST_TASK}

---

# GAME INFO
{GAME_INFO}

---

# STRATEGY
{strategy_text}

---

# HISTORY
## Key Notes/Facts derived by analyst itself
{key_facts}

## Last {getattr(deps, "max_history_turns", 5)} Turns Observable Step Logs
{step_logs}

## Previous Turn Analysis{prev_heading_suffix}
{previous_analysis}

---

# CURRENT GAME STATE
{current_state}

---

# OUTPUT
Use the AnalystCompactOutput schema with:
- analysis: clear narrative for executor with embedded highlights where helpful.
- key_points_for_executor: bullet reminders if any.
- key_facts: facts for future-self (concise).
- needs_replan: True only if conditions match strategist callbacks or the plan is invalidated.
- replan_reason: short reason if needs_replan is True.

## RESPONSE FORMAT
Respond with tool call 'final_result' using the AnalystCompactOutput schema only.
DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".
"""
