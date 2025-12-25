from typing import List, Literal, Union, Optional, Dict, Annotated, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO

load_dotenv()


# --- Action definitions ---
class MoveAction(BaseModel):
    """Command a unit to move one cell in a cardinal direction."""
    type: Literal["MOVE"] = "MOVE"
    entity_id: int = Field(description="ID of the unit to move")
    direction: Literal["UP", "DOWN", "LEFT", "RIGHT"] = Field(
        description="Cardinal direction to move (one cell)"
    )


class ShootAction(BaseModel):
    """Command a unit to fire at an enemy target."""
    type: Literal["SHOOT"] = "SHOOT"
    entity_id: int = Field(description="ID of the unit that will fire")
    target_id: int = Field(description="ID of the enemy unit to target")


class WaitAction(BaseModel):
    """Command a unit to hold position and skip this turn."""
    type: Literal["WAIT"] = "WAIT"
    entity_id: int = Field(description="ID of the unit that will wait")


class ToggleAction(BaseModel):
    """Toggle a unit's special ability or system on/off."""
    type: Literal["TOGGLE"] = "TOGGLE"
    entity_id: int = Field(description="ID of the unit to toggle")
    on: bool = Field(description="True to activate, False to deactivate")


Action = Annotated[Union[MoveAction, ShootAction, WaitAction, ToggleAction], Field(discriminator="type")]


class EntityAction(BaseModel):
    """A single unit's action with tactical justification."""
    reasoning: str = Field(
        description="Brief rationale clearly justifying why this action is chosen for this unit"
    )
    action: Action = Field(description="The action this unit will execute")


class TeamTurnPlan(BaseModel):
    """Complete turn plan with per-unit actions."""
    analysis: str = Field(
        description="Detailed step by step action-oriented analysis explaining key priorities for this turn. Carefully examine current situation, "
                    "enemy strategy, our strategy, inferred enemy actions and ally action implications, and decide best actions accordingly."
    )
    entity_actions: List[EntityAction] = Field(
        description="Ordered list of actions for each controllable unit, with justification"
    )


def _format_strategy(deps: GameDeps) -> str:
    if not deps.strategy_plan:
        return "No strategy provided yet."
    return deps.strategy_plan.to_text(include_analysis=False, include_callbacks=False)

def _format_entity_notes(entries: Dict[Any, Any]) -> str:
    if not entries:
        return "- None."
    lines: List[str] = []
    for entity_id in sorted(entries.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k)):
        raw = str(entries.get(entity_id, "")).strip()
        text = " ".join(raw.split()) if raw else "(none)"
        lines.append(f"- #{entity_id}: {text}")
    return "\n".join(lines)


def _latest_analyst(deps: GameDeps) -> Dict[str, Any]:
    if not deps.analyst_history:
        return {
            "analysis": "None yet.",
            "highlights": [],
            "action_implications": {},
            "action_inferences": {},
        }
    latest_turn = max(deps.analyst_history.keys())
    latest = deps.analyst_history[latest_turn]
    return {
        "analysis": latest.analysis or "None provided.",
        "highlights": latest.key_points_for_executor or [],
        "action_implications": latest.action_implications or {},
        "action_inferences": latest.action_inferences or {},
    }


EXECUTER_COMPACT_PROMPT = f"""
# ROLE
You are the Executer Agent. Read & analyse "strategist" and "analyst" insights carefully, then take legal actions for this turn.
But you have free will to micro-deviate from the plan if the current state demands it. Analyse & decide for yourself to win. You have the field control & responsibility.
Strategist & analyst doesn't see the full state, you do. You are the final decision maker.

---

# TASK
- Read the current strategy and unit roles, plus the analyst's latest notes.
- Pick one action per friendly unit using only legal actions implied by the current state.
- If uncertain or no good move exists, WAIT is acceptable.

---

# GAME INFO
{GAME_INFO}
"""
## RESPONSE FORMAT
#Respond with a tool call to 'final_result' with TeamTurnPlan.
#DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".


executer_agent = Agent[GameDeps, TeamTurnPlan](
    "openrouter:x-ai/grok-4.1-fast",#"openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    deps_type=GameDeps,
    output_type=TeamTurnPlan,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 16,
        openrouter_reasoning={"effort": "medium", "enabled":True},
    ),
    instructions=EXECUTER_COMPACT_PROMPT,
    output_retries=3,
)


@executer_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    deps = ctx.deps

    strategy_text = _format_strategy(deps)
    analyst = _latest_analyst(deps)
    highlights = "\n".join(f"- {h}" for h in analyst["highlights"]) if analyst["highlights"] else "- None."

    return f"""---

# STRATEGIST TELLS:
{strategy_text}

---

# ANALYST TELLS:
Analysis: 
{analyst['analysis']}

Action Inferences:
{_format_entity_notes(analyst['action_inferences'])}

Action Implications:
{_format_entity_notes(analyst['action_implications'])}

Key points for executor:
{highlights}

---

# CURRENT GAME STATE
{deps.current_state}

"""

# DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".
# RESPONSE FORMAT
# Return a tool call to 'final_result' with TeamTurnPlan.
