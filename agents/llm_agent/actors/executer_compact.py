from typing import List, Literal, Union, Optional, Dict, Annotated

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
        description="Brief rationale explaining how this action aligns with current strategy. Cite specific strategic priorities or threats that motivate this choice."
    )
    action: Action = Field(description="The action this unit will execute")


class TeamTurnPlan(BaseModel):
    """Complete turn plan with per-unit actions."""
    analysis: str = Field(
        description="Action-oriented analysis explaining key priorities for this turn. Reference specific elements from current strategy, analyst highlights, and game state that inform the chosen approach. Highlight what matters most and why."
    )
    entity_actions: List[EntityAction] = Field(
        description="Ordered list of actions for each controllable unit, with reasoning"
    )


def _format_strategy(deps: GameDeps) -> str:
    if not deps.strategy_plan:
        return "No strategy provided yet."
    return deps.strategy_plan.to_text(include_analysis=False, include_callbacks=False)


def _latest_analyst(deps: GameDeps) -> Dict[str, str]:
    if not deps.analyst_history:
        return {"analysis": "None yet.", "highlights": []}
    latest_turn = max(deps.analyst_history.keys())
    latest = deps.analyst_history[latest_turn]
    return {
        "analysis": latest.analysis or "None provided.",
        "highlights": latest.key_points_for_executor or [],
    }


EXECUTER_COMPACT_PROMPT = f"""
# ROLE
You are the Execution Commander. Convert the strategist's plan and analyst highlights into concrete, legal actions for this turn only.

---

# TASK
- Read the current strategy and unit roles, plus the analyst's latest notes.
- Pick one action per friendly unit using only legal actions implied by the current state.
- Keep it simple: no multi-turn plans, no speculative moves outside allowed actions.
- If uncertain or no good move exists, WAIT is acceptable.
- Do not restate full game rules; focus on decisions.

---

# GAME INFO
{GAME_INFO}

---

# RESPONSE FORMAT
Respond with a tool call to 'final_result' with TeamTurnPlan.
"""
#DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".


executer_compact_agent = Agent[GameDeps, TeamTurnPlan](
    "openrouter:openai/gpt-5-mini",#"openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    deps_type=GameDeps,
    output_type=TeamTurnPlan,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 16,
        openrouter_reasoning={"effort": "medium"},
    ),
    instructions=EXECUTER_COMPACT_PROMPT,
    output_retries=3,
)


@executer_compact_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    deps = ctx.deps

    strategy_text = _format_strategy(deps)
    analyst = _latest_analyst(deps)
    highlights = "\n".join(f"- {h}" for h in analyst["highlights"]) if analyst["highlights"] else "- None."

    return f"""
# STRATEGY (current)
{strategy_text}

---

# ANALYSES (latest)
Analysis: {analyst['analysis']}
Key points for executor:
{highlights}

---

# CURRENT GAME STATE
{deps.current_state}

---

# RESPONSE FORMAT
Return a tool call to 'final_result' with TeamTurnPlan.
"""

# DO NOT: Call 'final_result' with a placeholder text like "arguments_final_result".
