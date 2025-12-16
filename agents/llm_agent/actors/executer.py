import json
from typing import Literal, Union, List, Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO
from agents.llm_agent.prompts.tactics import TACTICAL_GUIDE


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


Action = Annotated[
    Union[MoveAction, ShootAction, WaitAction, ToggleAction],
    Field(discriminator="type"),
]


# --- Entity-level reasoning wrapper ---
class EntityAction(BaseModel):
    """A single unit's action with tactical justification."""
    reasoning: str = Field(
        description="Brief tactical rationale for this action aligned to current phase and role."
    )
    action: Action = Field(description="The action this unit will execute")


class TeamAction(BaseModel):
    """Complete turn plan with per-unit actions."""
    analysis: str = Field(
        description=(
            "Concise rationale tying chosen actions to the current phase objective and roles. "
            "Do not restate the full game rules."
        )
    )
    entity_actions: List[EntityAction] = Field(
        description="Ordered list of actions for each controllable unit, with reasoning"
    )


# --- System prompt template ---
EXECUTER_SYSTEM_PROMPT = f"""
You are the Execution Commander. Your purpose is to convert the strategist's current plan into concrete actions for each friendly unit this turn.
Your job is to pick the best action for each unit considering the current game state, the strategist's current phase objective and each unit's assigned role."""


# --- Agent definition ---
player = Agent(
    "openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    # model_settings=ModelSettings(
    #     temperature=0.7,
    #     top_p=0.8,
    #     max_tokens=32_000,
    #     extra_body={
    #         "top_k": 20,
    #         "min_p": 0,
    #         "repetition_penalty": 1.05,
    #     }
    # ),
    retries=3,
    output_retries=3,
    output_type=TeamAction,
    instructions=EXECUTER_SYSTEM_PROMPT,
)


@player.instructions
def execution_instructions(ctx: RunContext[GameDeps]) -> str:
    deps = ctx.deps or GameDeps()

    strategy_json = (
        json.dumps(deps.current_phase_strategy, indent=2, ensure_ascii=False)
        if deps.current_phase_strategy
        else "No current phase strategy."
    )

    roles = (
        "\n".join(f"- Unit {unit_id}: {role}" for unit_id, role in deps.entity_roles.items())
        if deps.entity_roles
        else "- No explicit roles provided."
    )

    return f"""
## CURRENT STRATEGY
{strategy_json}

### ROLES THIS PHASE
{roles}

Game rules are provided below. 
## GAME RULES
{GAME_INFO}

## RESPONSE FORMAT
Output the TeamAction schema only. Avoid narration outside the schema.
"""
