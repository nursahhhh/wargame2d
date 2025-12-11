import json
from typing import List, Literal, Optional

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic import BaseModel, Field

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.tactics import TACTICAL_GUIDE
from agents.llm_agent.prompts.analyst import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_PROMPT_TEMPLATE,
    HISTORY_SECTION_TEMPLATE,
)


class Position(BaseModel):
    x: int = Field(description="X coordinate on the grid.")
    y: int = Field(description="Y coordinate on the grid.")


class Action(BaseModel):
    """
    Flat action schema for simpler LLM outputs. Type is an enum; other fields are conditional by type.
    """

    type: Literal["MOVE", "SHOOT", "TOGGLE", "WAIT"] = Field(
        description="Action keyword. Allowed: MOVE | SHOOT | TOGGLE | WAIT.",
        examples=["MOVE", "SHOOT", "TOGGLE", "WAIT"],
    )
    direction: Optional[Literal["UP", "DOWN", "LEFT", "RIGHT"]] = Field(
        default=None,
        description="MOVE only: direction to move one cell.",
        examples=["UP", "DOWN", "LEFT", "RIGHT"],
    )
    destination: Optional[Position] = Field(
        default=None,
        description="MOVE only: destination after moving (x,y).",
    )
    target: Optional[int] = Field(
        default=None,
        description="SHOOT only: enemy unit id to target.",
    )
    on: Optional[bool] = Field(
        default=None,
        description="TOGGLE only: true to activate SAM radar/weapon system, false to go dark/stealth (SAM units only).",
    )

class ActionAnalysis(BaseModel):
    action: Action = Field(description="Specific action being considered for the unit.")
    implication: str = Field(description="Expected tactical effect or tradeoff of this action.")

class UnitInsight(BaseModel):
    unit_id: int = Field(description="Identifier for the unit in the current game_state.")
    role: str = Field(description="Role or mission context of the unit.")
    key_considerations: List[str] = Field(
        description="Bullet points on threats, resources, positioning, or timing relevant to this unit."
    )
    action_analysis: List[ActionAnalysis] = Field(
        description="Action options for the unit with their implications. Include all feasible options, even 'WAIT'."
    )

class GameAnalysis(BaseModel):
    unit_insights: List[UnitInsight] = Field(
        description="Unit-level analysis items. Start with the most threatened or impactful units."
    )
    spatial_status: str = Field(
        description="Short narrative of formation posture, positioning relative to enemies, and maneuver space."
    )
    critical_alerts: List[str] = Field(
        description="Ordered list of urgent risks that demand commander attention, prefixed with severity."
    )
    opportunities: List[str] = Field(
        description="Offensive or positional openings the team can exploit, prefixed with severity."
    )
    constraints: List[str] = Field(
        description="Key limitations such as ammo, detection gaps, terrain edges, or coordination risks."
    )
    situation_summary: str = Field(
        description="Overall tactical snapshot combining threats, openings, and intent for the next turn."
    )

analyst_agent = Agent(
    "openrouter:qwen/qwen3-coder:exacto",
    model_settings=ModelSettings(
        temperature=0.6,
        top_p=0.95,
        max_tokens=32_000,
        extra_body={
            "top_k": 20,
            "min_p": 0
        }
    ),
    deps_type=GameDeps,
    output_type=GameAnalysis,
    instructions=ANALYST_SYSTEM_PROMPT,
)

@analyst_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    history_section = ""
    if ctx.deps.step_info_list:
        history_section = HISTORY_SECTION_TEMPLATE.format(
            history_json=json.dumps(ctx.deps.step_info_list, default=str, indent=2)
        )

    game_info = {
        "current_turn": ctx.deps.current_turn_number,
        "multi_phase_strategy": ctx.deps.multi_phase_strategy,
        "current_phase_strategy": ctx.deps.current_phase_strategy,
        "entity_roles": ctx.deps.entity_roles,
        "callback_conditions": ctx.deps.callback_conditions,
    }

    return ANALYST_USER_PROMPT_TEMPLATE.format(
        game_state_json=json.dumps(ctx.deps.game_state, default=str, indent=2),
        game_info=json.dumps(game_info, default=str, indent=2),
        tactical_guide=TACTICAL_GUIDE,
        history_section=history_section.strip(),
    )
