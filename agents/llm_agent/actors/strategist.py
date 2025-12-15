from typing import List, Literal

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO
from agents.llm_agent.prompts.tactics import TACTICAL_GUIDE

load_dotenv()


class PhaseOutline(BaseModel):
    name: str = Field(
        description="Short label for the phase (e.g., 'Phase 1: Secure radar net')."
    )
    goal: str = Field(
        description="One-sentence objective for this phase, no execution detail."
    )


class CurrentPhasePlan(BaseModel):
    phase: str = Field(description="Name/number of the active phase.")
    objective: str = Field(
        description="High-level intent of the active phase in one sentence."
    )
    guidance: List[str] = Field(
        description=(
            "3-6 concise bullet points outlining the intended operational approach "
            "for this phase. Tactical guidance only-avoid unit-level orders."
        )
    )


class RoleAssignment(BaseModel):
    entity_id: int = Field(description="Friendly unit id.")
    role: str = Field(
        description="Single-sentence role for this unit for the current phase."
    )


class GamePlan(BaseModel):
    multi_phase_plan: List[PhaseOutline] = Field(
        description="2-4 phase outline covering the path to victory. Keep each goal concise."
    )
    current_phase_plan: CurrentPhasePlan = Field(
        description="Operational guidance for the active phase without unit-level orders."
    )
    roles: List[RoleAssignment] = Field(
        description="Role per friendly unit (one sentence each) aligned to the current phase."
    )
    callbacks: List[str] = Field(
        description=(
            "Specific observable conditions that should trigger a strategist callback."
        )
    )


STRATEGIST_SYSTEM_PROMPT = f"""
You are the Strategic Director for a 2D combat grid game. The user message contains the latest analysis and state snapshot. Your job is to translate it into a concise, multi-phase plan the execution agent can follow.

Use the GAME INFO and TACTICAL GUIDE as doctrine. 

## GAME INFO
{GAME_INFO}

## TACTICAL GUIDE
{TACTICAL_GUIDE}

## WHAT TO PRODUCE
- Multi-phase outline: 2-4 succinct phases from now to victory; no action detail.
- Current phase plan: Name + objective + 3-6 guidance bullets (operational intent, not per-unit orders).
- Roles: One-sentence objective per friendly unit id for this phase.
- Callback conditions: List specific observable triggers. Keep each condition terse and clear.

## RESPONSE FORMAT
Always response with tool properly formatted tool call ('final_result') using the GamePlan schema.

Stay concise, emphasize clarity, avoid jargon, and never output executable action commands.
"""


strategist_agent = Agent(
    "openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    model_settings=OpenRouterModelSettings(
        # temperature=0.7,
        # top_p=0.8,
        max_tokens=1024 * 32,
        openrouter_reasoning={"effort": "low"},
    ),
    deps_type=GameDeps,
    output_type=GamePlan,
    instructions=STRATEGIST_SYSTEM_PROMPT,
    output_retries=3,
)
