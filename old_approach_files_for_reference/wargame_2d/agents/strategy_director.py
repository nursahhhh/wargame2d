"""
commander_agent is used to read the high level game state and summaries etc... and create the game strategy
it is called at the beginning and also only if the executer agent has no trust left to the current gameplan working
to re-strategize!
"""

import json
from dataclasses import dataclass, field
from typing import Literal, List, Annotated, Union, Optional, Dict, Any, Tuple
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
import set_api_key
from wargame_2d.agents.fixed_prompts import GAME_INFO, STRATEGIC_GUIDELINES, DIRECTOR_TASK
from wargame_2d.agents.state_and_data import GameDeps


# --- Long-lived strategy pieces with explicit change-status ---
class TeamStrategy(BaseModel):
    text: str = Field(description="High-level short-term gameplan to win as a team.")
    call_me_back_if: str = Field(description="Clear concise set of conditions telling when do we re-strategize.")

class EntityDirective(BaseModel):
    entity_id: int
    directive: str = Field(description="High-level short-term standing role and order for this entity to execute the current strategy.")
    pseudo_code_directive: str = Field(description="A a clear, easy to understand  concept level pseudo-code-like  directive explaining how this entity should act on different cases. No need to specify the loop, focus on logic a single step.")

# --- Tactical plan ---
class FirstTacticalPlan(BaseModel):
    analysis: str = Field(description="Step-by-step analysis of current state, threats, opportunities")
    strategy: TeamStrategy
    entity_directives: List[EntityDirective]

# Get LLM actions
strategic_director = Agent(
   "openai:gpt-5-mini",
    deps_type=GameDeps,
    output_type=FirstTacticalPlan,
    instructions="You are a strategic director leading the RED team in a grid-based air combat simulation."
)

@strategic_director.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:
    return f"""
### TASK
{DIRECTOR_TASK}

{GAME_INFO}

{STRATEGIC_GUIDELINES}

### GAME STATE 
{ctx.deps.game_state}
"""
