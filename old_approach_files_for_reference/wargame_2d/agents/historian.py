import json
from dataclasses import dataclass, field
from typing import Literal, List, Annotated, Union, Optional, Dict, Any, Tuple
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from wargame_2d.agents.fixed_prompts import EXECUTER_TASK, GAME_INFO, STRATEGIC_GUIDELINES, ANALYST_TASK
from wargame_2d.agents.state_and_data import GameDeps
import set_api_key


class GameAnalysis(BaseModel):
    analysis: str = Field(description="Your game analysis/suggestions will be given to the field commander directly.")
    key_facts: str = Field(description="Key facts or events to remember for future-self, keep them well formatted and concise use turn numbers for the derived elements.")
    re_strategize: bool = Field(description="Whether or not we need to re-strategize as a team. It is a costly operation only return True when really needed.")
    re_strategize_reason: str = Field(description="Short explanation of why do you think re-strategize is needed. Otherwise keep it empty")

analyst_agent = Agent(
    "openai:gpt-5-mini",
    deps_type=GameDeps,
    output_type=GameAnalysis,
    instructions="You are an AI game analyst for the RED team in a grid-based air combat simulation."
)

@analyst_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:

    return f"""
### TASK
{ANALYST_TASK}


### GAME STATE 
{ctx.deps.game_state}
"""

