from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.prompts.game_info import GAME_INFO
from agents.llm_agent.prompts.tactics import TACTICAL_GUIDE

load_dotenv()


class UnitStrategy(BaseModel):
    entity_id: int = Field(description="Friendly unit id.")
    role: str = Field(description="One-sentence role/priority for this unit aligned to the plan; include stance if useful.")


class StrategyOutput(BaseModel):
    analysis: str = Field(description="Concise analysis of the current state (threats, advantages, constraints).")
    strategy: str = Field(description="High-level short-term gameplan to win as a team.")
    unit_strategies: List[UnitStrategy] = Field(description="Per-unit roles and postures for all alive friendlies.")
    call_me_back_if: List[str] = Field(description="Observable re-strategize triggers (concise conditions).")

    def to_text(self, include_analysis: bool = True) -> str:
        """
        Render a human-friendly string summary of the strategy output.

        Args:
            include_analysis: Whether to include the analysis section at the top.
        """
        lines: List[str] = []
        if include_analysis:
            lines.append("ANALYSIS")
            lines.append(self.analysis.strip())
            lines.append("")

        lines.append("TEAM STRATEGY")
        lines.append(self.strategy.strip())
        lines.append("")

        lines.append("UNIT STRATEGIES")
        for us in self.unit_strategies:
            lines.append(f"- #{us.entity_id}: {us.role.strip()}")
        lines.append("")

        lines.append("CALL ME BACK IF")
        for cond in self.call_me_back_if:
            lines.append(f"- {cond.strip()}")
        lines.append("")

        return "\n".join(lines).strip()


STRATEGIST_COMPACT_PROMPT = f"""
# ROLE
You are the Strategic Director for a 2D combat grid game. 

---

# DIRECTOR TASK
Analyze the current game rules and tactical state carefully. Identify the key advantages, disadvantages, 
and potential winning conditions. Then, develop a short-term strategic plan that covers:

1. A team-wide short-term strategy describing how the team should operate currently to achieve victory.

2. Individual unit strategies for each alive entity (e.g., AWACS, Aircraft, SAM, Decoy) that define their current roles, priorities, and coordination patterns.

3. Clear re-strategize triggers: Define specific conditions that would invalidate the current plan and require a new strategy (e.g., loss of critical units, mission objective completed, enemy formation changes). Re-strategizing is costly—only trigger when the situation fundamentally changes.

Act as a tactical director, not a field commander — focus on high-level, enduring strategy rather than turn-by-turn or micro-management decisions. 
Don't overcomplicate stuff, it is a simple game.

---

# REFERENCES
## GAME INFO
{GAME_INFO}

## TACTICAL GUIDE
{TACTICAL_GUIDE}

---

# OUTPUT
## EXPECTED OUTPUT (StrategyOutput)
- analysis: concise take on state (threats, advantages, constraints).
- strategy: team-level short-term gameplan (no micro orders).
- unit_strategies: per-unit role + posture for each alive friendly.
- call_me_back_if: observable, concise triggers to re-strategize.

---

## RESPONSE FORMAT
Return a tool call to 'final_result' using StrategyOutput schema only. No prose outside the tool call.
"""


strategist_compact_agent = Agent(
    "openrouter:deepseek/deepseek-v3.1-terminus:exacto",
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 32,
        openrouter_reasoning={"effort": "low"},
    ),
    deps_type=GameDeps,
    output_type=StrategyOutput,
    instructions=STRATEGIST_COMPACT_PROMPT,
    output_retries=3,
)
