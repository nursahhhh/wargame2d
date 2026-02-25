# commander_agent.py
from pathlib import Path
from typing import Optional
from .llm_utils import COMMANDER_TOOL, CommanderIntent, GameDeps
from pydantic_ai.agent import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

class CommanderAgent:
    def __init__(self, api_key: str, run_log_dir: Optional[Path] = None):
        self.api_key = api_key
        self.run_log_dir = run_log_dir

        # Initialize the Agent using COMMANDER_TOOL
        self.agent: Agent[GameDeps, CommanderIntent] = Agent[
            GameDeps, CommanderIntent
        ](
            model="openrouter:x-ai/grok-4.1-fast",
            deps_type=GameDeps,
            output_type=CommanderIntent,
            model_settings=OpenRouterModelSettings(
                api_key=self.api_key,
                max_tokens=16000,
                logfire_reasoning={"enabled": True, "effort": "high"}
            ),
            instructions="""
You are a tactical commander AI responsible for high-level strategic planning.
Focus on strategy, not low-level moves. Respond only using COMMANDER_TOOL format.
""",
            tools=[COMMANDER_TOOL],
            output_retries=3,
        )

    def decide_intent(
        self,
        intel,
        step: int,
        escalation_reason: Optional[str] = None,
        observed_shift: Optional[str] = None,
        urgency: Optional[int] = 3
    ) -> CommanderIntent:
        """
        Given the current game intel, return a high-level strategic intent.
        """
        # Wrap intel in GameDeps for the agent
        deps = GameDeps(intel=intel, step=step)

        # Run agent and get CommanderIntent
        result: CommanderIntent = self.agent.run(deps)

        # You can optionally attach escalation info to reasoning
        if not result.justification:
            result.justification = f"Escalation: {escalation_reason or 'N/A'}, urgency={urgency}"

        return result