# llm_hybrid_agent.py
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import os
import asyncio

from env.core.actions import Action
from env.core.types import Team
from env.world import WorldState
from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from ._prompt_formatter_ import PromptFormatter
from .llm_utils import GameDeps, executor_agent, TeamTurnPlan,commander_agent

@register_agent("llm_hybrid_agent")
class LLMHybridAgent(BaseAgent):
    def __init__(
        self, 
        team: Team, 
        name: str = "LLMHybridAgent", 
        api_key: Optional[str] = None, 
        memory_window: int = 5
    ):
        super().__init__(team, name)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.memory_window = memory_window
        self.recent_history: List[str] = []
        self.current_strategy: Optional[Dict[str, Any]] = None
        self.step_counter = 0
        self.prompt_formatter = PromptFormatter()

        # Optional: logging folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_log_dir = Path("llm_runs") / timestamp
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        # Attach the executor agent
        self.executor_agent = executor_agent
        self.commander_agent = commander_agent

    import asyncio

    def get_actions(
        self,
        state: Dict[str, Any],
        **kwargs
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:

        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)
        deps = GameDeps(intel=intel, step=self.step_counter)

        if self.step_counter == 0:
            commander_result = asyncio.run(
                self.commander_agent.run(
                    user_prompt="Generate overall strategy for this game.",
                    deps=deps
                )
            )
            self.current_strategy = commander_result.output

        executor_result = asyncio.run(
        self.executor_agent.run(
            user_prompt="Execute current strategy and decide actions.",
            deps={
                "intel": intel,
                "step": self.step_counter,
                "strategy": self.current_strategy
            }
        )
    )

      

        # Convert TeamTurnPlan → dict[int, Action]
        allowed_actions = self._build_allowed_actions_map(world,intel)
        plan = executor_result.output

        actions_dict: Dict[int, Action] = {}
        for ua in plan.actions:
            ent_id = ua.entity_id
            act_name = ua.action.upper()
            if ent_id not in allowed_actions:
                continue
            for act in allowed_actions[ent_id]:
                if act.type.name.upper() == act_name:
                    actions_dict[ent_id] = act
                    break

        metadata = {
            "parsed_actions": plan.actions,
            "current_strategy": self.current_strategy
        }

        self.step_counter += 1
        return actions_dict, metadata

    def _build_allowed_actions_map(self, world: WorldState, intel: TeamIntel) -> Dict[int, List[Action]]:
        """
        Build a mapping of entity_id -> List[Action] from current world state.
        """
        allowed_map: Dict[int, List[Action]] = {}
        for entity in intel.friendlies:
            if not entity.alive:
                continue
            allowed_map[entity.id] = list(entity.get_allowed_actions(world))
        return allowed_map

    # Optional helper to maintain memory / context
    def _build_context_prompt(self, current_prompt: str) -> str:
        history_text = "\n\n".join(self.recent_history[-self.memory_window:])
        self.recent_history.append(current_prompt)
        return f"Recent history:\n{history_text}\nCurrent situation:\n{current_prompt}"