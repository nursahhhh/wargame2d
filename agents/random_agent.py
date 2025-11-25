"""
Random agent implementation for testing and baseline comparison.

This agent makes random valid decisions for all its entities.
"""

import random
from typing import Dict, Any, Optional, TYPE_CHECKING
from env.core.actions import Action
from env.core.types import Team
from env.world import WorldState
from .base_agent import BaseAgent
from .registry import register_agent

if TYPE_CHECKING:
    from env.environment import StepInfo


@register_agent("random")
class RandomAgent(BaseAgent):
    """
    Agent that takes random actions.
    
    Decision process:
    - For each entity, sample uniformly from its allowed actions.
    
    This serves as a baseline for comparing learning agents.
    """
    
    def __init__(
        self, 
        team: Team, 
        name: str = None,
        seed: Optional[int] = None,
        **_: Any,
    ):
        """
        Initialize random agent.
        
        Args:
            team: Team to control
            name: Agent name (default: "RandomAgent")
            seed: Random seed for reproducibility (None = random)
        """
        super().__init__(team, name)
        self.rng = random.Random(seed)
    
    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """
        Generate random actions for all entities by sampling allowed actions.
        
        Args:
            state: Current game state
            step_info: Optional previous step resolution info (unused)
        
        Returns:
            Tuple of (actions, metadata)
        """
        world: WorldState = state["world"]
        actions = {}
        
        for entity in world.get_team_entities(self.team):
            if not entity.alive:
                continue
            allowed = entity.get_allowed_actions(world)
            if not allowed:
                continue
            actions[entity.id] = self.rng.choice(allowed)
        
        metadata = {
            "policy": "random",
            "actions_count": len(actions),
        }
        return actions, metadata
