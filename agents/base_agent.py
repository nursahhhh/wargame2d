"""
Base agent interface for the Grid Combat Environment.

All agents must implement this interface to interact with the environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
from env.core.actions import Action
from env.core.types import Team

if TYPE_CHECKING:
    from env.environment import StepInfo


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents observe the game state and produce actions for their entities.
    The environment handles fog-of-war through team views.
    
    Subclasses must implement:
    - get_actions(): Produce actions for all controlled entities
    
    Attributes:
        team: The team this agent controls (BLUE or RED)
        name: Agent name for logging/identification
    """
    
    def __init__(self, team: Team, name: str = None):
        """
        Initialize the agent.
        
        Args:
            team: Team this agent controls
            name: Optional name for the agent (defaults to class name)
        """
        self.team = team
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """
        Get actions for all controlled entities.
        
        This is called once per turn. The agent should:
        1. Extract relevant information from the state
        2. Use team_view for fog-of-war observations
        3. Optionally consume previous StepInfo (movement/combat/victory)
        4. Decide on actions for each entity
        5. Return (actions, metadata)
        
        State structure:
            {
                "world": WorldState,  # Contains team_views
                "config": {
                    "max_stalemate_turns": int,
                    "max_no_move_turns": int,
                    "check_missile_exhaustion": bool,
                }
            }
        step_info:
            Optional per-turn resolution info from the previous step.
            Agents must still respect fog-of-war if they use it.
        **kwargs:
            Reserved for future fields (e.g., history).
        
        To get your entities:
            world = state["world"]
            my_entities = world.get_team_entities(self.team)
        
        To get observations (fog-of-war):
            team_view = world.get_team_view(self.team)
            enemy_ids = team_view.get_enemy_ids(self.team)
            all_observations = team_view.get_all_observations()
        
        Args:
            state: Current game state from environment
        
        Returns:
            Tuple of:
                - Dict mapping entity_id to Action for each entity
                - Metadata dict (reasoning/logs/errors/etc.)
            
        Notes:
            - Must return actions for ALL alive entities on your team
            - Dead entities should not have actions
            - Use Action.wait() if no action desired
            - Invalid actions will be ignored by the environment
        """
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.team.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(team={self.team.name}, name='{self.name}')"
