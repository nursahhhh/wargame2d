"""
Base agent interface for the Grid Combat Environment.

All agents must implement this interface to interact with the environment.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from env.core.actions import Action
from env.core.types import Team


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
    def get_actions(self, state: Dict[str, Any]) -> Dict[int, Action]:
        """
        Get actions for all controlled entities.
        
        This is called once per turn. The agent should:
        1. Extract relevant information from the state
        2. Use team_view for fog-of-war observations
        3. Decide on actions for each entity
        4. Return a dict mapping entity_id -> Action
        
        State structure:
            {
                "world": WorldState,  # Contains team_views
                "config": {
                    "max_stalemate_turns": int,
                    "max_no_move_turns": int,
                    "check_missile_exhaustion": bool,
                }
            }
        
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
            Dict mapping entity_id to Action for each entity
            
        Notes:
            - Must return actions for ALL alive entities on your team
            - Dead entities should not have actions
            - Use Action.wait() if no action desired
            - Invalid actions will be ignored by the environment
        """
        pass
    
    def reset(self) -> None:
        """
        Reset agent state between episodes.
        
        Called at the start of each new game. Override if your agent
        maintains internal state that needs to be reset.
        """
        pass
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} ({self.team.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(team={self.team.name}, name='{self.name}')"

