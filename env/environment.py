"""
GridCombatEnv - Main environment interface.

This is the primary API for the Grid Combat Environment. It provides
a gym-like interface for running simulations and training agents.

Usage:
    from env import GridCombatEnv, Scenario
    from env.scenario import create_basic_battle
    
    env = GridCombatEnv()
    scenario = create_basic_battle()
    state = env.reset(scenario=scenario.to_dict())
    
    while not done:
        actions = get_actions(state)  # Your AI here
        state, rewards, done, info = env.step(actions)
    
    print(f"Winner: {state['world'].winner}")
    
State Structure:
    {
        "world": WorldState,  # Raw world object (use team views for fog-of-war)
        "tracking": {
            "turns_without_shooting": int,
            "turns_without_movement": int,
        },
        "config": {
            "max_stalemate_turns": int,
            "max_no_move_turns": int,
            "max_turns": int | None,
            "check_missile_exhaustion": bool,
        }
    }
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .core.types import Team, ActionType, GameResult
from .core.actions import Action
from .core.observations import Observation
from .entities.base import Entity
from .entities.sam import SAM
from .world import WorldState
from .mechanics import (
    SensorSystem, 
    MovementResolver, 
    CombatResolver, 
    VictoryConditions, 
    VictoryResult
)


@dataclass
class StepInfo:
    """
    Minimal metadata returned at the end of each step.
    
    Contains basic debugging/metadata information. All game state
    is in the returned state dict.
    """
    pass  # Currently empty, can add debug metadata if needed


class GridCombatEnv:
    """
    Grid Combat Environment - Main simulation interface.
    
    This class orchestrates all subsystems to provide a clean,
    gym-like interface for running combat simulations.
    
    The environment manages:
    - World state (entities, positions, team views)
    - Game mechanics (sensing, movement, combat)
    - Victory conditions
    - Turn counters and stalemate detection
    - Logging and history
    
    Attributes:
        world: Current world state
        turn: Current turn number
        verbose: Whether to print detailed logs
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the Grid Combat Environment.
        
        Args:
            verbose: Print detailed logs each turn (default: False)
        """
        # Logging settings
        self.verbose = verbose
        
        # World state (will be initialized in reset())
        self.world: Optional[WorldState] = None
        
        # Scenario config (will be set in reset())s
        self._max_stalemate_turns: Optional[int] = None
        self._max_no_move_turns: Optional[int] = None
        self._max_turns: Optional[int] = None
        self._check_missile_exhaustion: Optional[bool] = None
        
        # Mechanics modules (stateless, can be reused)
        self._sensors = SensorSystem()
        self._movement = MovementResolver()
        self._combat = CombatResolver()
        
        # Victory checker (will be initialized in reset())
        self._victory_checker: Optional[VictoryConditions] = None

    # For continuation games, we might reset the env with scenario and world and fix the initialization logic accordingly.
    def reset(
        self, 
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reset the environment with a scenario.
        
        The scenario contains all configuration including grid size,
        game rules, and entities. Scenarios are the ONLY way to configure
        the environment.
        
        Args:
            scenario: Dict from Scenario.to_dict():
                {
                    "config": {
                        "grid_width": int,
                        "grid_height": int,
                        "max_stalemate_turns": int,
                        "max_no_move_turns": int,
                        "max_turns": int | None,
                        "check_missile_exhaustion": bool,
                        "seed": int | None
                    },
                    "blue_entities": [entity1, entity2, ...],
                    "red_entities": [entity3, entity4, ...]
                }
        
        Returns:
            Initial state (same structure as step())
        
        Raises:
            ValueError: If scenario is missing required config
        """
        # Extract config from scenario
        if "config" not in scenario:
            raise ValueError("Scenario must contain 'config' dictionary")
        
        scenario_config = scenario["config"]
        
        # Extract grid settings
        grid_width = scenario_config["grid_width"]
        grid_height = scenario_config["grid_height"]
        seed = scenario_config.get("seed", None)
        
        # Extract victory condition settings
        self._max_stalemate_turns = scenario_config["max_stalemate_turns"]
        self._max_no_move_turns = scenario_config["max_no_move_turns"]
        self._max_turns = scenario_config.get("max_turns", None)
        self._check_missile_exhaustion = scenario_config["check_missile_exhaustion"]
        
        # Initialize victory checker with scenario config
        self._victory_checker = VictoryConditions(
            max_stalemate_turns=self._max_stalemate_turns,
            max_no_move_turns=self._max_no_move_turns,
            max_turns=self._max_turns,
            check_missile_exhaustion=self._check_missile_exhaustion
        )
        
        # Create new world
        self.world = WorldState(
            width=grid_width,
            height=grid_height,
            seed=seed
        )
        
        # Reset counters (stored in world)
        self.world.turn = 0
        self.world.turns_without_shooting = 0
        self.world.turns_without_movement = 0
        
        # Add entities from scenario
        for entity in scenario.get("blue_entities", []):
            self.world.add_entity(entity)
        for entity in scenario.get("red_entities", []):
            self.world.add_entity(entity)
        
        # Refresh observations after adding entities
        self._sensors.refresh_all_observations(self.world)
        
        # Return initial state
        return self._build_state()
    
    def step(
        self, 
        actions: Dict[int, Action]
    ) -> Tuple[Dict[str, Any], Dict[Team, float], bool, StepInfo]:
        """
        Execute one turn of the simulation.
        
        This is the main game loop. It processes all actions and returns
        the results.
        
        Game loop order:
        1. Pre-step housekeeping (SAM cooldowns)
        2. Movement + toggles + waits (counter updates handled internally)
        3. Re-sense (after movement)
        4. Combat (counter updates handled internally)
        5. Apply deaths
        6. Check victory
        7. Return results
        
        Args:
            actions: Map of entity_id -> Action
        
        Returns:
            Tuple of (state, rewards, done, info):
            - state: Dict - complete state (shared by both teams)
            - rewards: Dict[Team, float] - reward signal for each team
            - done: bool - whether game is over
            - info: StepInfo - minimal metadata
        
        Raises:
            RuntimeError: If reset() hasn't been called
        """
        if self.world is None:
            raise RuntimeError("Must call reset() before calling step()")
        
        self.world.turn += 1

        self._housekeeping() # Tick SAM cooldowns for now.

        # Resolve all (movement, toggles, waits) actions for a turn.
        # (Counter updates are handled inside resolve_actions)
        action_results = self._movement.resolve_actions(self.world, actions)

        # Resolve all combat actions (including death application) for a turn.
        # (Counter updates are handled inside resolve_combat)
        combat_results = self._combat.resolve_combat(self.world, actions)

        # Resolvers already updates the grid world in-place. Observations are for the fog of war.
        # So we are just updating, which team will see which entities after movement and combat.
        # Sense after resolutions
        self._sensors.refresh_all_observations(self.world)
        
        # Check victory
        victory_result = self._victory_checker.check_all(self.world)
        
        if victory_result.is_game_over:
            self.world.game_over = True
            self.world.winner = victory_result.winner
            self.world.game_over_reason = victory_result.reason
        
        # Build return values
        state = self._build_state()
        rewards = self._calculate_rewards(victory_result)
        info = StepInfo()
        
        return state, rewards, victory_result.is_game_over, info
    
    def _build_state(self) -> Dict[str, Any]:
        """
        Build complete state representation.
        
        The state includes the shared world object (fog-of-war is handled
        via team views), tracking counters, and configuration.
        
        Returns:
            Dictionary with world, tracking, and config
        """
        return {
            "world": self.world,
            "config": { # todo: maybe make those part of world.config?
                "max_stalemate_turns": self._max_stalemate_turns,
                "max_no_move_turns": self._max_no_move_turns,
                "max_turns": self._max_turns,
                "check_missile_exhaustion": self._check_missile_exhaustion,
            }
        }
    
    def _housekeeping(self) -> None:
        """Pre-turn housekeeping tasks."""
        # Tick SAM cooldowns
        for entity in self.world.get_alive_entities():
            if isinstance(entity, SAM):
                entity.tick_cooldown()
    
    def _calculate_rewards(self, victory_result: VictoryResult) -> Dict[Team, float]:
        """
        Calculate rewards for each team.
        
        Simple reward structure:
        - Win: +1.0
        - Loss: -1.0
        - Draw: 0.0
        - In progress: 0.0
        
        Can be extended for more sophisticated reward shaping.
        """
        if not victory_result.is_game_over:
            return {Team.BLUE: 0.0, Team.RED: 0.0}
        
        if victory_result.result == GameResult.BLUE_WINS:
            return {Team.BLUE: 1.0, Team.RED: -1.0}
        elif victory_result.result == GameResult.RED_WINS:
            return {Team.BLUE: -1.0, Team.RED: 1.0}
        else:  # DRAW
            return {Team.BLUE: 0.0, Team.RED: 0.0}
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the environment.
        """
        ...
    
    def close(self) -> None:
        """
        Clean up resources.
        
        Currently a no-op, but provided for gym compatibility.
        """
        pass
    
    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.world is not None and self.world.game_over
    
    @property
    def winner(self) -> Optional[Team]:
        """Get winner (None if draw or in progress)."""
        return self.world.winner if self.world else None
