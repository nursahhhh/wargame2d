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
from .scenario import Scenario
from .mechanics import (
    SensorSystem, 
    MovementResolver, 
    CombatResolver, 
    VictoryConditions, 
    VictoryResult,
    ActionResolutionResult,
    CombatResolutionResult,
)


@dataclass
class StepInfo:
    """
    Per-step metadata returned at the end of each step.
    
    Contains the raw movement and combat resolution outputs plus the
    victory check result. The full world is still returned in `state`.
    """
    movement: ActionResolutionResult
    combat: CombatResolutionResult
    victory: VictoryResult

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step info to a plain dict."""
        return {
            "movement": self.movement.to_dict(),
            "combat": self.combat.to_dict(),
            "victory": self.victory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepInfo":
        """Deserialize step info from a dict."""
        return cls(
            movement=ActionResolutionResult.from_dict(data["movement"]),
            combat=CombatResolutionResult.from_dict(data["combat"]),
            victory=VictoryResult.from_dict(data["victory"]),
        )


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
        self._scenario: Optional[Scenario] = None
        
        # Mechanics modules (stateless, can be reused)
        self._sensors = SensorSystem()
        self._movement = MovementResolver()
        self._combat = CombatResolver()
        
        # Victory checker (will be initialized in reset())
        self._victory_checker: Optional[VictoryConditions] = None

    # For continuation games, we might reset the env with scenario and world and fix the initialization logic accordingly.
    def reset(
        self, 
        scenario: Scenario | Dict[str, Any],
        world: WorldState | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Reset the environment with a scenario, optionally using an existing world.
        
        The scenario contains all configuration including grid size,
        game rules, and entities. Scenarios are the ONLY way to configure
        the environment. Optionally, an existing WorldState (or its dict)
        can be provided to resume from a saved state.
        
        Args:
            scenario: Scenario instance or Dict from Scenario.to_dict():
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
            world: Optional WorldState or dict (from WorldState.to_dict()).
                If provided, the environment will resume from this world
                instead of creating a new one from the scenario entities.
        
        Returns:
            Initial state (same structure as step())
        
        Raises:
            ValueError: If scenario is missing required config, or if the
                provided world grid does not match the scenario config.
        """
        # Normalize scenario input (supports Scenario or raw dict)
        if isinstance(scenario, Scenario):
            # Clone to avoid sharing mutable entities with caller
            scenario_obj = scenario.clone()
        else:
            if "config" not in scenario:
                raise ValueError("Scenario must contain 'config' dictionary")
            scenario_obj = Scenario.from_dict(scenario)
        
        self._scenario = scenario_obj
        
        # Initialize victory checker with scenario config
        self._victory_checker = VictoryConditions(
            max_stalemate_turns=scenario_obj.max_stalemate_turns,
            max_no_move_turns=scenario_obj.max_no_move_turns,
            max_turns=scenario_obj.max_turns,
            check_missile_exhaustion=scenario_obj.check_missile_exhaustion
        )
        
        if world is None:
            # Create new world from scenario
            self.world = WorldState(
                width=scenario_obj.grid_width,
                height=scenario_obj.grid_height,
                seed=scenario_obj.seed
            )
            
            # Reset counters (stored in world)
            self.world.turn = 0
            self.world.turns_without_shooting = 0
            self.world.turns_without_movement = 0
            
            # Add entities from scenario
            for entity in scenario_obj.blue_entities:
                self.world.add_entity(entity)
            for entity in scenario_obj.red_entities:
                self.world.add_entity(entity)
        else:
            # Resume from provided world (clone to avoid side effects)
            if isinstance(world, WorldState):
                world_obj = world.clone()
            else:
                world_obj = WorldState.from_dict(world)

            # Validate grid dimensions against scenario config
            grid = world_obj.grid
            if grid.width != scenario_obj.grid_width or grid.height != scenario_obj.grid_height:
                raise ValueError(
                    "Provided world grid size does not match scenario config: "
                    f"world=({grid.width}x{grid.height}), "
                    f"scenario=({scenario_obj.grid_width}x{scenario_obj.grid_height})"
                )

            self.world = world_obj
        
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
            - info: StepInfo - movement/combat/victory metadata
        
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
        info = StepInfo(
            movement=action_results,
            combat=combat_results,
            victory=victory_result
        )
        
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
            "config": {
                "max_stalemate_turns": self._scenario.max_stalemate_turns,
                "max_no_move_turns": self._scenario.max_no_move_turns,
                "max_turns": self._scenario.max_turns,
                "check_missile_exhaustion": self._scenario.check_missile_exhaustion,
            },
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
