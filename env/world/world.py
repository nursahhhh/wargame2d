"""
WorldState - Central game state manager.

The WorldState is the heart of the simulation. It:
- Manages all entities
- Coordinates game phases (sense, move, shoot)
- Delegates to mechanics modules for action resolution
- Tracks game state (turn counter, game over, etc.)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Any
import random

from .grid import Grid
from .team_view import TeamView
from ..entities.base import Entity
from ..core.types import Team, GridPos, GameResult
from ..core.actions import Action


class WorldState:
    """
    The central game state.

    WorldState manages:
    - The spatial grid
    - All entities
    - Per-team observation views
    - Game phase and turn tracking

    It does NOT handle:
    - Movement resolution (delegated to MovementResolver)
    - Combat resolution (delegated to CombatResolver)
    - Observation computation (delegated to SensorSystem)
    - Victory checking (delegated to VictoryConditions)
    """

    def __init__(
            self,
            width: int,
            height: int,
            seed: Optional[int] = None
    ):
        """
        Initialize a new world.

        Args:
            width: Grid width
            height: Grid height
            seed: Random seed for reproducibility
        """
        # Spatial grid
        self.grid = Grid(width, height)

        # Entity management
        self._entities: List[Entity] = []
        self._entities_by_id: Dict[int, Entity] = {}

        # Per-team intelligence
        self._team_views: Dict[Team, TeamView] = {
            Team.BLUE: TeamView(Team.BLUE),
            Team.RED: TeamView(Team.RED),
        }

        # Game state
        self.turn: int = 0
        self.game_over: bool = False
        self.winner: Optional[Team] = None
        self.game_over_reason: str = ""
        self.turns_without_shooting: int = 0
        self.turns_without_movement: int = 0

        # Random number generator
        self.rng = random.Random(seed)

        # Kill tracking (for this turn)
        self._pending_kills: Set[int] = set()

    # ========================================================================
    # ENTITY MANAGEMENT
    # ========================================================================

    def add_entity(self, entity: Entity) -> int:
        """
        Add an entity to the world.

        Args:
            entity: Entity to add

        Returns:
            Entity ID

        Raises:
            ValueError: If entity position is invalid or occupied
        """
        if not self.grid.in_bounds(entity.pos):
            raise ValueError(f"Entity position out of bounds: {entity.pos}")

        if self.is_position_occupied(entity.pos):
            raise ValueError(f"Position already occupied: {entity.pos}")

        self._entities.append(entity)
        self._entities_by_id[entity.id] = entity

        return entity.id

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID to look up

        Returns:
            Entity if found, None otherwise
        """
        return self._entities_by_id.get(entity_id)

    def get_all_entities(self) -> List[Entity]:
        """Get all entities (including dead ones)."""
        return self._entities.copy()

    def get_alive_entities(self) -> List[Entity]:
        """Get all living entities."""
        return [e for e in self._entities if e.alive]

    def get_team_entities(self, team: Team, alive_only: bool = True) -> List[Entity]:
        """
        Get entities belonging to a team.

        Args:
            team: Team to filter by
            alive_only: If True, only return living entities

        Returns:
            List of entities
        """
        entities = [e for e in self._entities if e.team == team]
        if alive_only:
            entities = [e for e in entities if e.alive]
        return entities

    def is_position_occupied(self, pos: GridPos) -> bool:
        """
        Check if a position is occupied by a living entity.

        Args:
            pos: Position to check

        Returns:
            True if occupied by living entity
        """
        return any(e.alive and e.pos == pos for e in self._entities)

    # ========================================================================
    # TEAM VIEW ACCESS
    # ========================================================================

    def get_team_view(self, team: Team) -> TeamView:
        """
        Get the observation view for a team.

        Args:
            team: Team to get view for

        Returns:
            TeamView for that team
        """
        return self._team_views[team]

    # ========================================================================
    # KILL TRACKING (state only - mechanics apply the kills)
    # ========================================================================
    def mark_for_kill(self, entity_id: int) -> None:
        """
        Mark an entity to be killed at end of turn.

        Args:
            entity_id: ID of entity to kill
        """
        self._pending_kills.add(entity_id)

    def get_pending_kills(self) -> Set[int]:
        """
        Get set of entity IDs marked for death.

        Returns:
            Set of entity IDs to be killed
        """
        return self._pending_kills.copy()

    def clear_pending_kills(self) -> None:
        """Clear all pending kills (after they've been applied)."""
        self._pending_kills.clear()

    # ========================================================================
    # UTILITY
    # ========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize world state to dictionary.

        Returns:
            JSON-serializable dictionary of complete game state
        """
        return {
            "grid": {
                "width": self.grid.width,
                "height": self.grid.height,
            },
            "entities": [entity.to_dict() for entity in self._entities],
            "team_views": {
                team.name: team_view.to_dict()
                for team, team_view in self._team_views.items()
            },
            "turn": self.turn,
            "game_over": self.game_over,
            "winner": self.winner.name if self.winner else None,
            "game_over_reason": self.game_over_reason,
            "turns_without_shooting": self.turns_without_shooting,
            "turns_without_movement": self.turns_without_movement,
            "rng_state": self.rng.getstate(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorldState:
        """
        Deserialize world state from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed WorldState
        """
        # Create empty world
        world = cls(
            width=data["grid"]["width"],
            height=data["grid"]["height"],
            seed=None
        )

        # Restore game state
        world.turn = data["turn"]
        world.game_over = data["game_over"]
        world.winner = Team[data["winner"]] if data["winner"] else None
        world.game_over_reason = data["game_over_reason"]
        world.turns_without_shooting = data.get("turns_without_shooting", 0)
        world.turns_without_movement = data.get("turns_without_movement", 0)
        
        # Convert rng_state to tuple (JSON converts tuples to lists)
        # Random state is: (version, (624 integers..., position), gauss_next)
        rng_state = data["rng_state"]
        if isinstance(rng_state, list):
            # Convert main tuple and nested tuple
            inner_tuple = tuple(rng_state[1]) if isinstance(rng_state[1], list) else rng_state[1]
            rng_state = (rng_state[0], inner_tuple, rng_state[2])
        world.rng.setstate(rng_state)

        # Reconstruct entities (they handle their own deserialization!)
        for entity_data in data["entities"]:
            entity = Entity.from_dict(entity_data)

            # Add to world (bypass validation since we're restoring state)
            world._entities.append(entity)
            world._entities_by_id[entity.id] = entity

        # Restore team views (if present in data - backward compatibility)
        if "team_views" in data:
            for team_name, team_view_data in data["team_views"].items():
                team = Team[team_name]
                world._team_views[team] = TeamView.from_dict(team_view_data)
        
        # Note: After loading, call SensorSystem.refresh_all_observations()
        # to regenerate derived state (observations, visible enemies, etc.)

        return world

    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> str:
        """
        Serialize to JSON.

        Args:
            filepath: If provided, write to file
            indent: JSON indentation (default: 2)

        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=indent, ensure_ascii=True)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, json_str: Optional[str] = None, filepath: Optional[str] = None) -> WorldState:
        """
        Deserialize from JSON.

        Args:
            json_str: JSON string to parse
            filepath: If provided, read from file instead

        Returns:
            Reconstructed WorldState

        Raises:
            ValueError: If neither json_str nor filepath provided
        """
        if filepath:
            with open(filepath, 'r') as f:
                json_str = f.read()

        if not json_str:
            raise ValueError("Must provide either json_str or filepath")

        data = json.loads(json_str)
        return cls.from_dict(data)

    def clone(self) -> WorldState:
        """
        Create a deep copy of this world state.

        Useful for simulation, look-ahead planning, or testing.

        Returns:
            Independent copy of this WorldState
        """
        return WorldState.from_dict(self.to_dict())

    def __str__(self) -> str:
        """String representation."""
        alive = len(self.get_alive_entities())
        total = len(self._entities)
        return f"WorldState(turn={self.turn}, entities={alive}/{total}, grid={self.grid})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"WorldState(grid={self.grid}, entities={len(self._entities)}, "
                f"turn={self.turn}, game_over={self.game_over})")