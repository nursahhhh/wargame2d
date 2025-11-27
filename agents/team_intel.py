from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from env.core.types import EntityKind, GridPos, Team, MoveDir
from env.entities.base import Entity
from env.mechanics import hit_probability
from env.world.grid import Grid
from env.world.team_view import TeamView

if TYPE_CHECKING:
    from env.world.world import WorldState


@dataclass(frozen=True)
class VisibleEnemy:
    """
    Fog-limited snapshot of a currently observed enemy.
    """

    id: int
    team: Team
    position: GridPos
    kind: EntityKind
    has_fired_before: bool
    seen_by: Set[int]


@dataclass(frozen=True)
class TeamIntel:
    """
    Safe, per-team view of the world for agent decision-making.

    - friendlies: full Entity objects (all fields are fair for your own team)
    - visible_enemies: limited snapshots built from observations
    """

    grid: Grid
    friendlies: List[Entity]
    visible_enemies: List[VisibleEnemy]
    friendly_ids: Set[int]
    visible_enemy_ids: Set[int]

    def get_friendly(self, entity_id: int) -> Optional[Entity]:
        return next((e for e in self.friendlies if e.id == entity_id), None)

    def get_enemy(self, entity_id: int) -> Optional[VisibleEnemy]:
        return next((e for e in self.visible_enemies if e.id == entity_id), None)

    def enemies_in_range(self, entity: Entity, max_range: float) -> List[VisibleEnemy]:
        """Return visible enemies within range of a friendly entity."""
        return [
            enemy
            for enemy in self.visible_enemies
            if self.grid.distance(entity.pos, enemy.position) <= max_range
        ]

    # ------------------------------------------------------------------
    # Foundational helpers
    # ------------------------------------------------------------------
    def friendly_positions(self, *, alive_only: bool = True) -> Dict[int, GridPos]:
        """Map of friendly entity_id -> position."""
        return {
            e.id: e.pos
            for e in self.friendlies
            if e.alive or not alive_only
        }

    def visible_enemy_positions(self) -> Dict[int, GridPos]:
        """Map of visible enemy entity_id -> position."""
        return {e.id: e.position for e in self.visible_enemies}

    def is_in_bounds(self, pos: GridPos) -> bool:
        """Check if position is inside the grid."""
        return self.grid.in_bounds(pos)

    def is_occupied(
        self,
        pos: GridPos,
        *,
        include_friendlies: bool = True,
        include_visible_enemies: bool = True,
        ignore_ids: Optional[Set[int]] = None,
        alive_only: bool = True,
    ) -> bool:
        """
        Check whether a position is occupied by known entities.

        Args:
            pos: Position to check
            include_friendlies: Whether to consider friendly units
            include_visible_enemies: Whether to consider currently observed enemies
            ignore_ids: Entity IDs to ignore (e.g., the moving entity itself)
            alive_only: Whether to treat dead friendlies as occupying space
        """
        ignore = ignore_ids or set()

        if include_friendlies:
            for friendly in self.friendlies:
                if friendly.id in ignore:
                    continue
                if (friendly.alive or not alive_only) and friendly.pos == pos:
                    return True

        if include_visible_enemies:
            for enemy in self.visible_enemies:
                if enemy.id in ignore:
                    continue
                if enemy.position == pos:
                    return True

        return False

    # ------------------------------------------------------------------
    # Targeting helpers
    # ------------------------------------------------------------------
    def nearest_visible_enemy(
        self, origin: GridPos
    ) -> Optional[tuple[VisibleEnemy, float]]:
        """
        Find the closest observed enemy to a position.

        Returns:
            Tuple of (enemy, distance) or None if no enemies are visible.
        """
        nearest: Optional[tuple[VisibleEnemy, float]] = None
        for enemy in self.visible_enemies:
            dist = self.grid.distance(origin, enemy.position)
            if nearest is None or dist < nearest[1]:
                nearest = (enemy, dist)
        return nearest

    def estimate_hit_probability(
        self,
        attacker: Entity,
        target: VisibleEnemy,
    ) -> Optional[float]:
        """
        Estimate hit probability for attacker -> target using attacker stats.

        Returns:
            Probability in [0,1] if attacker has required fields, else None.
        """
        required_fields = ("missile_max_range", "base_hit_prob", "min_hit_prob")
        if not all(hasattr(attacker, field) for field in required_fields):
            return None

        distance = self.grid.distance(attacker.pos, target.position)
        max_range = getattr(attacker, "missile_max_range")
        base = getattr(attacker, "base_hit_prob")
        min_p = getattr(attacker, "min_hit_prob")
        return hit_probability(distance=distance, max_range=max_range, base=base, min_p=min_p)

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    def move_toward(
        self,
        start: GridPos,
        target: GridPos,
        *,
        blocked: Optional[Set[GridPos]] = None,
        ignore_ids: Optional[Set[int]] = None,
    ) -> List[MoveDir]:
        """
        Suggest movement directions that reduce distance to a target.

        Directions are ordered to shrink the largest delta first (e.g., close
        a long horizontal gap before a short vertical one) and filtered to
        avoid leaving bounds or stepping onto occupied/blocked cells.
        """
        if start == target:
            return []

        blocked_positions = blocked or set()
        dx = target[0] - start[0]
        dy = target[1] - start[1]

        # Prioritize the axis with the larger absolute delta
        primary_axis_is_x = abs(dx) >= abs(dy)
        directions: List[MoveDir] = []

        if primary_axis_is_x and dx != 0:
            directions.append(MoveDir.RIGHT if dx > 0 else MoveDir.LEFT)
        if not primary_axis_is_x and dy != 0:
            directions.append(MoveDir.UP if dy > 0 else MoveDir.DOWN)

        # Secondary axis if there's still distance to close
        if primary_axis_is_x and dy != 0:
            directions.append(MoveDir.UP if dy > 0 else MoveDir.DOWN)
        if not primary_axis_is_x and dx != 0:
            directions.append(MoveDir.RIGHT if dx > 0 else MoveDir.LEFT)

        valid: List[MoveDir] = []
        for direction in directions:
            nx = start[0] + direction.delta[0]
            ny = start[1] + direction.delta[1]
            next_pos = (nx, ny)
            if not self.grid.in_bounds(next_pos):
                continue
            if next_pos in blocked_positions:
                continue
            if self.is_occupied(next_pos, ignore_ids=ignore_ids):
                continue
            valid.append(direction)

        return valid

    def move_away(
        self,
        start: GridPos,
        threat: GridPos,
        *,
        blocked: Optional[Set[GridPos]] = None,
        ignore_ids: Optional[Set[int]] = None,
    ) -> List[MoveDir]:
        """
        Suggest movement directions that increase distance from a threat.

        Orders directions to expand the largest separation axis first while
        respecting bounds and known occupancy.
        """
        blocked_positions = blocked or set()
        dx = start[0] - threat[0]
        dy = start[1] - threat[1]

        primary_axis_is_x = abs(dx) >= abs(dy)
        directions: List[MoveDir] = []

        if primary_axis_is_x and dx != 0:
            directions.append(MoveDir.RIGHT if dx > 0 else MoveDir.LEFT)
        if not primary_axis_is_x and dy != 0:
            directions.append(MoveDir.UP if dy > 0 else MoveDir.DOWN)
        if primary_axis_is_x and dy != 0:
            directions.append(MoveDir.UP if dy > 0 else MoveDir.DOWN)
        if not primary_axis_is_x and dx != 0:
            directions.append(MoveDir.RIGHT if dx > 0 else MoveDir.LEFT)

        current_distance = self.grid.distance(start, threat)
        valid: List[MoveDir] = []
        for direction in directions:
            nx = start[0] + direction.delta[0]
            ny = start[1] + direction.delta[1]
            next_pos = (nx, ny)
            if not self.grid.in_bounds(next_pos):
                continue
            if next_pos in blocked_positions:
                continue
            if self.is_occupied(next_pos, ignore_ids=ignore_ids):
                continue
            if self.grid.distance(next_pos, threat) <= current_distance:
                continue
            valid.append(direction)

        return valid

    @classmethod
    def build(cls, world: "WorldState", team: Team) -> "TeamIntel":
        """
        Construct a safe per-team intel view from the current world.
        """
        team_view: TeamView = world.get_team_view(team)
        friendlies = world.get_team_entities(team, alive_only=False)

        visible_enemies: List[VisibleEnemy] = []
        for obs in team_view.get_enemy_observations():
            visible_enemies.append(
                VisibleEnemy(
                    id=obs.entity_id,
                    team=obs.team,
                    position=obs.position,
                    kind=obs.kind,
                    has_fired_before=obs.has_fired_before,
                    seen_by=obs.seen_by,
                )
            )

        return cls(
            grid=world.grid,
            friendlies=friendlies,
            visible_enemies=visible_enemies,
            friendly_ids=team_view.get_friendly_ids(),
            visible_enemy_ids=team_view.get_enemy_ids(team),
        )
