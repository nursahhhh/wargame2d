from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from typing import Tuple, Literal
import math
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
    
    def enemy_fire_intensity(self, enemy: VisibleEnemy) -> float:
        """
        Temporal aggression signal.
        """
        return 0.7 if enemy.has_fired_before else 0.2

    def _enemy_is_grouped(self, enemy: VisibleEnemy, all_enemies: Iterable[VisibleEnemy],radius: float = 2.5) -> bool:
        """
        Pure geometric grouping check.
        """
        for other in all_enemies:
            if other.id == enemy.id:
                continue
            if self.grid.distance(enemy.position, other.position) <= radius:
                return True
        return False

    def enemy_threat_score(
        self,
        enemy: VisibleEnemy,
        reference_pos: GridPos,
    ) -> float:
        """
        Local threat score ∈ [0,1].
        Combines distance, firing behavior, and grouping.
        """
        distance = self.grid.distance(reference_pos, enemy.position)
        distance_factor = max(
            0.0, 1.0 - distance / max(self.grid.width, self.grid.height)
        )

        fire_factor = self.enemy_fire_intensity(enemy)
       

        return min(1.0, distance_factor + fire_factor )

    def pressure_around(
        self,
        entity: Entity,
        radius: float = 5.0,
    ) -> float:
        """
        Aggregated local danger around an entity.
        """
        pressure = 0.0
        for enemy in self.visible_enemies:
            if self.grid.distance(entity.pos, enemy.position) <= radius:
                pressure += self.enemy_threat_score(enemy, entity.pos)
        return min(1.0, pressure)

    def pressure_level(
        self,
        entity: Entity,
        radius: float = 5.0,
    ) -> str:
        """
        Discretized pressure for LLM consumption.
        """
        p = self.pressure_around(entity, radius)
        if p < 0.4:
            return "LOW"
        if p < 0.7:
            return "MEDIUM"
        return "HIGH"
    



    RadarThreat = Literal["INCREASING", "DECREASING", "NEUTRAL"]

    def radar_threat_trend(
        current_pos: GridPos,
        next_pos: GridPos,
        radar_enemy_pos: GridPos,
    ) -> RadarThreat:
        """
        Belief-based radar threat inference.

        Compares distance to a radar-capable enemy before and after an action.

        Returns:
            - "INCREASING"  : moving closer (higher radar risk)
            - "DECREASING"  : moving away (lower radar risk)
            - "NEUTRAL"     : no change
        """

        curr_dx = current_pos[0] - radar_enemy_pos[0]
        curr_dy = current_pos[1] - radar_enemy_pos[1]
        curr_dist = math.hypot(curr_dx, curr_dy)

        next_dx = next_pos[0] - radar_enemy_pos[0]
        next_dy = next_pos[1] - radar_enemy_pos[1]
        next_dist = math.hypot(next_dx, next_dy)

        if next_dist < curr_dist:
            return "INCREASING"
        elif next_dist > curr_dist:
            return "DECREASING"
        else:
            return "NEUTRAL"


    # ------------------------------------------------------------
    # Global aggression heuristic
    # ------------------------------------------------------------
    def aggression_level(
        self,
        *,
        base: float = 0.5,
        turn: int = 0,
    ) -> float:
        """
        Global aggression scalar ∈ [0,1].
        """
        aggression = base

        alive_friendlies = sum(1 for f in self.friendlies if f.alive)
        enemy_count = len(self.visible_enemies)

        if alive_friendlies > enemy_count:
            aggression += 0.1
        elif alive_friendlies < enemy_count:
            aggression -= 0.1

        if enemy_count > 0:
            fired_ratio = (
                sum(1 for e in self.visible_enemies if e.has_fired_before)
                / enemy_count
            )
            aggression -= 0.2 * fired_ratio

        aggression += min(0.2, turn * 0.01)

        return max(0.0, min(aggression, 1.0))

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
