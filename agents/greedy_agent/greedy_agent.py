"""
Greedy agent placeholder that will select locally optimal actions.

Decision logic:
- Patrol horizontally between an origin position and the map edge
- Chase visible enemies when spotted
- Shoot when the estimated hit probability clears a configured threshold
"""

from typing import Any, Dict, Optional, TYPE_CHECKING, Iterable, List

from env.core.actions import Action
from env.core.types import ActionType, EntityKind, MoveDir, Team
from ..base_agent import BaseAgent
from ..registry import register_agent
from ..team_intel import TeamIntel

if TYPE_CHECKING:
    from env.environment import StepInfo
    from env.world import WorldState
    from env.entities.base import Entity


@register_agent("greedy")
class GreedyAgent(BaseAgent):
    """
    Greedy policy with simple patrol/chase heuristics.
    """

    def __init__(
        self,
        team: Team,
        name: str | None = None,
        *,
        patrol_direction: str | MoveDir = "left",
        shoot_prob: float = 0.1,
        awacs_safe_distance: float = 5.0,
        **_: Any,
    ):
        """
        Initialize the greedy agent.

        Args:
            team: Team to control
            name: Optional agent name (default: "GreedyAgent")
            patrol_direction: Initial horizontal patrol direction ("left"/"right")
            shoot_prob: Minimum probability required to take a shot
            awacs_safe_distance: Minimum comfort radius before AWACS will reposition
        """
        super().__init__(team, name)
        self.initial_direction = self._parse_direction(patrol_direction)
        self.shoot_threshold = shoot_prob
        self.awacs_safe_distance = awacs_safe_distance

        # Per-entity state
        self._origins: Dict[int, tuple[int, int]] = {}
        self._patrol_targets: Dict[int, tuple[int, int]] = {}
        self._heading_home: Dict[int, bool] = {}

    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """
        Patrol horizontally, chase visible enemies, and shoot when odds are good.
        """
        world: "WorldState" = state["world"]
        intel = TeamIntel.build(world, self.team)

        actions: Dict[int, Action] = {}
        metadata: Dict[str, Any] = {"policy": "greedy", "injections": {**kwargs}}

        for entity in intel.friendlies:
            if not entity.alive:
                continue

            allowed = entity.get_allowed_actions(world)
            if not allowed:
                continue

            if entity.kind != EntityKind.AWACS and entity.can_move:
                self._ensure_patrol_state(
                    entity.id, entity.pos, intel.grid.width, intel.grid.height
                )

            action = self._shoot_if_viable(entity, intel, allowed)
            if action is None and entity.can_move:
                if entity.kind == EntityKind.AWACS:
                    action = self._awacs_evade(entity, intel, allowed)
                else:
                    action = self._chase_or_patrol(entity, intel, allowed)

            if action is None:
                action = self._first_allowed_wait(allowed)

            if action:
                actions[entity.id] = action

        metadata["actions_count"] = len(actions)
        return actions, metadata

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def _shoot_if_viable(
        self,
        entity: "Entity",
        intel: TeamIntel,
        allowed: Iterable[Action],
    ) -> Optional[Action]:
        """Pick the best allowed shot that clears the threshold."""
        best: tuple[int, float] | None = None  # (target_id, prob)
        for enemy in intel.visible_enemies:
            prob = intel.estimate_hit_probability(entity, enemy)
            if prob is None or prob < self.shoot_threshold:
                continue
            if best is None or prob > best[1]:
                best = (enemy.id, prob)

        if best is None:
            return None

        target_id = best[0]
        for action in allowed:
            if action.type == ActionType.SHOOT and action.params.get("target_id") == target_id:
                return action
        return None

    def _chase_or_patrol(
        self,
        entity: "Entity",
        intel: TeamIntel,
        allowed: List[Action],
    ) -> Optional[Action]:
        """Chase the nearest visible enemy or continue patrolling."""
        nearest = intel.nearest_visible_enemy(entity.pos)
        if nearest is not None:
            enemy, _ = nearest
            directions = intel.move_toward(entity.pos, enemy.position, ignore_ids={entity.id})
            action = self._pick_move_action(directions, allowed)
            if action:
                return action
            fallback = self._best_distance_move(entity.pos, enemy.position, allowed)
            if fallback:
                return fallback
            return self._first_allowed_wait(allowed)

        return self._patrol_move(entity, intel, allowed)

    def _patrol_move(
        self,
        entity: "Entity",
        intel: TeamIntel,
        allowed: List[Action],
    ) -> Optional[Action]:
        """Bounce between origin and the map edge in the initial direction."""
        entity_id = entity.id
        origin = self._origins[entity_id]
        boundary = self._patrol_targets[entity_id]
        heading_home = self._heading_home.get(entity_id, False)

        target = origin if heading_home else boundary

        # Flip direction when reaching an endpoint
        if entity.pos == boundary:
            heading_home = True
            target = origin
        elif entity.pos == origin and heading_home:
            heading_home = False
            target = boundary

        directions = intel.move_toward(entity.pos, target, ignore_ids={entity.id})

        # If blocked while heading outward, go back toward origin
        if not directions and not heading_home:
            heading_home = True
            target = origin
            directions = intel.move_toward(entity.pos, target, ignore_ids={entity.id})

        self._heading_home[entity_id] = heading_home
        return self._pick_move_action(directions, allowed) or self._first_allowed_wait(allowed)

    def _awacs_evade(
        self,
        entity: "Entity",
        intel: TeamIntel,
        allowed: List[Action],
    ) -> Optional[Action]:
        """AWACS stays put unless an enemy is visible, then moves away."""
        nearest = intel.nearest_visible_enemy(entity.pos)
        if nearest is None:
            return self._first_allowed_wait(allowed)

        enemy, distance = nearest
        if distance > self.awacs_safe_distance:
            return self._first_allowed_wait(allowed)
        if distance <= 1:
            return self._first_allowed_wait(allowed)

        directions = intel.move_away(entity.pos, enemy.position, ignore_ids={entity.id})
        return self._pick_move_action(directions, allowed) or self._first_allowed_wait(allowed)

    @staticmethod
    def _pick_move_action(directions: List[MoveDir], allowed: Iterable[Action]) -> Optional[Action]:
        """Select the first allowed move matching the preferred directions."""
        for direction in directions:
            for action in allowed:
                if action.type == ActionType.MOVE and action.params.get("dir") == direction:
                    return action
        return None

    @staticmethod
    def _first_allowed_wait(allowed: Iterable[Action]) -> Optional[Action]:
        """Return a WAIT action if available."""
        for action in allowed:
            if action.type == ActionType.WAIT:
                return action
        return None

    @staticmethod
    def _best_distance_move(
        current_pos: tuple[int, int],
        target_pos: tuple[int, int],
        allowed: Iterable[Action],
    ) -> Optional[Action]:
        """
        Pick a move that strictly reduces Manhattan distance to the target.
        """
        best_action: Optional[Action] = None
        best_dist: Optional[int] = None
        cur_dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])

        for action in allowed:
            if action.type != ActionType.MOVE:
                continue
            dir_param = action.params.get("dir")
            if not isinstance(dir_param, MoveDir):
                continue
            dx, dy = dir_param.delta
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            new_dist = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
            if new_dist < cur_dist and (best_dist is None or new_dist < best_dist):
                best_action = action
                best_dist = new_dist

        return best_action

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _ensure_patrol_state(
        self,
        entity_id: int,
        pos: tuple[int, int],
        grid_width: int,
        grid_height: int,
    ) -> None:
        """Initialize patrol endpoints for an entity if needed."""
        if entity_id not in self._origins:
            self._origins[entity_id] = pos

        if entity_id not in self._patrol_targets:
            x, y = pos
            if self.initial_direction == MoveDir.LEFT:
                boundary = (0, y)
            elif self.initial_direction == MoveDir.RIGHT:
                boundary = (grid_width - 1, y)
            elif self.initial_direction == MoveDir.UP:
                boundary = (x, grid_height - 1)
            elif self.initial_direction == MoveDir.DOWN:
                boundary = (x, 0)
            else:
                boundary = pos
            self._patrol_targets[entity_id] = boundary

    @staticmethod
    def _parse_direction(direction: str | MoveDir) -> MoveDir:
        """Normalize input to a MoveDir."""
        if isinstance(direction, MoveDir):
            return direction

        normalized = direction.lower()
        if normalized == "left":
            return MoveDir.LEFT
        if normalized == "right":
            return MoveDir.RIGHT
        if normalized == "up":
            return MoveDir.UP
        if normalized == "down":
            return MoveDir.DOWN

        raise ValueError(f"Unsupported patrol_direction: {direction}")
