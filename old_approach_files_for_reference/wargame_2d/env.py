from __future__ import annotations

"""
Grid Air Combat Environment (Fixed Y-Axis Version)
---------------------------------------------------------
KEY FIX: Y-axis now uses mathematical convention (Y+ = UP)
- Y=0 is at BOTTOM of grid
- Y increases UPWARD
- All logic uses mathematical coordinates
- Only rendering converts to screen coordinates
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any
import itertools
import math
import random
import json

# ----------------------------- Core Types ---------------------------------

GridPos = Tuple[int, int]


class Team(Enum):
    BLUE = "BLUE"
    RED = "RED"


class ActionType(Enum):
    WAIT = auto()
    MOVE = auto()
    SHOOT = auto()
    TOGGLE = auto()


class MoveDir(Enum):
    # FIXED: Mathematical coordinates - Y+ = UP
    UP = (0, 1)      # Move upward (increase Y)
    DOWN = (0, -1)   # Move downward (decrease Y)
    LEFT = (-1, 0)   # Move left (decrease X)
    RIGHT = (1, 0)   # Move right (increase X)

    @property
    def delta(self) -> Tuple[int, int]:
        return self.value


# Keep all other classes exactly the same - they don't need changes
# since they work with GridPos tuples, not specific coordinates

@dataclass
class Action:
    type: ActionType
    params: Dict = field(default_factory=dict)


# ----------------------------- Observations --------------------------------

@dataclass
class Observation:
    entity_id: int
    kind: str
    team: Team
    position: GridPos
    distance: float
    seen_by: Set[int] = field(default_factory=set)


# ----------------------------- Entity Base ---------------------------------

_entity_id_seq = itertools.count(1)


@dataclass
class Entity:
    team: Team
    pos: GridPos
    radar_range: float = 0.0
    kind: str = "entity"
    name: Optional[str] = None

    id: int = field(default_factory=lambda: next(_entity_id_seq))
    alive: bool = True
    last_action: Optional[Action] = None
    observations: List[Observation] = field(default_factory=list)

    can_move: bool = True
    can_shoot: bool = False

    def allowed_actions(self, world: "World") -> List[Action]:
        actions: List[Action] = [Action(ActionType.WAIT)]
        if self.can_move:
            for d in MoveDir:
                actions.append(Action(ActionType.MOVE, {"dir": d}))
        if self.can_shoot and self.alive:
            for e_id in world.command_center(self.team).visible_enemy_ids:
                actions.append(Action(ActionType.SHOOT, {"target_id": e_id}))
        return actions

    def perform(self, world: "World", action: Action) -> str:
        self.last_action = action
        if not self.alive:
            return f"{self.label()} is dead and cannot act."

        if action.type == ActionType.WAIT:
            return f"{self.label()} waits."

        if action.type == ActionType.MOVE and self.can_move:
            d: MoveDir = action.params.get("dir")
            if not isinstance(d, MoveDir):
                return f"{self.label()} invalid move params."
            dx, dy = d.delta
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if not world.in_bounds((nx, ny)):
                return f"{self.label()} bumps boundary."
            if world.is_occupied((nx, ny)):
                return f"{self.label()} blocked at {(nx, ny)}."
            self.pos = (nx, ny)
            return f"{self.label()} moves {d.name.lower()} to {self.pos}."

        if action.type == ActionType.SHOOT and self.can_shoot:
            target_id = action.params.get("target_id")
            tgt = world.entities_by_id.get(target_id)
            if not tgt or not tgt.alive:
                return f"{self.label()} cannot shoot: target invalid."
            if not world.command_center(self.team).can_target(target_id):
                return f"{self.label()} cannot shoot: target not observed."
            return f"{self.label()} has no weapon implementation."

        return f"{self.label()} cannot perform {action.type.name}."

    def label(self) -> str:
        nm = self.name or self.kind
        return f"{nm}#{self.id}({self.team.name})"

    def get_active_radar_range(self) -> float:
        """Get the effective radar range (can be overridden by subclasses)"""
        return self.radar_range


# -------------------------- Concrete Entities ------------------------------
# All entity classes remain the same - no coordinate-specific logic

@dataclass
class Shooter(Entity):
    can_shoot: bool = True
    missiles: int = 0
    missile_max_range: float = 5.0
    base_hit_prob: float = 0.8
    min_hit_prob: float = 0.1

    def perform(self, world: "World", action: Action) -> str:
        if action.type == ActionType.SHOOT:
            if self.missiles <= 0:
                return f"{self.label()} has no missiles."
            target_id = action.params.get("target_id")
            tgt = world.entities_by_id.get(target_id)
            if not tgt or not tgt.alive:
                return f"{self.label()} cannot shoot: bad target."
            if not world.command_center(self.team).can_target(target_id):
                return f"{self.label()} cannot shoot: target not observed."
            dist = world.distance(self.pos, tgt.pos)
            if dist > self.missile_max_range:
                return f"{self.label()} target out of range (d={dist:.1f})."

            p = world.hit_probability(
                distance=dist,
                max_range=self.missile_max_range,
                base=self.base_hit_prob,
                min_p=self.min_hit_prob,
            )
            self.missiles -= 1
            shot = random.random() <= p
            result = "HIT" if shot else "MISS"
            if shot:
                world.mark_kill(tgt)
            self.last_action = action
            return (
                f"{self.label()} fires at {tgt.label()} (d={dist:.1f}, p={p:.2f}) -> {result}."
            )
        return super().perform(world, action)


@dataclass
class Aircraft(Shooter):
    kind: str = "aircraft"
    radar_range: float = 5.0
    missiles: int = 2
    missile_max_range: float = 4.0


@dataclass
class AWACS(Entity):
    kind: str = "awacs"
    radar_range: float = 9.0
    can_move: bool = True
    can_shoot: bool = False


@dataclass
class Decoy(Entity):
    kind: str = "decoy"
    radar_range: float = 0.0
    can_move: bool = True
    can_shoot: bool = False


@dataclass
class SAM(Shooter):
    kind: str = "sam"
    can_move: bool = False
    missiles: int = 4
    missile_max_range: float = 6.0
    on: bool = False
    cooldown_steps: int = 5
    _cooldown: int = 0

    def get_active_radar_range(self) -> float:
        """SAM radar only active when ON"""
        return self.radar_range if self.on else 0.0

    def allowed_actions(self, world: "World") -> List[Action]:
        actions = [Action(ActionType.WAIT)]
        actions.append(Action(ActionType.TOGGLE, {"on": not self.on}))
        if self.on and self._cooldown == 0 and self.missiles > 0:
            for e_id in world.command_center(self.team).visible_enemy_ids:
                actions.append(Action(ActionType.SHOOT, {"target_id": e_id}))
        return actions

    def perform(self, world: "World", action: Action) -> str:
        if action.type == ActionType.TOGGLE:
            desired = action.params.get("on")
            if isinstance(desired, bool):
                self.on = desired
                return f"{self.label()} toggled {'ON' if self.on else 'OFF'}."
            return f"{self.label()} invalid toggle param."

        if action.type == ActionType.SHOOT:
            if not self.on:
                return f"{self.label()} is OFF."
            if self._cooldown > 0:
                return f"{self.label()} cooling down ({self._cooldown})."
            log = super().perform(world, action)
            if "fires" in log:
                self._cooldown = self.cooldown_steps
            return log

        return super().perform(world, action)

    def tick_cooldown(self):
        if self._cooldown > 0:
            self._cooldown -= 1


# ----------------------------- Command Center ------------------------------

class CommandCenter:
    def __init__(self, team: Team):
        self.team = team
        self.friendly_ids: Set[int] = set()
        self.visible_enemy_ids: Set[int] = set()
        self.observations: Dict[int, Observation] = {}
        self.enemy_firing_history: Dict[int, bool] = {}

    def record_enemy_firing(self, entity_id: int):
        """Record that an enemy entity has fired a missile."""
        self.enemy_firing_history[entity_id] = True

    def has_enemy_fired(self, entity_id: int) -> bool:
        """Check if an enemy has ever fired a missile."""
        return self.enemy_firing_history.get(entity_id, False)

    def reset_step(self, world: "World"):
        self.friendly_ids = {e.id for e in world.entities if e.team == self.team and e.alive}
        self.visible_enemy_ids = set()
        self.observations = {}

    def ingest(self, obs_list: Iterable[Observation]):
        for obs in obs_list:
            if obs.entity_id not in self.observations:
                self.observations[obs.entity_id] = obs
            else:
                self.observations[obs.entity_id].seen_by.update(obs.seen_by)
            if obs.team != self.team:
                self.visible_enemy_ids.add(obs.entity_id)

    def can_target(self, target_id: int) -> bool:
        return target_id in self.visible_enemy_ids

    def visible_observations(self, include_friendlies: bool = True) -> List[Observation]:
        obs = []
        for o in self.observations.values():
            if include_friendlies or o.team != self.team:
                obs.append(o)
        return obs


# ------------------------------- World -------------------------------------

class World:
    def __init__(self, width: int, height: int, seed: Optional[int] = 42*42, max_stalemate_turns: int = 60,
                 max_no_move_turns: int = 15):
        self.width = width
        self.height = height
        self.entities: List[Entity] = []
        self.entities_by_id: Dict[int, Entity] = {}
        self._kills: List[int] = []
        self._rng = random.Random(seed)
        self._cc: Dict[Team, CommandCenter] = {
            Team.BLUE: CommandCenter(Team.BLUE),
            Team.RED: CommandCenter(Team.RED),
        }
        self.default_base_hit = 0.8
        self.default_min_hit = 0.1
        self.game_over = False
        self.winner: Optional[Team] = None
        self.game_over_reason = ""

        self.max_stalemate_turns = max_stalemate_turns
        self.turns_without_shooting = 0
        self.total_turns = 0

        self.max_no_move_turns = max_no_move_turns
        self.turns_without_movement = 0

        self._team_observable_logs_history: Dict[Team, List[List[str]]] = {
            Team.BLUE: [],
            Team.RED: []
        }
        self._max_log_history: int = 3

    def in_bounds(self, p: GridPos) -> bool:
        x, y = p
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, p: GridPos) -> bool:
        return any(e.alive and e.pos == p for e in self.entities)

    def distance(self, a: GridPos, b: GridPos) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def add(self, e: Entity) -> int:
        if not self.in_bounds(e.pos):
            raise ValueError("Entity position out of bounds")
        self.entities.append(e)
        self.entities_by_id[e.id] = e
        return e.id

    def mark_kill(self, tgt: Entity):
        self._kills.append(tgt.id)

    def command_center(self, team: Team) -> CommandCenter:
        return self._cc[team]

    def _compute_entity_observations(self, e: Entity) -> List[Observation]:
        observations: List[Observation] = []
        if not e.alive:
            return observations

        active_radar = e.get_active_radar_range()
        if active_radar <= 0:
            return observations

        for other in self.entities:
            if not other.alive:
                continue
            if other.id == e.id:
                continue

            if isinstance(other, SAM) and not other.on:
                continue

            d = self.distance(e.pos, other.pos)
            if d <= active_radar:
                observed_kind = other.kind
                if isinstance(other, Decoy) and other.team != e.team:
                    observed_kind = 'aircraft'

                observations.append(
                    Observation(
                        entity_id=other.id,
                        kind=observed_kind,
                        team=other.team,
                        position=other.pos,
                        distance=d,
                        seen_by={e.id},
                    )
                )
        return observations

    def refresh_observations(self):
        for cc in self._cc.values():
            cc.reset_step(self)
        for e in self.entities:
            e.observations = self._compute_entity_observations(e)
            self._cc[e.team].ingest(e.observations)
        for team, cc in self._cc.items():
            for e in self.entities:
                if e.team == team and e.alive:
                    d = 0.0
                    obs = Observation(e.id, e.kind, e.team, e.pos, d, seen_by={e.id})
                    cc.ingest([obs])

    def hit_probability(self, *, distance: float, max_range: float, base: float = None, min_p: float = None) -> float:
        if base is None:
            base = self.default_base_hit
        if min_p is None:
            min_p = self.default_min_hit
        if max_range <= 0:
            return 0.0
        frac = max(0.0, min(1.0, 1.0 - (distance / max_range)))
        return max(min_p, base * frac)

    def check_game_over(self) -> bool:
        if self.game_over:
            return True

        blue_awacs_alive = any(isinstance(e, AWACS) and e.team == Team.BLUE and e.alive
                               for e in self.entities)
        red_awacs_alive = any(isinstance(e, AWACS) and e.team == Team.RED and e.alive
                              for e in self.entities)

        if not blue_awacs_alive and not red_awacs_alive:
            self.game_over = True
            self.winner = None
            self.game_over_reason = "Both AWACS destroyed - DRAW"
            return True
        elif not blue_awacs_alive:
            self.game_over = True
            self.winner = Team.RED
            self.game_over_reason = "BLUE AWACS destroyed - RED WINS"
            return True
        elif not red_awacs_alive:
            self.game_over = True
            self.winner = Team.BLUE
            self.game_over_reason = "RED AWACS destroyed - BLUE WINS"
            return True

        total_missiles = sum(e.missiles for e in self.entities
                             if isinstance(e, Shooter) and e.alive)
        if total_missiles == 0:
            self.game_over = True
            self.winner = None
            self.game_over_reason = "No missiles remaining - DRAW"
            return True

        if self.turns_without_shooting >= self.max_stalemate_turns:
            self.game_over = True
            self.winner = None
            self.game_over_reason = f"Stalemate - No missiles fired for {self.max_stalemate_turns} turns - DRAW"
            return True

        if self.turns_without_movement >= self.max_no_move_turns:
            self.game_over = True
            self.winner = None
            self.game_over_reason = f"Stagnation - No movement for {self.max_no_move_turns} turns - DRAW"
            return True

        return False

    def get_game_stats(self) -> Dict[str, Any]:
        """Get detailed game statistics."""
        stats = {
            "blue": {
                "total": 0,
                "alive": 0,
                "destroyed": 0,
                "by_type": {}
            },
            "red": {
                "total": 0,
                "alive": 0,
                "destroyed": 0,
                "by_type": {}
            },
            "missiles": {
                "blue_remaining": 0,
                "red_remaining": 0,
                "total_remaining": 0
            }
        }

        for e in self.entities:
            team_key = "blue" if e.team == Team.BLUE else "red"
            stats[team_key]["total"] += 1

            if e.alive:
                stats[team_key]["alive"] += 1
            else:
                stats[team_key]["destroyed"] += 1

            if e.kind not in stats[team_key]["by_type"]:
                stats[team_key]["by_type"][e.kind] = {"total": 0, "alive": 0, "destroyed": 0}

            stats[team_key]["by_type"][e.kind]["total"] += 1
            if e.alive:
                stats[team_key]["by_type"][e.kind]["alive"] += 1
            else:
                stats[team_key]["by_type"][e.kind]["destroyed"] += 1

            if isinstance(e, Shooter) and e.alive:
                if e.team == Team.BLUE:
                    stats["missiles"]["blue_remaining"] += e.missiles
                else:
                    stats["missiles"]["red_remaining"] += e.missiles

        stats["missiles"]["total_remaining"] = (stats["missiles"]["blue_remaining"] +
                                                stats["missiles"]["red_remaining"])

        return stats

    def get_game_over_summary(self) -> Dict[str, Any]:
        """Get complete game over summary as a JSON-serializable dict."""
        if not self.game_over:
            return {"game_over": False}

        stats = self.get_game_stats()

        summary = {
            "game_over": True,
            "winner": self.winner.name if self.winner else "DRAW",
            "reason": self.game_over_reason,
            "statistics": {
                "blue_team": {
                    "total_units": stats["blue"]["total"],
                    "surviving_units": stats["blue"]["alive"],
                    "destroyed_units": stats["blue"]["destroyed"],
                    "units_by_type": stats["blue"]["by_type"],
                    "missiles_remaining": stats["missiles"]["blue_remaining"]
                },
                "red_team": {
                    "total_units": stats["red"]["total"],
                    "surviving_units": stats["red"]["alive"],
                    "destroyed_units": stats["red"]["destroyed"],
                    "units_by_type": stats["red"]["by_type"],
                    "missiles_remaining": stats["missiles"]["red_remaining"]
                },
                "total_missiles_remaining": stats["missiles"]["total_remaining"]
            }
        }

        return summary

    def format_game_stats(self) -> str:
        """Format game statistics as a readable string."""
        stats = self.get_game_stats()

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("GAME STATISTICS")
        lines.append("=" * 60)

        for team_name in ["blue", "red"]:
            team_stats = stats[team_name]
            display_name = team_name.upper()

            lines.append(f"\n{display_name} TEAM:")
            lines.append(f"  Total Units: {team_stats['total']}")
            lines.append(f"  Alive: {team_stats['alive']}")
            lines.append(f"  Destroyed: {team_stats['destroyed']}")

            if team_stats['by_type']:
                lines.append(f"  By Type:")
                for unit_type, type_stats in sorted(team_stats['by_type'].items()):
                    lines.append(f"    {unit_type.upper()}: {type_stats['alive']}/{type_stats['total']} alive")

        lines.append(f"\nMISSILES REMAINING:")
        lines.append(f"  BLUE: {stats['missiles']['blue_remaining']}")
        lines.append(f"  RED: {stats['missiles']['red_remaining']}")
        lines.append(f"  TOTAL: {stats['missiles']['total_remaining']}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def step(self, actions: Dict[int, Action]) -> List[str]:
        """Execute a full game step"""
        logs: List[str] = []
        self._kills = []

        shooting_occurred = False

        _current_turn_logs = {
            Team.BLUE: [],
            Team.RED: []
        }

        # Housekeeping
        for e in self.entities:
            if isinstance(e, SAM):
                e.tick_cooldown()

        # Sense
        self.refresh_observations()

        # Movement & toggles
        alive_entities = [e for e in self.entities if e.alive]
        shuffled_entities = alive_entities.copy()
        self._rng.shuffle(shuffled_entities)

        _pos_before = {e.id: e.pos for e in shuffled_entities}

        for e in shuffled_entities:
            act = actions.get(e.id, Action(ActionType.WAIT))
            if act.type in (ActionType.MOVE, ActionType.TOGGLE, ActionType.WAIT):
                action_log = e.perform(self, act)
                logs.append(action_log)
                self._record_observable_action(e, action_log, _current_turn_logs)

        movement_occurred = any(e.pos != _pos_before.get(e.id) for e in shuffled_entities)

        # Re-sense after movement
        self.refresh_observations()

        # Shooting phase
        alive_entities = [e for e in self.entities if e.alive]
        shuffled_entities = alive_entities.copy()
        self._rng.shuffle(shuffled_entities)

        for e in shuffled_entities:
            act = actions.get(e.id)
            if act and act.type == ActionType.SHOOT:
                shooting_occurred = True
                action_log = e.perform(self, act)
                logs.append(action_log)
                self._record_observable_action(e, action_log, _current_turn_logs)

        # Apply deaths
        for entity_id in self._kills:
            tgt = self.entities_by_id.get(entity_id)
            if tgt:
                tgt.alive = False

        # Track stalemate / stagnation
        self.total_turns += 1
        if shooting_occurred:
            self.turns_without_shooting = 0
        else:
            self.turns_without_shooting += 1

        if movement_occurred:
            self.turns_without_movement = 0
        else:
            self.turns_without_movement += 1

        # Check game over
        if self.check_game_over():
            logs.append(f"\n*** {self.game_over_reason} ***")

        # Store logs
        for team in [Team.BLUE, Team.RED]:
            self._team_observable_logs_history[team].append(_current_turn_logs[team])
            if len(self._team_observable_logs_history[team]) > self._max_log_history:
                self._team_observable_logs_history[team].pop(0)

        return logs

    def _format_observable_log(self, log: str, observing_team: Team) -> str:
        import re
        pattern = r'([\w-]+)#(\d+)\((\w+)\)'

        def replace_entity(match):
            name = match.group(1)
            entity_id = int(match.group(2))
            team_str = match.group(3)

            entity = self.entities_by_id.get(entity_id)
            if not entity:
                return match.group(0)

            observed_type = entity.kind

            if entity.team != observing_team:
                cc = self.command_center(observing_team)
                obs = cc.observations.get(entity_id)
                if obs:
                    observed_type = obs.kind
                else:
                    observed_type = "Unknown"
            else:
                observed_type = entity.kind

            display_name = observed_type.capitalize()
            return f"{display_name}#{entity_id}({team_str})"

        formatted_log = re.sub(pattern, replace_entity, log)
        return formatted_log

    def _record_observable_action(self, entity: Entity, action_log: str, current_turn_logs: Dict[Team, List[str]]):
        for team in [Team.BLUE, Team.RED]:
            can_observe = False
            is_friendly = entity.team == team

            if is_friendly:
                can_observe = True
            else:
                cc = self.command_center(team)
                if entity.id in cc.visible_enemy_ids:
                    can_observe = True

            if can_observe:
                formatted_log = self._format_observable_log(action_log, team)
                prefix = "[ALLY]" if is_friendly else "[ENEMY]"
                final_log = f"{prefix} {formatted_log}"
                current_turn_logs[team].append(final_log)

    def get_team_observable_logs(self, team: Team, num_turns: int = None) -> List[List[str]]:
        history = self._team_observable_logs_history.get(team, [])
        if num_turns is None:
            return history
        return history[-num_turns:] if history else []

    def render(self, mode: str = "god", legend: bool = True) -> str:
        """
        Render the grid with MATHEMATICAL coordinates (Y+ = UP)
        Grid is printed top-to-bottom but represents Y going upward
        """
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        def put(p: GridPos, ch: str):
            x, y = p
            if self.in_bounds(p):
                # CRITICAL: Convert mathematical Y to screen Y for display
                screen_y = self.height - 1 - y  # Y=0 math -> bottom row, Y=max math -> top row
                grid[screen_y][x] = ch

        show_team: Optional[Team] = None
        show_all_radar = False
        if mode.lower() == "blue":
            show_team = Team.BLUE
        elif mode.lower() == "red":
            show_team = Team.RED
        else:
            show_all_radar = True

        def paint_radar_for(entity: Entity):
            r = int(math.floor(entity.get_active_radar_range()))
            if r <= 0 or not entity.alive:
                return
            ex, ey = entity.pos
            for y in range(max(0, ey - r), min(self.height, ey + r + 1)):
                for x in range(max(0, ex - r), min(self.width, ex + r + 1)):
                    if self.distance((x, y), (ex, ey)) <= entity.get_active_radar_range():
                        if self.in_bounds((x, y)):
                            screen_y = self.height - 1 - y
                            if grid[screen_y][x] == " ":
                                grid[screen_y][x] = "."

        if show_all_radar:
            for e in self.entities:
                paint_radar_for(e)
        elif show_team is not None:
            for e in self.entities:
                if e.team == show_team:
                    paint_radar_for(e)

        def icon(e: Entity) -> str:
            base = {
                "aircraft": "A",
                "awacs": "W",
                "decoy": "D",
                "sam": "S",
            }.get(e.kind, "E")
            if e.team == Team.RED:
                base = base.lower()
            if not e.alive:
                base = "x"
            return base

        visible_ids: Set[int] = set()
        if show_team is not None:
            visible_ids = set(self.command_center(show_team).observations.keys())
            for e in self.entities:
                if e.team == show_team and e.alive:
                    visible_ids.add(e.id)

        for e in self.entities:
            if not e.alive:
                if show_team is not None and e.team != show_team:
                    continue
            elif show_team is not None and e.id not in visible_ids:
                continue
            put(e.pos, icon(e))

        lines = ["+" + "-" * self.width + "+"]
        # Print top row first (high Y values)
        for screen_row in range(self.height):
            lines.append("|" + ''.join(grid[screen_row]) + "|")
        lines.append("+" + "-" * self.width + "+")

        if legend:
            lines.append(
                "Legend: A/W/D/S = BLUE Aircraft/AWACS/Decoy/SAM, a/w/d/s = RED. '.' = radar coverage. 'x' = destroyed.")
            lines.append("Coordinate System: X increases RIGHT, Y increases UP (mathematical convention)")
            for e in sorted(self.entities, key=lambda x: x.id):
                la = e.last_action.type.name if e.last_action else "-"
                extra = ""
                if isinstance(e, SAM):
                    extra = f" on={'ON' if e.on else 'OFF'} cd={e._cooldown}"
                if isinstance(e, Shooter):
                    extra = f" missiles={e.missiles}" + (extra if extra else "")
                lines.append(f" {e.label()} at {e.pos} [{la}]{extra}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("Fixed Grid Air Combat Environment - Mathematical Y-Axis")
    print("Y+ = UP (mathematical convention)")