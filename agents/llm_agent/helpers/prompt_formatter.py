"""
Structured JSON prompt formatter for LLM agents.

The formatter tries to mirror the desired draft output:
- Clear metadata describing the grid and coordinate system
- High-level situation summary with spatial aggregates
- Per-unit views that include nearby allies, threats, and allowed actions
- Threat assessment that uses configurable weapon assumptions for enemies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from env.core.actions import Action
from env.core.types import ActionType, EntityKind, MoveDir, Team
from env.entities.base import Entity
from env.mechanics.combat import hit_probability
from agents.team_intel import TeamIntel, VisibleEnemy


@dataclass
class WeaponProfile:
    """Assumed weapon characteristics for threat estimation."""

    max_range: float
    base_hit_prob: float
    min_hit_prob: float


def _default_enemy_profiles() -> Dict[EntityKind, WeaponProfile]:
    # Default ranges mirror scenario defaults (aircraft ~4, SAM ~6).
    return {
        EntityKind.AIRCRAFT: WeaponProfile(max_range=4.0, base_hit_prob=0.8, min_hit_prob=0.1),
        EntityKind.SAM: WeaponProfile(max_range=6.0, base_hit_prob=0.8, min_hit_prob=0.1),
        EntityKind.AWACS: WeaponProfile(max_range=0.0, base_hit_prob=0.0, min_hit_prob=0.0),
        EntityKind.DECOY: WeaponProfile(max_range=0.0, base_hit_prob=0.0, min_hit_prob=0.0),
        EntityKind.UNKNOWN: WeaponProfile(max_range=3.0, base_hit_prob=0.8, min_hit_prob=0.1),
    }


@dataclass
class TeamOrientation:
    """How this team defines forward/backward relative to the grid."""

    spawn_side: str  # e.g., "LEFT" or "RIGHT"
    forward: MoveDir
    backward: MoveDir
    note: str = "Forward = toward enemy base; Backward/Retreat = toward own base."


def _default_team_orientations() -> Dict[str, TeamOrientation]:
    """
    Defaults assume BLUE spawns left (forward=RIGHT) and RED spawns right (forward=LEFT).
    Configurable via PromptConfig.team_orientation_map to handle variant scenarios.
    """
    return {
        Team.BLUE.name: TeamOrientation(spawn_side="LEFT", forward=MoveDir.RIGHT, backward=MoveDir.LEFT),
        Team.RED.name: TeamOrientation(spawn_side="RIGHT", forward=MoveDir.LEFT, backward=MoveDir.RIGHT),
    }


@dataclass
class PromptConfig:
    """
    Tunable knobs for prompt shaping and threat estimation.

    - nearby_ally_radius / nearby_enemy_radius: distance caps for inclusion
    - grouping_radius: distance to consider enemies as a cluster
    - threat_close_radius: distance that counts as "close" for threat typing
    - enemy_weapon_profiles: assumed weapon stats per enemy kind (used for
      can_hit_range + hit_probability())
    - fallback_enemy_weapon_profile: used when an enemy kind has no explicit
      profile entry
    - team_orientation_map: how each team defines forward/backward (for prompt disambiguation)
    - include_orientation_metadata: include forward/backward definitions in metadata
    - include_direction_resolution: include a brief forward/backward hint for horizontal MOVE actions
    """

    nearby_ally_radius: float = 5.0
    nearby_enemy_radius: float = 6.0
    grouping_radius: float = 3.0
    threat_close_radius: float = 3.0
    threat_approach_buffer: float = 2.0  # How far outside enemy max range counts as "almost in range"
    include_hit_probabilities: bool = True
    include_casualties: bool = True
    enemy_weapon_profiles: Dict[EntityKind, WeaponProfile] = field(default_factory=_default_enemy_profiles)
    fallback_enemy_weapon_profile: WeaponProfile = field(
        default_factory=lambda: WeaponProfile(max_range=3.0, base_hit_prob=0.7, min_hit_prob=0.1)
    )
    team_orientation_map: Dict[str, TeamOrientation] = field(default_factory=_default_team_orientations)
    include_orientation_metadata: bool = True
    include_direction_resolution: bool = True


class PromptFormatter:
    """
    Convert intel + allowed actions into structured JSON for an LLM agent.
    """

    def build_prompt(
        self,
        *,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        config: Optional[PromptConfig] = None,
        turn_number: int = 0,
        team_name: Optional[str] = None,
        missing_enemies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the JSON prompt and return both the string and the underlying dict.
        """
        cfg = config or PromptConfig()
        friendly_positions = [e.pos for e in intel.friendlies if e.alive]
        enemy_positions = [e.position for e in intel.visible_enemies]
        move_conflicts = self._collect_move_conflicts(allowed_actions, intel)

        missing_enemies = missing_enemies or []

        situation = self._build_situation(intel, friendly_positions, enemy_positions, cfg, missing_enemies)
        # Add far-away enemies not covered by per-unit threat listings.
        situation["distant_enemies"] = self._get_distant_enemies(intel, friendly_positions, cfg)
        situation["missing_enemies"] = missing_enemies

        payload: Dict[str, Any] = {
            "metadata": self._build_metadata(intel, turn_number, team_name, cfg),
            "situation": situation,
            "units": [],
        }

        if cfg.include_casualties:
            payload["casualties"] = self._get_casualties(intel)

        grouped_map = self._enemy_group_map(intel.visible_enemies, intel, cfg.grouping_radius)

        for entity in intel.friendlies:
            if not entity.alive:
                continue
            payload["units"].append(
                self._build_unit_data(
                    entity=entity,
                    intel=intel,
                    allowed_actions=allowed_actions.get(entity.id, []),
                    cfg=cfg,
                    grouped_map=grouped_map,
                    move_conflicts=move_conflicts,
                )
            )

        return payload

    def _get_distant_enemies(
        self,
        intel: TeamIntel,
        friendly_positions: List[Tuple[int, int]],
        cfg: PromptConfig,
    ) -> List[Dict[str, Any]]:
        """
        List visible enemies that are outside the nearby enemy radius of all friendlies.
        Provides minimal info so they are not omitted from the prompt entirely.
        """
        distant: List[Dict[str, Any]] = []
        for enemy in intel.visible_enemies:
            if not friendly_positions:
                min_dist = None
            else:
                min_dist = min(intel.grid.distance(pos, enemy.position) for pos in friendly_positions)
            if min_dist is not None and min_dist <= cfg.nearby_enemy_radius:
                continue  # Already covered by at least one unit's threat listing
            distant.append(
                {
                    "enemy_id": enemy.id,
                    "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                    "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                    "position": {"x": enemy.position[0], "y": enemy.position[1]},
                    "nearest_friendly_distance": round(min_dist, 1) if isinstance(min_dist, (int, float)) else None,
                    "current_threat": "OUT_OF_RANGE",
                    "notes": "Outside all nearby threat radii; poses no immediate risk",
                }
            )
        return distant

    # ------------------------------------------------------------------ #
    # Metadata + situation
    # ------------------------------------------------------------------ #
    def _build_metadata(
        self,
        intel: TeamIntel,
        turn_number: int,
        team_name: Optional[str],
        cfg: PromptConfig,
    ) -> Dict[str, Any]:
        team = team_name or (intel.friendlies[0].team.name if intel.friendlies else "UNKNOWN")
        payload: Dict[str, Any] = {
            "turn": turn_number,
            "team": team,
            "grid_size": {
                "width": intel.grid.width,
                "height": intel.grid.height,
                "center": {
                    # Use integer cell coordinates; for even sizes this picks the lower/left center cell.
                    "x": intel.grid.width // 2,
                    "y": intel.grid.height // 2,
                },
            },
            "coordinate_system": {
                "description": "Standard Cartesian grid with origin at bottom-left",
                "x_axis": "Increases from LEFT to RIGHT (moving RIGHT increases x by 1)",
                "y_axis": "Increases from BOTTOM to TOP (moving UP increases y by 1)",
                "movement_rules": {
                    "UP": "y + 1 (north, towards top of grid)",
                    "DOWN": "y - 1 (south, towards bottom of grid)",
                    "RIGHT": "x + 1 (east, towards right edge)",
                    "LEFT": "x - 1 (west, towards left edge)",
                },
                "example": "If you are at (10, 5) and move RIGHT, you go to (11, 5). If you move UP, you go to (10, 6).",
            },
        }

        if cfg.include_orientation_metadata:
            orientation = self._get_orientation(team, cfg)
            payload["team_orientation"] = {
                "team": team,
                "spawn_side": orientation.spawn_side,
                "forward_direction": orientation.forward.name,
                "backward_direction": orientation.backward.name,
                "note": orientation.note,
                "orientation_by_team": {
                    t: {
                        "spawn_side": o.spawn_side,
                        "forward_direction": o.forward.name,
                        "backward_direction": o.backward.name,
                        "note": o.note,
                    }
                    for t, o in cfg.team_orientation_map.items()
                },
            }

        return payload

    def _build_situation(
        self,
        intel: TeamIntel,
        friendly_positions: List[Tuple[int, int]],
        enemy_positions: List[Tuple[int, int]],
        cfg: PromptConfig,
        missing_enemies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ally_center = self._calculate_center_of_mass(friendly_positions)
        enemy_center = self._calculate_center_of_mass(enemy_positions)
        formation_center_distance = None
        if ally_center and enemy_center:
            ally_pos = (ally_center["x"], ally_center["y"])
            enemy_pos = (enemy_center["x"], enemy_center["y"])
            formation_center_distance = round(intel.grid.distance(ally_pos, enemy_pos), 1)
        enemy_center_relative_to_allies = None
        if ally_center and enemy_center:
            dx = enemy_center["x"] - ally_center["x"]
            dy = enemy_center["y"] - ally_center["y"]
            enemy_center_relative_to_allies = {
                "relative_to": "ally_center_of_mass",
                "dx": dx,
                "dy": dy,
                "distance": formation_center_distance,
                "direction": self._get_cardinal_direction(dx, dy),
            }

        friendly_losses = self._summarize_losses(intel.friendlies)

        return {
            "friendly_forces": {
                "alive": sum(1 for e in intel.friendlies if e.alive),
                "armed": sum(1 for e in intel.friendlies if e.alive and self._is_armed(e)),
                "mobile": sum(1 for e in intel.friendlies if e.alive and e.can_move),
                "lost_units": friendly_losses,
            },
            "enemy_forces": {
                "visible_now": len(intel.visible_enemies),
                "visible_shooters": sum(1 for e in intel.visible_enemies if e.has_fired_before),
                "killed_units": None,  # Enemy casualties are not tracked in TeamIntel
                "missing_contacts": len(missing_enemies),
            },
            "spatial_analysis": {
                "ally_center_of_mass": ally_center,
                "enemy_center_of_mass": enemy_center,
                "enemy_center_relative_to_allies": enemy_center_relative_to_allies,
                "formation_center_distance": formation_center_distance,
                "ally_formation_spread": self._formation_spread(friendly_positions, ally_center, intel.grid),
                "enemy_formation_spread": self._formation_spread(enemy_positions, enemy_center, intel.grid),
            },
        }

    # ------------------------------------------------------------------ #
    # Unit-centric blocks
    # ------------------------------------------------------------------ #
    def _build_unit_data(
        self,
        *,
        entity: Entity,
        intel: TeamIntel,
        allowed_actions: List[Action],
        cfg: PromptConfig,
        grouped_map: Dict[int, List[int]],
        move_conflicts: Dict[Tuple[int, int], List[int]],
    ) -> Dict[str, Any]:
        unit_data = {
            "unit_id": entity.id,
            "type": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": {"x": entity.pos[0], "y": entity.pos[1]},
            "capabilities": self._get_capabilities(entity),
            "nearby_allies": {
                "reference_unit": entity.id,
                "radius_checked": cfg.nearby_ally_radius,
                "allies": self._get_nearby_allies(entity, intel, cfg),
            },
            "threats": self._get_threats(entity, intel, cfg, grouped_map),
            "actions": self._get_actions(entity, allowed_actions, intel, cfg, move_conflicts),
        }
        return unit_data

    def _get_capabilities(self, entity: Entity) -> Dict[str, Any]:
        shoot_state = self._shoot_state(entity)
        missiles = getattr(entity, "missiles", None)
        weapon_range = getattr(entity, "missile_max_range", None)

        capabilities: Dict[str, Any] = {
            "can_move": entity.can_move,
            "can_shoot": shoot_state["can_shoot_now"],
        }

        if entity.kind == EntityKind.SAM:
            capabilities["is_sam_on"] = bool(getattr(entity, "on", False))

        if missiles is not None:
            capabilities["missiles_remaining"] = missiles
        if weapon_range is not None:
            capabilities["weapon_max_range"] = weapon_range
        if shoot_state["status"] != "READY":
            capabilities["shoot_status"] = shoot_state["status"]
        if shoot_state["reason"]:
            capabilities["shoot_status_reason"] = shoot_state["reason"]
        if shoot_state["cooldown_remaining"] > 0:
            capabilities["cooldown_turns_remaining"] = shoot_state["cooldown_remaining"]

        radar_range = getattr(entity, "get_active_radar_range", lambda: None)()
        if radar_range is not None:
            capabilities["radar_range"] = radar_range

        return capabilities

    def _get_nearby_allies(
        self,
        entity: Entity,
        intel: TeamIntel,
        cfg: PromptConfig,
    ) -> List[Dict[str, Any]]:
        allies: List[Dict[str, Any]] = []
        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue
            distance = intel.grid.distance(entity.pos, other.pos)
            if distance > cfg.nearby_ally_radius:
                continue
            dx = other.pos[0] - entity.pos[0]
            dy = other.pos[1] - entity.pos[1]
            shoot_state = self._shoot_state(other)
            allies.append(
                {
                    "unit_id": other.id,
                    "type": other.kind.name if hasattr(other.kind, "name") else str(other.kind),
                    "position": {"x": other.pos[0], "y": other.pos[1]},
                    "relative_position": {
                        "relative_to_unit": entity.id,
                        "dx": dx,
                        "dy": dy,
                        "distance": round(distance, 1),
                        "direction": self._get_cardinal_direction(dx, dy),
                    },
                    "can_shoot": shoot_state["can_shoot_now"],
                    "missiles_remaining": getattr(other, "missiles", None),
                }
            )
        allies.sort(key=lambda a: a["relative_position"]["distance"])
        return allies

    def _get_threats(
        self,
        entity: Entity,
        intel: TeamIntel,
        cfg: PromptConfig,
        grouped_map: Dict[int, List[int]],
    ) -> Dict[str, Any]:
        detected: List[Dict[str, Any]] = []
        for enemy in intel.visible_enemies:
            distance = intel.grid.distance(entity.pos, enemy.position)
            if distance > cfg.nearby_enemy_radius:
                continue
            dx = enemy.position[0] - entity.pos[0]
            dy = enemy.position[1] - entity.pos[1]

            our_engagement = self._our_engagement(entity, enemy, intel, distance, cfg)
            their_engagement, threat_type, risk_level, safety_margin = self._their_engagement(
                enemy, distance, cfg
            )
            is_shooter = self._enemy_can_shoot(enemy, cfg)
            detected.append(
                {
                    "enemy_id": enemy.id,
                    "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                    "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                    "position": {"x": enemy.position[0], "y": enemy.position[1]},
                    "relative_position": {
                        "relative_to_unit": entity.id,
                        "dx": dx,
                        "dy": dy,
                        "distance": round(distance, 1),
                        "direction": self._get_cardinal_direction(dx, dy),
                    },
                    "has_fired_before": enemy.has_fired_before,
                    "is_shooter": is_shooter,
                    "grouped_with": grouped_map.get(enemy.id, []),
                    "threat_type": threat_type,
                    "risk_level": risk_level,
                    "safety_distance_margin": safety_margin,
                }
            )
            if our_engagement is not None:
                detected[-1]["our_engagement"] = our_engagement
            if their_engagement is not None:
                detected[-1]["their_engagement"] = their_engagement

        detected.sort(key=lambda t: t["relative_position"]["distance"])

        return {
            "reference_unit": entity.id,
            "radius_checked": cfg.nearby_enemy_radius,
            "grouping_radius": cfg.grouping_radius,
            "detected_enemies": detected,
        }

    def _our_engagement(
        self,
        entity: Entity,
        enemy: VisibleEnemy,
        intel: TeamIntel,
        distance: float,
        cfg: PromptConfig,
    ) -> Optional[Dict[str, Any]]:
        # Skip engagement details if the unit cannot shoot or has no missiles left.
        shoot_state = self._shoot_state(entity)
        missiles = getattr(entity, "missiles", None)
        if not shoot_state["can_shoot_now"]:
            return None

        max_range = getattr(entity, "missile_max_range", None)
        in_range = max_range is not None and distance <= max_range
        hit_prob = None
        if cfg.include_hit_probabilities and in_range:
            hit_prob = intel.estimate_hit_probability(entity, enemy)
        return {
            "we_can_shoot": True,
            "in_our_range": in_range,
            "our_hit_probability": round(hit_prob, 3) if isinstance(hit_prob, float) else None,
            "out_of_our_range_by": round(distance - max_range, 1) if max_range and not in_range else None,
        }

    def _their_engagement(
        self,
        enemy: VisibleEnemy,
        distance: float,
        cfg: PromptConfig,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[float]]:
        profile = self._get_enemy_profile(enemy.kind, cfg)
        if profile.max_range <= 0:
            return None, "UNARMED", "SAFE", None

        we_are_in_range = distance <= profile.max_range and profile.max_range > 0
        estimated = None
        if cfg.include_hit_probabilities:
            if we_are_in_range:
                estimated = hit_probability(
                    distance=distance,
                    max_range=profile.max_range,
                    base=profile.base_hit_prob,
                    min_p=profile.min_hit_prob,
                )
            else:
                estimated = 0.0  # Out of range, so effective hit probability is zero
        # Classify threat level using distance to assumed max range.
        danger_range = profile.max_range
        caution_range = profile.max_range + cfg.threat_approach_buffer
        safety_margin = round(max(distance - danger_range, 0), 1)
        if distance <= danger_range:
            threat_type = "ARMED_IN_RANGE"
            risk_level = "DANGER"
        elif distance <= caution_range:
            threat_type = "ARMED_ALMOST_IN_RANGE"
            risk_level = "CAUTION"
        else:
            threat_type = "ARMED_OUT_OF_RANGE"
            risk_level = "SAFE"

        return (
            {
                "assumed_enemy_weapon_range": profile.max_range,
                "we_are_in_their_range": we_are_in_range,
                "estimated_their_hit_probability": round(estimated, 2) if isinstance(estimated, float) else None,
            },
            threat_type,
            risk_level,
            safety_margin,
        )

    def _get_actions(
        self,
        entity: Entity,
        actions: List[Action],
        intel: TeamIntel,
        cfg: PromptConfig,
        move_conflicts: Dict[Tuple[int, int], List[int]],
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for action in actions:
            entry: Dict[str, Any] = {"type": action.type.name, "entity_id": entity.id}

            if action.type == ActionType.MOVE:
                direction: MoveDir = action.params.get("dir")  # type: ignore[assignment]
                entry["direction"] = direction.name if isinstance(direction, MoveDir) else direction
                destination = self._calculate_new_position(entity.pos, direction)
                entry["destination"] = {"x": destination[0], "y": destination[1]}
                blockage = self._describe_move_blockage(entity, destination, intel, move_conflicts)
                if blockage:
                    entry["blockage"] = blockage
                if cfg.include_direction_resolution:
                    orientation_hint = self._orientation_hint(entity, direction, cfg)
                    if orientation_hint:
                        entry["orientation_hint"] = orientation_hint
            elif action.type == ActionType.SHOOT:
                target_id = action.params.get("target_id")
                entry["target_id"] = target_id
            elif action.type == ActionType.TOGGLE:
                entry["on"] = action.params.get("on")
            # WAIT has no extra fields

            formatted.append(entry)

        return formatted

    # ------------------------------------------------------------------ #
    # Casualties + helpers
    # ------------------------------------------------------------------ #
    def _get_casualties(self, intel: TeamIntel) -> Dict[str, Any]:
        friendly = []
        for entity in intel.friendlies:
            if entity.alive:
                continue
            friendly.append(
                {
                    "unit_id": entity.id,
                    "type": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
                    "last_position": {"x": entity.pos[0], "y": entity.pos[1]},
                }
            )
        return {"friendly": friendly, "enemy": []}

    def _summarize_losses(self, friendlies: Iterable[Entity]) -> Optional[str]:
        lost: Dict[str, int] = {}
        for entity in friendlies:
            if entity.alive:
                continue
            name = entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind)
            lost[name] = lost.get(name, 0) + 1
        if not lost:
            return None
        parts = [f"{count} {kind}" for kind, count in lost.items()]
        return ", ".join(parts)

    def _calculate_center_of_mass(self, positions: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        if not positions:
            return None
        avg_x = sum(pos[0] for pos in positions) / len(positions)
        avg_y = sum(pos[1] for pos in positions) / len(positions)
        # Round to nearest grid cell for simpler consumption in prompts.
        return {"x": int(round(avg_x)), "y": int(round(avg_y))}

    def _formation_spread(
        self,
        positions: List[Tuple[int, int]],
        center: Optional[Dict[str, float]],
        grid,
    ) -> Optional[float]:
        if not positions or not center:
            return None
        center_pos = (center["x"], center["y"])
        distances = [grid.distance(pos, center_pos) for pos in positions]
        if not distances:
            return None
        return round(sum(distances) / len(distances), 1)

    def _get_cardinal_direction(self, dx: float, dy: float) -> str:
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return "SAME_POSITION"
        vertical = "UP" if dy > 0 else "DOWN"
        horizontal = "RIGHT" if dx > 0 else "LEFT"
        if abs(dx) < 0.5:
            return vertical
        if abs(dy) < 0.5:
            return horizontal
        return f"{vertical}_{horizontal}"

    def _enemy_group_map(
        self,
        enemies: Iterable[VisibleEnemy],
        intel: TeamIntel,
        radius: float,
    ) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = {}
        enemy_list = list(enemies)
        for i, enemy in enumerate(enemy_list):
            for other in enemy_list[i + 1 :]:
                if intel.grid.distance(enemy.position, other.position) <= radius:
                    grouped.setdefault(enemy.id, []).append(other.id)
                    grouped.setdefault(other.id, []).append(enemy.id)
        return grouped

    def _get_enemy_profile(self, kind: EntityKind, cfg: PromptConfig) -> WeaponProfile:
        """
        Fetch the assumed weapon profile for an enemy kind using the configured map.
        """
        return cfg.enemy_weapon_profiles.get(kind, cfg.fallback_enemy_weapon_profile)

    def _collect_move_conflicts(
        self,
        allowed_actions: Dict[int, List[Action]],
        intel: TeamIntel,
    ) -> Dict[Tuple[int, int], List[int]]:
        """
        Build a map of destination -> list[entity_id] for all ally MOVE actions.

        This lets us warn the LLM about multiple friendlies aiming for the same cell.
        """
        conflicts: Dict[Tuple[int, int], List[int]] = {}
        for entity_id, actions in allowed_actions.items():
            friendly = intel.get_friendly(entity_id)
            if friendly is None or not friendly.alive:
                continue
            for action in actions:
                if action.type != ActionType.MOVE:
                    continue
                direction: MoveDir = action.params.get("dir")  # type: ignore[assignment]
                dest = self._calculate_new_position(friendly.pos, direction)
                conflicts.setdefault(dest, []).append(entity_id)
        return conflicts

    def _enemy_can_shoot(self, enemy: VisibleEnemy, cfg: PromptConfig) -> bool:
        """
        Infer enemy shooting capability from the assumed weapon profile.
        """
        # A positive weapon range in the assumed profile is treated as "armed".
        profile = self._get_enemy_profile(enemy.kind, cfg)
        return profile.max_range > 0

    def _calculate_new_position(self, current_pos: Tuple[int, int], direction: Any) -> Tuple[int, int]:
        if isinstance(direction, MoveDir):
            delta = direction.delta
        else:
            direction_map = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            delta = direction_map.get(str(direction).upper(), (0, 0))
        return current_pos[0] + delta[0], current_pos[1] + delta[1]

    def _cooldown_remaining(self, entity: Entity) -> int:
        cooldown = getattr(entity, "_cooldown", 0)
        try:
            return max(int(cooldown), 0)
        except (TypeError, ValueError):
            return 0

    def _shoot_state(self, entity: Entity) -> Dict[str, Any]:
        """
        Normalize shooting readiness so cooldowns are reflected in prompts.
        """
        cooldown_remaining = self._cooldown_remaining(entity)
        missiles = getattr(entity, "missiles", None)

        # Default state assumes the unit can shoot right now.
        state: Dict[str, Any] = {
            "can_shoot_now": bool(entity.can_shoot),
            "status": "READY",
            "reason": None,
            "cooldown_remaining": cooldown_remaining,
        }

        if not entity.can_shoot:
            state.update({"can_shoot_now": False, "status": "UNARMED"})
            return state

        if cooldown_remaining > 0:
            state.update(
                {
                    "can_shoot_now": False,
                    "status": "COOLING_DOWN",
                    "reason": f"Cooling down for {cooldown_remaining} more turn(s)",
                }
            )
            return state

        if missiles is not None and missiles <= 0:
            state.update({"can_shoot_now": False, "status": "OUT_OF_MISSILES", "reason": "No missiles remaining"})
            return state

        return state

    def _is_armed(self, entity: Entity) -> bool:
        return bool(getattr(entity, "missiles", 0)) or entity.can_shoot

    def _describe_move_blockage(
        self,
        mover: Entity,
        destination: Tuple[int, int],
        intel: TeamIntel,
        move_conflicts: Dict[Tuple[int, int], List[int]],
    ) -> Optional[Dict[str, Any]]:
        """
        Explain whether a move is blocked or risky based on known occupancy and
        other allies targeting the same cell.
        """
        occupant_friendly: Optional[Entity] = None
        occupant_enemy: Optional[VisibleEnemy] = None

        occupied = intel.is_occupied(destination, ignore_ids={mover.id})
        if occupied:
            for friendly in intel.friendlies:
                if friendly.id == mover.id or not friendly.alive:
                    continue
                if friendly.pos == destination:
                    occupant_friendly = friendly
                    break

            if occupant_friendly is None:
                for enemy in intel.visible_enemies:
                    if enemy.position == destination:
                        occupant_enemy = enemy
                        break

        shared_with = [eid for eid in move_conflicts.get(destination, []) if eid != mover.id]

        details: Dict[str, Any] = {}
        reasons: List[str] = []
        severity: Optional[str] = None  # "BLOCKED" or "RISK"

        if occupant_enemy:
            severity = "BLOCKED"
            reasons.append(f"Visible enemy at destination (id={occupant_enemy.id})")
            details["occupied_by_enemy_id"] = occupant_enemy.id

        if occupant_friendly:
            if not occupant_friendly.can_move:
                severity = "BLOCKED"
                reasons.append(f"Immobile friendly occupying cell (id={occupant_friendly.id})")
            else:
                severity = severity or "RISK"
                reasons.append(f"Friendly currently in cell (id={occupant_friendly.id}) and may or may not move away")
            details["occupied_by_friendly_id"] = occupant_friendly.id
            details["occupied_by_friendly_can_move"] = occupant_friendly.can_move

        if shared_with:
            severity = severity or "RISK"
            reasons.append(f"Other allies also planning to move here: {shared_with}")
            details["shared_destination_with"] = shared_with

        if not reasons:
            return None

        details["severity"] = severity or "RISK"
        details["reason"] = "; ".join(reasons)
        return details

    def _get_orientation(self, team_name: str, cfg: PromptConfig) -> TeamOrientation:
        return cfg.team_orientation_map.get(
            team_name,
            TeamOrientation(spawn_side="UNKNOWN", forward=MoveDir.RIGHT, backward=MoveDir.LEFT),
        )

    def _orientation_hint(
        self,
        entity: Entity,
        direction: Any,
        cfg: PromptConfig,
    ) -> Optional[str]:
        """
        For horizontal moves, add a concise note when the direction aligns with the team's forward or backward orientation.
        """
        team_name = entity.team.name if hasattr(entity.team, "name") else str(entity.team)
        orientation = self._get_orientation(team_name, cfg)
        dir_obj = direction if isinstance(direction, MoveDir) else MoveDir.__members__.get(str(direction).upper())

        if not isinstance(dir_obj, MoveDir):
            return None
        if dir_obj == orientation.forward:
            return f"{dir_obj.name} means FORWARD for {team_name} team"
        if dir_obj == orientation.backward:
            return f"{dir_obj.name} means BACKWARD for {team_name} team"
        return None
