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
from env.core.types import ActionType, EntityKind, MoveDir
from env.entities.base import Entity
from env.mechanics.combat import hit_probability
from ..team_intel import TeamIntel, VisibleEnemy


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
        EntityKind.UNKNOWN: WeaponProfile(max_range=3.0, base_hit_prob=0.6, min_hit_prob=0.1),
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
    """

    nearby_ally_radius: float = 5.0
    nearby_enemy_radius: float = 8.0
    grouping_radius: float = 2.5
    threat_close_radius: float = 3.0
    include_hit_probabilities: bool = True
    include_casualties: bool = True
    enemy_weapon_profiles: Dict[EntityKind, WeaponProfile] = field(default_factory=_default_enemy_profiles)
    fallback_enemy_weapon_profile: WeaponProfile = field(
        default_factory=lambda: WeaponProfile(max_range=3.0, base_hit_prob=0.7, min_hit_prob=0.1)
    )


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
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build the JSON prompt and return both the string and the underlying dict.
        """
        cfg = config or PromptConfig()
        friendly_positions = [e.pos for e in intel.friendlies if e.alive]
        enemy_positions = [e.position for e in intel.visible_enemies]

        payload: Dict[str, Any] = {
            "metadata": self._build_metadata(intel, turn_number, team_name),
            "situation": self._build_situation(intel, friendly_positions, enemy_positions, cfg),
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
                )
            )

        json_string = json.dumps(payload, indent=2)
        return json_string, payload

    # ------------------------------------------------------------------ #
    # Metadata + situation
    # ------------------------------------------------------------------ #
    def _build_metadata(self, intel: TeamIntel, turn_number: int, team_name: Optional[str]) -> Dict[str, Any]:
        return {
            "turn": turn_number,
            "team": team_name or (intel.friendlies[0].team.name if intel.friendlies else "UNKNOWN"),
            "grid_size": {
                "width": intel.grid.width,
                "height": intel.grid.height,
                "center": {
                    "x": intel.grid.width / 2,
                    "y": intel.grid.height / 2,
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

    def _build_situation(
        self,
        intel: TeamIntel,
        friendly_positions: List[Tuple[int, int]],
        enemy_positions: List[Tuple[int, int]],
        cfg: PromptConfig,
    ) -> Dict[str, Any]:
        ally_center = self._calculate_center_of_mass(friendly_positions)
        enemy_center = self._calculate_center_of_mass(enemy_positions)
        formation_separation = (
            self._euclidean_distance(ally_center, enemy_center) if ally_center and enemy_center else None
        )

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
            },
            "spatial_analysis": {
                "ally_center_of_mass": ally_center,
                "enemy_center_of_mass": enemy_center,
                "formation_separation": formation_separation,
                "ally_formation_spread": self._formation_spread(friendly_positions, ally_center),
                "enemy_formation_spread": self._formation_spread(enemy_positions, enemy_center),
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
    ) -> Dict[str, Any]:
        threats = self._get_threats(entity, intel, cfg, grouped_map)
        nearby_allies = self._get_nearby_allies(entity, intel, cfg)

        unit_data = {
            "unit_id": entity.id,
            "type": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": {"x": entity.pos[0], "y": entity.pos[1]},
            "capabilities": self._get_capabilities(entity),
            "nearby_allies": {
                "reference_unit": entity.id,
                "radius_checked": cfg.nearby_ally_radius,
                "allies": nearby_allies,
            },
            "threats": threats,
            "actions": self._get_actions(entity, allowed_actions, intel, cfg),
        }

        unit_data["tactical_assessment"] = self._assess_unit_situation(threats, nearby_allies)
        return unit_data

    def _get_capabilities(self, entity: Entity) -> Dict[str, Any]:
        return {
            "can_move": entity.can_move,
            "can_shoot": entity.can_shoot,
            "missiles_remaining": getattr(entity, "missiles", None),
            "weapon_max_range": getattr(entity, "missile_max_range", None),
            "radar_range": getattr(entity, "get_active_radar_range", lambda: None)(),
        }

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
                    "can_shoot": self._is_armed(other),
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
            their_engagement, threat_type = self._their_engagement(enemy, distance, cfg)

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
                    "is_shooter": enemy.has_fired_before,
                    "grouped_with": grouped_map.get(enemy.id, []),
                    "threat_type": threat_type,
                    "our_engagement": our_engagement,
                    "their_engagement": their_engagement,
                }
            )

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
    ) -> Dict[str, Any]:
        max_range = getattr(entity, "missile_max_range", None)
        in_range = max_range is not None and distance <= max_range
        hit_prob = None
        if cfg.include_hit_probabilities and in_range:
            hit_prob = intel.estimate_hit_probability(entity, enemy)
        return {
            "we_can_shoot": bool(entity.can_shoot and getattr(entity, "missiles", 0) > 0),
            "in_our_range": in_range,
            "our_hit_probability": round(hit_prob, 3) if isinstance(hit_prob, float) else None,
            "out_of_our_range_by": round(distance - max_range, 1) if max_range and not in_range else None,
        }

    def _their_engagement(
        self,
        enemy: VisibleEnemy,
        distance: float,
        cfg: PromptConfig,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        profile = cfg.enemy_weapon_profiles.get(enemy.kind, cfg.fallback_enemy_weapon_profile)
        we_are_in_range = distance <= profile.max_range and profile.max_range > 0
        estimated = None
        if cfg.include_hit_probabilities and we_are_in_range:
            estimated = hit_probability(
                distance=distance,
                max_range=profile.max_range,
                base=profile.base_hit_prob,
                min_p=profile.min_hit_prob,
            )
        threat_type = None
        if we_are_in_range and distance <= cfg.threat_close_radius:
            threat_type = "ARMED_AND_CLOSE"
        elif we_are_in_range:
            threat_type = "ARMED_IN_RANGE"
        elif profile.max_range > 0:
            threat_type = "ARMED_OUT_OF_RANGE"

        return (
            {
                "assumed_enemy_weapon_range": profile.max_range,
                "we_are_in_their_range": we_are_in_range,
                "estimated_their_hit_probability": round(estimated, 2) if isinstance(estimated, float) else None,
            },
            threat_type,
        )

    def _get_actions(
        self,
        entity: Entity,
        actions: List[Action],
        intel: TeamIntel,
        cfg: PromptConfig,
    ) -> List[Dict[str, Any]]:
        action_list: List[Dict[str, Any]] = []
        for action in actions:
            data: Dict[str, Any] = {"type": action.type.name, "parameters": action.to_dict()["params"]}
            if action.type == ActionType.SHOOT:
                target_id = action.params.get("target_id")
                target = intel.get_enemy(target_id)
                if target:
                    distance = intel.grid.distance(entity.pos, target.position)
                    dx = target.position[0] - entity.pos[0]
                    dy = target.position[1] - entity.pos[1]
                    data["target"] = {
                        "enemy_id": target_id,
                        "type": target.kind.name if hasattr(target.kind, "name") else str(target.kind),
                        "position": {"x": target.position[0], "y": target.position[1]},
                        "relative_position": {"dx": dx, "dy": dy, "distance": round(distance, 1), "direction": self._get_cardinal_direction(dx, dy)},
                    }
                    if cfg.include_hit_probabilities:
                        hit_prob = intel.estimate_hit_probability(entity, target)
                        data["hit_probability"] = round(hit_prob, 3) if isinstance(hit_prob, float) else None
            elif action.type == ActionType.MOVE:
                direction: MoveDir = action.params.get("dir")  # type: ignore[assignment]
                data["direction"] = direction.name if isinstance(direction, MoveDir) else direction
                new_pos = self._calculate_new_position(entity.pos, direction)
                data["destination"] = {"x": new_pos[0], "y": new_pos[1]}
            action_list.append(data)
        return action_list

    def _assess_unit_situation(
        self,
        threats: Dict[str, Any],
        nearby_allies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        detected = threats.get("detected_enemies", [])
        immediate = [
            t
            for t in detected
            if t.get("threat_type") in ("ARMED_AND_CLOSE",) or (t.get("their_engagement", {}).get("we_are_in_their_range") and t["relative_position"]["distance"] <= threats.get("grouping_radius", 0))
        ]
        return {
            "nearby_threat_count": len(detected),
            "nearby_ally_count": len(nearby_allies),
            "threat_status": "DANGER" if immediate else ("CAUTION" if detected else "SAFE"),
        }

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
        return {"x": round(avg_x, 1), "y": round(avg_y, 1)}

    def _formation_spread(
        self,
        positions: List[Tuple[int, int]],
        center: Optional[Dict[str, float]],
    ) -> Optional[float]:
        if not positions or not center:
            return None
        distances = [((pos[0] - center["x"]) ** 2 + (pos[1] - center["y"]) ** 2) ** 0.5 for pos in positions]
        if not distances:
            return None
        return round(sum(distances) / len(distances), 1)

    def _euclidean_distance(self, pos1: Optional[Dict[str, float]], pos2: Optional[Dict[str, float]]) -> Optional[float]:
        if not pos1 or not pos2:
            return None
        dx = pos1["x"] - pos2["x"]
        dy = pos1["y"] - pos2["y"]
        return round((dx ** 2 + dy ** 2) ** 0.5, 1)

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

    def _calculate_new_position(self, current_pos: Tuple[int, int], direction: Any) -> Tuple[int, int]:
        if isinstance(direction, MoveDir):
            delta = direction.delta
        else:
            direction_map = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            delta = direction_map.get(str(direction).upper(), (0, 0))
        return current_pos[0] + delta[0], current_pos[1] + delta[1]

    def _is_armed(self, entity: Entity) -> bool:
        return bool(getattr(entity, "missiles", 0)) or entity.can_shoot
