"""
Lightweight, human-friendly state formatter for the LLM agent.

The output mirrors the earlier concise schema: team/turn summary, friendly
units with nearby context and available actions, visible enemies, last-known
enemies, and simple battlefield metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from env.core.actions import Action
from env.core.types import ActionType, EntityKind, MoveDir, Team
from env.world import WorldState
from env.mechanics.combat import hit_probability

from agents.team_intel import TeamIntel, VisibleEnemy


@dataclass
class WeaponProfile:
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
    spawn_side: str
    forward: MoveDir
    backward: MoveDir
    note: str = "Forward = toward enemy base; Backward/Retreat = toward own base."


def _default_team_orientations() -> Dict[str, TeamOrientation]:
    """
    Defaults assume BLUE spawns left (forward=RIGHT) and RED spawns right (forward=LEFT).
    """
    return {
        Team.BLUE.name: TeamOrientation(spawn_side="LEFT", forward=MoveDir.RIGHT, backward=MoveDir.LEFT),
        Team.RED.name: TeamOrientation(spawn_side="RIGHT", forward=MoveDir.LEFT, backward=MoveDir.RIGHT),
    }


@dataclass
class StateFormatterConfig:
    nearby_unit_distance: float = 3.0
    nearby_enemy_distance: float = 5.0
    threat_approach_buffer: float = 2.0  # how far outside enemy range counts as CAUTION
    grouping_radius: float = 3.0  # reserved for future enemy grouping hints
    include_dead_entities: bool = True
    include_orientation_metadata: bool = True
    team_orientation_map: Dict[str, TeamOrientation] = field(default_factory=_default_team_orientations)
    enemy_weapon_profiles: Dict[EntityKind, WeaponProfile] = field(default_factory=_default_enemy_profiles)
    fallback_enemy_weapon_profile: WeaponProfile = field(
        default_factory=lambda: WeaponProfile(max_range=3.0, base_hit_prob=0.7, min_hit_prob=0.1)
    )


class StateFormatter:
    def __init__(self, config: Optional[StateFormatterConfig] = None) -> None:
        self.config = config or StateFormatterConfig()

    def build_state(
        self,
        *,
        world: WorldState,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        turn: int,
        team: Team,
        missing_enemies: List[Dict[str, Any]],
        casualties: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        friendly_units = [
            self._friendly_unit_entry(entity, intel, allowed_actions.get(entity.id, []))
            for entity in intel.friendlies
            if entity.alive
        ]

        friendly_positions = [e.pos for e in intel.friendlies if e.alive]
        enemy_units = [
            self._visible_enemy_entry(enemy, friendly_positions, intel) for enemy in intel.visible_enemies
        ]

        summary = {
            "turn": turn,
            "friendlies_alive": len(friendly_units),
            "visible_enemies": len(enemy_units),
            "missing_enemies": len(missing_enemies),
        }

        dead_entities: Optional[List[Dict[str, Any]]] = None
        if self.config.include_dead_entities:
            dead_entities = self._dead_entities(intel, casualties)

        unit_details = [
            self._friendly_unit_structured(entity, intel, allowed_actions.get(entity.id, []), self.config)
            for entity in intel.friendlies
            if entity.alive
        ]

        payload: Dict[str, Any] = {
            "team": team.name,
            "turn_summary": summary,
            "friendly_units": friendly_units,
            "friendly_units_detailed": unit_details,
            "enemy_units": enemy_units,
            "distant_visible_enemies": self._distant_enemies(intel, self.config),
            "last_known_enemies": missing_enemies,
            "battlefield": {
                "width": world.grid.width,
                "height": world.grid.height,
                "center": {"x": world.grid.width // 2, "y": world.grid.height // 2},
            },
            "forces_snapshot": self._forces_snapshot(intel, enemy_units, dead_entities),
            "potential_move_conflicts": self._structured_move_conflicts(
                self._collect_move_conflicts(allowed_actions, intel)
            ),
            "coordinate_system": self._coordinate_system_lines(),
        }

        if self.config.include_dead_entities and dead_entities is not None:
            payload["dead_entities"] = dead_entities
        if casualties is not None:
            payload["casualties"] = casualties
        if self.config.include_orientation_metadata:
            payload["team_orientation"] = self._orientation_payload(team.name)

        return payload

    def build_state_string(
        self,
        *,
        world: WorldState,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        turn: int,
        team: Team,
        missing_enemies: List[Dict[str, Any]],
        casualties: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> str:
        """
        Produce a single formatted string version of the state.
        """
        state = self.build_state(
            world=world,
            intel=intel,
            allowed_actions=allowed_actions,
            turn=turn,
            team=team,
            missing_enemies=missing_enemies,
            casualties=casualties,
        )

        lines: List[str] = []
        lines.append(f"### Tactical State Snapshot - Team {state['team']} (Turn {state['turn_summary']['turn']})")
        lines.append("")
        # Battlefield / coordinates / orientation
        lines.append("#### Battlefield & Coordinates")
        lines.append(
            f"- **Map:** {state['battlefield']['width']}x{state['battlefield']['height']} "
            f"(center=({state['battlefield']['center']['x']}, {state['battlefield']['center']['y']}))"
        )
        lines.append("- **Coordinate system:**")
        lines.extend([f"  {line}" for line in self._coordinate_system_lines()])
        if self.config.include_orientation_metadata:
            lines.append("- **Team orientation:**")
            lines.extend([f"  {line}" for line in self._orientation_lines(team)])
        lines.append("")

        forces = self._forces_snapshot(intel, state["enemy_units"], state.get("dead_entities"))
        lines.append("#### Situation Summary")
        lines.append(
            f"- **Friendly forces:** alive={forces['friendly_alive']}, armed={forces['friendly_armed']}, mobile={forces['friendly_mobile']}, lost={forces['friendly_lost'] or 'none'}"
        )
        lines.append(
            f"- **Enemy forces:** visible_now={forces['enemy_visible']}, visible_shooters={forces['enemy_visible_shooters']}, killed={forces['enemy_killed'] or 'none'}"
        )
        lines.append("")

        if state["last_known_enemies"]:
            lines.append("#### Missing Enemies")
            for missing in state["last_known_enemies"]:
                last_pos = missing.get("last_seen_position") or {}
                lines.append(
                    f"- id={missing.get('id')} type={missing.get('type')} "
                    f"last_seen=({last_pos.get('x')}, {last_pos.get('y')}) "
                    f"turns_since_seen={missing.get('turns_since_seen')}"
                )

        dead_entities = state.get("dead_entities") or []
        if dead_entities:
            lines.append("")
            lines.append("#### Casualties")
            our_team = team.name
            friendly_losses = [d for d in dead_entities if d.get("team") == our_team]
            enemy_losses = [d for d in dead_entities if d.get("team") != our_team]

            if friendly_losses:
                lines.append("- **Friendly casualties:**")
                for dead in friendly_losses:
                    killer = dead.get("killed_by") or {}
                    parts = [
                        f"- Turn {dead.get('killed_on_turn')}:",  # could be None; still prints None if missing
                        f"Ally {dead.get('type')} #{dead.get('id')}",
                        f"killed at ({dead.get('death_position', {}).get('x')}, {dead.get('death_position', {}).get('y')})",
                    ]
                    if killer:
                        parts.append(
                            f"by Enemy {killer.get('type')} #{killer.get('id')}"
                        )
                    lines.append(" ".join(filter(None, parts)))

            if enemy_losses:
                lines.append("- **Enemy casualties:**")
                for dead in enemy_losses:
                    killer = dead.get("killed_by") or {}
                    parts = [
                        f"- Turn {dead.get('killed_on_turn')}:",
                        f"Enemy {dead.get('type')} #{dead.get('id')}",
                        f"killed at ({dead.get('death_position', {}).get('x')}, {dead.get('death_position', {}).get('y')})",
                    ]
                    if killer:
                        parts.append(
                            f"by Ally {killer.get('type')} #{killer.get('id')}"
                        )
                    lines.append(" ".join(filter(None, parts)))

        unit_sections = self._unit_sections(intel, allowed_actions, self.config)
        if unit_sections:
            lines.append("")
            lines.append("#### Ally Units & Options")
            lines.extend(unit_sections)
            lines.append("")  # Spacer after terminology for readability
            lines.extend(self._terminology_lines())

        distant = self._distant_enemies(intel, self.config)
        if distant:
            lines.append("")
            lines.append(f"#### Distant Visible Enemies (>{self.config.nearby_enemy_distance})")
            for entry in distant:
                parts = [
                    f"- Enemy #{entry['enemy_id']} {entry['type']} at ({entry['position']['x']}, {entry['position']['y']})",
                    f"closest_ally_dist={entry.get('nearest_friendly_distance')}",
                ]
                if entry.get("risk_level"):
                    parts.append(f"risk={entry['risk_level']}")
                if entry.get("threat_type"):
                    parts.append(f"threat={entry['threat_type']}")
                if entry.get("threat_type") == "UNARMED":
                    parts.append("cannot shoot")
                lines.append(" ".join(parts))

        move_conflicts = self._collect_move_conflicts(allowed_actions, intel)
        collision_section = self._format_collision_section(move_conflicts)
        if collision_section:
            lines.append("")
            lines.append("#### Potential Move Collisions (Allies)")
            lines.append("Note: multiple allies could enter the same cell next turn.")
            lines.extend(collision_section)

        return "\n".join(lines)

    def _friendly_unit_entry(
        self,
        entity,
        intel: TeamIntel,
        allowed_actions: List[Action],
    ) -> Dict[str, Any]:
        nearby = self._nearby_units(entity, intel)
        available_actions = self._format_available_actions(entity, allowed_actions, intel)
        return {
            "id": entity.id,
            "type": getattr(entity.kind, "name", str(entity.kind)),
            "position": {"x": entity.pos[0], "y": entity.pos[1]},
            "capabilities": {
                "can_move": entity.can_move,
                "can_shoot": entity.can_shoot,
                "missiles_remaining": getattr(entity, "missiles", None),
                "weapon_range": getattr(entity, "missile_max_range", None),
                "is_sam_on": getattr(entity, "on", None),
            },
            "nearby": nearby,
            "available_actions": available_actions,
        }

    def _nearby_units(self, entity, intel: TeamIntel) -> Dict[str, Any]:
        allies: List[Dict[str, Any]] = []
        enemies: List[Dict[str, Any]] = []

        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue
            dist = intel.grid.distance(entity.pos, other.pos)
            if dist <= self.config.nearby_unit_distance:
                allies.append(
                    {
                        "id": other.id,
                        "type": getattr(other.kind, "name", str(other.kind)),
                        "distance": round(dist, 1),
                        "relative": self._relative_position(entity.pos, other.pos),
                    }
                )

        for enemy in intel.visible_enemies:
            dist = intel.grid.distance(entity.pos, enemy.position)
            if dist <= self.config.nearby_unit_distance:
                enemies.append(
                    {
                        "id": enemy.id,
                        "type": getattr(enemy.kind, "name", str(enemy.kind)),
                        "distance": round(dist, 1),
                        "relative": self._relative_position(entity.pos, enemy.position),
                    }
                )

        allies.sort(key=lambda x: x["distance"])
        enemies.sort(key=lambda x: x["distance"])

        return {
            "close_friendlies": allies,
            "visible_enemies": enemies,
        }

    def _visible_enemy_entry(
        self,
        enemy: VisibleEnemy,
        friendly_positions: List[Tuple[int, int]],
        intel: TeamIntel,
    ) -> Dict[str, Any]:
        nearest_dist = None
        if friendly_positions:
            nearest_dist = min(intel.grid.distance(pos, enemy.position) for pos in friendly_positions)

        return {
            "id": enemy.id,
            "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
            "type": getattr(enemy.kind, "name", str(enemy.kind)),
            "position": {"x": enemy.position[0], "y": enemy.position[1]},
            "distance_from_nearest_friendly": round(nearest_dist, 1) if nearest_dist is not None else None,
            "detected_by": list(enemy.seen_by),
        }

    def _dead_entities(
        self,
        intel: TeamIntel,
        casualties: Optional[Dict[str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        """
        Dead-entity list constrained to what our team plausibly knows.
        """
        if casualties:
            # Prefer enriched casualty records if provided
            dead: List[Dict[str, Any]] = []
            dead.extend(casualties.get("friendly", []))
            dead.extend(casualties.get("enemy", []))
            return dead

        # Fallback: only include our own dead friendlies (no fog-breaking enemy info).
        dead: List[Dict[str, Any]] = []
        for entity in intel.friendlies:
            if entity.alive:
                continue
            dead.append(
                {
                    "id": entity.id,
                    "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
                    "type": getattr(entity.kind, "name", str(entity.kind)),
                    "death_position": {"x": entity.pos[0], "y": entity.pos[1]},
                }
            )
        return dead

    def _forces_snapshot(
        self,
        intel: TeamIntel,
        enemy_units: List[Dict[str, Any]],
        dead_entities: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        friendly_alive = sum(1 for e in intel.friendlies if e.alive)
        friendly_armed = sum(
            1
            for e in intel.friendlies
            if e.alive and (getattr(e, "missiles", 0) or getattr(e, "can_shoot", False))
        )
        friendly_mobile = sum(1 for e in intel.friendlies if e.alive and getattr(e, "can_move", False))
        friendly_lost = sum(1 for e in intel.friendlies if not e.alive) or None

        enemy_visible = len(enemy_units)
        enemy_visible_shooters = sum(1 for e in enemy_units if e.get("type") in ("AIRCRAFT", "SAM"))
        enemy_killed = (
            sum(1 for d in (dead_entities or []) if d.get("team") != intel.friendlies[0].team.name)
            if intel.friendlies
            else None
        )

        return {
            "friendly_alive": friendly_alive,
            "friendly_armed": friendly_armed,
            "friendly_mobile": friendly_mobile,
            "friendly_lost": friendly_lost,
            "enemy_visible": enemy_visible,
            "enemy_visible_shooters": enemy_visible_shooters,
            "enemy_killed": enemy_killed,
        }

    def _coordinate_system_lines(self) -> List[str]:
        return [
            "- Standard Cartesian grid with origin at bottom-left",
            "- X increases LEFT -> RIGHT; Y increases BOTTOM -> TOP",
            "- Movement: UP (y+1), DOWN (y-1), RIGHT (x+1), LEFT (x-1)",
            "- Example: at (10,5), RIGHT -> (11,5); UP -> (10,6)",
        ]

    def _orientation_lines(self, team: Team) -> List[str]:
        orientation = self._get_orientation(team.name)
        other_team = Team.RED if team == Team.BLUE else Team.BLUE
        #other_orientation = self._get_orientation(other_team.name)
        lines = [
            f"- {team.name}: spawn_side={orientation.spawn_side}, **forward={orientation.forward.name}**, **backward={orientation.backward.name}**",
            #f"- {other_team.name}: spawn_side={other_orientation.spawn_side}, forward={other_orientation.forward.name}, backward={other_orientation.backward.name}",
            f"- **Note: {orientation.note}**",
        ]
        return lines

    def _orientation_payload(self, team_name: str) -> Dict[str, Any]:
        current = self._get_orientation(team_name)
        return {
            "team": team_name,
            "spawn_side": current.spawn_side,
            "forward_direction": current.forward.name,
            "backward_direction": current.backward.name,
            "note": current.note,
            "orientation_by_team": {
                name: {
                    "spawn_side": ori.spawn_side,
                    "forward_direction": ori.forward.name,
                    "backward_direction": ori.backward.name,
                    "note": ori.note,
                }
                for name, ori in self.config.team_orientation_map.items()
            },
        }

    def _get_orientation(self, team_name: str) -> TeamOrientation:
        return self.config.team_orientation_map.get(
            team_name,
            TeamOrientation(spawn_side="UNKNOWN", forward=MoveDir.RIGHT, backward=MoveDir.LEFT),
        )

    def _format_available_actions(
        self,
        entity,
        actions: List[Action],
        intel: TeamIntel,
    ) -> Dict[str, Any]:
        shoot_state = self._shoot_state(entity)
        formatted: Dict[str, Any] = {
            "can_wait": False,
            "movement_options": [],
            "shooting_options": [],
            "blocked_shooting_options": [],
            "toggle_option": None,
            "can_move": bool(getattr(entity, "can_move", False)),
        }

        if formatted["can_move"]:
            allowed_dirs = {
                (action.params.get("dir").name if isinstance(action.params.get("dir"), MoveDir) else str(action.params.get("dir")))
                for action in actions
                if action.type == ActionType.MOVE
            }

            # Movement (include blocked reasons for clarity)
            for direction in [MoveDir.UP, MoveDir.DOWN, MoveDir.LEFT, MoveDir.RIGHT]:
                destination = self._calculate_destination(entity.pos, direction)
                reason, blocker_id = self._move_block_reason(destination, entity, intel)
                formatted["movement_options"].append(
                    {
                        "direction": direction.name,
                        "destination": {"x": destination[0], "y": destination[1]},
                        "allowed": direction.name in allowed_dirs and reason is None,
                        "blocked_reason": reason,
                        "blocked_by_id": blocker_id,
                    }
                )

        # Shooting (only allowed targets; add hit estimate if possible)
        for action in actions:
            if action.type == ActionType.WAIT:
                formatted["can_wait"] = True
            elif action.type == ActionType.SHOOT:
                target_id = action.params.get("target_id")
                # If the unit cannot shoot right now (e.g., SAM cooling), treat this as blocked.
                if not shoot_state["can_shoot_now"]:
                    formatted["blocked_shooting_options"].append(
                        {
                            "target_id": target_id,
                            "reason": shoot_state.get("reason") or shoot_state.get("status"),
                        }
                    )
                    continue
                entry = {
                    "target_id": target_id,
                    "target_type": None,
                    "hit_probability": None,
                }
                enemy = intel.get_enemy(target_id) if target_id is not None else None
                if enemy:
                    entry["target_type"] = getattr(enemy.kind, "name", str(enemy.kind))
                    hit_prob = intel.estimate_hit_probability(entity, enemy)
                    entry["hit_probability"] = round(hit_prob, 3) if isinstance(hit_prob, float) else None
                formatted["shooting_options"].append(entry)
            elif action.type == ActionType.TOGGLE:
                formatted["toggle_option"] = {
                    "current_state": "ON" if getattr(entity, "on", False) else "OFF",
                    "toggle_to": "OFF" if getattr(entity, "on", False) else "ON",
                }
            else:
                formatted.setdefault("other_actions", []).append({"type": action.type.name, "raw": action.params})

        return formatted

    def _friendly_unit_structured(
        self,
        entity,
        intel: TeamIntel,
        actions: List[Action],
        cfg: StateFormatterConfig,
    ) -> Dict[str, Any]:
        """
        Structured snapshot paralleling the unit block used in the string formatter.
        """
        shoot_state = self._shoot_state(entity)
        radar_active = getattr(entity, "get_active_radar_range", lambda: None)()
        radar_nominal = getattr(entity, "radar_range", None)
        sam_status = None
        if getattr(entity, "kind", None) == EntityKind.SAM:
            sam_status = {
                "on": bool(getattr(entity, "on", False)),
                "cooldown_remaining": getattr(entity, "_cooldown", 0),
            }

        return {
            "id": entity.id,
            "type": getattr(entity.kind, "name", str(entity.kind)),
            "position": {"x": entity.pos[0], "y": entity.pos[1]},
            "capabilities": {
                "can_move": bool(getattr(entity, "can_move", False)),
                "can_shoot": bool(getattr(entity, "can_shoot", False)),
                "missiles_remaining": getattr(entity, "missiles", None),
                "weapon_range": getattr(entity, "missile_max_range", None),
                "radar_range": radar_nominal,
                "radar_active_range": radar_active,
            },
            "status": {
                "shoot_state": shoot_state,
                "sam_status": sam_status,
            },
            "nearby_allies": self._get_nearby_allies(entity, intel, cfg),
            "threats": self._get_threats(entity, intel, cfg),
            "available_actions": self._format_available_actions(entity, actions, intel),
        }

    def _unit_sections(
        self,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        cfg: StateFormatterConfig,
    ) -> List[str]:
        sections: List[str] = []
        for entity in intel.friendlies:
            if not entity.alive:
                continue
            sections.extend(
                self._format_unit_block(
                    entity,
                    intel,
                    allowed_actions.get(entity.id, []),
                    cfg,
                )
            )
            sections.append("")  # spacer between units
        return sections

    def _format_unit_block(
        self,
        entity,
        intel: TeamIntel,
        actions: List[Action],
        cfg: StateFormatterConfig,
    ) -> List[str]:
        lines: List[str] = []
        unit_label = getattr(entity.kind, "name", str(entity.kind))
        box_width = 78

        def top_bar(title: str) -> str:
            prefix = f"╔═══ {title} "
            fill = max(box_width - len(prefix) - 1, 0)
            return prefix + "═" * fill

        def bottom_bar() -> str:
            return "╚" + "═" * (box_width - 2) + "╝"

        def pad(content: str) -> str:
            return f"║ {content}"

        lines.append(top_bar(f"ALLY Unit #{entity.id} - {unit_label}"))
        # General info
        lines.append(pad(f"**Position:** x={entity.pos[0]}, y={entity.pos[1]}"))
        caps = []
        if entity.can_move:
            caps.append("Mobile")
        if entity.can_shoot:
            caps.append("Armed")
        missiles = getattr(entity, "missiles", None)
        if missiles is not None:
            caps.append(f"Missiles={missiles}")
        radar_active = getattr(entity, "get_active_radar_range", lambda: None)()
        radar_nominal = getattr(entity, "radar_range", None)
        if radar_nominal is not None and radar_nominal > 0:
            if radar_active and radar_active > 0:
                caps.append(f"Radar({radar_nominal}) active")
            else:
                caps.append(f"Radar({radar_nominal}) off")
        weapon_range = getattr(entity, "missile_max_range", None)
        if weapon_range is not None:
            caps.append(f"WeaponRange={weapon_range}")
        if caps:
            lines.append(pad(f"**Capabilities:** {', '.join(caps)}"))
        shoot_state = self._shoot_state(entity)
        if shoot_state.get("status") != "READY":
            note_parts = [shoot_state.get("status")]
            if shoot_state.get("reason"):
                note_parts.append(shoot_state["reason"])
            lines.append(pad(f"**Shoot status:** {' - '.join(filter(None, note_parts))}"))
        # SAM specific status
        if getattr(entity, "kind", None) == EntityKind.SAM:
            sam_on = bool(getattr(entity, "on", False))
            cooldown = getattr(entity, "_cooldown", 0)
            if sam_on:
                if cooldown > 0:
                    lines.append(pad(f"**SAM Status:** ON, cooling down ({cooldown} turn(s) remaining)"))
                else:
                    lines.append(pad("**SAM Status:** ON & READY (visible to enemies)"))
            else:
                lines.append(pad("**SAM Status:** OFF (not emitting/visible; radar off; cannot shoot while off)"))

        lines.append("║")

        # Nearby allies
        nearby_allies = self._get_nearby_allies(entity, intel, cfg)
        if nearby_allies:
            lines.append(pad(f"**▶ Nearby Allies (within {cfg.nearby_unit_distance}):**"))
            for a in nearby_allies:
                rel = a["relative_position"]
                lines.append(
                    pad(
                        f"  • #{a['unit_id']} ({a['type']}) rel={rel['direction']} "
                        f"(dx={rel['dx']}, dy={rel['dy']}, dist={rel['distance']})"
                    )
                )
        else:
            lines.append(pad("**▶ Nearby Allies:** none"))

        lines.append("║")

        # Threats (visible enemies within radius)
        threats = self._get_threats(entity, intel, cfg)
        detected = threats["detected_enemies"]
        if detected:
            lines.append(pad(f"**⚠ Threats (within {cfg.nearby_enemy_distance}):**"))
            for threat in detected:
                rel = threat["relative_position"]
                risk_level = threat.get("risk_level") or "UNKNOWN"
                threat_type = threat.get("threat_type") or "UNKNOWN"
                desc_parts = [
                    f"risk={risk_level} threat={threat_type} Enemy #{threat['enemy_id']} {threat['type']}",
                ]
                if threat.get("type") == "AIRCRAFT":
                    if threat.get("has_fired_before"):
                        desc_parts.append("is_decoy=no (confirmed shooter)")
                    else:
                        desc_parts.append("is_decoy=possible (unconfirmed)")
                desc_parts.append(
                    f"rel={rel['direction']} (dx={rel['dx']}, dy={rel['dy']}, dist={rel['distance']})"
                )
                if threat.get("threat_type") == "UNARMED":
                    desc_parts.append("cannot shoot")
                if threat.get("our_engagement"):
                    oe = threat["our_engagement"]
                    if oe.get("in_our_range"):
                        desc_parts.append(f"our_hit≈{oe.get('our_hit_probability')} (if we shoot)")
                    elif oe.get("out_of_our_range_by") is not None:
                        desc_parts.append(f"out_of_range_by={oe['out_of_our_range_by']}")
                if threat.get("their_engagement"):
                    te = threat["their_engagement"]
                    if te.get("we_are_in_their_range"):
                        desc_parts.append(f"enemy_hit≈{te.get('estimated_their_hit_probability')} (if they shoot)")
                    else:
                        desc_parts.append(f"safe_margin={threat.get('safety_distance_margin')}")
                lines.append(pad("  • " + "; ".join(str(p) for p in desc_parts if p)))
        else:
            lines.append(pad(f"**⚠ Threats (within {cfg.nearby_enemy_distance}):** none"))

        lines.append("║")

        # Actions (brief)
        available = self._format_available_actions(entity, actions, intel)
        lines.append(pad("**📋 Available Actions:**"))
        move_lines = self._format_move_actions(available.get("movement_options", []), available.get("can_move", False))
        for ml in move_lines:
            lines.append(pad(f"  • {ml}"))

        shoot_lines = self._format_shoot_actions(available.get("shooting_options", []))
        if shoot_lines:
            for sl in shoot_lines:
                lines.append(pad(f"  • {sl}"))

        toggle = available.get("toggle_option")
        if toggle:
            lines.append(pad(f"  • TOGGLE to {toggle.get('toggle_to')} (currently {toggle.get('current_state')})"))

        if available.get("can_wait"):
            lines.append(pad("  • WAIT"))
        if not (move_lines or shoot_lines or toggle or available.get("can_wait")):
            lines.append(pad("  • none"))

        blocked_shots = available.get("blocked_shooting_options", [])
        if blocked_shots:
            lines.append(pad("  • **Blocked Shots:**"))
            for bs in blocked_shots:
                reason = bs.get("reason")
                lines.append(pad(f"    - SHOOT #{bs.get('target_id')} (blocked: {reason or 'not ready'})"))

        # Blocked moves (informational, not offered as available)
        blocked_moves = [
            m for m in available.get("movement_options", []) if not m.get("allowed") and m.get("blocked_reason")
        ]
        if blocked_moves:
            lines.append(pad("  • **Blocked Moves:**"))
            for m in blocked_moves:
                reason = m.get("blocked_reason")
                blocker = m.get("blocked_by_id")
                blocker_type = None
                if blocker is not None:
                    blocker_ent = intel.get_friendly(blocker) or intel.get_enemy(blocker)
                    if blocker_ent:
                        blocker_type = getattr(blocker_ent.kind, "name", str(blocker_ent.kind))

                if reason == "blocked_by_enemy_immobile" and blocker:
                    lines.append(
                        pad(
                            f"    - {m['direction']} (enemy #{blocker} {blocker_type or ''} occupying; likely hard block)".rstrip()
                        )
                    )
                elif reason == "blocked_by_enemy_maybe_moves" and blocker:
                    lines.append(
                        pad(
                            f"    - {m['direction']} (enemy #{blocker} {blocker_type or ''} currently there; could vacate)".rstrip()
                        )
                    )
                elif reason == "blocked_by_friendly_immobile" and blocker:
                    lines.append(
                        pad(
                            f"    - {m['direction']} (ally #{blocker} {blocker_type or ''} immobile; hard block)".rstrip()
                        )
                    )
                elif reason == "blocked_by_friendly_maybe_moves" and blocker:
                    lines.append(
                        pad(
                            f"    - {m['direction']} (ally #{blocker} {blocker_type or ''} currently there; opens if they move away)".rstrip()
                        )
                    )
                elif reason == "out_of_bounds":
                    lines.append(pad(f"    - {m['direction']} (out of bounds)"))
                elif reason:
                    lines.append(pad(f"    - {m['direction']} ({reason})"))

        lines.append(bottom_bar())

        return lines

    def _terminology_lines(self) -> List[str]:
        """Reusable legend to keep unit sections concise but unambiguous."""
        return [
            "- **Terminology:**",
            "  - **rel=<DIR> (dx, dy, dist):** direction and offsets from this unit; x: left-/right+, y: down-/up+; dist=straight-line (grid cells).",
            "  - **Risk/threat:** DANGER=can shoot; CAUTION=almost in range; SAFE=out of range; UNARMED=cannot shoot.",
            "  - **closest_ally_dist:** straight-line distance from that enemy to the nearest friendly (grid cells).",
            "  - **Safe margin:** cells we are outside the enemy's assumed max range (armed enemies only).",
            "  - **is_decoy (aircraft only):** no = has fired before; possible = not confirmed yet.",
            "  - **our_hit / enemy_hit:** estimated hit probability if we shoot them now / if they shoot us now.",
            "  - **out_of_range_by:** how many cells we are short of our own weapon range for that target.",
            "  - **Move collisions:** cells multiple allies could move into next turn (avoid duplicate selection).",
        ]

    # ------------------------------------------------------------------ #
    # Per-unit helper calculations
    # ------------------------------------------------------------------ #
    def _get_nearby_allies(
        self,
        entity,
        intel: TeamIntel,
        cfg: StateFormatterConfig,
    ) -> List[Dict[str, Any]]:
        allies: List[Dict[str, Any]] = []
        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue
            distance = intel.grid.distance(entity.pos, other.pos)
            if distance > cfg.nearby_unit_distance:
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
                    "can_shoot": getattr(other, "can_shoot", False),
                    "missiles_remaining": getattr(other, "missiles", None),
                }
            )
        allies.sort(key=lambda a: a["relative_position"]["distance"])
        return allies

    def _get_threats(
        self,
        entity,
        intel: TeamIntel,
        cfg: StateFormatterConfig,
    ) -> Dict[str, Any]:
        detected: List[Dict[str, Any]] = []
        for enemy in intel.visible_enemies:
            distance = intel.grid.distance(entity.pos, enemy.position)
            if distance > cfg.nearby_enemy_distance:
                continue
            dx = enemy.position[0] - entity.pos[0]
            dy = enemy.position[1] - entity.pos[1]

            our_engagement = self._our_engagement(entity, enemy, intel, distance, cfg)
            their_engagement, threat_type, risk_level, safety_margin = self._their_engagement(
                enemy, distance, cfg
            )
            detected.append(
                {
                    "enemy_id": enemy.id,
                    "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                    "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                    "relative_position": {
                        "relative_to_unit": entity.id,
                        "dx": dx,
                        "dy": dy,
                        "distance": round(distance, 1),
                        "direction": self._get_cardinal_direction(dx, dy),
                    },
                    "has_fired_before": enemy.has_fired_before,
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
            "radius_checked": cfg.nearby_enemy_distance,
            "detected_enemies": detected,
        }

    def _our_engagement(
        self,
        entity,
        enemy: VisibleEnemy,
        intel: TeamIntel,
        distance: float,
        cfg: StateFormatterConfig,
    ) -> Optional[Dict[str, Any]]:
        shoot_state = self._shoot_state(entity)
        if not shoot_state["can_shoot_now"]:
            return None

        max_range = getattr(entity, "missile_max_range", None)
        in_range = max_range is not None and distance <= max_range
        hit_prob = None
        if in_range:
            hit_prob = intel.estimate_hit_probability(entity, enemy)
        return {
            "we_can_shoot": True,
            "in_our_range": in_range,
            "our_hit_probability": round(hit_prob, 2) if isinstance(hit_prob, float) else None,
            "out_of_our_range_by": round(distance - max_range, 1) if max_range and not in_range else None,
        }

    def _their_engagement(
        self,
        enemy: VisibleEnemy,
        distance: float,
        cfg: StateFormatterConfig,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[float]]:
        profile = self._get_enemy_profile(enemy.kind, cfg)
        if profile.max_range <= 0:
            return None, "UNARMED", "SAFE", None

        we_are_in_range = distance <= profile.max_range and profile.max_range > 0
        estimated = None
        if we_are_in_range:
            estimated = hit_probability(
                distance=distance,
                max_range=profile.max_range,
                base=profile.base_hit_prob,
                min_p=profile.min_hit_prob,
            )
        else:
            estimated = 0.0

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

    def _shoot_state(self, entity) -> Dict[str, Any]:
        cooldown = getattr(entity, "_cooldown", 0)
        missiles = getattr(entity, "missiles", None)

        state: Dict[str, Any] = {
            "can_shoot_now": bool(getattr(entity, "can_shoot", False)),
            "status": "READY",
            "reason": None,
            "cooldown_remaining": cooldown if isinstance(cooldown, int) else 0,
        }

        if not state["can_shoot_now"]:
            state.update({"status": "UNARMED"})
            return state

        if cooldown and cooldown > 0:
            state.update(
                {"can_shoot_now": False, "status": "COOLING_DOWN", "reason": f"Cooling down {cooldown} turn(s)"}
            )
            return state

        if missiles is not None and missiles <= 0:
            state.update({"can_shoot_now": False, "status": "OUT_OF_MISSILES", "reason": "No missiles remaining"})
            return state

        return state

    def _get_enemy_profile(self, kind: EntityKind, cfg: StateFormatterConfig) -> WeaponProfile:
        return cfg.enemy_weapon_profiles.get(kind, cfg.fallback_enemy_weapon_profile)

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

    def _calculate_destination(self, current_pos: Tuple[int, int], direction: Any) -> Tuple[int, int]:
        delta = (0, 0)
        if isinstance(direction, MoveDir):
            delta = direction.delta
        else:
            direction_map = {"UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            delta = direction_map.get(str(direction).upper(), (0, 0))
        return current_pos[0] + delta[0], current_pos[1] + delta[1]

    def _move_block_reason(
        self,
        destination: Tuple[int, int],
        mover,
        intel: TeamIntel,
    ) -> Tuple[Optional[str], Optional[int]]:
        if not intel.grid.in_bounds(destination):
            return "out_of_bounds", None

        for friendly in intel.friendlies:
            if friendly.id == mover.id or not friendly.alive:
                continue
            if friendly.pos == destination:
                if getattr(friendly, "can_move", False):
                    return "blocked_by_friendly_maybe_moves", friendly.id
                return "blocked_by_friendly_immobile", friendly.id

        for enemy in intel.visible_enemies:
            if enemy.position == destination:
                can_move = getattr(enemy, "can_move", True)
                if can_move:
                    return "blocked_by_enemy_maybe_moves", enemy.id
                return "blocked_by_enemy_immobile", enemy.id

        return None, None

    def _format_move_actions(self, move_options: List[Dict[str, Any]], can_move: bool) -> List[str]:
        if not can_move:
            return []
        allowed = [m["direction"] for m in move_options if m.get("allowed")]
        if not allowed:
            return []
        return ["MOVE: " + "/".join(allowed)]

    def _format_shoot_actions(self, shooting_options: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        for shot in shooting_options:
            target_id = shot.get("target_id")
            hit = shot.get("hit_probability")
            if hit is not None:
                lines.append(f"SHOOT #{target_id} (hit≈{hit})")
            else:
                lines.append(f"SHOOT #{target_id}")
        return lines

    def _distant_enemies(
        self,
        intel: TeamIntel,
        cfg: StateFormatterConfig,
    ) -> List[Dict[str, Any]]:
        distant: List[Dict[str, Any]] = []
        friendly_positions = [e.pos for e in intel.friendlies if e.alive]
        if not friendly_positions:
            return distant
        for enemy in intel.visible_enemies:
            min_dist = min(intel.grid.distance(pos, enemy.position) for pos in friendly_positions)
            if min_dist <= cfg.nearby_enemy_distance:
                continue
            their_engagement, threat_type, risk_level, safety_margin = self._their_engagement(
                enemy, min_dist, cfg
            )
            distant.append(
                {
                    "enemy_id": enemy.id,
                    "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                    "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                    "position": {"x": enemy.position[0], "y": enemy.position[1]},
                    "nearest_friendly_distance": round(min_dist, 1),
                    "threat_type": threat_type,
                    "risk_level": risk_level,
                    "safety_distance_margin": safety_margin,
                    "their_engagement": their_engagement,
                }
            )
        return distant

    def _relative_position(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        parts: List[str] = []
        if dx > 0:
            parts.append(f"{dx} right")
        elif dx < 0:
            parts.append(f"{abs(dx)} left")
        if dy > 0:
            parts.append(f"{dy} up")
        elif dy < 0:
            parts.append(f"{abs(dy)} down")
        return ", ".join(parts) if parts else "same position"

    def _collect_move_conflicts(
        self,
        allowed_actions: Dict[int, List[Action]],
        intel: TeamIntel,
    ) -> Dict[Tuple[int, int], List[Tuple[int, str, str]]]:
        """
        Build a map of destination -> list[(entity_id, type, direction)] for all ally MOVE actions.

        This flags cells that multiple friendlies could move into (potential collision risk).
        """
        conflicts: Dict[Tuple[int, int], List[Tuple[int, str, str]]] = {}
        for entity_id, actions in allowed_actions.items():
            friendly = intel.get_friendly(entity_id)
            if friendly is None or not friendly.alive:
                continue
            for action in actions:
                if action.type != ActionType.MOVE:
                    continue
                direction: MoveDir = action.params.get("dir")  # type: ignore[assignment]
                dest = self._calculate_destination(friendly.pos, direction)
                ent_type = getattr(friendly.kind, "name", str(friendly.kind))
                conflicts.setdefault(dest, []).append((entity_id, ent_type, direction.name))
        # Keep only contested destinations (2+)
        return {dest: tuples for dest, tuples in conflicts.items() if len(tuples) > 1}

    def _format_collision_section(
        self,
        move_conflicts: Dict[Tuple[int, int], List[Tuple[int, str, str]]],
    ) -> List[str]:
        lines: List[str] = []
        for dest, ids in sorted(move_conflicts.items()):
            details = []
            for eid, etype, direction in ids:
                details.append(f"#{eid} {etype} via {direction}")
            lines.append(
                f"- Cell ({dest[0]}, {dest[1]}): " + "; ".join(details) + " -> collision risk (same destination)"
            )
        return lines

    def _structured_move_conflicts(
        self, move_conflicts: Dict[Tuple[int, int], List[Tuple[int, str, str]]]
    ) -> List[Dict[str, Any]]:
        """Structured version of potential move collisions."""
        structured: List[Dict[str, Any]] = []
        for dest, ids in sorted(move_conflicts.items()):
            structured.append(
                {
                    "destination": {"x": dest[0], "y": dest[1]},
                    "contenders": [
                        {"entity_id": eid, "type": etype, "direction": direction} for eid, etype, direction in ids
                    ],
                }
            )
        return structured
