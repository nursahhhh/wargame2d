"""
Simple AI Hook System for Grid Air Combat
-----------------------------------------
Provides a clean interface for AI to observe game state and return actions.
Output is simple JSON-like dicts for easy LLM integration.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

from pydantic_ai import Agent

from wargame_2d.agent import TacticalPlan, create_llm_prompt, tactical_plan_to_action_dicts
from wargame_2d.agents.historian import analyst_agent
from wargame_2d.agents.state_and_data import COMBAT_GAME_DEPS
from wargame_2d.agents.strategy_director import strategic_director
from wargame_2d.agents.executer import executer_agent
from wargame_2d.env import World, Team, Entity, Action, ActionType, MoveDir, Shooter, SAM
from wargame_2d.controllers import ControllerRegistry, ControllerType, get_controller_actions
import random
import json
import set_api_key



@dataclass
class GameStateConfig:
    """Configuration for game state perception and threat assessment"""

    # Proximity thresholds
    nearby_unit_distance: float = 5.0  # Distance to consider units as "nearby"
    grouped_enemy_distance: float = 5.0  # Distance to consider enemies as "grouped"

    # Threat assessment distances
    very_close_threat_distance: float = 3.0  # Distance for "VERY CLOSE" threat
    close_threat_distance: float = 5.0  # Distance for "CLOSE" threat

    # Weapon range assumptions (for threat assessment when exact stats unknown)
    assumed_aircraft_range: float = 3.0  # Assumed max range for enemy aircraft
    assumed_sam_range: float = 4.0  # Assumed max range for enemy SAMs

    # Display options
    show_enemy_intel: bool = False  # Show exact enemy missile counts and SAM status
    show_dead_entities: bool = True  # ← ADD THIS LINE

    # Hit probability defaults (for calculations when exact stats unavailable)
    default_base_hit_prob: float = 0.8
    default_min_hit_prob: float = 0.1


# Global default config (can be overridden)
DEFAULT_CONFIG = GameStateConfig()


def get_game_state(world: World, team: Team, config: GameStateConfig = None, log_history_turns: int = 3) -> Dict[
    str, Any]:
    """
    Get complete observable game state for a team in LLM-friendly format.

    Args:
        world: Game world
        team: Team to get state for
        config: Configuration for perception and threat assessment (uses DEFAULT_CONFIG if None)
        log_history_turns: Number of recent turns to include in history (default: 3)
    """
    if config is None:
        config = DEFAULT_CONFIG

    cc = world.command_center(team)

    # Initialize historical observations tracking if not exists
    if not hasattr(cc, '_historical_observations'):
        cc._historical_observations = {}

    # Friendly units with complete tactical information
    friendly_units = []
    for entity in world.entities:
        if entity.team == team and entity.alive:
            unit_info = {
                "id": entity.id,
                "type": entity.kind,
                "position": {"x": entity.pos[0], "y": entity.pos[1]},
                "capabilities": {
                    "can_move": entity.can_move,
                    "can_shoot": entity.can_shoot,
                    "radar_range": entity.radar_range,
                },
                "nearby": _get_nearby_units(world, entity, team, config),
            }

            # Add shooter-specific info
            if hasattr(entity, 'missiles'):
                unit_info["weapons"] = {
                    "missiles_remaining": entity.missiles,
                    "missile_max_range": entity.missile_max_range,
                }

            # Add SAM-specific info
            if hasattr(entity, 'on'):
                unit_info["sam_status"] = {
                    "is_on": entity.on,
                    "cooldown_remaining": entity._cooldown,
                    "cooldown_duration": entity.cooldown_steps,
                    "ready_to_fire": entity.on and entity._cooldown == 0,
                }

            # Calculate and add available actions for this unit
            unit_info["available_actions"] = _get_available_actions_for_unit(world, entity, team)

            friendly_units.append(unit_info)

    # Currently visible enemy units
    visible_enemy_ids = set()
    enemy_units = []
    for obs in cc.observations.values():
        if obs.team != team:
            visible_enemy_ids.add(obs.entity_id)

            # Update historical observations with current sighting
            cc._historical_observations[obs.entity_id] = {
                "type": obs.kind,
                "position": {"x": obs.position[0], "y": obs.position[1]},
            }

            enemy_info = {
                "id": obs.entity_id,
                "type": obs.kind,  # This is the OBSERVED type (decoys appear as 'aircraft')
                "position": {"x": obs.position[0], "y": obs.position[1]},
                "distance_from_nearest_friendly": round(obs.distance, 2),
                "detected_by": list(obs.seen_by),
                "nearby": _get_nearby_units_for_observation(world, obs, team, config),
            }

            # Don't add threat_assessment here - will be done during formatting
            # (Or add the new one if you want it in the game_state dict)

            # Always show if entity has fired (helps identify non-decoys)
            actual_entity = world.entities_by_id.get(obs.entity_id)
            if actual_entity:
                missile_intel = _estimate_missiles_fired(actual_entity)

                # Check command center's firing history
                has_fired_before = cc.has_enemy_fired(obs.entity_id)

                # If entity has missiles or has fired before, include intel
                if missile_intel["has_missiles"] or has_fired_before:
                    enemy_info["missile_intel"] = {
                        "has_ever_fired": has_fired_before or missile_intel["has_ever_fired"],
                        "definitely_not_decoy": has_fired_before or missile_intel["has_ever_fired"],
                    }

                    # Add detailed intel if config allows
                    if config.show_enemy_intel:
                        enemy_info["missile_intel"]["missiles_remaining"] = missile_intel["missiles_remaining"]
                        enemy_info["missile_intel"]["recently_fired"] = missile_intel["recently_fired"]

                        if missile_intel["is_sam"]:
                            enemy_info["missile_intel"]["sam_on"] = missile_intel["sam_on"]
                            enemy_info["missile_intel"]["sam_cooldown"] = missile_intel["sam_cooldown"]

            enemy_units.append(enemy_info)

    # Clean up historical observations for dead enemies FIRST (before building last_known_enemies)
    dead_enemy_ids = [
        entity_id for entity_id in cc._historical_observations.keys()
        if not (actual_entity := world.entities_by_id.get(entity_id)) or not actual_entity.alive
    ]

    for dead_id in dead_enemy_ids:
        del cc._historical_observations[dead_id]

    # THEN build last known positions of enemies we've seen before but can't see now
    # (This now automatically excludes dead enemies since they were just removed)
    last_known_enemies = []
    for entity_id, last_obs in cc._historical_observations.items():
        # Only include if:
        # 1. Not currently visible
        # 2. Entity is on enemy team (already verified by cleanup above)
        if entity_id not in visible_enemy_ids:
            actual_entity = world.entities_by_id.get(entity_id)
            # This check is now redundant (cleanup ensures entity is alive) but kept for safety
            if actual_entity and actual_entity.team != team:
                last_known_enemies.append({
                    "id": entity_id,
                    "type": last_obs["type"],
                    "last_seen_position": last_obs["position"],
                })

    # Tactical situation summary
    situation = {
        "friendly_count": len(friendly_units),
        "enemy_count": len(enemy_units),
        "last_known_enemy_count": len(last_known_enemies),
        "friendly_shooters": sum(1 for u in friendly_units if u["capabilities"]["can_shoot"]),
        "enemy_shooters_visible": sum(1 for u in enemy_units if u["type"] in ["aircraft", "sam"]),
        "radar_coverage": _calculate_radar_coverage(world, team),
    }

    # Get observable logs from last N turns
    log_history = world.get_team_observable_logs(team, num_turns=log_history_turns)


    # Collect dead entities if configured
    dead_entities = []
    if config.show_dead_entities:
        for entity in world.entities:
            if not entity.alive:
                dead_info = {
                    "id": entity.id,
                    "team": entity.team.name,
                    "type": entity.kind,
                    "death_position": {"x": entity.pos[0], "y": entity.pos[1]},
                }
                dead_entities.append(dead_info)

    return {
        "team": team.name,
        "turn_summary": situation,
        "friendly_units": friendly_units,
        "enemy_units": enemy_units,
        "last_known_enemies": last_known_enemies,
        "dead_entities": dead_entities,
        "turn_history": log_history,
        "battlefield": {
            "width": world.width,
            "height": world.height,
            "center": {"x": world.width // 2, "y": world.height // 2}
        }
    }

def _get_relative_position(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    """
    Get relative position description in terms of movement directions.
    Returns a string like "2 right, 3 up" or "at same location"
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]

    if dx == 0 and dy == 0:
        return "same location"

    parts = []

    # Horizontal
    if dx > 0:
        parts.append(f"{dx} right")
    elif dx < 0:
        parts.append(f"{abs(dx)} left")

    # Vertical
    if dy > 0:
        parts.append(f"{dy} down")
    elif dy < 0:
        parts.append(f"{abs(dy)} up")

    return ", ".join(parts)


def _get_nearby_units(world: World, entity: Entity, team: Team, config: GameStateConfig) -> Dict[str, Any]:
    """
    Get nearby friendly and enemy units for spatial awareness.
    Uses team-wide radar coverage, not just the entity's own radar.
    """
    nearby = {
        "close_friendlies": [],
        "visible_enemies": [],
    }

    cc = world.command_center(team)

    # Close friendlies (within configured distance, regardless of this entity's radar)
    for other in world.entities:
        if other.team == team and other.alive and other.id != entity.id:
            dist = world.distance(entity.pos, other.pos)
            if dist <= config.nearby_unit_distance:
                relative_pos = _get_relative_position(entity.pos, other.pos)
                nearby["close_friendlies"].append({
                    "id": other.id,
                    "type": other.kind,
                    "distance": round(dist, 1),
                    "relative_position": relative_pos,
                })

    # Visible enemies (detected by ANY team radar and within configured distance of this entity)
    for obs in cc.observations.values():
        if obs.team != team:
            dist = world.distance(entity.pos, obs.position)
            if dist <= config.nearby_unit_distance:
                relative_pos = _get_relative_position(entity.pos, obs.position)
                nearby["visible_enemies"].append({
                    "id": obs.entity_id,
                    "type": obs.kind,  # Observed type (decoys show as aircraft)
                    "distance": round(dist, 1),
                    "relative_position": relative_pos,
                })

    # Sort by distance (closest first)
    nearby["close_friendlies"].sort(key=lambda x: x["distance"])
    nearby["visible_enemies"].sort(key=lambda x: x["distance"])

    return nearby


def _get_nearby_units_for_observation(world: World, obs, team: Team, config: GameStateConfig) -> Dict[str, Any]:
    """
    Get nearby units for an observed enemy (from our perspective).
    """
    nearby = {
        "close_to_our_units": [],
        "near_other_enemies": [],
    }

    # Close to our units (within configured distance)
    for friendly in world.entities:
        if friendly.team == team and friendly.alive:
            dist = world.distance(obs.position, friendly.pos)
            if dist <= config.nearby_unit_distance:
                relative_pos = _get_relative_position(obs.position, friendly.pos)
                nearby["close_to_our_units"].append({
                    "id": friendly.id,
                    "type": friendly.kind,
                    "distance": round(dist, 1),
                    "relative_position": relative_pos,
                })

    # Near other visible enemies (within configured distance)
    cc = world.command_center(team)
    for other_obs in cc.observations.values():
        if other_obs.team != team and other_obs.entity_id != obs.entity_id:
            dist = world.distance(obs.position, other_obs.position)
            if dist <= config.grouped_enemy_distance:
                relative_pos = _get_relative_position(obs.position, other_obs.position)
                nearby["near_other_enemies"].append({
                    "id": other_obs.entity_id,
                    "type": other_obs.kind,
                    "distance": round(dist, 1),
                    "relative_position": relative_pos,
                })

    # Sort by distance (closest first)
    nearby["close_to_our_units"].sort(key=lambda x: x["distance"])
    nearby["near_other_enemies"].sort(key=lambda x: x["distance"])

    return nearby


def _estimate_missiles_fired(entity) -> Dict[str, Any]:
    """
    Get missile firing information for an enemy entity.
    We assume teams can detect and track missile launches (observable intel).
    Returns missile count and recent firing activity.
    For SAMs, also includes cooldown status.
    """
    intel = {
        "has_missiles": False,
        "missiles_remaining": None,
        "recently_fired": False,
        "has_ever_fired": False,  # NEW: Track if entity has EVER fired
        "is_sam": False,
        "sam_on": None,
        "sam_cooldown": None,
    }

    # Check if entity has missile capability
    if hasattr(entity, 'missiles'):
        intel["has_missiles"] = True
        intel["missiles_remaining"] = entity.missiles

        # Check if recently fired (this turn)
        if hasattr(entity, 'last_action') and entity.last_action:
            if entity.last_action.type == ActionType.SHOOT:
                intel["recently_fired"] = True

        # NEW: Track if entity has EVER fired by checking missile count vs max
        # If current missiles < starting missiles, it has fired before
        if hasattr(entity, 'missiles'):
            starting_missiles = 2  # Default for aircraft
            if isinstance(entity, SAM):
                starting_missiles = 4
            elif hasattr(entity, '__class__') and 'Aircraft' in entity.__class__.__name__:
                starting_missiles = 2

            if entity.missiles < starting_missiles:
                intel["has_ever_fired"] = True

    # SAM-specific intel (observable: on/off state and cooldown)
    if isinstance(entity, SAM):
        intel["is_sam"] = True
        intel["sam_on"] = entity.on
        intel["sam_cooldown"] = entity._cooldown

    return intel


# REPLACE the _get_available_actions_for_unit function in ai_controller.py
# This fixes the decoy leakage in blocked move descriptions

def _get_available_actions_for_unit(world: World, entity: Entity, team: Team) -> Dict[str, Any]:
    """
    Get all available actions for a specific unit with detailed information.
    Uses observable entity types (decoys appear as aircraft to enemies).
    """
    actions = {
        "can_wait": True,
        "movement_options": [],
        "shooting_options": [],
        "toggle_option": None,
    }

    # Movement options
    if entity.can_move:
        for direction in [MoveDir.UP, MoveDir.DOWN, MoveDir.LEFT, MoveDir.RIGHT]:
            dx, dy = direction.delta
            new_pos = (entity.pos[0] + dx, entity.pos[1] + dy)

            move_option = {
                "direction": direction.name,
                "destination": {"x": new_pos[0], "y": new_pos[1]},
                "valid": False,
                "blocked_reason": None,
            }

            # Check if out of bounds
            if not world.in_bounds(new_pos):
                move_option["blocked_reason"] = "out_of_bounds"
                move_option["valid"] = False
            # Check if occupied by another unit
            elif world.is_occupied(new_pos):
                # Find what's blocking
                blocking_entity = None
                for e in world.entities:
                    if e.alive and e.pos == new_pos:
                        blocking_entity = e
                        break

                if blocking_entity:
                    if blocking_entity.team == team:
                        # Friendly blocking - show actual type
                        move_option["blocked_reason"] = f"blocked_by_friendly_{blocking_entity.kind}"
                        move_option["blocked_entity_id"] = blocking_entity.id
                    else:
                        # Enemy blocking - show OBSERVED type (decoys appear as aircraft)
                        cc = world.command_center(team)
                        obs = cc.observations.get(blocking_entity.id)

                        if obs:
                            # Use observed kind (decoys appear as aircraft to enemies)
                            observed_kind = obs.kind
                            move_option["blocked_reason"] = f"blocked_by_enemy_{observed_kind}"
                            move_option["blocked_entity_id"] = blocking_entity.id
                        else:
                            # Enemy not observable (shouldn't happen if path is blocked, but handle it)
                            move_option["blocked_reason"] = "blocked_by_enemy_unknown"
                            move_option["blocked_entity_id"] = blocking_entity.id
                else:
                    move_option["blocked_reason"] = "blocked_by_unknown"
                move_option["valid"] = False
            else:
                # Path is clear
                move_option["valid"] = True

            actions["movement_options"].append(move_option)

    # Shooting options
    if entity.can_shoot and isinstance(entity, Shooter) and entity.missiles > 0:
        # Check if SAM and handle special restrictions
        can_shoot_now = True
        if isinstance(entity, SAM):
            can_shoot_now = entity.on and entity._cooldown == 0

        if can_shoot_now:
            cc = world.command_center(team)
            for target_id in cc.visible_enemy_ids:
                target = world.entities_by_id.get(target_id)
                if target and target.alive:
                    distance = world.distance(entity.pos, target.pos)

                    if distance <= entity.missile_max_range:
                        hit_prob = world.hit_probability(
                            distance=distance,
                            max_range=entity.missile_max_range,
                            base=entity.base_hit_prob,
                            min_p=entity.min_hit_prob
                        )

                        relative_pos = _get_relative_position(entity.pos, target.pos)

                        # Use OBSERVED type for target (decoys appear as aircraft)
                        obs = cc.observations.get(target_id)
                        observed_type = obs.kind if obs else target.kind

                        actions["shooting_options"].append({
                            "target_id": target_id,
                            "target_type": observed_type,  # Use observed type
                            "target_position": {"x": target.pos[0], "y": target.pos[1]},
                            "distance": round(distance, 2),
                            "relative_position": relative_pos,
                            "hit_probability": round(hit_prob, 3),
                            "hit_probability_percent": round(hit_prob * 100, 1),
                            "in_range": True,
                        })

    # Toggle option for SAMs
    if isinstance(entity, SAM):
        actions["toggle_option"] = {
            "current_state": "ON" if entity.on else "OFF",
            "can_toggle_to": "OFF" if entity.on else "ON",
            "description": f"Turn SAM {'OFF' if entity.on else 'ON'}",
        }

    return actions


def _assess_threat_detailed(world: World, obs, friendly_unit, team: Team, config: GameStateConfig) -> Dict[str, Any]:
    """
    Comprehensive threat assessment using ONLY observable information.

    Args:
        world: Game world (for distance calculations)
        obs: Observation of enemy unit
        friendly_unit: The friendly unit dict assessing this threat (for tactical context)
        team: Our team
        config: Configuration with assumed enemy capabilities
    """
    threat = {
        "level": "UNKNOWN",
        "proximity": "DISTANT",
        "can_we_shoot": False,
        "can_they_shoot": False,
        "our_engagement": {},
        "their_threat": {},
        "tactical_summary": ""
    }

    # ✅ FIX: Calculate distance from THIS unit to the enemy
    if friendly_unit:
        friendly_pos = (friendly_unit['position']['x'], friendly_unit['position']['y'])
        enemy_pos = obs.position
        distance = world.distance(friendly_pos, enemy_pos)
    else:
        # Fallback to observation distance if no unit context
        distance = obs.distance

    # === PROXIMITY ASSESSMENT ===
    if distance < config.very_close_threat_distance:
        threat["proximity"] = "VERY CLOSE"
    elif distance < config.close_threat_distance:
        threat["proximity"] = "CLOSE"
    else:
        threat["proximity"] = "DISTANT"

    # === ENEMY TYPE CLASSIFICATION ===
    enemy_can_shoot = obs.kind in ["aircraft", "sam"]

    if enemy_can_shoot:
        threat["level"] = "HIGH"
    elif obs.kind == "awacs":
        threat["level"] = "MEDIUM"
    else:  # decoy
        threat["level"] = "LOW"

    # === OUR OFFENSIVE CAPABILITY (using OUR actual stats) ===
    if friendly_unit and 'weapons' in friendly_unit:
        our_range = friendly_unit['weapons']['missile_max_range']
        our_missiles = friendly_unit['weapons']['missiles_remaining']

        if our_missiles > 0:
            if distance <= our_range:
                # We CAN shoot - calculate hit probability
                threat["can_we_shoot"] = True

                base_prob = config.default_base_hit_prob
                min_prob = config.default_min_hit_prob
                frac = max(0.0, min(1.0, 1.0 - (distance / our_range)))
                hit_prob = max(min_prob, base_prob * frac)

                threat["our_engagement"] = {
                    "in_range": True,
                    "distance": round(distance, 1),
                    "max_range": our_range,
                    "hit_probability": round(hit_prob, 3),
                    "hit_percent": round(hit_prob * 100, 1),
                    "missiles_available": our_missiles
                }
            else:
                # Out of range - show how far to close
                threat["can_we_shoot"] = False
                threat["our_engagement"] = {
                    "in_range": False,
                    "distance": round(distance, 1),
                    "max_range": our_range,
                    "need_to_close": round(distance - our_range, 1),
                    "missiles_available": our_missiles
                }
        else:
            # No ammo
            threat["our_engagement"] = {
                "in_range": False,
                "no_ammo": True
            }

    # === ENEMY THREAT TO US (using ASSUMED enemy capabilities) ===
    if enemy_can_shoot:
        # Use assumed ranges based on observed type
        assumed_range = config.assumed_sam_range if obs.kind == "sam" else config.assumed_aircraft_range

        if distance <= assumed_range:
            # We are IN their assumed range - DANGER!
            threat["can_they_shoot"] = True

            # Estimate their hit probability using assumptions
            base_prob = config.default_base_hit_prob
            min_prob = config.default_min_hit_prob
            frac = max(0.0, min(1.0, 1.0 - (distance / assumed_range)))
            estimated_hit_prob = max(min_prob, base_prob * frac)

            threat["their_threat"] = {
                "in_their_range": True,
                "distance": round(distance, 1),
                "assumed_range": assumed_range,
                "estimated_hit_probability": round(estimated_hit_prob, 3),
                "estimated_hit_percent": round(estimated_hit_prob * 100, 1),
                "warning": "DANGER - In enemy firing range"
            }
        else:
            # We are SAFE (outside assumed range)
            threat["can_they_shoot"] = False
            threat["their_threat"] = {
                "in_their_range": False,
                "distance": round(distance, 1),
                "assumed_range": assumed_range,
                "safe_margin": round(distance - assumed_range, 1),
                "warning": f"SAFE - Keep distance >{assumed_range}"
            }
    else:
        # Enemy can't shoot (AWACS/Decoy)
        threat["their_threat"] = {
            "in_their_range": False,
            "no_weapons": True,
            "warning": "Enemy cannot shoot"
        }

    # === TACTICAL SUMMARY (one-liner for quick assessment) ===
    summary_parts = []

    if threat["can_they_shoot"]:
        summary_parts.append(f"⚠️ DANGER: In enemy range")
    else:
        summary_parts.append(f"✓ Safe from enemy fire")

    if threat["can_we_shoot"]:
        summary_parts.append(f"Can engage ({threat['our_engagement']['hit_percent']:.0f}% hit)")
    elif threat.get("our_engagement", {}).get("need_to_close"):
        summary_parts.append(f"Out of range (close {threat['our_engagement']['need_to_close']:.1f} cells)")

    threat["tactical_summary"] = " | ".join(summary_parts)

    return threat
def _format_threat_display(enemy_info: Dict, threat: Dict, relative_position: str = None, distance_override: float = None,
                           is_sam: bool = False, sam_is_off: bool = False) -> List[str]:
    """
    Format threat information into clean, readable lines.
    Returns list of formatted strings.

    Args:
        enemy_info: Dictionary with enemy information
        threat: Dictionary with threat assessment
        relative_position: Optional relative position string (e.g., "2 right, 3 up")
        distance_override: Optional distance to display (overrides distance_from_nearest_friendly)
                          Use this when showing threat from a specific unit's perspective
        is_sam: Whether the unit viewing this threat is a SAM
        sam_is_off: Whether the SAM is currently OFF (only relevant if is_sam=True)
    """
    lines = []

    # Main threat header with relative position
    threat_icon = "🔴" if threat['level'] == "HIGH" else "🟡" if threat['level'] == "MEDIUM" else "🟢"

    # Add relative position if available
    relative_pos_str = ""
    if relative_position:
        relative_pos_str = f" → {relative_position}"

    # Use override distance if provided (for unit-specific displays), otherwise use observation distance
    display_distance = distance_override if distance_override is not None else enemy_info[
        'distance_from_nearest_friendly']

    lines.append(
        f"║   {threat_icon} Enemy #{enemy_info['id']} ({enemy_info['type'].upper()}) "
        f"at (x={enemy_info['position']['x']}, y={enemy_info['position']['y']}){relative_pos_str} "
        f"- {threat['proximity']} [dist={display_distance:.1f}]"
    )

    # Tactical summary (one-liner) - modified for SAM OFF state
    if is_sam and sam_is_off:
        # Override tactical summary for OFF SAM
        lines.append(f"║      ✓ You are SAFE (SAM OFF - enemy cannot detect/shoot you)")
    else:
        lines.append(f"║      {threat['tactical_summary']}")

    # === OUR OFFENSIVE CAPABILITY ===
    if threat.get('our_engagement'):
        engagement = threat['our_engagement']

        if engagement.get('no_ammo'):
            shot_prefix = "Our Shot (if toggle ON)" if is_sam and sam_is_off else "Our Shot"
            lines.append(f"║      • {shot_prefix}: NO AMMO")
        elif engagement.get('in_range'):
            # Add conditional prefix for SAM OFF
            shot_prefix = "Our Shot (if toggle ON)" if is_sam and sam_is_off else "Our Shot"
            lines.append(
                f"║      • {shot_prefix}: IN RANGE - "
                f"{engagement['hit_percent']:.1f}% hit @ {engagement['distance']:.1f} cells "
                f"(max range: {engagement['max_range']:.1f})"
            )
        else:
            shot_prefix = "Our Shot (if toggle ON)" if is_sam and sam_is_off else "Our Shot"
            lines.append(
                f"║      • {shot_prefix}: OUT OF RANGE - "
                f"@ {engagement['distance']:.1f} cells, can shoot ≤{engagement['max_range']:.1f} "
                f"(close {engagement['need_to_close']:.1f} more)"
            )

    # === ENEMY THREAT TO US ===
    if threat.get('their_threat'):
        enemy_threat = threat['their_threat']

        if enemy_threat.get('no_weapons'):
            lines.append(f"║      • Their Shot: UNARMED ({enemy_info['type']} cannot shoot)")
        elif enemy_threat.get('in_their_range'):
            lines.append(
                f"║      • Their Shot: ⚠️ IN RANGE - "
                f"~{enemy_threat['estimated_hit_percent']:.1f}% estimated hit "
                f"(assumed range: {enemy_threat['assumed_range']:.1f})"
            )
        else:
            lines.append(
                f"║      • Their Shot: SAFE - "
                f"@ {enemy_threat['distance']:.1f} cells, they shoot ≤{enemy_threat['assumed_range']:.1f} "
                f"(safe margin: +{enemy_threat['safe_margin']:.1f})"
            )

    # === ADDITIONAL INTEL ===
    # Show firing history if available
    if enemy_info.get('missile_intel', {}).get('has_ever_fired'):
        lines.append(f"║      • Intel: ⚠️ HAS FIRED - Confirmed threat, NOT a decoy")

    # Show grouping
    if enemy_info.get('nearby', {}).get('near_other_enemies'):
        group_ids = [str(u['id']) for u in enemy_info['nearby']['near_other_enemies']]
        lines.append(f"║      • Grouped with: Enemy #{', #'.join(group_ids)}")

    return lines


def _calculate_radar_coverage(world: World, team: Team) -> Dict[str, Any]:
    """
    Calculate radar coverage statistics for tactical awareness.
    """
    total_cells = world.width * world.height
    covered_cells = set()

    for entity in world.entities:
        if entity.team == team and entity.alive and entity.radar_range > 0:
            # SAM radar only counts when ON
            if isinstance(entity, SAM) and not entity.on:
                continue

            # Calculate covered cells (approximate)
            ex, ey = entity.pos
            r = int(entity.radar_range)
            for y in range(max(0, ey - r), min(world.height, ey + r + 1)):
                for x in range(max(0, ex - r), min(world.width, ex + r + 1)):
                    dist = ((x - ex) ** 2 + (y - ey) ** 2) ** 0.5
                    if dist <= entity.radar_range:
                        covered_cells.add((x, y))

    coverage_percent = (len(covered_cells) / total_cells) * 100

    return {
        "covered_cells": len(covered_cells),
        "total_cells": total_cells,
        "coverage_percent": round(coverage_percent, 1),
    }


def _extract_entity_id_from_log(log: str) -> int:
    """Extract entity ID from log message for sorting."""
    import re
    match = re.search(r'#(\d+)', log)
    return int(match.group(1)) if match else 999999

def _format_turn_history_for_display(turn_history: List[List[str]]) -> str:
    """
    Format turn history with clear separation of ally vs enemy actions.

    Args:
        turn_history: List of turns, each containing log messages

    Returns:
        Formatted history string
    """
    if not turn_history:
        return ""

    lines = []
    lines.append("")
    lines.append("RECENT TURN HISTORY (Observable Actions):")
    lines.append("=" * 80)

    total_turns = len(turn_history)
    for turn_idx, turn_logs in enumerate(turn_history):
        turns_ago = total_turns - turn_idx - 1

        if turns_ago == 0:
            turn_label = "LAST TURN"
        elif turns_ago == 1:
            turn_label = "2 TURNS AGO"
        elif turns_ago == 2:
            turn_label = "3 TURNS AGO"
        else:
            turn_label = f"{turns_ago + 1} TURNS AGO"

        lines.append(f"\n{turn_label}:")
        lines.append("-" * 80)

        if not turn_logs:
            lines.append("  • No observable actions this turn")
            continue

        # Separate ally and enemy actions
        ally_actions = []
        enemy_actions = []

        for log in turn_logs:
            # Remove prefix for display
            clean_log = log.replace("[ALLY] ", "").replace("[ENEMY] ", "")

            if "[ALLY]" in log:
                ally_actions.append(clean_log)
            elif "[ENEMY]" in log:
                enemy_actions.append(clean_log)

        # Sort by entity ID for consistent ordering
        ally_actions.sort(key=_extract_entity_id_from_log)
        enemy_actions.sort(key=_extract_entity_id_from_log)

        # Display ally actions first
        if ally_actions:
            lines.append("")
            lines.append("  ╔══ OUR ACTIONS ══")
            for action in ally_actions:
                lines.append(f"  ║ {action}")
            lines.append("  ╚" + "═" * 40)

        # Then enemy actions
        if enemy_actions:
            lines.append("")
            lines.append("  ╔══ ENEMY ACTIONS (Observed) ══")
            for action in enemy_actions:
                lines.append(f"  ║ {action}")
            lines.append("  ╚" + "═" * 40)

        if not ally_actions and not enemy_actions:
            lines.append("  • No observable actions this turn")

    lines.append("=" * 80)

    return "\n".join(lines)

# COMPLETE REPLACEMENT for format_game_state_for_llm in ai_controller.py
# This removes the OLD turn history code that was causing duplication

def format_game_state_for_llm(game_state: Dict[str, Any], world: World, config: GameStateConfig = None) -> tuple[str, str]:
    """
    Format the game state into a readable text format optimized for LLM comprehension.
    Consolidates tactical information around friendly units for better context.

    Args:
        game_state: The game state dictionary
        world: The game world object
        config: Configuration (uses DEFAULT_CONFIG if None)

    Returns:
        tuple[str, str]: (main_game_state, turn_history)
            - main_game_state: Current tactical situation without turn history
            - turn_history: Formatted turn history string (empty if no history)
    """
    if config is None:
        config = DEFAULT_CONFIG

    lines = []
    lines.append("=" * 80)
    lines.append(f"TACTICAL SITUATION REPORT - {game_state['team']} TEAM")
    lines.append("=" * 80)
    lines.append("")

    # Situation summary
    summary = game_state["turn_summary"]
    lines.append("SITUATION OVERVIEW:")
    lines.append(f"  Friendly Units: {summary['friendly_count']} ({summary['friendly_shooters']} with weapons)")
    lines.append(f"  Enemy Units Detected: {summary['enemy_count']} ({summary['enemy_shooters_visible']} shooters)")
    if summary.get('last_known_enemy_count', 0) > 0:
        lines.append(f"  Lost Contact With: {summary['last_known_enemy_count']} enemy unit(s)")
    # lines.append(f"  Radar Coverage: {summary['radar_coverage']['coverage_percent']}% of battlefield")
    lines.append("")

    # Turn history - Format separately but don't add to main lines
    turn_history = game_state.get('turn_history', [])
    turn_history_str = ""
    if turn_history:
        turn_history_str = _format_turn_history_for_display(turn_history)

    # Battlefield info
    bf = game_state["battlefield"]
    lines.append(
        f"BATTLEFIELD: {bf['width']}x{bf['height']} grid (center at x={bf['center']['x']}, y={bf['center']['y']})")
    lines.append("")

    # Build enemy lookup for enhanced information
    enemy_lookup = {e['id']: e for e in game_state['enemy_units']}

    # Extract team from game_state
    team_str = game_state['team']
    team = Team.BLUE if team_str == 'BLUE' else Team.RED
    cc = world.command_center(team)

    # Friendly units with tactical context
    lines.append("=" * 80)
    lines.append("TACTICAL UNIT OVERVIEW & ENGAGEMENT OPTIONS:")
    lines.append("=" * 80)

    for unit in game_state["friendly_units"]:
        lines.append(f"\n╔═══ ALLY Unit #{unit['id']} - {unit['type'].upper()} ═══")
        lines.append(f"║ Position: x={unit['position']['x']}, y={unit['position']['y']}")

        # Capabilities
        caps = unit['capabilities']
        cap_list = []
        if caps['can_move']:
            cap_list.append("Mobile")
        if caps['can_shoot']:
            cap_list.append("Armed")
        if caps['radar_range'] > 0:
            cap_list.append(f"Radar({caps['radar_range']})")
        lines.append(f"║ Capabilities: {', '.join(cap_list)}")

        # Weapons
        if 'weapons' in unit:
            w = unit['weapons']
            lines.append(f"║ Armament: {w['missiles_remaining']} missiles (max_range={w['missile_max_range']})")

        # SAM status
        if 'sam_status' in unit:
            s = unit['sam_status']

            # Determine status with detailed information
            if s['is_on']:
                if s['ready_to_fire']:
                    status = "ON & READY (You can shoot, but enemies can detect and shoot you)"
                else:
                    remaining = s['cooldown_remaining']
                    total = s['cooldown_duration']
                    status = f"ON but COOLING DOWN ({remaining}/{total} turns remaining - Cannot shoot yet, but still VISIBLE to enemies)"
            else:
                status = "OFF (✓ SAFE - You're invisible & cannot be shot, but you cannot shoot either)"

            lines.append(f"║ SAM Status: {status}")

        # Nearby friendlies
        nearby = unit['nearby']
        if nearby['close_friendlies']:
            lines.append(f"║")
            lines.append(f"║ ▶ Nearby Allies ({len(nearby['close_friendlies'])}):")
            for f in nearby['close_friendlies']:
                lines.append(
                    f"║   • Ally Unit #{f['id']} ({f['type']}) at (x={unit['position']['x'] + _parse_relative_delta(f['relative_position'])[0]}, " +
                    f"y={unit['position']['y'] + _parse_relative_delta(f['relative_position'])[1]}) " +
                    f"→ {f['relative_position']} [dist={f['distance']}]")

        # Nearby enemies with enhanced information
        if nearby['visible_enemies']:
            lines.append(f"║")

            # Check if this is a SAM unit
            is_sam = unit.get('type') == 'sam'
            sam_is_off = is_sam and unit.get('sam_status', {}).get('is_on') == False

            # Add context-aware header
            if is_sam and sam_is_off:
                lines.append(
                    f"║ ⚠ Detected Threats ({len(nearby['visible_enemies'])}) - [SAM OFF: Threats can't target you]:")
            else:
                lines.append(f"║ ⚠ Detected Threats ({len(nearby['visible_enemies'])}):")

            for e in nearby['visible_enemies']:
                enemy_full = enemy_lookup.get(e['id'])

                if enemy_full:
                    # Get observation object
                    obs = cc.observations.get(e['id'])

                    if obs:  # Make sure observation exists
                        # NEW: Use enhanced threat assessment
                        threat = _assess_threat_detailed(
                            world,  # ✅ Now available
                            obs,  # ✅ Observation object
                            unit,  # ✅ Current friendly unit
                            team,  # ✅ Now available
                            config  # ✅ Already available
                        )

                        # Format and display with SAM-specific context
                        threat_lines = _format_threat_display(
                            enemy_full,
                            threat,
                            e.get('relative_position'),
                            e.get('distance'),              # ← ADD THIS LINE!
                            is_sam=is_sam,
                            sam_is_off=sam_is_off
                        )
                        lines.extend(threat_lines)
                        lines.append("║")  # Blank line between enemies
        # Available actions
        lines.append(f"║")
        lines.append(f"║ 📋 Available Actions:")
        actions = unit['available_actions']

        # Movement - show both valid and blocked with reasons
        if actions['movement_options']:
            valid_moves = [m for m in actions['movement_options'] if m['valid']]
            blocked_moves = [m for m in actions['movement_options'] if not m['valid']]

            if valid_moves:
                move_dirs = [m['direction'] for m in valid_moves]
                lines.append(f"║   • MOVE: {', '.join(move_dirs)}")

            if blocked_moves:
                lines.append(f"║   • BLOCKED MOVES:")
                for m in blocked_moves:
                    reason = m.get('blocked_reason', 'unknown')

                    # Format reason for display
                    if reason == "out_of_bounds":
                        reason_text = "map boundary"
                    elif reason.startswith("blocked_by_friendly_"):
                        unit_type = reason.replace("blocked_by_friendly_", "")
                        entity_id = m.get('blocked_entity_id', '?')
                        reason_text = f"friendly {unit_type} (#{entity_id})"
                    elif reason.startswith("blocked_by_enemy_"):
                        unit_type = reason.replace("blocked_by_enemy_", "")
                        entity_id = m.get('blocked_entity_id', '?')
                        reason_text = f"enemy {unit_type} (#{entity_id})"
                    else:
                        reason_text = reason.replace("_", " ")

                    lines.append(f"║     ✗ {m['direction']}: {reason_text}")
        elif unit['capabilities']['can_move']:
            lines.append(f"║   • MOVE: All directions blocked!")

        # Shooting with detailed targets
        if actions['shooting_options']:
            lines.append(f"║   • SHOOT: {len(actions['shooting_options'])} target(s) in range")
            for shot in actions['shooting_options']:
                lines.append(
                    f"║     → Enemy #{shot['target_id']} ({shot['target_type'] if shot['target_type'] != 'decoy' else 'aircraft'}) " +
                    f"at (x={shot['target_position']['x']}, y={shot['target_position']['y']}) " +
                    f"- Relative: {shot['relative_position']} [dist={shot['distance']:.1f}] " +
                    f"Hit: {shot['hit_probability_percent']:.1f}%"
                )

        # Toggle
        if actions['toggle_option']:
            lines.append(f"║   • TOGGLE: {actions['toggle_option']['description']}")

        lines.append(f"║   • WAIT: Skip turn")
        lines.append(f"╚{'═' * 78}")

    # Unengaged enemy units (not near any friendly)
    unengaged_enemies = [
        e for e in game_state['enemy_units']
        if not e['nearby']['close_to_our_units']
    ]

    if unengaged_enemies:
        lines.append("")
        lines.append("=" * 80)
        lines.append("UNENGAGED ENEMY UNITS (Not Near Any Friendly):")
        lines.append("=" * 80)
        for enemy in unengaged_enemies:
            # Simple threat level based on type (no unit context since they're unengaged)
            threat_level = "HIGH" if enemy['type'] in ['aircraft', 'sam'] else "MEDIUM" if enemy[
                                                                                               'type'] == 'awacs' else "LOW"

            lines.append(f"\nEnemy #{enemy['id']} - {enemy['type'].upper()} (Estimated Threat: {threat_level})")
            lines.append(f"  Position: x={enemy['position']['x']}, y={enemy['position']['y']}")
            lines.append(f"  Nearest Friendly Distance: {enemy['distance_from_nearest_friendly']:.1f} cells")

            # Show grouping
            if enemy['nearby'].get('near_other_enemies'):
                group_desc = [f"#{u['id']}({u['type']})" for u in enemy['nearby']['near_other_enemies']]
                lines.append(f"  Grouped with: {', '.join(group_desc)}")

    # Last known enemy positions (lost contact)
    if game_state.get('last_known_enemies'):
        lines.append("")
        lines.append("=" * 80)
        lines.append("⚠ LOST CONTACT - Last Known Positions:")
        lines.append("=" * 80)
        lines.append("These enemies are ALIVE but out of radar. They may have moved or hidden.")
        lines.append("")
        for enemy in game_state['last_known_enemies']:
            lines.append(f"  • Enemy #{enemy['id']} ({enemy['type'].upper()}) - " +
                         f"Last seen: x={enemy['last_seen_position']['x']}, y={enemy['last_seen_position']['y']}")

    # Dead entities (casualties of war)
    if game_state.get('dead_entities') and config.show_dead_entities:
        lines.append("")
        lines.append("=" * 80)
        lines.append("💀 CASUALTIES - Dead Units (No Actions Available):")
        lines.append("=" * 80)
        lines.append("These units are ELIMINATED from the battlefield. Listed for situational awareness only.")
        lines.append("")

        # Separate by team
        our_team = game_state['team']
        our_dead = [e for e in game_state['dead_entities'] if e['team'] == our_team]
        enemy_dead = [e for e in game_state['dead_entities'] if e['team'] != our_team]

        if our_dead:
            lines.append("⚫ Our Fallen Units:")
            for dead in our_dead:
                lines.append(
                    f"  • Unit #{dead['id']} ({dead['type'].upper()}) - "
                    f"Killed at: x={dead['death_position']['x']}, y={dead['death_position']['y']}"
                )
            lines.append("")

        if enemy_dead:
            lines.append("✓ Enemy Casualties:")
            for dead in enemy_dead:
                lines.append(
                    f"  • Unit #{dead['id']} ({dead['type'].upper()}) - "
                    f"Eliminated at: x={dead['death_position']['x']}, y={dead['death_position']['y']}"
                )

        lines.append("")
        lines.append("⚠ NOTE: Dead units CANNOT take actions and pose NO threat.")

    lines.append("")
    lines.append("=" * 80)

    main_state = "\n".join(lines)
    return main_state, turn_history_str


def _parse_relative_delta(relative_position: str) -> tuple[int, int]:
    """
    Parse relative position string like "2 right, 3 up" into (dx, dy) delta.
    Returns (0, 0) if parsing fails.
    """
    try:
        dx, dy = 0, 0
        parts = relative_position.lower().split(', ')

        for part in parts:
            tokens = part.strip().split()
            if len(tokens) != 2:
                continue

            amount = int(tokens[0])
            direction = tokens[1]

            if direction == 'right':
                dx += amount
            elif direction == 'left':
                dx -= amount
            elif direction == 'down':
                dy += amount
            elif direction == 'up':
                dy -= amount

        return (dx, dy)
    except:
        return (0, 0)

def random_ai_actions(world: World, team: Team) -> Dict[int, Dict[str, Any]]:
    """
    Simple random AI that returns actions as a dict.
    Returns: {entity_id: action_dict}

    Action dict format:
    - {"type": "WAIT"}
    - {"type": "MOVE", "direction": "UP"}  # UP, DOWN, LEFT, RIGHT
    - {"type": "SHOOT", "target_id": 123}
    - {"type": "TOGGLE", "on": true}
    """
    actions = {}

    cc = world.command_center(team)

    for entity in world.entities:
        if entity.team != team or not entity.alive:
            continue

        # Random decision
        choice = random.choice(["wait", "move", "shoot"])

        if choice == "wait":
            actions[entity.id] = {"type": "WAIT"}

        elif choice == "move" and entity.can_move:
            # Try random direction
            directions = ["UP", "DOWN", "LEFT", "RIGHT"]
            direction = random.choice(directions)
            actions[entity.id] = {"type": "MOVE", "direction": direction}

        elif choice == "shoot" and entity.can_shoot and hasattr(entity, 'missiles'):
            # Check if SAM - must be ON and not in cooldown
            if hasattr(entity, 'on') and (not entity.on or entity._cooldown > 0):
                actions[entity.id] = {"type": "WAIT"}
                continue

            # Check if has ammo
            if entity.missiles <= 0:
                actions[entity.id] = {"type": "WAIT"}
                continue

            # Try to shoot at random visible enemy
            if cc.visible_enemy_ids:
                target_id = random.choice(list(cc.visible_enemy_ids))
                target = world.entities_by_id.get(target_id)

                if target and target.alive:
                    dist = world.distance(entity.pos, target.pos)
                    if dist <= entity.missile_max_range:
                        actions[entity.id] = {"type": "SHOOT", "target_id": target_id}
                    else:
                        actions[entity.id] = {"type": "WAIT"}
                else:
                    actions[entity.id] = {"type": "WAIT"}
            else:
                actions[entity.id] = {"type": "WAIT"}

        else:
            actions[entity.id] = {"type": "WAIT"}

    return actions


def convert_to_engine_actions(action_dicts: Dict[int, Dict[str, Any]]) -> Dict[int, Action]:
    """
    Convert simple action dicts to engine Action objects.
    """
    engine_actions = {}

    for entity_id, action_dict in action_dicts.items():
        action_type = action_dict.get("type", "WAIT")

        if action_type == "WAIT":
            engine_actions[entity_id] = Action(ActionType.WAIT)

        elif action_type == "MOVE":
            direction_str = action_dict.get("direction", "UP")
            direction_map = {
                "UP": MoveDir.UP,
                "DOWN": MoveDir.DOWN,
                "LEFT": MoveDir.LEFT,
                "RIGHT": MoveDir.RIGHT,
            }
            direction = direction_map.get(direction_str, MoveDir.UP)
            engine_actions[entity_id] = Action(ActionType.MOVE, {"dir": direction})

        elif action_type == "SHOOT":
            target_id = action_dict.get("target_id")
            if target_id is not None:
                engine_actions[entity_id] = Action(ActionType.SHOOT, {"target_id": target_id})

        elif action_type == "TOGGLE":
            on_state = action_dict.get("on", True)
            engine_actions[entity_id] = Action(ActionType.TOGGLE, {"on": on_state})

    return engine_actions


def get_ai_actions(world: World,
                   team: Team = Team.RED,
                   controller_type: ControllerType = ControllerType.RULE_BASED,
                   controller_params: Dict[str, Any] = None) -> Tuple[Dict[int, Action], Optional[Dict[str, Any]]]:
    """
    Main AI hook function - now supports multiple controller types.

    Args:
        world: Game world
        team: Team to control
        controller_type: Type of controller to use (RULE_BASED, LLM, RANDOM, etc.)
        controller_params: Parameters to pass to the controller

    Returns:
        Tuple of (actions, llm_output)
        - actions: Dict of entity_id -> Action
        - llm_output: Dict with LLM reasoning (None for non-LLM controllers)
    """
    if controller_params is None:
        controller_params = {"min_shoot_prob": 0.30}

    # Get actions from the specified controller
    engine_actions, llm_output = get_controller_actions(world, team, controller_type, controller_params)

    # Print actions taken (if any)
    # if engine_actions:
    #     print(f"\n[{controller_type.value.upper()}] Generated {len(engine_actions)} actions for {team.name}:")
    #     for eid, action in engine_actions.items():
    #         print(f"  Unit #{eid}: {action.type.name}")

    return engine_actions, llm_output

def rule_based_team_ai(world: World, team: Team, min_shoot_prob: float = 0.30) -> Dict[int, Dict[str, Any]]:
    """
    Generalized rule-based AI for a team (Blue or Red).

    Differences between teams:
    - Blue (assumed 'blue' in team.name) patrols RIGHT by default.
    - Red (anything else) patrols LEFT by default.

    Other rules same as original:
      - AWACS waits
      - SAM: ensure ON, wait if cooldown/no missiles, shoot best target with prob >= min_hit_prob
      - Aircraft/Decoy: shoot if good shot; otherwise move toward closest visible enemy (or patrol when none visible).
    """
    actions: Dict[int, Dict[str, Any]] = {}
    cc = world.command_center(team)

    # Determine primary patrol direction based on team identity.
    team_name = getattr(team, "name", str(team)).lower()
    # if 'blue' -> primary patrol RIGHT (Blue on LHS moves right), else LEFT (Red moves left)
    primary_patrol_dir = "RIGHT" if team_name == "blue" else "LEFT"
    opposite_patrol_dir = "LEFT" if primary_patrol_dir == "RIGHT" else "RIGHT"

    direction_map = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    for entity in world.entities:
        if entity.team != team or not entity.alive:
            continue

        # AWACS: always wait
        if entity.kind == "awacs":
            actions[entity.id] = {"type": "WAIT"}
            continue

        # SAM: ensure ON, handle cooldown/ammo, shoot best target with prob >= min_hit_prob
        if entity.kind == "sam":
            sam = entity

            # Toggle ON if off
            if not getattr(sam, "on", True):
                actions[entity.id] = {"type": "TOGGLE", "on": True}
                continue

            # Cooldown check (use attribute if available)
            if getattr(sam, "_cooldown", getattr(sam, "cooldown", 0)) > 0:
                actions[entity.id] = {"type": "WAIT"}
                continue

            # Missiles
            if getattr(sam, "missiles", 0) <= 0:
                actions[entity.id] = {"type": "WAIT"}
                continue

            best_target = None
            best_prob = 0.0

            for target_id in cc.visible_enemy_ids:
                target = world.entities_by_id.get(target_id)
                if not target or not target.alive:
                    continue

                dist = world.distance(sam.pos, target.pos)
                if dist <= getattr(sam, "missile_max_range", float("inf")):
                    prob = world.hit_probability(
                        distance=dist,
                        max_range=getattr(sam, "missile_max_range", 1.0),
                        base=getattr(sam, "base_hit_prob", 0.0),
                        min_p=getattr(sam, "min_hit_prob", 0.0),
                    )

                    if prob >= min_shoot_prob and prob > best_prob:
                        best_prob = prob
                        best_target = target_id

            if best_target:
                actions[entity.id] = {"type": "SHOOT", "target_id": best_target}
            else:
                actions[entity.id] = {"type": "WAIT"}
            continue

        # Decoy / Aircraft behavior
        if entity.kind in ("decoy", "aircraft"):
            # Find closest visible enemy (and its distance)
            closest_enemy = None
            closest_dist = float("inf")

            for target_id in cc.visible_enemy_ids:
                target = world.entities_by_id.get(target_id)
                if not target or not target.alive:
                    continue

                dist = world.distance(entity.pos, target.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_enemy = target

            # Shooting: if can shoot and has missiles, pick best target with prob >= min_hit_prob
            if getattr(entity, "can_shoot", False) and getattr(entity, "missiles", 0) > 0:
                best_target = None
                best_prob = 0.0

                for target_id in cc.visible_enemy_ids:
                    target = world.entities_by_id.get(target_id)
                    if not target or not target.alive:
                        continue

                    dist = world.distance(entity.pos, target.pos)
                    if dist <= getattr(entity, "missile_max_range", float("inf")):
                        prob = world.hit_probability(
                            distance=dist,
                            max_range=getattr(entity, "missile_max_range", 1.0),
                            base=getattr(entity, "base_hit_prob", 0.0),
                            min_p=getattr(entity, "min_hit_prob", 0.0),
                        )

                        if prob >= min_shoot_prob and prob > best_prob:
                            best_prob = prob
                            best_target = target_id

                if best_target:
                    actions[entity.id] = {"type": "SHOOT", "target_id": best_target}
                    continue

            # Movement logic
            if getattr(entity, "can_move", False):
                if closest_enemy:
                    # If adjacent or already at attack range distance <= 1, wait
                    if closest_dist <= 1:
                        actions[entity.id] = {"type": "WAIT"}
                        continue

                    # Move toward the closest enemy: prefer axis with larger distance
                    dx = closest_enemy.pos[0] - entity.pos[0]
                    dy = closest_enemy.pos[1] - entity.pos[1]

                    if abs(dx) >= abs(dy):
                        direction = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        direction = "DOWN" if dy > 0 else "UP"

                    delta = direction_map[direction]
                    new_pos = (entity.pos[0] + delta[0], entity.pos[1] + delta[1])

                    if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                        actions[entity.id] = {"type": "MOVE", "direction": direction}
                    else:
                        # Try alternative directions in deterministic order
                        for alt_dir in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            if alt_dir == direction:
                                continue
                            delta = direction_map[alt_dir]
                            new_pos = (entity.pos[0] + delta[0], entity.pos[1] + delta[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": alt_dir}
                                break
                        else:
                            actions[entity.id] = {"type": "WAIT"}
                else:
                    # No enemy visible -> patrol using team-specific primary direction
                    current_x = entity.pos[0]

                    if primary_patrol_dir == "RIGHT":
                        # Try move right until edge, then go left
                        if current_x < world.width - 1:
                            new_pos = (current_x + 1, entity.pos[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": "RIGHT"}
                                continue
                        # if blocked or at right edge, try alternatives then move left
                        # Try alternatives (UP, DOWN, LEFT)
                        for alt_dir in ["UP", "DOWN", "LEFT"]:
                            delta = direction_map[alt_dir]
                            new_pos = (entity.pos[0] + delta[0], entity.pos[1] + delta[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": alt_dir}
                                break
                        else:
                            # Finally try to move left (patrol opposite)
                            new_pos = (current_x - 1, entity.pos[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": "LEFT"}
                            else:
                                actions[entity.id] = {"type": "WAIT"}
                    else:
                        # primary_patrol_dir == "LEFT"
                        if current_x > 0:
                            new_pos = (current_x - 1, entity.pos[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": "LEFT"}
                                continue
                        # if blocked or at left edge, try alternatives then move right
                        for alt_dir in ["UP", "DOWN", "RIGHT"]:
                            delta = direction_map[alt_dir]
                            new_pos = (entity.pos[0] + delta[0], entity.pos[1] + delta[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": alt_dir}
                                break
                        else:
                            # Finally try to move right (patrol opposite)
                            new_pos = (current_x + 1, entity.pos[1])
                            if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
                                actions[entity.id] = {"type": "MOVE", "direction": "RIGHT"}
                            else:
                                actions[entity.id] = {"type": "WAIT"}
            else:
                actions[entity.id] = {"type": "WAIT"}
            continue

    return actions


# ============================================================================
# Register Controllers
# ============================================================================

def _rule_based_controller(world: World, team: Team, **kwargs) -> Tuple[Dict[int, Action], None]:
    """Rule-based controller wrapper"""
    min_shoot_prob = kwargs.get("min_shoot_prob", 0.30)
    action_dicts = rule_based_team_ai(world, team, min_shoot_prob)
    return convert_to_engine_actions(action_dicts), None


def _random_controller(world: World, team: Team, **kwargs) -> Tuple[Dict[int, Action], None]:
    """Random controller wrapper"""
    action_dicts = random_ai_actions(world, team)
    return convert_to_engine_actions(action_dicts), None


def _llm_controller(world: World, team: Team, **kwargs) -> Tuple[Dict[int, Action], Optional[Dict[str, Any]]]:
    """
    LLM controller wrapper

    Returns:
        Tuple of (engine_actions, llm_output_record)
    """
    min_shoot_prob = kwargs.get("min_shoot_prob", 0.30)
    log_history_turns = kwargs.get("log_history_turns", 3)  # NEW: configurable

    # Get game state with configurable history
    game_state = get_game_state(world, team, log_history_turns=log_history_turns)
    # ✅ PASS world to the formatter
    formatted_state, formatted_turn_history = format_game_state_for_llm(game_state, world)

    COMBAT_GAME_DEPS.game_state = formatted_state

    if team == team.RED:
        print("\n" + formatted_state)

    # STRATEGIST
    if COMBAT_GAME_DEPS.current_turn_number == 0:
        resp = strategic_director.run_sync("Analyse the game status carefully and come up with winning strategy.", deps=COMBAT_GAME_DEPS)
        COMBAT_GAME_DEPS.team_strategy = resp.output.strategy.text
        COMBAT_GAME_DEPS.entity_directives = resp.output.entity_directives
        COMBAT_GAME_DEPS.re_strategize_when = resp.output.strategy.call_me_back_if
        COMBAT_GAME_DEPS.world = world

    # TODO: If entity is dead, remove it's directives?
    entity_directives_str = "\n".join(
        f"- Entity {d.entity_id}:\n'''\n{d.pseudo_code_directive}'''\n---\n"
        for d in COMBAT_GAME_DEPS.entity_directives
        if d.entity_id not in ([entity_dict["id"] for entity_dict in game_state.get("dead_entities", [])])
    )

    # ANALYST
    key_facts_text = "\n---".join(COMBAT_GAME_DEPS.key_facts) if COMBAT_GAME_DEPS.key_facts else None
    previous_turn = COMBAT_GAME_DEPS.turn_summaries[-1] if COMBAT_GAME_DEPS.turn_summaries else None
    analyst_prompt = f"""
TURN NO: {COMBAT_GAME_DEPS.current_turn_number} (You can use it for your analysis and new key facts to make it clear for future)
Analyse the game status carefully. Your last turn analysis along your old key facts so far (to help you understand the history) :

**Recent Turn History (Observable Actions)**
{formatted_turn_history}
---
**Your Previous Turn Analysis:**
{previous_turn}
---
**Your Key Facts:**
{key_facts_text}
---
**The Team Strategy:**
{COMBAT_GAME_DEPS.team_strategy}

**Re-strategize When:***
{COMBAT_GAME_DEPS.re_strategize_when}
"""

    resp = analyst_agent.run_sync(analyst_prompt, deps=COMBAT_GAME_DEPS)
    analysis, key_facts, re_strategize, re_strategize_reason = resp.output.analysis, resp.output.key_facts, resp.output.re_strategize, resp.output.re_strategize_reason
    COMBAT_GAME_DEPS.turn_summaries.append(analysis)
    COMBAT_GAME_DEPS.key_facts.append(key_facts)


    # RE-STRATEGIZE
    if re_strategize:
        restrategize_prompt = f"""
It is time to re-adjust to the current game plan. Give me a new strategy based on current conditions.

### Previous Strategy :
**Team Strategy**:
{COMBAT_GAME_DEPS.team_strategy}
---
**Entity Level Directives:**
{entity_directives_str}
---
**You Asked for a Callback When:**
{COMBAT_GAME_DEPS.re_strategize_when}
---
-> Remember currently you might have less resources compared to old strategy, consider the current game state, for the new strategy.

### Current Game Analysis by the Analyst:
{analysis}
### Reason to Restrategize by the Analyst:
{re_strategize_reason}
"""
        resp = strategic_director.run_sync(restrategize_prompt, deps=COMBAT_GAME_DEPS)
        COMBAT_GAME_DEPS.team_strategy = resp.output.strategy.text
        COMBAT_GAME_DEPS.entity_directives = resp.output.entity_directives
        COMBAT_GAME_DEPS.re_strategize_when = resp.output.strategy.call_me_back_if
        COMBAT_GAME_DEPS.restrategized = True

    # EXECUTER
    strategic_context_message = (
        f"As your strategy director, here are the high level strategy for you and for each entity. "
        f"I believe in you on the field.\n\n"
        f"**Team Strategy**: {COMBAT_GAME_DEPS.team_strategy}\n\n---\n"
        f"**Entity Level Strategies:**\n{entity_directives_str}\n\n---\n"
        f"**Game Analysis from the Analyst**:\n{analysis}\n\n"
    )
    result = executer_agent.run_sync(strategic_context_message, deps=COMBAT_GAME_DEPS)

    # Get both action dicts and structured output
    action_dicts, llm_output_record = tactical_plan_to_action_dicts(result.output, print_output=True)
    engine_actions = convert_to_engine_actions(action_dicts)

    COMBAT_GAME_DEPS.current_turn_number +=1

    # Following is just to record executer context for better debugging on recorder side
    if llm_output_record is None:
        llm_output_record = {}
    from wargame_2d.agents.fixed_prompts import EXECUTER_TASK, GAME_INFO, STRATEGIC_GUIDELINES
    executor_system_context = f"""
You are an AI field commander leading the RED team in a grid-based air combat simulation with given strategical orders

### TASK
{EXECUTER_TASK}

{GAME_INFO}

{STRATEGIC_GUIDELINES}

### GAME STATE 
{COMBAT_GAME_DEPS.game_state}
"""
    llm_output_record["executor_context"] = executor_system_context +"\n\n"+ strategic_context_message
    return engine_actions, llm_output_record


# def _code_generation_controller(world: World, team: Team, **kwargs) -> Tuple[
#     Dict[int, Action], Optional[Dict[str, Any]]]:
#     """
#     Code generation controller - generates Python code via LLM, then executes it.
#     Code is cached after first generation.
#     """
#     from wargame_2d.code_controller import get_code_gen_controller
#
#     # Get parameters
#     regenerate = kwargs.get("regenerate_code", False)
#     user_prompt = kwargs.get("user_prompt", "Generate intelligent controller code for tactical combat.")
#     system_prompt = kwargs.get("system_prompt", None)
#     additional_context = kwargs.get("additional_context", "")
#     save_code_path = kwargs.get("save_code_path", None)
#     log_history_turns = kwargs.get("log_history_turns", 3)
#
#     # Get game state
#     game_state_dict = get_game_state(world, team, log_history_turns=log_history_turns)
#     game_state_formatted = format_game_state_for_llm(game_state_dict)
#
#     # Get controller instance
#     controller = get_code_gen_controller()
#
#     # Generate code if needed
#     if not controller.is_ready or regenerate:
#         # Set system prompt if provided
#         if system_prompt:
#             from wargame_2d.code_controller import code_generation_agent
#             code_generation_agent._system_prompt = system_prompt
#
#         # Generate and compile
#         try:
#             controller.generate_and_compile(
#                 user_prompt=user_prompt,
#                 game_state_formatted=game_state_formatted,
#                 additional_context=additional_context
#             )
#         except Exception as e:
#             print(f"[CODE GEN] Failed to generate/compile: {e}")
#             return {}, None
#
#     # Execute controller
#     action_dicts = controller.execute(game_state_dict)
#
#
#     # Save code if requested
#     if save_code_path:
#         controller.save_code(save_code_path)
#
#     # Convert to engine actions
#     engine_actions = convert_to_engine_actions(action_dicts)
#
#     # LLM output record (minimal - just metadata)
#     llm_output = {
#         "controller_type": "code_generation",
#         # "code_length": len(controller._generated_code) if controller._generated_code else 0,
#         # "num_actions": len(action_dicts)
#     }
#
#     return engine_actions, llm_output



# Register all controllers
ControllerRegistry.register(ControllerType.RULE_BASED, _rule_based_controller)
ControllerRegistry.register(ControllerType.LLM, _llm_controller)
ControllerRegistry.register(ControllerType.RANDOM, _random_controller)
#ControllerRegistry.register(ControllerType.CODE_GEN, _code_generation_controller)