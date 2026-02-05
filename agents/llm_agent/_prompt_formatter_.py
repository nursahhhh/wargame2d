"""
Draft helpers for shaping game state into an LLM-friendly prompt.

This formatter is intentionally conservative:
- It NEVER mutates game state
- It only reads what exists
- Missing fields are handled safely (None)

Updated version:
- Replaces boolean `has_fired_before` with fire-behavior abstraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from env.core.actions import Action
from env.core.types import ActionType,MoveDir
from env.entities.base import Entity
from env.entities import SAM
from ..team_intel import TeamIntel, VisibleEnemy


@dataclass
class PromptConfig:
    nearby_ally_radius: float = 3.0
    nearby_enemy_radius: float = 5.0
    grouping_radius: float = 2.5
    include_hit_probabilities: bool = True
    awacs_stats = None


class PromptFormatter:

    """
    Convert TeamIntel + allowed actions into:
    - structured payload (dict)
    - human-readable prompt (string)
    """

    def _json_safe(self, value: Any) -> Any:
        """
        Convert Python-native structures (tuple, etc.)
        into JSON/LLM-friendly ones without mutating game state.
        """
        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, list):
            return [self._json_safe(v) for v in value]

        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}

        return value

    def build_prompt(
        self,
        *,
        intel: TeamIntel,
        step :int,
        allowed_actions: Dict[int, List[Action]],
        config: Optional[PromptConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        cfg = config or PromptConfig()

        payload: Dict[str, Any] = {
            "grid": {
                "width": intel.grid.width,
                "height": intel.grid.height,
            },
            "global_state": {
            "aggression": round(intel.aggression_level(turn=step), 2
        )
    },
            "friendlies": [],
        }

        for entity in intel.friendlies:
            if not entity.alive:
                continue

            summary = self._summarize_entity(entity)
            summary["capabilities"] = self._capabilities(entity)
            summary["nearby_allies"] = self._nearby_allies(
                entity, intel, cfg.nearby_ally_radius
            )
            summary["nearby_enemies"] = self._nearby_enemies(
                entity, intel, cfg.nearby_enemy_radius, cfg.include_hit_probabilities
            )
        
            summary["grouped_with_allies"] = (
                any(a["distance"] <= cfg.grouping_radius for a in summary["nearby_allies"])
                if summary["nearby_allies"]
                else False
            )
            summary["pressure"] = {
                "value": round(intel.pressure_around(entity), 2),
                "level": intel.pressure_level(entity),
            }


            payload["friendlies"].append(summary)
        json_payload = self._json_safe(payload)
        prompt = self._format_prompt(json_payload)

        return prompt, payload


    # Entity summaries
    def _summarize_entity(self, entity: Entity) -> Dict[str, Any]:
        return {
            "id": entity.id,
            "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
            "kind": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": entity.pos,
        }

    def _capabilities(self, entity: Entity) -> Dict[str, Any]:
        caps=  {
            "mobile": entity.can_move,
            "armed": bool(getattr(entity, "missiles", 0)) or entity.can_shoot,
            "missiles": getattr(entity, "missiles", None),
            "missile_max_range": getattr(entity, "missile_max_range", None),
            "base_hit_prob": getattr(entity, "base_hit_prob", None),
            "min_hit_prob": getattr(entity, "min_hit_prob", None),
            "radar_range": entity.get_active_radar_range(),
        }


        # SAM-specific
        if hasattr(entity, "is_toggled") or  isinstance(entity, SAM): 
            caps["is_radar_active"] = getattr(entity, "is_toggled", False)
            caps["activation_range"] = getattr(entity, "activation_range", None)
            caps["can_shoot_when_active"] = getattr(entity, "can_shoot", False)
        return caps

    # Nearby allies
  
    def _nearby_allies(
        self,
        entity: Entity,
        intel: TeamIntel,
        radius: float,
    ) -> List[Dict[str, Any]]:

        allies: List[Dict[str, Any]] = []

        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue

            distance = intel.grid.distance(entity.pos, other.pos)
            if distance <= radius:
                allies.append(
                    {
                        "id": other.id,
                        "kind": other.kind.name if hasattr(other.kind, "name") else str(other.kind),
                        "position": other.pos,
                        "distance": distance,
                        "armed": bool(getattr(other, "missiles", 0)) or other.can_shoot,
                    }
                )
        return allies

    # Nearby enemies 
    def _nearby_enemies(
        self,
        entity: Entity,
        intel: TeamIntel,
        radius: float,
        include_hit_probs: bool,
    ) -> List[Dict[str, Any]]:

        enemies: List[Dict[str, Any]] = []

        for enemy in intel.visible_enemies:
            distance = intel.grid.distance(entity.pos, enemy.position)
            if distance > radius:
                continue

            fire_total = getattr(enemy, "fire_count_total", None)
            fire_recent = getattr(enemy, "fire_count_last_k", None)
            fire_delta = getattr(enemy, "last_fire_step_delta", None)

            entry: Dict[str, Any] = {
                "id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "kind": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "position": enemy.position,
                "distance": distance,
                "threat_score": round( intel.enemy_threat_score(enemy, entity.pos), 2),
                "fire_behavior": {
                    "total_shots": fire_total,
                    "recent_shots": fire_recent,
                    "last_fired_steps_ago": fire_delta,
                },
                "armed": None}
            if len(intel.visible_enemies) >1:
                entry["grouped"] = intel. _enemy_is_grouped(enemy=enemy, all_enemies=intel.visible_enemies,radius= radius) 
                    
                
            

            if include_hit_probs:
                entry["our_hit_prob"] = intel.estimate_hit_probability(entity, enemy)
                entry["their_hit_prob"] = None
            
            if entity.name == "AWACS" :
                    entry["enemy_proximity_trend"] = TeamIntel.radar_threat_trend(entity,enemy)
            
            enemies.append(entry)

        return enemies


    
        

        
    def _summarize_actions(world, awacs_id, candidate_actions):
        awacs = world.entities[awacs_id]
        awacs_pos = awacs.position

        enemy_awacs = world.enemy_awacs
        enemy_pos = enemy_awacs.position
        enemy_radar_range = enemy_awacs.radar_range

        summaries = []
        def estimate_next_position(pos, action):
            x, y = pos

            if action.type != ActionType.MOVE:
                return pos

            raw_dir = action.params.get("dir")

            if isinstance(raw_dir, MoveDir):
                move_dir = raw_dir
            elif isinstance(raw_dir, str):
                try:
                    move_dir = MoveDir[raw_dir]
                except KeyError:
                    return pos
            else:
                return pos

            dx, dy = move_dir.delta
            return (x + dx, y + dy)

        for action in candidate_actions:
            next_pos = estimate_next_position(awacs_pos, action)

    
            summaries.append({
                "action_type": action.type,
                "direction": getattr(action, "direction", None),
                "current_pos": awacs_pos,
                "estimated_next_pos": next_pos,
                "distance_to_enemy_awacs": round(
                    ((next_pos[0] - enemy_pos[0]) ** 2 +
                    (next_pos[1] - enemy_pos[1]) ** 2) ** 0.5,
                    2
                )
            })

        return summaries
        

    # ===============================================================
    # Prompt formatting
    # ===============================================================
    def _format_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Strategic prompt for LLM.
        Emphasizes win condition, long-term safety and role-based reasoning.
        """

        lines: List[str] = []

        # ==================================================
        # GLOBAL STRATEGIC CONTEXT
        # ==================================================
        if "global_state" in payload:
            aggr = payload["global_state"]["aggression"]
            lines.append(
                f"\nGLOBAL AGGRESSION ESTIMATE: {aggr} "
                "(higher = safer to pressure, lower = defensive posture)"
            )

           


        # ==================================================
        # FRIENDLY UNITS
        # ==================================================
        lines.append("\n=== FRIENDLY UNITS ===")

        for friendly in payload["friendlies"]:
            lines.append(
                f"\nUnit {friendly['kind']}#{friendly['id']} at {friendly['position']}"
            )

            # Capabilities
            cap = friendly["capabilities"]
            lines.append(
                f"- Capabilities: mobile={cap['mobile']}, armed={cap['armed']}, radar={cap['radar_range']}"
            )

            # Pressure & threat
            pressure = friendly.get("pressure_level")
            pressure_score = friendly.get("pressure_score")

            if pressure is not None:
                lines.append(
                    f"- Local pressure: {pressure}"
                    + (f" ({pressure_score:.2f})" if pressure_score is not None else "")
                )

            # Nearby enemies (contextual, not raw)
            enemies = friendly.get("nearby_enemies", [])
            if enemies:
                lines.append("- Nearby threats:")
                for e in enemies:
                    reason = []
                    if e.get("has_fired_before"):
                        reason.append("has fired before")
                    if e.get("grouped"):
                        reason.append("grouped")
                    if "distance" in e:
                        reason.append(f"d={e['distance']:.1f}")

                    reason_txt = ", ".join(reason)
                    lines.append(
                        f"  • {e['kind']}#{e['id']} at {e['position']} ({reason_txt})"
                    )
            else:
                lines.append("- Nearby threats: none")
      
            # Allowed actions
            actions = friendly.get("allowed_actions", [])
            if actions:
                lines.append("- Available actions:")
                for a in actions:
                    if a.type == "MOVE":
                        d=a["params"]["dir"]
                        lines.append(f"  • MOVE {d} (risk unknown) ")
                    else:
                        lines.append(f"  •  {a['type']}")



        return "\n".join(lines)


    def _format_enemy_line(self, enemy: Dict[str, Any]) -> str:
        hit_prob = enemy.get("our_hit_prob")
        hit_text = f", our_hit_prob={hit_prob:.2f}" if isinstance(hit_prob, float) else ""
        return (
            f"{enemy['kind']}#{enemy['id']}@{enemy['position']} d={enemy['distance']:.1f}"
            f"{hit_text}"
        )

    def _format_action_line(self, action: Dict[str, Any]) -> str:
        if action["type"] == ActionType.SHOOT.name:
            target = action.get("target_id")
            distance = action.get("distance")
            hit_prob = action.get("our_hit_prob")
            detail_parts = []
            if target is not None:
                detail_parts.append(f"target={target}")
            if distance is not None:
                detail_parts.append(f"d={distance:.1f}")
            if isinstance(hit_prob, float):
                detail_parts.append(f"p_hit={hit_prob:.2f}")
                detail = ", ".join(detail_parts)
            return f"SHOOT({detail})"
        if action["type"] == ActionType.MOVE.name:
            return f"MOVE(dir={action['params'].get('dir')})"
        if action["type"] == ActionType.TOGGLE.name:
            return f"TOGGLE(on={action['params'].get('on')})"
        return action["type"]