from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from env.core.actions import Action, ActionType, MoveDir
from env.entities.base import Entity
from env.entities import SAM
from ..team_intel import TeamIntel


class PromptConfig(BaseModel):
    nearby_ally_radius: float = 3.0
    nearby_enemy_radius: float = 5.0
    grouping_radius: float = 2.5
    include_hit_probabilities: bool = True
    awacs_stats: Optional[Dict[str, Any]] = None


class PromptFormatter:

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            return list(map(self._json_safe, value))
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        return value

    def build_prompt(
        self,
        *,
        intel: TeamIntel,
        step: int,
        allowed_actions: Dict[int, List[Action]],
        config: Optional[PromptConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        cfg = config or PromptConfig()
        payload: Dict[str, Any] = {
            "grid": {"width": intel.grid.width, "height": intel.grid.height},
            "step": step,
            "global_state": {"aggression": round(intel.aggression_level(turn=step), 2)},
            "friendlies": [],
        }

        for entity in intel.friendlies:
            if not entity.alive:
                continue
            summary = self._summarize_entity(entity, intel, cfg)
            payload["friendlies"].append(summary)

        json_payload = self._json_safe(payload)
        prompt = self._format_prompt(json_payload)
        return prompt, payload

    def _summarize_entity(self, entity: Entity, intel: TeamIntel, cfg: PromptConfig) -> Dict[str, Any]:
        summary = {
            "id": entity.id,
            "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
            "kind": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": entity.pos,
            "capabilities": self._capabilities(entity),
            "nearby_allies": self._nearby_allies(entity, intel, cfg.nearby_ally_radius),
            "nearby_enemies": self._nearby_enemies(entity, intel, cfg.nearby_enemy_radius, cfg.include_hit_probabilities),
        }
        summary["grouped_with_allies"] = any(a["distance"] <= cfg.grouping_radius for a in summary["nearby_allies"]) if summary["nearby_allies"] else False
        summary["pressure"] = {
            "value": round(intel.pressure_around(entity), 2),
            "level": intel.pressure_level(entity),
        }
        return summary

    def _capabilities(self, entity: Entity) -> Dict[str, Any]:
        caps = {
            "mobile": entity.can_move,
            "armed": bool(getattr(entity, "missiles", 0)) or entity.can_shoot,
            "missiles": getattr(entity, "missiles", None),
            "missile_max_range": getattr(entity, "missile_max_range", None),
            "base_hit_prob": getattr(entity, "base_hit_prob", None),
            "min_hit_prob": getattr(entity, "min_hit_prob", None),
            "radar_range": entity.get_active_radar_range(),
        }
        if hasattr(entity, "is_toggled") or isinstance(entity, SAM):
            caps.update({
                "is_radar_active": getattr(entity, "is_toggled", False),
                "activation_range": getattr(entity, "activation_range", None),
                "can_shoot_when_active": getattr(entity, "can_shoot", False),
            })
        return caps

    def _nearby_allies(self, entity: Entity, intel: TeamIntel, radius: float) -> List[Dict[str, Any]]:
        allies = []
        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue
            dist = intel.grid.distance(entity.pos, other.pos)
            if dist <= radius:
                allies.append({
                    "id": other.id,
                    "kind": other.kind.name if hasattr(other.kind, "name") else str(other.kind),
                    "position": other.pos,
                    "distance": dist,
                    "armed": bool(getattr(other, "missiles", 0)) or other.can_shoot,
                })
        return allies

    def _nearby_enemies(self, entity: Entity, intel: TeamIntel, radius: float, include_hit_probs: bool) -> List[Dict[str, Any]]:
        enemies = []
        for enemy in intel.visible_enemies:
            distance = intel.grid.distance(entity.pos, enemy.position)
            if distance > radius:
                continue
            entry = {
                "id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "kind": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "position": enemy.position,
                "distance": distance,
                "threat_score": round(intel.enemy_threat_score(enemy, entity.pos), 2),
                "fire_behavior": {
                    "total_shots": getattr(enemy, "fire_count_total", None),
                    "recent_shots": getattr(enemy, "fire_count_last_k", None),
                    "last_fired_steps_ago": getattr(enemy, "last_fire_step_delta", None),
                },
                "armed": None,
            }
            if include_hit_probs:
                entry["our_hit_prob"] = intel.estimate_hit_probability(entity, enemy)
            enemies.append(entry)
        return enemies

    def _format_prompt(self, payload: Dict[str, Any]) -> str:
        lines = [f"Turn: {payload.get('step', 0)}", f"Grid: {payload['grid']['width']}x{payload['grid']['height']}"]
        for f in payload["friendlies"]:
            lines.append(f"Unit {f['kind']}#{f['id']} at {f['position']}")
        return "\n".join(lines)