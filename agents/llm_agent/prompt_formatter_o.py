"""
Draft helpers for shaping game state into an LLM-friendly prompt.

Usage sketch (inside an agent):

    formatter = PromptFormatter()
    prompt, payload = formatter.build_prompt(
        intel=intel,
        allowed_actions=allowed_actions,  # mapping entity_id -> list[Action]
        config=PromptConfig(
            nearby_ally_radius=3.0,
            nearby_enemy_radius=5.0,
            grouping_radius=2.5,
        ),
    )

`payload` is a structured dictionary you can further post-process or serialize
to JSON; `prompt` is a human-readable string assembled from that payload.

This module is intentionally conservative and introspective: it only uses
fields that already exist on entities/visible enemies and falls back to None
when data is missing, so it can evolve safely as we add richer stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from env.core.actions import Action
from env.core.types import ActionType
from env.entities.base import Entity
from ..team_intel import TeamIntel, VisibleEnemy


@dataclass
class PromptConfig:
    """
    Tunable knobs that shape proximity calculations and grouping.
    """

    nearby_ally_radius: float = 3.0
    nearby_enemy_radius: float = 5.0
    grouping_radius: float = 2.5
    include_hit_probabilities: bool = True


class PromptFormatter:
    """
    Convert intel + allowed actions into a structured payload and readable prompt.
    """

    def build_prompt(
        self,
        *,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        config: Optional[PromptConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a string prompt plus structured payload for all friendly entities.
        """
        cfg = config or PromptConfig()
        payload = {
            "grid": {"width": intel.grid.width, "height": intel.grid.height},
            "friendlies": [],
        }

        for entity in intel.friendlies:
            if not entity.alive:
                continue

            friendly_summary = self._summarize_entity(entity)
            friendly_summary["capabilities"] = self._capabilities(entity)
            friendly_summary["nearby_allies"] = self._nearby_allies(
                entity, intel, cfg.nearby_ally_radius
            )
            friendly_summary["nearby_enemies"] = self._nearby_enemies(
                entity, intel, cfg.nearby_enemy_radius, cfg.include_hit_probabilities
            )
            friendly_summary["grouped_with_allies"] = (
                len(friendly_summary["nearby_allies"]) > 0
                and any(
                    ally["distance"] <= cfg.grouping_radius
                    for ally in friendly_summary["nearby_allies"]
                )
            )
            friendly_summary["allowed_actions"] = self._summarize_actions(
                entity, allowed_actions.get(entity.id, []), intel, cfg.include_hit_probabilities
            )

            payload["friendlies"].append(friendly_summary)

        prompt = self._format_prompt(payload)
        return prompt, payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _summarize_entity(self, entity: Entity) -> Dict[str, Any]:
        return {
            "id": entity.id,
            "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
            "kind": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": entity.pos,
        }

    def _capabilities(self, entity: Entity) -> Dict[str, Any]:
        """Best-effort capability snapshot using known attributes."""
        return {
            "mobile": entity.can_move,
            "armed": bool(getattr(entity, "missiles", 0)) or entity.can_shoot,
            "missiles": getattr(entity, "missiles", None),
            "missile_max_range": getattr(entity, "missile_max_range", None),
            "base_hit_prob": getattr(entity, "base_hit_prob", None),
            "min_hit_prob": getattr(entity, "min_hit_prob", None),
            "radar_range": entity.get_active_radar_range(),
        }

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
            entry: Dict[str, Any] = {
                "id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "kind": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "position": enemy.position,
                "distance": distance,
                "has_fired_before": enemy.has_fired_before,
                "armed": None,  # unknown for visible enemies without intel
                "grouped": self._is_grouped(enemy, intel.visible_enemies, intel, radius),
            }
            if include_hit_probs:
                entry["our_hit_prob"] = intel.estimate_hit_probability(entity, enemy)
                entry["their_hit_prob"] = None  # Placeholder; we lack enemy weapon stats
            enemies.append(entry)
        return enemies

    def _is_grouped(
        self,
        enemy: VisibleEnemy,
        all_enemies: Iterable[VisibleEnemy],
        intel: TeamIntel,
        radius: float,
    ) -> bool:
        for other in all_enemies:
            if other.id == enemy.id:
                continue
            if intel.grid.distance(enemy.position, other.position) <= radius:
                return True
        return False

    def _summarize_actions(
        self,
        entity: Entity,
        actions: List[Action],
        intel: TeamIntel,
        include_hit_probs: bool,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for action in actions:
            summary = {
                "type": action.type.name,
                "params": action.to_dict()["params"],
            }
            if action.type == ActionType.SHOOT:
                target_id = action.params.get("target_id")
                target = intel.get_enemy(target_id) or intel.get_friendly(target_id)
                summary["target_id"] = target_id
                if target:
                    target_pos = target.position if isinstance(target, VisibleEnemy) else target.pos  # type: ignore[attr-defined]
                    distance = intel.grid.distance(entity.pos, target_pos)
                    summary["target_position"] = target_pos
                    summary["distance"] = distance
                    if include_hit_probs and isinstance(target, VisibleEnemy):
                        summary["our_hit_prob"] = intel.estimate_hit_probability(entity, target)
            summaries.append(summary)
        return summaries

    def _format_prompt(self, payload: Dict[str, Any]) -> str:
        """Turn the structured payload into a readable text block."""
        lines: List[str] = []
        lines.append(
            f"Grid: width={payload['grid']['width']}, height={payload['grid']['height']}"
        )
        for friendly in payload["friendlies"]:
            lines.append(
                f"\nAlly {friendly['kind']}#{friendly['id']} at {friendly['position']} "
                f"(mobile={friendly['capabilities']['mobile']}, armed={friendly['capabilities']['armed']}, "
                f"radar={friendly['capabilities']['radar_range']})"
            )
            if friendly["nearby_allies"]:
                allies_str = ", ".join(
                    f"{ally['kind']}#{ally['id']}@{ally['position']} d={ally['distance']:.1f}"
                    for ally in friendly["nearby_allies"]
                )
                lines.append(f"  Nearby allies: {allies_str}")
            if friendly["nearby_enemies"]:
                enemies_str = "; ".join(
                    self._format_enemy_line(enemy) for enemy in friendly["nearby_enemies"]
                )
                lines.append(f"  Nearby threats: {enemies_str}")

            if friendly["allowed_actions"]:
                actions_str = "; ".join(self._format_action_line(a) for a in friendly["allowed_actions"])
                lines.append(f"  Allowed actions: {actions_str}")
            else:
                lines.append("  Allowed actions: none")
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

