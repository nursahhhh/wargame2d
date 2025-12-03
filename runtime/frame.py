from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from env.core.actions import Action
from env.core.types import Team
from env.environment import StepInfo
from env.world import WorldState


@dataclass
class Frame:
    """
    Immutable snapshot of a single turn, with helpers to serialize for transport.
    """

    world: WorldState | None
    actions: Optional[Mapping[int, Action]] = None
    action_metadata: Optional[Mapping[str, Any]] = None
    step_info: Optional[StepInfo] = None
    done: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the frame into a JSON-friendly dictionary.
        """
        frame: Dict[str, Any] = {
            "turn": self.world.turn,
            "world": self.world.to_dict(),
            "entities": self._serialize_entities(self.world),
            "observations": self._serialize_observations(self.world),
        }

        actions_payload = self._serialize_actions(self.actions or {})
        if actions_payload:
            frame["actions"] = actions_payload
        if self.action_metadata is not None:
            frame["action_metadata"] = dict(self.action_metadata)
        if self.step_info is not None:
            frame["step_info"] = self.step_info.to_dict()
        if self.done is not None:
            frame["done"] = self.done

        return frame

    @staticmethod
    def _serialize_observations(world: WorldState) -> Dict[str, Any]:
        """
        Serialize per-team observations for fog-of-war consumers.

        Includes observed entities (with spoofed kinds), friendly IDs, and
        visible enemy IDs. Visible positions are omitted to keep payloads
        lean; UI can still filter entities by IDs.
        """
        observations: Dict[str, Any] = {}
        for team in Team:
            view = world.get_team_view(team)
            obs_list = view.get_all_observations()

            observations[team.name.lower()] = {
                "entities": [obs.to_dict() for obs in obs_list],
                "friendly_ids": sorted(view.get_friendly_ids()),
                "visible_enemy_ids": sorted(view.get_enemy_ids(team)),
            }

        return observations

    @staticmethod
    def _serialize_entities(world: WorldState) -> List[Dict[str, Any]]:
        """
        Serialize entities for the frontend without altering canonical world dict.
        """
        serialized: List[Dict[str, Any]] = []

        for entity in world.get_all_entities():
            data: Dict[str, Any] = {
                "id": entity.id,
                "team": entity.team.name,
                "kind": entity.kind.value,
                "type": entity.__class__.__name__,
                "name": entity.name,
                "position": list(entity.pos),
                "alive": entity.alive,
                "is_alive": entity.alive,
                "can_move": entity.can_move,
                "can_shoot": entity.can_shoot,
                "radar_range": getattr(entity, "radar_range", 0),
                "active_radar": entity.get_active_radar_range(),
                "missiles": getattr(entity, "missiles", None),
                "missile_max_range": getattr(entity, "missile_max_range", None),
            }

            # SAM-specific fields (optional)
            data["radar_on"] = getattr(entity, "on", None)
            data["cooldown_remaining"] = getattr(entity, "_cooldown", None)

            serialized.append(data)

        return serialized

    @staticmethod
    def _serialize_actions(actions: Mapping[int, Action]) -> list[Dict[str, Any]]:
        """Serialize action map to a list for easy iteration client-side."""
        serialized: list[Dict[str, Any]] = []
        for entity_id, action in actions.items():
            serialized.append(
                {
                    "entity_id": entity_id,
                    "type": action.type.name,
                    "params": action.to_dict().get("params", {}),
                    "label": str(action),
                }
            )
        return serialized
