"""
Decoy entity - a deceptive unit that appears as an aircraft to enemies.

Decoys:
- Appear as aircraft to enemy radar
- Can move around
- Have no radar (cannot detect enemies)
- Cannot shoot (no weapons)
- Used to confuse enemy targeting
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, TYPE_CHECKING, Dict, Any

from .base import Entity
from ..core.types import Team, GridPos, EntityKind, MoveDir
from ..core.actions import Action

if TYPE_CHECKING:
    from ..world.world import WorldState


@dataclass
class Decoy(Entity):
    """
    A decoy unit that appears as an aircraft to enemies.

    Decoys are used for deception - they look like aircraft on enemy radar
    but have no offensive capability. The SensorSystem handles making
    decoys appear as aircraft to enemy observations.

    Attributes:
        radar_range: Always 0.0 (decoys cannot detect anything)
    """

    kind: EntityKind = EntityKind.DECOY
    can_move: bool = True
    can_shoot: bool = False
    radar_range: float = 0.0  # Decoys have no radar

    def get_allowed_actions(self, world: WorldState) -> List[Action]:
        """
        Get all actions this decoy can perform that are feasible.
        
        This checks:
        - Entity-level constraints (alive, can_move)
        - World-state constraints that are knowable (bounds)
        
        Decoys can only wait or move - they have no weapons or sensors.
        """
        if not self.alive:
            return []

        actions = [Action.wait()]

        # Decoys can move to position themselves - only include moves that stay in bounds
        if self.can_move:
            for direction in MoveDir:
                dx, dy = direction.delta
                new_pos = (self.pos[0] + dx, self.pos[1] + dy)
                if world.grid.in_bounds(new_pos):
                    actions.append(Action.move(direction))

        # No shooting - decoys are unarmed
        # No radar - decoys cannot detect enemies
        return actions

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Decoy:
        """Construct decoy from dictionary."""
        return cls(
            team=Team[data["team"]],
            pos=tuple(data["pos"]),
            name=data["name"],
        )