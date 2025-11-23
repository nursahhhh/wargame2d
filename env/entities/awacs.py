"""
AWACS entity - Airborne Warning and Control System.

AWACS is a surveillance aircraft that:
- Has extended radar range for detecting enemies
- Can move around the battlefield
- Cannot shoot (no weapons)
- Is a high-value target (winning condition tied to its survival)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Dict, Any

from .base import Entity
from ..core.types import Team, GridPos, EntityKind, MoveDir
from ..core.actions import Action
from ..core.validation import validate_action_in_world

if TYPE_CHECKING: # False at run-time
    from ..world.world import WorldState


@dataclass
class AWACS(Entity):
    """
    An airborne early warning and control aircraft.

    AWACS provides extended radar coverage for its team but cannot attack.
    The game ends when an AWACS is destroyed.

    Radar range must be explicitly specified when creating an AWACS as a keyword argument.
    Type-defining attributes (kind, can_move, can_shoot) have defaults.
    """

    # Type-defining fields (have defaults)
    kind: EntityKind = EntityKind.AWACS
    can_move: bool = True
    can_shoot: bool = False  # AWACS cannot attack
    
    # Instance-specific stats (NO defaults - must be specified)
    # Using kw_only to allow required fields after fields with defaults
    radar_range: float = field(kw_only=True)

    def get_allowed_actions(self, world: WorldState) -> List[Action]:
        """
        Get all actions this AWACS can perform that are feasible.
        
        This checks:
        - Entity-level constraints (alive, can_move)
        - World-state constraints that are knowable (bounds)
        
        AWACS can only wait or move - it has no weapons.
        """
        if not self.alive:
            return []

        actions = []

        wait_action = Action.wait()
        if validate_action_in_world(world, self, wait_action).valid:
            actions.append(wait_action)

        # AWACS can move - only include moves that stay in bounds
        if self.can_move:
            for direction in MoveDir:
                dx, dy = direction.delta
                new_pos = (self.pos[0] + dx, self.pos[1] + dy)
                move_action = Action.move(direction)
                if validate_action_in_world(world, self, move_action).valid:
                    actions.append(move_action)

        # No shooting - AWACS is unarmed
        return actions

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> AWACS:
        """Construct AWACS from dictionary."""
        return cls(
            team=Team[data["team"]],
            pos=tuple(data["pos"]),
            name=data["name"],
            radar_range=data["radar_range"],
        )
