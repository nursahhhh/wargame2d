from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Dict, Any

from .base import Entity
from ..core.types import Team, GridPos, EntityKind, MoveDir
from ..core.actions import Action
from ..core.validation import validate_action_in_world

if TYPE_CHECKING:
    from ..world.world import WorldState


@dataclass
class Aircraft(Entity):
    """
    A mobile fighter aircraft with missiles and radar.
    
    All stats must be explicitly specified when creating an aircraft.
    Type-defining attributes (kind, can_move, can_shoot) have defaults.
    Combat stats must be provided explicitly as keyword arguments.
    """
    
    # Type-defining fields (have defaults)
    kind: EntityKind = EntityKind.AIRCRAFT
    can_move: bool = True
    can_shoot: bool = True
    
    # Instance-specific stats (NO defaults - must be specified)
    # Using kw_only to allow required fields after fields with defaults
    missiles: int = field(kw_only=True)
    radar_range: float = field(kw_only=True) # it actually has default on base class, but we want to force user to specify it
    missile_max_range: float = field(kw_only=True)
    base_hit_prob: float = field(kw_only=True)
    min_hit_prob: float = field(kw_only=True)
    
    def __post_init__(self):
        """Validate aircraft-specific parameters."""
        super().__post_init__()
        if self.missiles < 0:
            raise ValueError(f"Missiles cannot be negative: {self.missiles}")
        if self.missile_max_range <= 0:
            raise ValueError(f"Missile range must be positive: {self.missile_max_range}")
        if not 0 <= self.base_hit_prob <= 1:
            raise ValueError(f"Base hit probability must be in [0,1]: {self.base_hit_prob}")
        if not 0 <= self.min_hit_prob <= 1:
            raise ValueError(f"Min hit probability must be in [0,1]: {self.min_hit_prob}")
    
    def get_allowed_actions(self, world: WorldState) -> List[Action]:
        """
        Get all actions this aircraft can perform that are feasible.
        
        This checks:
        - Entity-level constraints (alive, capabilities, resources)
        - World-state constraints that are knowable (bounds, range, visibility)
        
        Does NOT check:
        - Position occupation (resolved simultaneously with other entities)
        """
        if not self.alive:
            return []  # Dead entities can't act
        
        actions = []

        wait_action = Action.wait()
        if validate_action_in_world(world, self, wait_action).valid:
            actions.append(wait_action)
        
        # Movement - only include moves that stay in bounds
        if self.can_move:
            for direction in MoveDir:
                move_action = Action.move(direction)
                if validate_action_in_world(world, self, move_action).valid:
                    actions.append(move_action)
        
        # Shooting - only include targets in range and visible
        if self.can_shoot and self.missiles > 0:
            view = world.get_team_view(self.team)
            visible_enemy_ids = view.get_enemy_ids(self.team)
            
            for target_id in visible_enemy_ids:
                target = world.get_entity(target_id)
                if target and target.alive:
                    shoot_action = Action.shoot(target_id)
                    if validate_action_in_world(world, self, shoot_action).valid:
                        actions.append(shoot_action)
        
        return actions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize aircraft to dictionary."""
        data = super().to_dict()
        # Add aircraft-specific fields
        data.update({
            "missiles": self.missiles,
            "missile_max_range": self.missile_max_range,
            "base_hit_prob": self.base_hit_prob,
            "min_hit_prob": self.min_hit_prob,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Aircraft:
        """Construct aircraft from dictionary."""
        return cls(
            team=Team[data["team"]],
            pos=tuple(data["pos"]),
            name=data["name"],
            radar_range=data["radar_range"],
            missiles=data["missiles"],
            missile_max_range=data["missile_max_range"],
            base_hit_prob=data["base_hit_prob"],
            min_hit_prob=data["min_hit_prob"],
        )
