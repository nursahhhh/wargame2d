"""
SAM entity - Surface-to-Air Missile system.

SAMs are defensive units that:
- Cannot move (stationary)
- Can toggle radar on/off
- Only visible when radar is ON
- Have a cooldown after firing
- Can shoot at aircraft within range
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, TYPE_CHECKING, Dict, Any

from .base import Entity
from ..core.types import Team, GridPos, EntityKind, ActionValidation
from ..core.actions import Action
from ..core.validation import validate_action_in_world

if TYPE_CHECKING:
    from ..world.world import WorldState


@dataclass
class SAM(Entity):
    """
    A stationary surface-to-air missile system.

    SAMs have unique mechanics:
    - Cannot move (stationary defense)
    - Radar can be toggled on/off
    - Only detectable by enemies when radar is ON
    - Has a cooldown period after firing
    - Good range and missile capacity

    All stats must be explicitly specified when creating a SAM as keyword arguments.

    Attributes:
        missiles: Number of missiles available (required)
        radar_range: Detection range when ON (required)
        missile_max_range: Maximum firing range (required)
        base_hit_prob: Hit probability at close range (required)
        min_hit_prob: Minimum hit probability at max range (required)
        cooldown_steps: Turns to wait after firing (required)
        on: Whether radar is currently active (default: False)
        _cooldown: Current cooldown counter (internal, default: 0)
    """

    # Type-defining fields (have defaults)
    kind: EntityKind = EntityKind.SAM
    can_move: bool = False  # SAMs are stationary
    can_shoot: bool = True

    # SAM-specific mechanics (have reasonable defaults)
    on: bool = False  # Radar starts OFF
    _cooldown: int = 0  # Current cooldown counter
    
    # Instance-specific stats (NO defaults - must be specified)
    # Using kw_only to allow required fields after fields with defaults
    missiles: int = field(kw_only=True)
    radar_range: float = field(kw_only=True)
    missile_max_range: float = field(kw_only=True)
    base_hit_prob: float = field(kw_only=True)
    min_hit_prob: float = field(kw_only=True)
    cooldown_steps: int = field(kw_only=True)

    def __post_init__(self):
        """Validate SAM-specific parameters."""
        super().__post_init__()
        if self.missiles < 0:
            raise ValueError(f"Missiles cannot be negative: {self.missiles}")
        if self.missile_max_range <= 0:
            raise ValueError(f"Missile range must be positive: {self.missile_max_range}")
        if self.cooldown_steps < 0:
            raise ValueError(f"Cooldown steps cannot be negative: {self.cooldown_steps}")

    def get_active_radar_range(self) -> float:
        """
        SAM radar is only active when turned ON.

        This is critical for stealth mechanics - SAMs are invisible
        to enemies when their radar is off.

        Returns:
            Radar range if ON, 0.0 if OFF
        """
        return self.radar_range if self.on else 0.0

    def get_allowed_actions(self, world: WorldState) -> List[Action]:
        """
        Get all actions this SAM can perform that are feasible.
        
        This checks:
        - Entity-level constraints (alive, radar on, cooldown, missiles)
        - World-state constraints that are knowable (range, visibility)
        
        SAMs can:
        - Always toggle radar on/off
        - Shoot when: radar ON, not cooling down, has missiles, target in range
        - Cannot move (stationary)
        """
        if not self.alive:
            return []

        actions = []

        wait_action = Action.wait()
        if validate_action_in_world(world, self, wait_action).valid:
            actions.append(wait_action)

        # SAMs can always toggle their radar
        toggle_action = Action.toggle(on=not self.on)
        if validate_action_in_world(world, self, toggle_action).valid:
            actions.append(toggle_action)

        # Can only shoot if radar is ON, not cooling down, and has missiles
        if self.on and self._cooldown == 0 and self.missiles > 0:
            view = world.get_team_view(self.team)
            visible_enemy_ids = view.get_enemy_ids(self.team)
            
            # Only include targets in range
            for target_id in visible_enemy_ids:
                target = world.get_entity(target_id)
                if target and target.alive:
                    shoot_action = Action.shoot(target_id)
                    if validate_action_in_world(world, self, shoot_action).valid:
                        actions.append(shoot_action)

        return actions
    
    def _validate_shoot(self, world: WorldState, action: Action) -> ActionValidation:
        """
        Validate a SHOOT action for SAM (includes SAM-specific checks).
        
        SAMs have additional requirements:
        - Radar must be ON
        - Must not be cooling down
        """
        # First check base requirements (alive, can_shoot, missiles, target_id)
        base_validation = super()._validate_shoot(world, action)
        if not base_validation.valid:
            return base_validation
        
        # SAM-specific: radar must be ON
        if not self.on:
            return ActionValidation.fail(
                "RADAR_OFF",
                f"{self.label()} radar is OFF"
            )
        
        # SAM-specific: must not be cooling down
        if self._cooldown > 0:
            return ActionValidation.fail(
                "COOLING_DOWN",
                f"{self.label()} cooling down ({self._cooldown} turns remaining)"
            )
        
        return ActionValidation.success()

    def tick_cooldown(self) -> None:
        """
        Decrease cooldown counter by 1 (called each turn).

        This should be called by the World during housekeeping phase.
        """
        if self._cooldown > 0:
            self._cooldown -= 1

    def start_cooldown(self) -> None:
        """
        Start the cooldown timer after firing.

        This should be called by CombatResolver after a successful shot.
        """
        self._cooldown = self.cooldown_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SAM to dictionary."""
        data = super().to_dict()
        # Add SAM-specific fields
        data.update({
            "missiles": self.missiles,
            "missile_max_range": self.missile_max_range,
            "base_hit_prob": self.base_hit_prob,
            "min_hit_prob": self.min_hit_prob,
            "on": self.on,
            "cooldown_steps": self.cooldown_steps,
            "_cooldown": self._cooldown,
        })
        return data

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> SAM:
        """Construct SAM from dictionary."""
        sam = cls(
            team=Team[data["team"]],
            pos=tuple(data["pos"]),
            name=data["name"],
            radar_range=data["radar_range"],
            missiles=data["missiles"],
            missile_max_range=data["missile_max_range"],
            base_hit_prob=data["base_hit_prob"],
            min_hit_prob=data["min_hit_prob"],
            on=data["on"],
            cooldown_steps=data["cooldown_steps"],
        )
        # Restore cooldown counter
        sam._cooldown = data["_cooldown"]
        return sam
