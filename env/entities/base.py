from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict, Any, Type

from ..core.types import Team, GridPos, EntityKind, ActionValidation, ActionType, MoveDir
from ..core.actions import Action
from ..utils.id_generator import get_next_entity_id

if TYPE_CHECKING:
    from ..world.world import WorldState


@dataclass
class Entity(ABC):
    """
    Abstract base class for all game entities.

    Entities exist on the grid, belong to a team, and can perform actions.
    Subclasses must implement get_allowed_actions() to define their behavior.
    """

    # Required attributes (NO defaults)
    team: Team
    pos: GridPos

    # Entity metadata (have defaults)
    kind: EntityKind = EntityKind.UNKNOWN
    name: Optional[str] = None

    # Auto-generated / internal state (have defaults)
    id: int = field(default_factory=get_next_entity_id)
    alive: bool = True
    last_action: Optional[Action] = None

    # Capability flags (have defaults, set by subclasses)
    can_move: bool = True
    can_shoot: bool = False
    
    # NOTE: radar_range removed from base - defined in subclasses as required

    def __post_init__(self):
        """Validate entity after initialization."""
        if self.radar_range < 0:
            raise ValueError(f"Radar range cannot be negative: {self.radar_range}")

    @abstractmethod
    def get_allowed_actions(self, world: WorldState) -> List[Action]:
        """
        Get list of valid actions this entity can perform.

        Args:
            world: Current world state (for checking visibility, etc.)

        Returns:
            List of Action objects this entity can perform.
            Returns empty list if entity is dead.
        """
        pass

    def validate_action(self, world: WorldState, action: Action) -> ActionValidation:
        """
        Efficiently validate if this entity can perform an action.
        
        This method performs entity-level validation without generating all possible
        actions. It checks:
        - Entity state (alive, capabilities)
        - Resource availability (missiles)
        - Entity-specific constraints (SAM radar/cooldown)
        - Basic parameter validity
        
        World-state dependent checks (range, bounds, collisions) are handled
        by the resolvers.
        
        Args:
            world: Current world state
            action: The action to validate
        
        Returns:
            ActionValidation with detailed result
        """
        # Check if alive
        if not self.alive:
            return ActionValidation.fail(
                "ENTITY_DEAD",
                f"{self.label()} is dead and cannot act"
            )
        
        # Validate based on action type
        if action.type == ActionType.WAIT:
            return ActionValidation.success(f"{self.label()} can wait")
        
        elif action.type == ActionType.MOVE:
            return self._validate_move(world, action)
        
        elif action.type == ActionType.SHOOT:
            return self._validate_shoot(world, action)
        
        elif action.type == ActionType.TOGGLE:
            return self._validate_toggle(world, action)
        
        else:
            return ActionValidation.fail(
                "UNKNOWN_ACTION",
                f"{self.label()} unknown action type {action.type}"
            )
    
    def _validate_move(self, world: WorldState, action: Action) -> ActionValidation:
        """Validate a MOVE action (entity-level checks only)."""
        if not self.can_move:
            return ActionValidation.fail(
                "NO_CAPABILITY",
                f"{self.label()} cannot move (immobile)"
            )
        
        direction = action.params.get("dir")
        if not isinstance(direction, MoveDir):
            return ActionValidation.fail(
                "INVALID_DIRECTION",
                f"{self.label()} invalid movement direction"
            )
        
        return ActionValidation.success()
    
    def _validate_shoot(self, world: WorldState, action: Action) -> ActionValidation:
        """Validate a SHOOT action (entity-level checks only)."""
        if not self.can_shoot:
            return ActionValidation.fail(
                "NO_CAPABILITY",
                f"{self.label()} cannot shoot (no weapons)"
            )
        
        # Check if entity has missiles attribute (Aircraft, SAM)
        if not hasattr(self, 'missiles'):
            return ActionValidation.fail(
                "NO_CAPABILITY",
                f"{self.label()} has no weapon implementation"
            )
        
        if self.missiles <= 0:
            return ActionValidation.fail(
                "NO_MISSILES",
                f"{self.label()} has no missiles"
            )
        
        # Check target_id parameter exists
        target_id = action.params.get("target_id")
        if not isinstance(target_id, int):
            return ActionValidation.fail(
                "INVALID_TARGET",
                f"{self.label()} invalid target ID"
            )
        
        # Subclasses can override for additional checks (SAM radar, cooldown)
        return ActionValidation.success()
    
    def _validate_toggle(self, world: WorldState, action: Action) -> ActionValidation:
        """Validate a TOGGLE action (entity-level checks only)."""
        from ..entities.sam import SAM
        
        if not isinstance(self, SAM):
            return ActionValidation.fail(
                "NOT_SAM",
                f"{self.label()} cannot toggle (not a SAM)"
            )
        
        on = action.params.get("on")
        if not isinstance(on, bool):
            return ActionValidation.fail(
                "INVALID_TOGGLE",
                f"{self.label()} invalid toggle parameter"
            )
        
        return ActionValidation.success()

    def get_active_radar_range(self) -> float:
        """
        Get the current effective radar range.

        This can be overridden by subclasses (e.g., SAM radar on/off).
        By default, returns the base radar_range.

        Returns:
            Current radar range (0 if radar is inactive)
        """
        return self.radar_range

    def label(self) -> str:
        """
        Get a human-readable label for this entity.

        Returns:
            String like "Aircraft#1(BLUE)" or "awacs#3(RED)"
        """
        display_name = self.name if self.name else self.kind.value
        return f"{display_name}#{self.id}({self.team.name})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize entity to dictionary.

        Subclasses can override to add type-specific fields.

        Returns:
            JSON-serializable dictionary
        """
        data = {
            "type": self.__class__.__name__,
            "id": self.id,
            "team": self.team.name,
            "pos": list(self.pos),
            "kind": self.kind.value,
            "name": self.name,
            "alive": self.alive,
            "radar_range": self.radar_range,
            "can_move": self.can_move,
            "can_shoot": self.can_shoot,
        }

        # Serialize last action if present
        if self.last_action:
            data["last_action"] = self.last_action.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Entity:
        """
        Deserialize entity from dictionary.

        This is a factory method that creates the correct entity subclass.
        Subclasses should override _from_dict_impl() instead.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed entity (correct subclass)
        """
        from ..core.actions import Action

        # Get the correct entity class
        entity_cls = cls._get_entity_class(data["type"])

        # Let subclass build itself
        entity = entity_cls._from_dict_impl(data)

        # Restore base entity state
        entity.id = data["id"]
        entity.alive = data["alive"]

        if "last_action" in data:
            entity.last_action = Action.from_dict(data["last_action"])

        return entity

    @classmethod
    def _get_entity_class(cls, type_name: str) -> Type[Entity]:
        """
        Get entity class from type name.

        Args:
            type_name: Name of entity class (e.g., "Aircraft")

        Returns:
            Entity class

        Raises:
            ValueError: If type name is unknown
        """
        # Import here to avoid circular imports
        from . import Aircraft, AWACS, SAM, Decoy

        entity_classes = {
            "Aircraft": Aircraft,
            "AWACS": AWACS,
            "SAM": SAM,
            "Decoy": Decoy,
        }

        if type_name not in entity_classes:
            raise ValueError(f"Unknown entity type: {type_name}")

        return entity_classes[type_name]


    @classmethod
    @abstractmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Entity:
        """
        Construct entity from dictionary data.

        Subclasses MUST implement this to handle their specific fields.

        Args:
            data: Dictionary from to_dict()

        Returns:
            New entity instance (id and alive will be set by from_dict)
        """
        pass  # Force subclasses to implement

    def to_json(self) -> str:
        """Serialize entity to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Entity:
        """Deserialize entity from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """String representation for debugging."""
        status = "alive" if self.alive else "dead"
        return f"{self.label()} at {self.pos} [{status}]"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"{self.__class__.__name__}(id={self.id}, team={self.team}, "
                f"pos={self.pos}, alive={self.alive})")