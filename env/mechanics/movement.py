"""
MovementResolver - Movement action resolution.

This module handles:
- Validating movement actions
- Checking bounds and collision
- Applying position changes
- Generating movement logs
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Tuple
from dataclasses import dataclass

from ..core.types import ActionType
from ..core.actions import Action
from ..core.validation import validate_action_in_world

if TYPE_CHECKING:
    from ..world.world import WorldState
    from ..entities.base import Entity


@dataclass
class MovementResult:
    """
    Result of resolving a single movement action.
    
    Attributes:
        entity_id: ID of entity that moved (or tried to)
        success: Whether movement succeeded
        old_pos: Position before movement
        new_pos: Position after movement (same as old if failed)
        failure_reason: Optional machine-readable reason code when movement fails
    """
    entity_id: int
    success: bool
    old_pos: tuple[int, int]
    new_pos: tuple[int, int]
    failure_reason: str | None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize movement result to a plain dict."""
        return {
            "entity_id": self.entity_id,
            "success": self.success,
            "old_pos": self.old_pos,
            "new_pos": self.new_pos,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MovementResult":
        """Deserialize a movement result from a dict."""
        return cls(
            entity_id=data["entity_id"],
            success=data["success"],
            old_pos=tuple(data["old_pos"]),
            new_pos=tuple(data["new_pos"]),
            failure_reason=data.get("failure_reason"),
        )


@dataclass
class ActionResolutionResult:
    """
    Complete result of resolving all actions for a turn.
    
    Attributes:
        movement_results: Results from all movement actions
        logs: Combined logs in execution order (including skipped/invalid)
        movement_occurred: True if at least one entity successfully moved
    """
    movement_results: List[MovementResult]
    logs: List[str]
    movement_occurred: bool

    # Maybe add here a serialization/de-serialization logic.
    # Also I might want to see all logs in a single list.

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the action resolution result to a dict."""
        return {
            "movement_results": [r.to_dict() for r in self.movement_results],
            "logs": self.logs,
            "movement_occurred": self.movement_occurred,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionResolutionResult":
        """Deserialize an action resolution result from a dict."""
        return cls(
            movement_results=[MovementResult.from_dict(r) for r in data.get("movement_results", [])],
            logs=data.get("logs", []),
            movement_occurred=data.get("movement_occurred", False),
        )


class MovementResolver:
    """
    Stateless resolver for movement actions.
    
    The MovementResolver:
    - Validates movement requests
    - Checks grid bounds
    - Checks collisions
    - Applies position changes
    - Generates logs
    
    All methods are stateless - they don't modify the resolver itself.
    """
    
    def __init__(self):
        """Initialize the movement resolver."""
        pass  # Stateless, no initialization needed
    
    def resolve_actions(
        self,
        world: WorldState,
        actions: Mapping[int, Action],
        randomize_order: bool = True
    ) -> ActionResolutionResult:
        """
        Resolve all actions (movement, toggles, waits) for a turn.
        
        This is the main entry point for action resolution. It handles
        all action types in a single pass and returns comprehensive results.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> Action. Non-Action values are ignored with a log.
            randomize_order: If True, shuffle movement order to prevent ID bias
        
        Returns:
            ActionResolutionResult with all outcomes
        """
        alive_entities = {entity.id: entity for entity in world.get_alive_entities()}

        movement_queue: List[tuple[Entity, Action]] = []
        movement_results: List[MovementResult] = []
        logs: List[str] = []

        for entity_id, action in actions.items():
            if not isinstance(action, Action):
                logs.append(f"Invalid action for entity {entity_id}; ignoring")
                continue

            entity = alive_entities.get(entity_id)
            if entity is None:
                logs.append(f"Action provided for unknown or dead entity {entity_id}; ignoring")
                continue

            if action.type == ActionType.MOVE:
                movement_queue.append((entity, action))
            elif action.type == ActionType.TOGGLE:
                logs.append(self._resolve_toggle(entity, action))
            elif action.type == ActionType.WAIT:
                logs.append(f"{entity.label()} waits")
            else:
                # SHOOT and other action types are handled elsewhere
                logs.append(
                    f"{entity.label()} action {action.type.name} ignored in movement phase"
                )

        # Randomize movement order to prevent ID bias
        if randomize_order:
            world.rng.shuffle(movement_queue)

        # Process each movement
        for entity, action in movement_queue:
            result, log_message = self.resolve_single(world, entity, action)
            movement_results.append(result)
            logs.append(log_message)

        movement_occurred = any(result.success for result in movement_results)

        # Update movement counter
        if movement_occurred:
            world.turns_without_movement = 0
        else:
            world.turns_without_movement += 1

        return ActionResolutionResult(
            movement_results=movement_results,
            logs=logs,
            movement_occurred=movement_occurred
        )
    
    def resolve_single(
        self, 
        world: WorldState, 
        entity: Entity, 
        action: Action
    ) -> Tuple[MovementResult, str]:
        """
        Resolve a single movement action.
        
        This method now uses shared validation first (entity-level + bounds),
        then performs world-state checks (collisions).
        
        Args:
            world: Current world state (modified in-place)
            entity: Entity attempting to move
            action: Movement action
        
        Returns:
            Tuple of (MovementResult, log message)
        """
        old_pos = entity.pos
        
        # Use shared validation first (checks alive, can_move, direction, bounds)
        validation = validate_action_in_world(world, entity, action)
        if not validation.valid:
            result = MovementResult(
                entity_id=entity.id,
                success=False,
                old_pos=old_pos,
                new_pos=old_pos,
                failure_reason=validation.error_code
            )
            return result, validation.message
        
        # Extract direction (we know it's valid from validation)
        direction = action.params.get("dir")
        
        # Calculate new position
        dx, dy = direction.delta
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # Check collision - TRULY DYNAMIC (position might have just been occupied)
        if world.is_position_occupied(new_pos):
            log_message = f"{entity.label()} blocked by another entity at {new_pos}"
            result = MovementResult(
                entity_id=entity.id,
                success=False,
                old_pos=old_pos,
                new_pos=old_pos,
                failure_reason="COLLISION"
            )
            return result, log_message
        
        # Movement is valid - apply it
        entity.pos = new_pos
        
        result = MovementResult(
            entity_id=entity.id,
            success=True,
            old_pos=old_pos,
            new_pos=new_pos,
            failure_reason=None
        )
        log_message = f"{entity.label()} moves {direction.name} to {new_pos}"
        return result, log_message
    
    def _resolve_toggle(self, entity: Entity, action: Action) -> str:
        """
        Resolve a single TOGGLE action (SAM radar).
        
        This method now uses entity-level validation first.
        
        Args:
            entity: Entity toggling
            action: Toggle action
        
        Returns:
            Log message
        """
        from ..entities.sam import SAM
        
        # Use entity-level validation (checks if SAM, valid parameter)
        # Note: We need a world reference, but toggle doesn't use it
        # This is a limitation - we'll handle it gracefully
        if not isinstance(entity, SAM):
            return f"{entity.label()} cannot toggle (not a SAM)"
        
        desired_state = action.params.get("on")
        if not isinstance(desired_state, bool):
            return f"{entity.label()} invalid toggle parameter"
        
        # Apply toggle
        entity.on = desired_state
        
        state_str = "ON" if desired_state else "OFF"
        return f"{entity.label()} radar toggled {state_str}"
    
