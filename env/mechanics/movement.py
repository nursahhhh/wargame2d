"""
MovementResolver - Movement action resolution.

This module handles:
- Validating movement actions
- Checking bounds and collision
- Applying position changes
- Generating movement logs
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Tuple
from dataclasses import dataclass

from ..core.types import ActionType, MoveDir
from ..core.actions import Action

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
        log: Human-readable log message
    """
    entity_id: int
    success: bool
    old_pos: tuple[int, int]
    new_pos: tuple[int, int]
    log: str


@dataclass
class ActionResolutionResult:
    """
    Complete result of resolving all actions for a turn.
    
    Attributes:
        movement_results: Results from all movement actions
        toggle_logs: Logs from all toggle actions
        wait_logs: Logs from all wait actions
        movement_occurred: True if at least one entity successfully moved
    """
    movement_results: List[MovementResult]
    toggle_logs: List[str]
    wait_logs: List[str]
    movement_occurred: bool


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
        actions: Dict[int, Action],
        randomize_order: bool = True
    ) -> ActionResolutionResult:
        """
        Resolve all actions (movement, toggles, waits) for a turn.
        
        This is the main entry point for action resolution. It handles
        all action types in a single pass and returns comprehensive results.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
            randomize_order: If True, shuffle movement order to prevent ID bias
        
        Returns:
            ActionResolutionResult with all outcomes
        """
        # Resolve movement
        movement_results = self.resolve_all(world, actions, randomize_order)
        
        # Resolve toggles
        toggle_logs = self.resolve_toggles(world, actions)
        
        # Resolve waits
        wait_logs = self.resolve_waits(world, actions)
        
        # Check if any movement occurred
        movement_occurred = self.has_movement_occurred(movement_results)
        
        # Update movement counter
        if movement_occurred:
            world.turns_without_movement = 0
        else:
            world.turns_without_movement += 1
        
        return ActionResolutionResult(
            movement_results=movement_results,
            toggle_logs=toggle_logs,
            wait_logs=wait_logs,
            movement_occurred=movement_occurred
        )
    
    def resolve_all(
        self, 
        world: WorldState, 
        actions: Dict[int, Action],
        randomize_order: bool = True
    ) -> List[MovementResult]:
        """
        Resolve all movement actions for a turn.
        
        Movement actions are processed in random order to prevent
        bias from entity ID ordering.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
            randomize_order: If True, shuffle order to prevent ID bias
        
        Returns:
            List of MovementResult objects (one per movement action)
        """
        results = []
        
        # Get all entities that want to move
        moving_entities = []
        for entity in world.get_alive_entities():
            action = actions.get(entity.id)
            if action and action.type == ActionType.MOVE:
                moving_entities.append(entity)
        
        # Randomize order to prevent ID bias
        if randomize_order:
            world.rng.shuffle(moving_entities)
        
        # Process each movement
        for entity in moving_entities:
            action = actions[entity.id]
            result = self.resolve_single(world, entity, action)
            results.append(result)
        
        return results
    
    def resolve_single(
        self, 
        world: WorldState, 
        entity: Entity, 
        action: Action
    ) -> MovementResult:
        """
        Resolve a single movement action.
        
        This method now uses entity-level validation first, then performs
        world-state checks (bounds, collisions).
        
        Args:
            world: Current world state (modified in-place)
            entity: Entity attempting to move
            action: Movement action
        
        Returns:
            MovementResult with outcome
        """
        old_pos = entity.pos
        
        # Use entity-level validation first (checks alive, can_move, direction)
        validation = entity.validate_action(world, action)
        if not validation.valid:
            return MovementResult(
                entity_id=entity.id,
                success=False,
                old_pos=old_pos,
                new_pos=old_pos,
                log=validation.message
            )
        
        # Extract direction (we know it's valid from validation)
        direction = action.params.get("dir")
        
        # Calculate new position
        dx, dy = direction.delta
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # Note: Bounds checking removed - get_allowed_actions() already filters
        # out-of-bounds moves. If you need defensive validation for actions that
        # didn't come from get_allowed_actions(), you can add it back here.
        
        # Check collision - TRULY DYNAMIC (position might have just been occupied)
        if world.is_position_occupied(new_pos):
            return MovementResult(
                entity_id=entity.id,
                success=False,
                old_pos=old_pos,
                new_pos=old_pos,
                log=f"{entity.label()} blocked by another entity at {new_pos}"
            )
        
        # Movement is valid - apply it
        entity.pos = new_pos
        entity.last_action = action
        
        return MovementResult(
            entity_id=entity.id,
            success=True,
            old_pos=old_pos,
            new_pos=new_pos,
            log=f"{entity.label()} moves {direction.name} to {new_pos}"
        )
    
    def resolve_toggles(
        self,
        world: WorldState,
        actions: Dict[int, Action]
    ) -> List[str]:
        """
        Resolve TOGGLE actions (SAM radar on/off).
        
        Toggles are processed separately from movement since they
        don't affect positioning.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
        
        Returns:
            List of log messages
        """
        logs = []
        
        for entity in world.get_alive_entities():
            action = actions.get(entity.id)
            if not action or action.type != ActionType.TOGGLE:
                continue
            
            # Handle toggle action
            log = self._resolve_toggle(entity, action)
            logs.append(log)
        
        return logs
    
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
        from ..world.world import WorldState
        
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
        entity.last_action = action
        
        state_str = "ON" if desired_state else "OFF"
        return f"{entity.label()} radar toggled {state_str}"
    
    def resolve_waits(
        self,
        world: WorldState,
        actions: Dict[int, Action]
    ) -> List[str]:
        """
        Resolve WAIT actions.
        
        Wait actions don't do anything, but we track them for logging
        and action history.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
        
        Returns:
            List of log messages
        """
        logs = []
        
        for entity in world.get_alive_entities():
            action = actions.get(entity.id)
            if not action or action.type != ActionType.WAIT:
                continue
            
            entity.last_action = action
            logs.append(f"{entity.label()} waits")
        
        return logs
    
    def has_movement_occurred(self, results: List[MovementResult]) -> bool:
        """
        Check if any movement actually happened this turn.
        
        Used for stagnation detection.
        
        Args:
            results: Movement results from resolve_all()
        
        Returns:
            True if at least one entity successfully moved
        """
        return any(result.success for result in results)
    
    def get_movement_summary(self, results: List[MovementResult]) -> Dict[str, int]:
        """
        Get summary statistics about movement.
        
        Args:
            results: Movement results from resolve_all()
        
        Returns:
            Dictionary with statistics
        """
        return {
            "attempted": len(results),
            "successful": sum(1 for r in results if r.success),
            "blocked": sum(1 for r in results if not r.success and "blocked" in r.log),
            "out_of_bounds": sum(1 for r in results if not r.success and "boundary" in r.log),
        }

