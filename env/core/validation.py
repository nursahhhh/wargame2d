"""
Shared action validation helpers.

This module combines entity-level validation with world-dependent checks so
both the resolvers and any "what can I do?" queries use the same rules.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import ActionValidation, ActionType

if TYPE_CHECKING:
    from ..world.world import WorldState
    from ..entities.base import Entity
    from .actions import Action


def validate_action_in_world(world: WorldState, entity: Entity, action: Action) -> ActionValidation:
    """
    Validate an action using both entity-level and world-dependent rules.

    This reuses `entity.validate_action` for static checks (alive, ammo,
    cooldown, parameters) and then applies dynamic rules that depend on
    the current world state (bounds, target validity, visibility, range).
    """
    base_validation = entity.validate_action(world, action)
    if not base_validation.valid:
        return base_validation

    if action.type == ActionType.MOVE:
        direction = action.params.get("dir")
        dx, dy = direction.delta
        new_pos = (entity.pos[0] + dx, entity.pos[1] + dy)
        if not world.grid.in_bounds(new_pos):
            return ActionValidation.fail(
                "OUT_OF_BOUNDS",
                f"{entity.label()} cannot move {direction.name} (out of bounds)"
            )
        return ActionValidation.success()

    if action.type == ActionType.SHOOT:
        target_id = action.params.get("target_id")
        target = world.get_entity(target_id)
        if not target or not target.alive:
            return ActionValidation.fail(
                "INVALID_TARGET",
                f"{entity.label()} target invalid or dead"
            )

        team_view = world.get_team_view(entity.team)
        if not team_view.can_target(target_id):
            return ActionValidation.fail(
                "NOT_VISIBLE",
                f"{entity.label()} cannot target {target.label()} (not observed)"
            )

        max_range = getattr(entity, "missile_max_range", None)
        if max_range is None or max_range <= 0:
            return ActionValidation.fail(
                "NO_CAPABILITY",
                f"{entity.label()} has no missile range configured"
            )

        distance = world.grid.distance(entity.pos, target.pos)
        if distance > max_range:
            return ActionValidation.fail(
                "OUT_OF_RANGE",
                f"{entity.label()} target out of range ({distance:.1f} > {max_range:.1f})"
            )

        return ActionValidation.success()

    # WAIT / TOGGLE (nothing dynamic beyond entity-level for now)
    return base_validation
