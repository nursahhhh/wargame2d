import json
import math
from dataclasses import dataclass, field
from typing import Literal, List, Annotated, Union, Optional, Dict, Any, Tuple
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field

from wargame_2d.agents.fixed_prompts import EXECUTER_TASK, GAME_INFO, STRATEGIC_GUIDELINES
from wargame_2d.agents.movement_tool import calculate_movement
from wargame_2d.agents.state_and_data import GameDeps
import set_api_key


# --- Action definitions ---
class MoveAction(BaseModel):
    type: Literal["MOVE"]
    entity_id: int = Field(description="ID of the unit to move")
    direction: Literal["UP", "DOWN", "LEFT", "RIGHT"]

class ShootAction(BaseModel):
    type: Literal["SHOOT"]
    entity_id: int
    target_id: int

class WaitAction(BaseModel):
    type: Literal["WAIT"]
    entity_id: int

class ToggleAction(BaseModel):
    type: Literal["TOGGLE"]
    entity_id: int
    on: bool

Action = Annotated[
    Union[MoveAction, ShootAction, WaitAction, ToggleAction],
    Field(discriminator="type")
]

# --- Entity-level reasoning wrapper ---
class EntityAction(BaseModel):
    reasoning: str = Field(description="Short explanation for why this action is chosen for this entity")
    action: Action = Field(description="The action to execute")

class TacticalPlan(BaseModel):
    analysis: str = Field(description="Step-by-step analysis of the current game state, threats, and opportunities and how to blend the given director strategy with those.")
    strategy: str = Field(description="Mid to low level tactics broken-down from the high level strategy.")
    entity_actions: List[EntityAction] = Field(description="List of actions and reasoning per entity")
    confidence: Optional[float] = Field(ge=0, le=1, description="Confidence score how how well the plan works so far 1 means everything works perfect and you have full trust on winning, 0 means everything is going out of plan and it looks like we'll lose.")

executer_agent = Agent(
    "openai:gpt-5-mini",
    deps_type=GameDeps,
    output_type=TacticalPlan,
    instructions="You are an AI field commander leading the RED team in a grid-based air combat simulation with given strategical orders"
)

@executer_agent.instructions
def full_prompt(ctx: RunContext[GameDeps]) -> str:

    return f"""
### TASK
{EXECUTER_TASK}

{GAME_INFO}

### GAME STATE 
{ctx.deps.game_state}
"""


# @executer_agent.tool
# def move_towards_or_away(
#         ctx: RunContext[GameDeps],
#         source_id: int,
#         target_id: Optional[int] = None,
#         target_x: Optional[int] = None,
#         target_y: Optional[int] = None,
#         direction: str = "towards"
# ) -> Dict[str, Any]:
#     """
#     Calculate the best movement action(s) to move a unit towards or away from a target.
#
#     Determines optimal movement directions for a unit to either approach or retreat from
#     a target entity or position. Considers blocked positions and provides multiple equally
#     good options when applicable.
#
#     Args:
#         source_id: The ID of the entity that should move.
#         target_id: The ID of the target entity (optional if target_x/y provided).
#         target_x: The x-coordinate of the target position (optional if target_id provided).
#         target_y: The y-coordinate of the target position (optional if target_id provided).
#         direction: Movement strategy - either "towards" or "away" (default: "towards").
#
#     Returns:
#         Dict containing recommended_action, all_options, current_distance, positions, and status.
#         The recommended_action can be directly used as a move action.
#
#     Example:
#         >>> move_towards_or_away(ctx, source_id=12, target_id=45, direction="towards")
#         {'recommended_action': {'type': 'MOVE', 'direction': 'RIGHT'}, ...}
#     """
#     return calculate_movement(ctx, source_id, target_id, target_x, target_y, direction)

# @executer_agent.tool
# def move_towards(ctx: RunContext[GameDeps], source_id: int, target_x: int, target_y: int) -> Dict[str, Any]:
#     """
#     Calculate the best movement action to move a unit closer to a target position.
#
#     This tool determines the optimal direction (UP, DOWN, LEFT, RIGHT) to move a unit
#     towards a specified target position by analyzing the distance on each axis and
#     choosing the direction that reduces the total distance most effectively.
#
#     Args:
#         source_id: The ID of the entity that should move towards the target.
#         target_x: The x-coordinate of the target position.
#         target_y: The y-coordinate of the target position.
#
#     Returns:
#         Dict[str, Any]: An action dictionary with the following structure:
#             - {"type": "MOVE", "direction": "UP"|"DOWN"|"LEFT"|"RIGHT"} if movement is possible
#             - {"type": "WAIT"} if already adjacent (distance <= 1) or no valid move exists
#
#     Example:
#         >>> move_towards(ctx, source_id=12, target_x=15, target_y=8)
#         {"type": "MOVE", "direction": "RIGHT"}
#
#     Notes:
#         - If the unit is already within distance 1 of the target, returns WAIT action
#         - Prioritizes the axis with the larger distance difference
#         - If the preferred direction is blocked, tries alternative directions
#         - Returns WAIT if all directions are blocked or out of bounds
#     """
#     world = ctx.deps.world
#
#     # Get the source entity
#     entity = world.entities_by_id.get(source_id)
#     if not entity or not entity.alive or not entity.can_move:
#         return {"type": "WAIT"}
#
#     source_pos = entity.pos
#
#     # Calculate distance
#     dx = target_x - source_pos[0]
#     dy = target_y - source_pos[1]
#     distance = math.hypot(dx, dy)
#
#     # If already adjacent or at target, wait
#     if distance <= 1:
#         return {"type": "WAIT"}
#
#     # Determine primary direction based on larger distance component
#     direction_map = {
#         "UP": (0, -1),
#         "DOWN": (0, 1),
#         "LEFT": (-1, 0),
#         "RIGHT": (1, 0),
#     }
#
#     # Choose direction that reduces distance most
#     if abs(dx) >= abs(dy):
#         # Move horizontally first
#         primary_direction = "RIGHT" if dx > 0 else "LEFT"
#         alternative_directions = ["UP", "DOWN", "RIGHT" if dx <= 0 else "LEFT"]
#     else:
#         # Move vertically first
#         primary_direction = "DOWN" if dy > 0 else "UP"
#         alternative_directions = ["LEFT", "RIGHT", "DOWN" if dy <= 0 else "UP"]
#
#     # Try primary direction
#     delta = direction_map[primary_direction]
#     new_pos = (source_pos[0] + delta[0], source_pos[1] + delta[1])
#     if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
#         return {"type": "MOVE", "direction": primary_direction}
#
#     # Try alternative directions
#     for alt_dir in alternative_directions:
#         delta = direction_map[alt_dir]
#         new_pos = (source_pos[0] + delta[0], source_pos[1] + delta[1])
#         if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
#             return {"type": "MOVE", "direction": alt_dir}
#
#     # All directions blocked
#     return {"type": "WAIT"}
#
#
# @executer_agent.tool
# def move_away_from(ctx: RunContext[GameDeps], source_id: int, target_x: int, target_y: int) -> Dict[str, Any]:
#     """
#     Calculate the best movement action to move a unit away from a target position.
#
#     This tool determines the optimal direction (UP, DOWN, LEFT, RIGHT) to move a unit
#     away from a specified target position by analyzing the distance on each axis and
#     choosing the direction that increases the total distance most effectively.
#
#     Args:
#         source_id: The ID of the entity that should move away from the target.
#         target_x: The x-coordinate of the position to avoid.
#         target_y: The y-coordinate of the position to avoid.
#
#     Returns:
#         Dict[str, Any]: An action dictionary with the following structure:
#             - {"type": "MOVE", "direction": "UP"|"DOWN"|"LEFT"|"RIGHT"} if movement is possible
#             - {"type": "WAIT"} if no valid move exists or already at maximum safe distance
#
#     Example:
#         >>> move_away_from(ctx, source_id=12, target_x=15, target_y=8)
#         {"type": "MOVE", "direction": "LEFT"}
#
#     Notes:
#         - Prioritizes the axis with the larger distance difference to maximize separation
#         - If the preferred direction is blocked, tries alternative directions
#         - Useful for retreating from threats or maintaining safe distance from enemies
#         - Returns WAIT if all directions are blocked or out of bounds
#     """
#     world = ctx.deps.world
#
#     # Get the source entity
#     entity = world.entities_by_id.get(source_id)
#     if not entity or not entity.alive or not entity.can_move:
#         return {"type": "WAIT"}
#
#     source_pos = entity.pos
#
#     # Calculate distance components
#     dx = target_x - source_pos[0]
#     dy = target_y - source_pos[1]
#
#     # Determine primary direction based on larger distance component
#     # Move in OPPOSITE direction to increase distance
#     direction_map = {
#         "UP": (0, -1),
#         "DOWN": (0, 1),
#         "LEFT": (-1, 0),
#         "RIGHT": (1, 0),
#     }
#
#     # Choose direction that increases distance most
#     if abs(dx) >= abs(dy):
#         # Move horizontally away
#         primary_direction = "LEFT" if dx > 0 else "RIGHT"  # Opposite of towards
#         alternative_directions = ["UP", "DOWN", "LEFT" if dx <= 0 else "RIGHT"]
#     else:
#         # Move vertically away
#         primary_direction = "UP" if dy > 0 else "DOWN"  # Opposite of towards
#         alternative_directions = ["LEFT", "RIGHT", "UP" if dy <= 0 else "DOWN"]
#
#     # Try primary direction
#     delta = direction_map[primary_direction]
#     new_pos = (source_pos[0] + delta[0], source_pos[1] + delta[1])
#     if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
#         return {"type": "MOVE", "direction": primary_direction}
#
#     # Try alternative directions (prefer perpendicular movement)
#     for alt_dir in alternative_directions:
#         delta = direction_map[alt_dir]
#         new_pos = (source_pos[0] + delta[0], source_pos[1] + delta[1])
#         if world.in_bounds(new_pos) and not world.is_occupied(new_pos):
#             return {"type": "MOVE", "direction": alt_dir}
#
#     # All directions blocked
#     return {"type": "WAIT"}