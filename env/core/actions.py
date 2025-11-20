"""
Action definitions and utilities.

Actions represent commands given to entities. This module provides:
- Action dataclass
- Action validation
- Action factory methods
- Action serialization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json

from .types import ActionType, MoveDir


@dataclass
class Action:
    """
    An action that can be performed by an entity.

    Actions consist of a type and optional parameters. The parameters
    are validated based on the action type.

    Use static factory methods for convenient construction:
        - Action.wait()
        - Action.move(direction)
        - Action.shoot(target_id)
        - Action.toggle(on)

    Or construct directly:
        - Action(ActionType.WAIT)
        - Action(ActionType.MOVE, {"dir": MoveDir.UP})
    """

    type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate action parameters after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate that parameters match the action type.

        Raises:
            ValueError: If parameters are invalid for the action type
        """
        if self.type == ActionType.WAIT:
            if self.params:
                raise ValueError("WAIT action should have no parameters")

        elif self.type == ActionType.MOVE:
            if "dir" not in self.params:
                raise ValueError("MOVE action requires 'dir' parameter")
            if not isinstance(self.params["dir"], MoveDir):
                raise ValueError(f"'dir' must be a MoveDir enum, got {type(self.params['dir'])}")

        elif self.type == ActionType.SHOOT:
            if "target_id" not in self.params:
                raise ValueError("SHOOT action requires 'target_id' parameter")
            if not isinstance(self.params["target_id"], int):
                raise ValueError(f"'target_id' must be an int, got {type(self.params['target_id'])}")

        elif self.type == ActionType.TOGGLE:
            if "on" not in self.params:
                raise ValueError("TOGGLE action requires 'on' parameter")
            if not isinstance(self.params["on"], bool):
                raise ValueError(f"'on' must be a bool, got {type(self.params['on'])}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert action to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the action
        """
        params_dict = {}
        for key, value in self.params.items():
            if isinstance(value, MoveDir):
                params_dict[key] = value.name
            else:
                params_dict[key] = value

        return {
            "type": self.type.name,
            "params": params_dict
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Action:
        """
        Create an action from a dictionary.

        Args:
            data: Dictionary containing 'type' and 'params'

        Returns:
            Action instance

        Raises:
            ValueError: If dictionary format is invalid
        """
        if "type" not in data:
            raise ValueError("Action dictionary must contain 'type'")

        action_type = ActionType[data["type"]]
        params = data.get("params", {})

        # Convert string direction back to enum
        if "dir" in params and isinstance(params["dir"], str):
            params["dir"] = MoveDir[params["dir"]]

        return cls(type=action_type, params=params)

    def to_json(self) -> str:
        """Convert action to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Action:
        """Create action from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.type == ActionType.WAIT:
            return "WAIT"
        elif self.type == ActionType.MOVE:
            return f"MOVE {self.params['dir'].name}"
        elif self.type == ActionType.SHOOT:
            return f"SHOOT target={self.params['target_id']}"
        elif self.type == ActionType.TOGGLE:
            return f"TOGGLE {'ON' if self.params['on'] else 'OFF'}"
        return f"{self.type.name}({self.params})"


    # FACTORY METHODS
    @staticmethod
    def wait() -> Action:
        """
        Create a WAIT action.

        Returns:
            Action that makes the entity wait and do nothing this turn.
        """
        return Action(ActionType.WAIT)

    @staticmethod
    def move(direction: MoveDir) -> Action:
        """
        Create a MOVE action.

        Args:
            direction: Direction to move (UP, DOWN, LEFT, RIGHT)

        Returns:
            Action that moves the entity in the specified direction.
        """
        return Action(ActionType.MOVE, {"dir": direction})

    @staticmethod
    def shoot(target_id: int) -> Action:
        """
        Create a SHOOT action.

        Args:
            target_id: ID of the target entity to shoot at

        Returns:
            Action that fires at the specified target.
        """
        return Action(ActionType.SHOOT, {"target_id": target_id})

    @staticmethod
    def toggle(on: bool) -> Action:
        """
        Create a TOGGLE action for SAM radar.

        Args:
            on: True to turn radar on, False to turn it off

        Returns:
            Action that toggles radar state.
        """
        return Action(ActionType.TOGGLE, {"on": on})





