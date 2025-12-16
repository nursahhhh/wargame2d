"""
AI Controller Registry for Grid Air Combat
Allows flexible controller selection per team.
"""
from enum import Enum
from typing import Dict, Any, Callable, Tuple, Optional
from wargame_2d.env import World, Team, Action


class ControllerType(Enum):
    """Available controller types"""
    HUMAN = "human"
    RULE_BASED = "rule_based"
    LLM = "llm"
    RANDOM = "random"
    CODE_GEN = "code_generation"


class ControllerRegistry:
    """Registry for AI controllers"""

    _controllers: Dict[ControllerType, Callable] = {}

    @classmethod
    def register(cls, controller_type: ControllerType, func: Callable):
        """Register a controller function"""
        cls._controllers[controller_type] = func

    @classmethod
    def get(cls, controller_type: ControllerType) -> Callable:
        """Get a controller function"""
        if controller_type not in cls._controllers:
            raise ValueError(f"Controller type {controller_type} not registered")
        return cls._controllers[controller_type]

    @classmethod
    def list_available(cls) -> list[ControllerType]:
        """List all available controller types"""
        return list(cls._controllers.keys())


def get_controller_actions(
        world: World,
        team: Team,
        controller_type: ControllerType,
        controller_params: Dict[str, Any] = None
) -> Tuple[Dict[int, Action], Optional[Dict[str, Any]]]:
    """
    Main entry point for getting actions from any controller type.

    Args:
        world: Game world
        team: Team to control
        controller_type: Type of controller to use
        controller_params: Optional parameters for the controller

    Returns:
        Tuple of (actions, llm_output)
        - actions: Dict of entity_id -> Action
        - llm_output: Dict with LLM reasoning (None for non-LLM controllers)
    """
    if controller_params is None:
        controller_params = {}

    # Human control is handled by the UI, return empty dict
    if controller_type == ControllerType.HUMAN:
        return {}, None

    controller_func = ControllerRegistry.get(controller_type)
    return controller_func(world, team, **controller_params)