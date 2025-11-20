"""
Mechanics module - Action resolution systems.

This module provides stateless resolvers for game actions:
- SensorSystem: Computes what entities observe
- MovementResolver: Resolves movement actions
- CombatResolver: Resolves shooting/combat actions
- VictoryConditions: Checks game ending conditions

All resolvers are stateless - they take WorldState and return results
without modifying their own state.
"""

from .sensors import SensorSystem
from .movement import MovementResolver, ActionResolutionResult
from .combat import CombatResolver, CombatResolutionResult, hit_probability
from .victory import VictoryConditions, VictoryResult

__all__ = [
    "SensorSystem",
    "MovementResolver",
    "ActionResolutionResult",
    "CombatResolver",
    "CombatResolutionResult",
    "hit_probability",
    "VictoryConditions",
    "VictoryResult",
]

