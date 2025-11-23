"""
Core type definitions for the Grid Combat Environment.

This module contains all fundamental types, enums, and constants used
throughout the system. No logic, just pure data structures.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Tuple
from dataclasses import dataclass

# ============================================================================
# SPATIAL TYPES
# ============================================================================

# Grid position: (x, y) where:
# - X increases to the RIGHT
# - Y increases UPWARD
# - Origin (0, 0) is at BOTTOM-LEFT
GridPos = Tuple[int, int]


class Team(Enum):
    """Team affiliation for entities."""
    BLUE = "BLUE"
    RED = "RED"

    def __str__(self) -> str:
        return self.value

    @property
    def opponent(self) -> Team:
        """Get the opposing team."""
        return Team.RED if self == Team.BLUE else Team.BLUE


# ============================================================================
# ACTIONS
# ============================================================================

class ActionType(Enum):
    """Types of actions entities can perform."""
    WAIT = auto()  # Do nothing this turn
    MOVE = auto()  # Move in a direction
    SHOOT = auto()  # Fire at a target
    TOGGLE = auto()  # Toggle radar on/off (SAM specific)

    def __str__(self) -> str:
        return self.name


class MoveDir(Enum):
    """
    Movement directions using mathematical coordinates (Y+ = UP).
    Each direction provides a delta tuple (dx, dy).
    """
    UP = (0, 1)  # Move upward (increase Y)
    DOWN = (0, -1)  # Move downward (decrease Y)
    LEFT = (-1, 0)  # Move left (decrease X)
    RIGHT = (1, 0)  # Move right (increase X)

    @property
    def delta(self) -> Tuple[int, int]:
        """Get the (dx, dy) movement delta."""
        return self.value

    def __str__(self) -> str:
        return self.name


# ============================================================================
# ENTITY KINDS
# ============================================================================

class EntityKind(Enum):
    """Types of entities in the game."""
    AIRCRAFT = "aircraft"
    AWACS = "awacs"
    SAM = "sam"
    DECOY = "decoy"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    @property
    def icon(self) -> str:
        """Get the display icon for this entity kind."""
        return {
            EntityKind.AIRCRAFT: "A",
            EntityKind.AWACS: "W",
            EntityKind.SAM: "S",
            EntityKind.DECOY: "D",
            EntityKind.UNKNOWN: "U"
        }[self]


# ============================================================================
# GAME RESULT
# ============================================================================

class GameResult(Enum):
    """Possible game outcomes."""
    IN_PROGRESS = "in_progress"
    BLUE_WINS = "blue_wins"
    RED_WINS = "red_wins"
    DRAW = "draw"

    def __str__(self) -> str:
        return self.value.replace("_", " ").title()


# ============================================================================
# ACTION VALIDATION
# ============================================================================

@dataclass
class ActionValidation:
    """
    Structured result of validating an action.
    
    This provides detailed information about why an action is valid or invalid,
    enabling better error messages and debugging.
    
    Attributes:
        valid: Whether the action is valid
        error_code: Machine-readable error code (None if valid)
        message: Human-readable message explaining the result
    
    Error codes:
        - "ENTITY_DEAD": Entity is not alive
        - "NO_CAPABILITY": Entity lacks the basic capability (can_move/can_shoot)
        - "NO_MISSILES": No missiles remaining
        - "RADAR_OFF": SAM radar is off
        - "COOLING_DOWN": SAM is in cooldown period
        - "INVALID_DIRECTION": Movement direction is invalid
        - "INVALID_TARGET": Target ID is invalid or target is dead
        - "OUT_OF_BOUNDS": Movement would leave grid bounds
        - "NOT_VISIBLE": Target is not currently observable
        - "OUT_OF_RANGE": Target is outside weapon range
        - "NOT_SAM": Only SAMs can toggle radar
        - "INVALID_TOGGLE": Toggle parameter is invalid
    """
    valid: bool
    error_code: str | None = None
    message: str = ""
    
    @staticmethod
    def success(message: str = "") -> ActionValidation:
        """Create a validation success result."""
        return ActionValidation(valid=True, error_code=None, message=message)
    
    @staticmethod
    def fail(error_code: str, message: str) -> ActionValidation:
        """Create a validation failure result."""
        return ActionValidation(valid=False, error_code=error_code, message=message)
