"""
Grid Combat Environment - A 2D turn-based air combat simulation.

This package provides a modular, testable combat simulation environment
suitable for reinforcement learning and game AI research.

Quick Start:
    from env import GridCombatEnv, Scenario, create_mixed_scenario
    from env.core import Team, Action, MoveDir
    
    # Create environment and scenario
    env = GridCombatEnv()
    scenario = create_mixed_scenario()
    
    # Reset with scenario
    state = env.reset(scenario=scenario.to_dict())
    
    # Run simulation
    actions = {}  # entity_id -> Action mapping
    state, rewards, done, info = env.step(actions)
"""

__version__ = "2.0.0"

# Main environment interface
from .environment import GridCombatEnv, StepInfo

# Scenario system
from .scenario import (
    Scenario,
    create_mixed_scenario,
)

# Core types available at package level
from .core import (
    GridPos,
    Team,
    ActionType,
    MoveDir,
    EntityKind,
    GameResult,
)

__all__ = [
    # Main interface
    "GridCombatEnv",
    "StepInfo",
    
    # Scenario system
    "Scenario",
    "create_mixed_scenario",
    
    # Core types
    "GridPos",
    "Team",
    "ActionType",
    "MoveDir",
    "EntityKind",
    "GameResult",
]