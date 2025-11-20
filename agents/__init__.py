"""
Agent interface and implementations for the Grid Combat Environment.

This module provides:
- BaseAgent: Abstract interface for all agents
- RandomAgent: Simple random action agent for testing
"""

from .base_agent import BaseAgent
from .random_agent import RandomAgent

__all__ = [
    "BaseAgent",
    "RandomAgent",
]

