"""
Agent interface and implementations for the Grid Combat Environment.

This module provides:
- BaseAgent: Abstract interface for all agents
- RandomAgent: Simple random action agent for testing
"""

from .base_agent import BaseAgent
from .factory import PreparedAgent, create_agent_from_spec, normalize_agent_input
from .registry import register_agent, resolve_agent_class
from .spec import AgentSpec
from .random_agent import RandomAgent

__all__ = [
    "BaseAgent",
    "AgentSpec",
    "PreparedAgent",
    "create_agent_from_spec",
    "normalize_agent_input",
    "register_agent",
    "resolve_agent_class",
    "RandomAgent",
]
