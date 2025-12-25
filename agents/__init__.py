"""
Agent interface and implementations for the Grid Combat Environment.

This module provides:
- BaseAgent: Abstract interface for all agents
- RandomAgent: Simple random action agent for testing
"""

from .base_agent import BaseAgent
from .factory import create_agent_from_spec

from .registry import register_agent, resolve_agent_class
from .spec import AgentSpec
from .random_agent import RandomAgent
from .greedy_agent import GreedyAgent
from .llm_agent import LLMAgent
from .team_intel import TeamIntel, VisibleEnemy

__all__ = [
    "BaseAgent",
    "AgentSpec",
    "create_agent_from_spec",
    "register_agent",
    "resolve_agent_class",
    "RandomAgent",
    "GreedyAgent",
    "LLMAgent",
    "TeamIntel",
    "VisibleEnemy",
]
