from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base_agent import BaseAgent
from .registry import resolve_agent_class
from .spec import AgentSpec


@dataclass
class PreparedAgent:
    """Agent instance paired with static per-episode command/act params."""
    agent: BaseAgent
    commands: Dict[str, Any]
    act_params: Dict[str, Any]


def create_agent_from_spec(spec: AgentSpec) -> PreparedAgent:
    """Instantiate an agent from an AgentSpec."""
    cls = resolve_agent_class(spec.type)

    init_kwargs = dict(spec.init_params)
    init_kwargs.setdefault("team", spec.team)
    if spec.name is not None:
        init_kwargs.setdefault("name", spec.name)

    agent = cls(**init_kwargs)
    if not isinstance(agent, BaseAgent):
        raise TypeError(f"Agent {cls} is not a BaseAgent")

    return PreparedAgent(
        agent=agent,
        commands=spec.commands or {},
        act_params=spec.act_params or {},
    )


def normalize_agent_input(
    agent_input: BaseAgent | AgentSpec | Dict[str, Any],
) -> BaseAgent | AgentSpec:
    """
    Accept an agent instance, AgentSpec, or raw dict and normalize to a usable form.
    Raw dicts are converted to AgentSpec.
    """
    if isinstance(agent_input, BaseAgent):
        return agent_input
    if isinstance(agent_input, AgentSpec):
        return agent_input
    if isinstance(agent_input, dict):
        return AgentSpec.from_dict(agent_input)
    raise TypeError("agent_input must be BaseAgent, AgentSpec, or dict")
