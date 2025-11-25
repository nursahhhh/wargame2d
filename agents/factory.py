from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base_agent import BaseAgent
from .registry import resolve_agent_class
from .spec import AgentSpec


@dataclass
class PreparedAgent:
    """Agent instance paired with static per-episode parameters."""
    agent: BaseAgent
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
        act_params=spec.act_params or {},
    )
