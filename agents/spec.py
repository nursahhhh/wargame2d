from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from env.core.types import Team


@dataclass
class AgentSpec:
    """
    Serializable description of an agent.
    
    This is intended for configuration files (scenarios, UI, etc.) so that
    agents can be instantiated dynamically by a factory/registry.
    """
    type: str
    team: Team
    name: Optional[str] = None
    init_params: Dict[str, Any] = field(default_factory=dict)
    commands: Dict[str, Any] = field(default_factory=dict)
    act_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dict."""
        return {
            "type": self.type,
            "team": self.team.name,
            "name": self.name,
            "init_params": self.init_params,
            "commands": self.commands,
            "act_params": self.act_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpec":
        """Construct from a dict (e.g., loaded from JSON)."""
        team_raw = data.get("team")
        if team_raw is None:
            raise ValueError("AgentSpec requires 'team'")
        team = Team[team_raw] if isinstance(team_raw, str) else team_raw
        return cls(
            type=data["type"],
            team=team,
            name=data.get("name"),
            init_params=data.get("init_params", {}) or {},
            commands=data.get("commands", {}) or {},
            act_params=data.get("act_params", {}) or {},
        )

    def with_team(self, team: Team) -> "AgentSpec":
        """Return a copy with the provided team."""
        return AgentSpec(
            type=self.type,
            team=team,
            name=self.name,
            init_params=self.init_params,
            commands=self.commands,
            act_params=self.act_params,
        )
