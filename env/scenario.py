"""
Scenario system for creating and managing game setups.

Provides type-safe Python definitions for scenarios and JSON serialization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING
import json
import time

from infra.paths import PROJECT_ROOT, SCENARIO_STORAGE_DIR
from .entities.base import Entity
from .entities import Aircraft, AWACS, SAM, Decoy
from .core.types import Team
from agents import AgentSpec



class Scenario:
    """
    A complete, self-contained scenario definition.
    
    A scenario includes EVERYTHING needed to initialize an environment:
    - Grid dimensions
    - Game rules (stalemate thresholds, victory conditions)
    - Random seed
    - All entities with explicit stats
    
    This makes scenarios reproducible and portable.
    
    Example (entities carry their own team):
        scenario = Scenario(
            grid_width=20,
            grid_height=20,
            entities=[
                Aircraft(team=Team.BLUE, pos=(2, 10), missiles=4, ...),
                Aircraft(team=Team.RED, pos=(18, 10), missiles=4, ...),
            ],
        )
        scenario.save_json("my_scenario.json")
        scenario = Scenario.load_json("my_scenario.json")
"""
    
    def __init__(
        self,
        grid_width: int = 20,
        grid_height: int = 20,
        max_stalemate_turns: int = 60,
        max_no_move_turns: int = 15,
        max_turns: Optional[int] = None,
        check_missile_exhaustion: bool = True,
        seed: Optional[int] = None,
        entities: Optional[List[Entity]] = None,
        agents: Optional[List["AgentSpec"]] = None,
    ):
        """
        Initialize a scenario with game configuration.
        
        Args:
            grid_width: Width of the grid (columns)
            grid_height: Height of the grid (rows)
            max_stalemate_turns: Turns without shooting before declaring a draw
            max_no_move_turns: Turns without movement before declaring a draw
            max_turns: Optional hard cap on total turns before declaring a draw
            check_missile_exhaustion: End game early if all missiles are gone
            seed: Random seed for reproducibility (None = random)
            entities: Optional list of entities (team carried on each entity)
            agents: Optional list of AgentSpec (one per team)
        """
        # Game configuration
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_stalemate_turns = max_stalemate_turns
        self.max_no_move_turns = max_no_move_turns
        self.max_turns = max_turns
        self.check_missile_exhaustion = check_missile_exhaustion
        self.seed = seed
        self.agents: Optional[List["AgentSpec"]] = agents
        
        # Entities
        self.entities: List[Entity] = []
        if entities:
            for entity in entities:
                self.add_entity(entity)
    
    def add_entity(self, entity: Entity) -> Scenario:
        """Add an entity (team must be set on the entity)."""
        if entity.team not in (Team.BLUE, Team.RED):
            raise ValueError(f"Entity team must be BLUE or RED, got {entity.team}")
        self.entities.append(entity)
        return self

    def clone(self) -> Scenario:
        """
        Create a deep copy of this scenario (including entities).
        
        Useful for running multiple environments from the same base scenario
        without sharing mutable entity objects.
        """
        return Scenario.from_json_dict(self.to_json_dict())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for env.reset().
        
        Returns:
            Dict with config, entities, and optional agent specs
        """
        data = {
            "config": {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "max_stalemate_turns": self.max_stalemate_turns,
            "max_no_move_turns": self.max_no_move_turns,
            "max_turns": self.max_turns,
            "check_missile_exhaustion": self.check_missile_exhaustion,
            "seed": self.seed,
        },
            "entities": self.entities,
        }
        if self.agents is not None:
            data["agents"] = self._serialize_agents(self.agents)
        return data
    
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.
        
        Returns:
            JSON-serializable dictionary
        """
        data = {
            "config": {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "max_stalemate_turns": self.max_stalemate_turns,
            "max_no_move_turns": self.max_no_move_turns,
            "max_turns": self.max_turns,
            "check_missile_exhaustion": self.check_missile_exhaustion,
            "seed": self.seed,
        },
            "entities": [e.to_dict() for e in self.entities],
        }
        if self.agents is not None:
            data["agents"] = self._serialize_agents(self.agents)
        return data
    
    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> Scenario:
        """
        Deserialize from JSON-compatible dictionary.
        
        Args:
            data: Dictionary from to_json_dict()
        
        Returns:
            Reconstructed Scenario
        """
        # Extract config
        config = data.get("config", {})
        scenario = cls(
            grid_width=config.get("grid_width", 20),
            grid_height=config.get("grid_height", 20),
            max_stalemate_turns=config.get("max_stalemate_turns", 60),
            max_no_move_turns=config.get("max_no_move_turns", 15),
            max_turns=config.get("max_turns"),
            check_missile_exhaustion=config.get("check_missile_exhaustion", True),
            seed=config.get("seed"),
            agents=cls._deserialize_agents(data.get("agents")),
        )
        
        # Load entities
        for entity_data in data.get("entities", []):
            scenario.entities.append(Entity.from_dict(entity_data))
        
        return scenario

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Scenario:
        """
        Deserialize from the in-memory scenario dict used by env.reset().
        
        This accepts both entity objects and JSON-like entity dicts. Entities
        are deep-copied to avoid sharing mutable instances.
        """
        config = data.get("config", {})
        scenario = cls(
            grid_width=config.get("grid_width", 20),
            grid_height=config.get("grid_height", 20),
            max_stalemate_turns=config.get("max_stalemate_turns", 60),
            max_no_move_turns=config.get("max_no_move_turns", 15),
            max_turns=config.get("max_turns"),
            check_missile_exhaustion=config.get("check_missile_exhaustion", True),
            seed=config.get("seed"),
            agents=cls._deserialize_agents(data.get("agents")),
        )

        def _to_entity(e: Any) -> Entity:
            if isinstance(e, Entity):
                return Entity.from_dict(e.to_dict())
            return Entity.from_dict(e)

        for entity_data in data.get("entities", []):
            scenario.entities.append(_to_entity(entity_data))

        return scenario

    @staticmethod
    def _serialize_agents(agents: List["AgentSpec"]) -> List[Dict[str, Any]]:
        """
        Best-effort serialization for agent specs; uses to_dict when available.
        """
        # Local import to avoid circular imports during module load
        from agents import AgentSpec
        serialized: List[Dict[str, Any]] = []
        for value in agents:
            serialized.append(value.to_dict() if isinstance(value, AgentSpec) else value)  # type: ignore[arg-type]
        return serialized
    
    @staticmethod
    def _deserialize_agents(data: Any) -> Optional[List["AgentSpec"]]:
        if data is None:
            return None
        # Local import to avoid circular imports during module load
        from agents import AgentSpec
        agents_list: List[AgentSpec] = []
        for value in data:
            if isinstance(value, AgentSpec):
                agents_list.append(value)
            elif isinstance(value, dict):
                agents_list.append(AgentSpec.from_dict(value))
            else:
                raise TypeError(f"Agent definition must be AgentSpec or dict, got {type(value)}")
        return agents_list
    
    def save_json(self, filepath: str | Path | None = None, indent: int = 2) -> None:
        """
        Save scenario to JSON file.
        
        Args:
            filepath: Path to save to. If None, saves under storage/scenarios with a timestamped name.
            indent: JSON indentation (default: 2)
        """
        if filepath is None:
            base_dir = SCENARIO_STORAGE_DIR
            base_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = base_dir / f"scenario_{timestamp}.json"
        else:
            filepath = Path(filepath)
            if not filepath.is_absolute():
                filepath = PROJECT_ROOT / filepath
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_json_dict(), f, indent=indent, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: str | Path) -> Scenario:
        """
        Load scenario from JSON file.
        
        Args:
            filepath: Path to load from
        
        Returns:
            Loaded Scenario
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_json_dict(data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Scenario(entities={len(self.entities)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Scenario(entities={self.entities})"


# =============================================================================
# SCENARIO BUILDERS (Examples/Templates)
# =============================================================================

def create_mixed_scenario() -> Scenario:
    """
    Create a complex mixed scenario.
    
    All entity types, realistic positioning on a 20x20 grid.
    """
    return Scenario(
        grid_width=20,
        grid_height=13,
        max_stalemate_turns=60,
        max_no_move_turns=100,
        max_turns=50,
        seed=42,
        agents=[AgentSpec(team=Team.BLUE, type="random", name="Blue Random Agent"),
                AgentSpec(team=Team.RED, type="random", name="Red Random Agent")],
        entities=[
            # Blue team - Combined arms
            AWACS(
                team=Team.BLUE, pos=(1, 10),
                radar_range=9.0
            ),
            Aircraft(
                team=Team.BLUE, pos=(5, 10),
                radar_range=5.0,
                missiles=4,
                missile_max_range=4.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1
            ),
            Aircraft(
                team=Team.BLUE, pos=(5, 12),
                radar_range=5.0,
                missiles=4,
                missile_max_range=4.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1
            ),
            SAM(
                team=Team.BLUE, pos=(2, 2),
                radar_range=8.0,
                missiles=6,
                missile_max_range=6.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1,
                cooldown_steps=5,
                on=True
            ),
            # Red team - Combined arms
            AWACS(
                team=Team.RED, pos=(19, 10),
                radar_range=9.0
            ),
            Aircraft(
                team=Team.RED, pos=(15, 10),
                radar_range=5.0,
                missiles=4,
                missile_max_range=4.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1
            ),
            Aircraft(
                team=Team.RED, pos=(15, 8),
                radar_range=5.0,
                missiles=4,
                missile_max_range=4.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1
            ),
            Decoy(team=Team.RED, pos=(16, 10)),
            SAM(
                team=Team.RED, pos=(18, 12),
                radar_range=8.0,
                missiles=6,
                missile_max_range=6.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1,
                cooldown_steps=5,
                on=False
            )
        ]
    )
