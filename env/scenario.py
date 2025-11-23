"""
Scenario system for creating and managing game setups.

Provides type-safe Python definitions for scenarios and JSON serialization.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .entities.base import Entity
from .entities import Aircraft, AWACS, SAM, Decoy
from .core.types import Team


class Scenario:
    """
    A complete, self-contained scenario definition.
    
    A scenario includes EVERYTHING needed to initialize an environment:
    - Grid dimensions
    - Game rules (stalemate thresholds, victory conditions)
    - Random seed
    - All entities with explicit stats
    
    This makes scenarios reproducible and portable.
    
    Example:
        # Option 1: Pass entities to constructor
        scenario = Scenario(
            grid_width=20,
            grid_height=20,
            max_stalemate_turns=60,
            max_no_move_turns=15,
            seed=42,
            blue_entities=[
                Aircraft(team=Team.BLUE, pos=(2, 10), missiles=4, ...),
                Aircraft(team=Team.BLUE, pos=(4, 10), missiles=4, ...)
            ],
            red_entities=[
                Aircraft(team=Team.RED, pos=(18, 10), missiles=4, ...)
            ]
        )
        
        # Option 2: Build incrementally
        scenario = Scenario(grid_width=20, grid_height=20, seed=42)
        scenario.add_blue(Aircraft(
            team=Team.BLUE, pos=(2, 10),
            missiles=4, radar_range=5.0,
            missile_max_range=4.0,
            base_hit_prob=0.8, min_hit_prob=0.1
        ))
        scenario.add_red(Aircraft(team=Team.RED, pos=(18, 10), ...))
        
        # Save to JSON
        scenario.save_json("my_scenario.json")
        
        # Load from JSON
        scenario = Scenario.load_json("my_scenario.json")
        
        # Use in environment
        env = GridCombatEnv()
        state = env.reset(scenario=scenario.to_dict())
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
        blue_entities: Optional[List[Entity]] = None,
        red_entities: Optional[List[Entity]] = None
    ):
        """
        Initialize a scenario with game configuration.
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            max_stalemate_turns: Max turns without shooting before draw
            max_no_move_turns: Max turns without movement before draw
            max_turns: Optional hard cap on total turns before draw
            check_missile_exhaustion: Check if all missiles depleted
            seed: Random seed for reproducibility (None = random)
            blue_entities: Optional list of Blue team entities
            red_entities: Optional list of Red team entities
        """
        # Game configuration
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_stalemate_turns = max_stalemate_turns
        self.max_no_move_turns = max_no_move_turns
        self.max_turns = max_turns
        self.check_missile_exhaustion = check_missile_exhaustion
        self.seed = seed
        
        # Entities
        self.blue_entities: List[Entity] = []
        self.red_entities: List[Entity] = []
        
        # Add entities if provided
        if blue_entities:
            for entity in blue_entities:
                self.add_blue(entity)
        
        if red_entities:
            for entity in red_entities:
                self.add_red(entity)
    
    def add_blue(self, entity: Entity) -> Scenario:
        """
        Add a Blue team entity.
        
        Args:
            entity: Entity to add
        
        Returns:
            self (for method chaining)
        """
        if entity.team != Team.BLUE:
            raise ValueError(f"Entity must be Team.BLUE, got {entity.team}")
        self.blue_entities.append(entity)
        return self
    
    def add_red(self, entity: Entity) -> Scenario:
        """
        Add a Red team entity.
        
        Args:
            entity: Entity to add
        
        Returns:
            self (for method chaining)
        """
        if entity.team != Team.RED:
            raise ValueError(f"Entity must be Team.RED, got {entity.team}")
        self.red_entities.append(entity)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for env.reset().
        
        Returns:
            Dict with config and entities
        """
        return {
            "config": {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "max_stalemate_turns": self.max_stalemate_turns,
            "max_no_move_turns": self.max_no_move_turns,
            "max_turns": self.max_turns,
            "check_missile_exhaustion": self.check_missile_exhaustion,
            "seed": self.seed,
        },
            "blue_entities": self.blue_entities,
            "red_entities": self.red_entities
        }
    
    def to_json_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.
        
        Returns:
            JSON-serializable dictionary
        """
        return {
            "config": {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "max_stalemate_turns": self.max_stalemate_turns,
            "max_no_move_turns": self.max_no_move_turns,
            "max_turns": self.max_turns,
            "check_missile_exhaustion": self.check_missile_exhaustion,
            "seed": self.seed,
        },
            "blue_entities": [e.to_dict() for e in self.blue_entities],
            "red_entities": [e.to_dict() for e in self.red_entities]
        }
    
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
            seed=config.get("seed")
        )
        
        # Load entities
        for entity_data in data.get("blue_entities", []):
            entity = Entity.from_dict(entity_data)
            scenario.blue_entities.append(entity)
        
        for entity_data in data.get("red_entities", []):
            entity = Entity.from_dict(entity_data)
            scenario.red_entities.append(entity)
        
        return scenario
    
    def save_json(self, filepath: str | Path, indent: int = 2) -> None:
        """
        Save scenario to JSON file.
        
        Args:
            filepath: Path to save to
            indent: JSON indentation (default: 2)
        """
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
        return (f"Scenario(blue={len(self.blue_entities)}, "
                f"red={len(self.red_entities)})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Scenario(blue_entities={self.blue_entities}, "
                f"red_entities={self.red_entities})")


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
        # Blue team - Combined arms
        blue_entities=[
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
            )
        ],
        # Red team - Combined arms
        red_entities=[
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
