"""
Observation definitions and utilities.

Observations represent what an entity knows about other entities in the world.
This module provides:
- Observation dataclass
- Observation filtering
- Observation transformation
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List, Dict, Optional
import json

from .types import Team, GridPos, EntityKind


@dataclass
class Observation:
    """
    An observation of an entity by another entity.

    Observations contain information about what one entity can "see" about
    another. The information may be incomplete or deceptive (e.g., decoys
    appear as aircraft to enemies).

    Attributes:
        entity_id: ID of the observed entity
        kind: Type of entity (may be deceptive for decoys observed by enemies)
        team: Team affiliation
        position: Grid position
        distance: Distance from observer
        seen_by: Set of entity IDs that can see this entity
    """

    entity_id: int
    kind: EntityKind
    team: Team
    position: GridPos
    distance: float
    seen_by: Set[int] = field(default_factory=set)

    def is_friendly(self, observer_team: Team) -> bool:
        """Check if this observation is of a friendly entity."""
        return self.team == observer_team

    def is_enemy(self, observer_team: Team) -> bool:
        """Check if this observation is of an enemy entity."""
        return self.team != observer_team

    def to_dict(self) -> Dict:
        """
        Convert observation to a JSON-serializable dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "entity_id": self.entity_id,
            "kind": self.kind.value,  # Serialize enum as string value
            "team": self.team.value,
            "position": list(self.position),
            "distance": self.distance,
            "seen_by": list(self.seen_by)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> Observation:
        """
        Create observation from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Observation instance
        """
        return cls(
            entity_id=data["entity_id"],
            kind=EntityKind(data["kind"]),  # Deserialize string back to enum
            team=Team(data["team"]),
            position=tuple(data["position"]),
            distance=data["distance"],
            seen_by=set(data["seen_by"])
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Observation:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """Human-readable representation."""
        return (f"Obs(id={self.entity_id}, {self.kind.value}, {self.team.name}, "
                f"pos={self.position}, dist={self.distance:.1f})")


# ============================================================================
# OBSERVATION COLLECTIONS
# ============================================================================

@dataclass
class ObservationSet:
    """
    A collection of observations for a team or entity.

    Provides convenient filtering and querying of observations.
    """

    observations: Dict[int, Observation] = field(default_factory=dict)

    def add(self, obs: Observation) -> None:
        """
        Add or merge an observation.

        If observation already exists, merges the seen_by sets.
        """
        if obs.entity_id in self.observations:
            self.observations[obs.entity_id].seen_by.update(obs.seen_by)
        else:
            self.observations[obs.entity_id] = obs

    def add_many(self, obs_list: List[Observation]) -> None:
        """Add multiple observations."""
        for obs in obs_list:
            self.add(obs)

    def get(self, entity_id: int) -> Optional[Observation]:
        """Get observation by entity ID."""
        return self.observations.get(entity_id)

    def contains(self, entity_id: int) -> bool:
        """Check if entity is observed."""
        return entity_id in self.observations

    def filter_by_team(self, team: Team) -> List[Observation]:
        """Get all observations of a specific team."""
        return [obs for obs in self.observations.values() if obs.team == team]

    def filter_by_kind(self, kind: EntityKind) -> List[Observation]:
        """Get all observations of a specific kind."""
        return [obs for obs in self.observations.values() if obs.kind == kind]

    def filter_by_kinds(self, kinds: Set[EntityKind]) -> List[Observation]:
        """Get all observations matching any of the specified kinds."""
        return [obs for obs in self.observations.values() if obs.kind in kinds]

    def get_enemy_ids(self, observer_team: Team) -> Set[int]:
        """Get IDs of all observed enemy entities."""
        return {
            obs.entity_id
            for obs in self.observations.values()
            if obs.is_enemy(observer_team)
        }

    def get_closest_enemy(self, observer_team: Team) -> Optional[Observation]:
        """Get the closest enemy observation."""
        enemies = [obs for obs in self.observations.values()
                   if obs.is_enemy(observer_team)]
        return min(enemies, key=lambda obs: obs.distance) if enemies else None

    def get_friendly_ids(self, observer_team: Team) -> Set[int]:
        """Get IDs of all observed friendly entities."""
        return {
            obs.entity_id
            for obs in self.observations.values()
            if obs.is_friendly(observer_team)
        }

    def all(self) -> List[Observation]:
        """Get all observations as a list."""
        return list(self.observations.values())

    def clear(self) -> None:
        """Clear all observations."""
        self.observations.clear()

    def __len__(self) -> int:
        """Number of unique entities observed."""
        return len(self.observations)

    def __contains__(self, entity_id: int) -> bool:
        """Check if entity ID is in observations."""
        return entity_id in self.observations

    def __iter__(self):
        """Iterate over observations."""
        return iter(self.observations.values())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_observations(obs_list: List[Observation]) -> List[Observation]:
    """
    Merge observations of the same entity, combining seen_by sets.

    Args:
        obs_list: List of observations to merge

    Returns:
        List of merged observations (one per unique entity)
    """
    obs_set = ObservationSet()
    obs_set.add_many(obs_list)
    return obs_set.all()


def filter_observations(
    obs_list: List[Observation],
    *,
    teams: Optional[List[Team]] = None,
    kinds: Optional[List[EntityKind]] = None,
    max_distance: Optional[float] = None,
    min_distance: Optional[float] = None,
) -> List[Observation]:
    """
    Filter observations based on multiple criteria.

    Args:
        obs_list: Observations to filter
        teams: Include only these teams (None = all)
        kinds: Include only these entity kinds (None = all)
        max_distance: Maximum distance (None = no limit)
        min_distance: Minimum distance (None = no limit)

    Returns:
        Filtered list of observations
    """
    filtered = obs_list

    if teams is not None:
        filtered = [obs for obs in filtered if obs.team in teams]

    if kinds is not None:
        filtered = [obs for obs in filtered if obs.kind in kinds]

    if max_distance is not None:
        filtered = [obs for obs in filtered if obs.distance <= max_distance]

    if min_distance is not None:
        filtered = [obs for obs in filtered if obs.distance >= min_distance]

    return filtered