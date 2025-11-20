"""
TeamView - Per-team observation and intelligence system.

Each team has a TeamView that aggregates observations from all friendly
entities and tracks which enemies are visible for targeting.

This was previously called CommandCenter but renamed for clarity.
"""

from __future__ import annotations
from typing import Set, Dict, Optional, Any
from ..core.types import Team
from ..core.observations import Observation, ObservationSet


class TeamView:
    """
    Aggregated intelligence view for one team.

    The TeamView:
    - Collects observations from all friendly entities
    - Tracks which enemies are visible (for targeting)
    - Provides querying interface for AI/decision-making

    Attributes:
        team: The team this view belongs to
    """

    def __init__(self, team: Team):
        """
        Initialize a team view.

        Args:
            team: The team this view represents
        """
        self.team = team
        self._observations = ObservationSet()
        self._friendly_ids: Set[int] = set()
        self._visible_enemy_ids: Set[int] = set()

        # Track enemy firing history for strategic decision-making
        self._enemy_firing_history: Dict[int, bool] = {}

    def reset(self) -> None:
        """Clear all observations and tracking (called each turn)."""
        self._observations.clear()
        self._friendly_ids.clear()
        self._visible_enemy_ids.clear()
        # Note: firing history persists across turns

    def add_friendly_id(self, entity_id: int) -> None:
        """
        Register a friendly entity ID.

        Args:
            entity_id: ID of friendly entity
        """
        self._friendly_ids.add(entity_id)

    def add_observation(self, obs: Observation) -> None:
        """
        Add a single observation.

        Args:
            obs: Observation to add
        """
        self._observations.add(obs)

        # Track enemy visibility
        if obs.team != self.team:
            self._visible_enemy_ids.add(obs.entity_id)

    def add_observations(self, obs_list: list[Observation]) -> None:
        """
        Add multiple observations.

        Args:
            obs_list: List of observations to add
        """
        for obs in obs_list:
            self.add_observation(obs)

    def can_target(self, entity_id: int) -> bool:
        """
        Check if an entity can be targeted (is visible enemy).

        Args:
            entity_id: Entity ID to check

        Returns:
            True if entity is a visible enemy
        """
        return entity_id in self._visible_enemy_ids

    def get_observation(self, entity_id: int) -> Optional[Observation]:
        """
        Get observation of a specific entity.

        Args:
            entity_id: Entity ID to look up

        Returns:
            Observation if available, None otherwise
        """
        return self._observations.get(entity_id)

    def get_all_observations(self) -> list[Observation]:
        """Get all observations (friendly and enemy)."""
        return self._observations.all()

    def get_friendly_observations(self) -> list[Observation]:
        """Get observations of friendly entities."""
        return self._observations.filter_by_team(self.team)

    def get_enemy_observations(self) -> list[Observation]:
        """Get observations of enemy entities."""
        return [obs for obs in self._observations.all() if obs.team != self.team]

    def get_friendly_ids(self) -> Set[int]:
        """Get set of all friendly entity IDs."""
        return self._friendly_ids.copy()

    def get_enemy_ids(self, observer_team: Team) -> Set[int]:
        """
        Get set of all visible enemy entity IDs.

        Args:
            observer_team: Team doing the observing (for compatibility)

        Returns:
            Set of visible enemy IDs
        """
        return self._visible_enemy_ids.copy()

    def record_enemy_fired(self, entity_id: int) -> None:
        """
        Record that an enemy has fired a weapon.

        This is useful for strategic AI - knowing which enemies
        have revealed themselves by firing.

        Args:
            entity_id: Enemy entity that fired
        """
        self._enemy_firing_history[entity_id] = True

    def has_enemy_fired(self, entity_id: int) -> bool:
        """
        Check if an enemy has ever fired.

        Args:
            entity_id: Enemy entity to check

        Returns:
            True if entity has fired at least once
        """
        return self._enemy_firing_history.get(entity_id, False)

    def __len__(self) -> int:
        """Number of entities observed."""
        return len(self._observations)

    def __str__(self) -> str:
        """String representation."""
        return (f"TeamView({self.team.name}: "
                f"{len(self._friendly_ids)} friendly, "
                f"{len(self._visible_enemy_ids)} enemies visible)")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"TeamView(team={self.team}, "
                f"friendly_ids={self._friendly_ids}, "
                f"visible_enemy_ids={self._visible_enemy_ids})")

    # ========================================================================
    # SERIALIZATION
    # ========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize TeamView state to dictionary.

        Note: Only persistent state (_enemy_firing_history) is serialized.
        Derived state (observations, friendly_ids, visible_enemy_ids) is
        regenerated after loading via SensorSystem.refresh_all_observations().

        Returns:
            Dictionary with persistent state
        """
        return {
            "team": self.team.name,
            "enemy_firing_history": {
                str(entity_id): fired
                for entity_id, fired in self._enemy_firing_history.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TeamView:
        """
        Deserialize TeamView from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed TeamView with persistent state restored
        """
        team = Team[data["team"]]
        team_view = cls(team)

        # Restore firing history
        team_view._enemy_firing_history = {
            int(entity_id): fired
            for entity_id, fired in data["enemy_firing_history"].items()
        }

        # Note: Observations will be regenerated by SensorSystem
        return team_view