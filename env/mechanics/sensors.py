"""
SensorSystem - Observation and radar computation.

This module handles:
- Computing what each entity observes
- Handling decoy deception
- SAM radar visibility (only when ON)
- Aggregating observations per team
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List

from ..core.types import EntityKind
from ..core.observations import Observation

if TYPE_CHECKING:
    from ..world.world import WorldState
    from ..entities.base import Entity
    from ..entities.sam import SAM
    from ..entities.decoy import Decoy


class SensorSystem:
    """
    Stateless system for computing entity observations.
    
    The SensorSystem:
    - Computes what each entity can "see" based on radar range
    - Handles decoy deception (decoys appear as aircraft to enemies)
    - Handles SAM visibility (SAMs only detectable when radar is ON)
    - Aggregates observations into team views
    
    All methods are stateless - they don't modify the SensorSystem itself.
    """
    
    def __init__(self):
        """Initialize the sensor system."""
        pass  # Stateless, no initialization needed
    
    def refresh_all_observations(self, world: WorldState) -> None:
        """
        Compute and update observations for all entities and teams.
        
        This is the main entry point - call this each turn to update
        what everyone can see.
        
        Process:
        1. Reset all team views
        2. Compute observations for each living entity
        3. Aggregate observations into team views
        4. Add self-observations (entities always see themselves)
        
        Args:
            world: Current world state (modified in-place)
        """
        # Step 1: Reset team views for this turn
        from ..core.types import Team
        for team in [Team.BLUE, Team.RED]:
            world.get_team_view(team).reset()
        
        # Step 2: Register friendly IDs
        for entity in world.get_alive_entities():
            team_view = world.get_team_view(entity.team)
            team_view.add_friendly_id(entity.id)
        
        # Step 3: Compute observations for each entity
        for observer in world.get_alive_entities():
            observations = self.compute_entity_observations(world, observer)
            
            # Store on entity (for easy access)
            observer.observations = observations
            
            # Add to team view
            team_view = world.get_team_view(observer.team)
            team_view.add_observations(observations)
        
        # Step 4: Add self-observations (entities always see themselves)
        for entity in world.get_alive_entities():
            self_obs = Observation(
                entity_id=entity.id,
                kind=entity.kind,
                team=entity.team,
                position=entity.pos,
                distance=0.0,
                seen_by={entity.id}
            )
            team_view = world.get_team_view(entity.team)
            team_view.add_observation(self_obs)
    
    def compute_entity_observations(
        self, 
        world: WorldState, 
        observer: Entity
    ) -> List[Observation]:
        """
        Compute what a single entity can observe.
        
        Args:
            world: Current world state
            observer: Entity doing the observing
        
        Returns:
            List of observations of other entities
        """
        observations = []
        
        # Dead entities can't observe
        if not observer.alive:
            return observations
        
        # Get active radar range (handles SAM on/off)
        active_radar = observer.get_active_radar_range()
        if active_radar <= 0:
            return observations
        
        # Check all other entities
        for target in world.get_all_entities():
            # Can't observe self
            if target.id == observer.id:
                continue
            
            # Can't observe dead entities
            if not target.alive:
                continue
            
            # Special case: SAMs with radar OFF are invisible
            if self._is_sam_invisible(target):
                continue
            
            # Check if in radar range
            distance = world.grid.distance(observer.pos, target.pos)
            if distance > active_radar:
                continue
            
            # Determine apparent kind (handles decoy deception)
            apparent_kind = self._get_apparent_kind(target, observer)
            
            # Create observation
            obs = Observation(
                entity_id=target.id,
                kind=apparent_kind,
                team=target.team,
                position=target.pos,
                distance=distance,
                seen_by={observer.id}
            )
            observations.append(obs)
        
        return observations
    
    def _is_sam_invisible(self, entity: Entity) -> bool:
        """
        Check if a SAM is invisible (radar OFF).
        
        SAMs are only visible to enemies when their radar is ON.
        When OFF, they're completely invisible.
        
        Args:
            entity: Entity to check
        
        Returns:
            True if entity is a SAM with radar OFF
        """
        # Check if it's a SAM
        if entity.kind != EntityKind.SAM:
            return False
        
        # Import here to avoid circular dependency at module level
        from ..entities.sam import SAM
        
        # Check if radar is OFF
        if isinstance(entity, SAM):
            return not entity.on
        
        return False
    
    def _get_apparent_kind(self, target: Entity, observer: Entity) -> EntityKind:
        """
        Get the apparent kind of a target (handles decoy deception).
        
        Decoys appear as aircraft to enemies, but friendlies see the truth.
        
        Args:
            target: Entity being observed
            observer: Entity doing the observing
        
        Returns:
            EntityKind as it appears to the observer
        """
        # If observing own team, see the truth
        if target.team == observer.team:
            return target.kind
        
        # Enemy decoys appear as aircraft
        if target.kind == EntityKind.DECOY:
            return EntityKind.AIRCRAFT
        
        # Everything else appears as it is
        return target.kind
    
    def get_entities_in_radar_range(
        self, 
        world: WorldState, 
        observer: Entity
    ) -> List[Entity]:
        """
        Get all entities within an observer's radar range.
        
        Utility method for AI/decision-making.
        
        Args:
            world: Current world state
            observer: Entity doing the observing
        
        Returns:
            List of entities within radar range (excluding self)
        """
        active_radar = observer.get_active_radar_range()
        if active_radar <= 0:
            return []
        
        entities_in_range = []
        for entity in world.get_alive_entities():
            if entity.id == observer.id:
                continue
            
            distance = world.grid.distance(observer.pos, entity.pos)
            if distance <= active_radar:
                entities_in_range.append(entity)
        
        return entities_in_range
    
    def can_observe(
        self, 
        world: WorldState, 
        observer: Entity, 
        target: Entity
    ) -> bool:
        """
        Check if an observer can currently see a target.
        
        Utility method for validation and AI.
        
        Args:
            world: Current world state
            observer: Entity doing the observing
            target: Entity to check visibility for
        
        Returns:
            True if observer can see target
        """
        # Can't observe if either is dead
        if not observer.alive or not target.alive:
            return False
        
        # Can't observe self
        if observer.id == target.id:
            return False
        
        # Check radar range
        active_radar = observer.get_active_radar_range()
        if active_radar <= 0:
            return False
        
        distance = world.grid.distance(observer.pos, target.pos)
        if distance > active_radar:
            return False
        
        # Check if target is invisible (SAM with radar OFF)
        if self._is_sam_invisible(target):
            return False
        
        return True

