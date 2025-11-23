"""
CombatResolver - Combat and shooting action resolution.

This module handles:
- Validating shooting actions
- Calculating hit probabilities
- Resolving missile hits/misses
- Managing missile counts
- Tracking kills
- SAM cooldown management
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List
from dataclasses import dataclass
import random

from ..core.types import ActionType
from ..core.actions import Action
from ..core.validation import validate_action_in_world

if TYPE_CHECKING:
    from ..world.world import WorldState
    from ..entities.base import Entity


def hit_probability(
    *,
    distance: float,
    max_range: float,
    base: float,
    min_p: float,
) -> float:
    """
    Calculate the probability of hitting a target at a given distance.

    Hit probability decreases linearly with distance:
    - At distance 0: probability = base
    - At distance = max_range: probability = min_p
    - Beyond max_range: probability = min_p (clamped)

    The formula is: max(min_p, base * (1 - distance/max_range))

    Args:
        distance: Distance to target (must be >= 0)
        max_range: Maximum effective range (must be > 0 for non-zero probability)
        base: Base hit probability at distance 0 (e.g., 0.8)
        min_p: Minimum hit probability at max range (e.g., 0.1)

    Returns:
        Hit probability in range [min_p, base], or 0.0 if max_range <= 0

    Examples:
        >>> hit_probability(distance=0.0, max_range=10.0, base=0.8, min_p=0.1)
        0.8
        >>> hit_probability(distance=5.0, max_range=10.0, base=0.8, min_p=0.1)
        0.45
        >>> hit_probability(distance=10.0, max_range=10.0, base=0.8, min_p=0.1)
        0.1
        >>> hit_probability(distance=15.0, max_range=10.0, base=0.8, min_p=0.1)
        0.1
    """

    # Validate inputs
    if distance < 0:
        raise ValueError(f"Distance cannot be negative: {distance}")
    if max_range <= 0:
        return 0.0
    if not 0 <= base <= 1:
        raise ValueError(f"Base probability must be in [0, 1]: {base}")
    if not 0 <= min_p <= 1:
        raise ValueError(f"Min probability must be in [0, 1]: {min_p}")
    if min_p > base:
        raise ValueError(f"Min probability ({min_p}) cannot exceed base ({base})")

    # Calculate linear interpolation
    # At distance 0: frac = 1.0, prob = base
    # At distance = max_range: frac = 0.0, prob = min_p
    frac = max(0.0, min(1.0, 1.0 - (distance / max_range)))
    probability = max(min_p, base * frac)

    return probability


@dataclass
class CombatResult:
    """
    Result of resolving a single combat action.
    
    Attributes:
        attacker_id: ID of entity that fired
        target_id: ID of target entity (None if invalid)
        success: Whether shot was fired (not whether it hit)
        hit: Whether shot hit target (None if not fired)
        distance: Distance to target (None if invalid)
        hit_probability: Calculated hit probability (None if not fired)
        target_killed: Whether target was killed
        log: Human-readable log message
    """
    attacker_id: int
    target_id: int | None
    success: bool
    hit: bool | None
    distance: float | None
    hit_probability: float | None
    target_killed: bool
    log: str


@dataclass
class CombatResolutionResult:
    """
    Complete result of resolving all combat for a turn.
    
    Attributes:
        combat_results: Results from all shooting actions
        death_logs: Logs from entities being killed
        killed_entity_ids: List of entity IDs that were killed
        combat_occurred: True if at least one shot was fired
    """
    combat_results: List[CombatResult]
    death_logs: List[str]
    killed_entity_ids: List[int]
    combat_occurred: bool


class CombatResolver:
    """
    Stateless resolver for combat actions.
    
    The CombatResolver:
    - Validates shooting actions
    - Calculates hit probabilities using physics module
    - Rolls for hits/misses
    - Manages ammunition
    - Handles SAM cooldowns
    - Marks kills
    
    All methods are stateless - they don't modify the resolver itself.
    """
    
    def __init__(self, rng: random.Random | None = None):
        """
        Initialize the combat resolver.
        
        Args:
            rng: Random number generator (optional, for deterministic testing)
        """
        self._rng = rng  # Optional for deterministic testing
    
    def resolve_combat(
        self,
        world: WorldState,
        actions: Dict[int, Action],
        randomize_order: bool = True
    ) -> CombatResolutionResult:
        """
        Resolve all combat actions for a turn, including death application.
        
        This is the main entry point for combat resolution. It handles
        all shooting actions and applies resulting deaths in a single pass.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
            randomize_order: If True, shuffle order to prevent ID bias
        
        Returns:
            CombatResolutionResult with all outcomes
        """
        # Resolve all combat actions
        combat_results = self.resolve_all(world, actions, randomize_order)
        
        # Apply deaths from combat
        death_logs, killed_entity_ids = self.apply_pending_deaths(world)
        
        # Check if any combat occurred
        combat_occurred = self.has_combat_occurred(combat_results)
        
        # Update combat counter
        if combat_occurred:
            world.turns_without_shooting = 0
        else:
            world.turns_without_shooting += 1
        
        return CombatResolutionResult(
            combat_results=combat_results,
            death_logs=death_logs,
            killed_entity_ids=killed_entity_ids,
            combat_occurred=combat_occurred
        )
    
    def resolve_all(
        self,
        world: WorldState,
        actions: Dict[int, Action],
        randomize_order: bool = True
    ) -> List[CombatResult]:
        """
        Resolve all combat actions for a turn.
        
        Combat actions are processed in random order to prevent
        bias from entity ID ordering.
        
        Args:
            world: Current world state (modified in-place)
            actions: Map of entity_id -> action
            randomize_order: If True, shuffle order to prevent ID bias
        
        Returns:
            List of CombatResult objects (one per shooting action)
        """
        results = []
        
        # Get all entities that want to shoot
        shooting_entities = []
        for entity in world.get_alive_entities():
            action = actions.get(entity.id)
            if action and action.type == ActionType.SHOOT:
                shooting_entities.append(entity)
        
        # Randomize order to prevent ID bias
        if randomize_order:
            world.rng.shuffle(shooting_entities)
        
        # Process each shot
        for entity in shooting_entities:
            action = actions[entity.id]
            result = self.resolve_single(world, entity, action)
            results.append(result)
        
        return results
    
    def resolve_single(
        self,
        world: WorldState,
        attacker: Entity,
        action: Action
    ) -> CombatResult:
        """
        Resolve a single shooting action.
        
        This method now uses shared validation first (entity-level + world checks),
        then performs resolution (hit probability, missile consumption, cooldown).
        
        Args:
            world: Current world state (modified in-place)
            attacker: Entity attempting to shoot
            action: Shooting action
        
        Returns:
            CombatResult with outcome
        """
        # Use shared validation first (entity-level + world-dependent checks)
        validation = validate_action_in_world(world, attacker, action)
        if not validation.valid:
            return CombatResult(
                attacker_id=attacker.id,
                target_id=action.params.get("target_id"),
                success=False,
                hit=None,
                distance=None,
                hit_probability=None,
                target_killed=False,
                log=validation.message
            )
        
        # Get target (we know target_id is valid from validation)
        target_id = action.params.get("target_id")
        target = world.get_entity(target_id)
        
        # Safety: target should exist after validation, but guard just in case
        if not target or not target.alive:
            return CombatResult(
                attacker_id=attacker.id,
                target_id=target_id,
                success=False,
                hit=None,
                distance=None,
                hit_probability=None,
                target_killed=False,
                log=f"{attacker.label()} target invalid or dead"
            )
        
        # Calculate distance for hit probability calculation
        distance = world.grid.distance(attacker.pos, target.pos)
        
        # Calculate hit probability
        prob = hit_probability(
            distance=distance,
            max_range=attacker.missile_max_range,
            base=attacker.base_hit_prob,
            min_p=attacker.min_hit_prob
        )
        
        # Use world RNG or our own
        rng = self._rng if self._rng is not None else world.rng
        
        # Roll for hit
        roll = rng.random()
        hit = roll <= prob
        
        # Consume missile
        attacker.missiles -= 1
        
        # Handle SAM cooldown
        from ..entities.sam import SAM
        if isinstance(attacker, SAM):
            attacker.start_cooldown()
        
        # Apply kill if hit
        target_killed = False
        if hit:
            world.mark_for_kill(target.id)
            target_killed = True
        
        # Record that enemy fired (for team intelligence)
        enemy_team_view = world.get_team_view(target.team)
        enemy_team_view.record_enemy_fired(attacker.id)
        
        # Generate log
        hit_str = "HIT" if hit else "MISS"
        log = (
            f"{attacker.label()} fires at {target.label()} "
            f"(d={distance:.1f}, p={prob:.2f}, roll={roll:.2f}) -> {hit_str}"
        )
        
        return CombatResult(
            attacker_id=attacker.id,
            target_id=target_id,
            success=True,
            hit=hit,
            distance=distance,
            hit_probability=prob,
            target_killed=target_killed,
            log=log
        )
    
    def has_combat_occurred(self, results: List[CombatResult]) -> bool:
        """
        Check if any combat actually happened this turn.
        
        Used for stalemate detection.
        
        Args:
            results: Combat results from resolve_all()
        
        Returns:
            True if at least one shot was fired
        """
        return any(result.success for result in results)
    
    def get_combat_summary(self, results: List[CombatResult]) -> Dict[str, int]:
        """
        Get summary statistics about combat.
        
        Args:
            results: Combat results from resolve_all()
        
        Returns:
            Dictionary with statistics
        """
        fired_results = [r for r in results if r.success]
        
        return {
            "attempted": len(results),
            "fired": len(fired_results),
            "hits": sum(1 for r in fired_results if r.hit),
            "misses": sum(1 for r in fired_results if not r.hit),
            "kills": sum(1 for r in fired_results if r.target_killed),
            "blocked": len(results) - len(fired_results),
        }
    
    def get_total_missiles_remaining(self, world: WorldState) -> int:
        """
        Count total missiles remaining across all entities.
        
        Utility for victory condition checking.
        
        Args:
            world: Current world state
        
        Returns:
            Total number of missiles
        """
        total = 0
        for entity in world.get_alive_entities():
            if hasattr(entity, 'missiles'):
                total += entity.missiles
        return total

    def apply_pending_deaths(self, world: WorldState) -> tuple[List[str], List[int]]:
        """
        Apply all pending kills marked during combat.
        
        This method extracts the kill application logic that was previously
        in WorldState.apply_kills(). WorldState now only stores the pending
        kills, this method performs the actual state mutation and logging.
        
        Args:
            world: Current world state
        
        Returns:
            Tuple of (death_logs, killed_entity_ids)
                death_logs: Human-readable messages about deaths
                killed_entity_ids: List of entity IDs that were killed
        """
        logs = []
        killed_ids = []
        
        for entity_id in world.get_pending_kills():
            entity = world.get_entity(entity_id)
            if entity and entity.alive:
                entity.alive = False
                logs.append(f"{entity.label()} was destroyed!")
                killed_ids.append(entity_id)
        
        # Clear pending kills after applying
        world.clear_pending_kills()
        
        return logs, killed_ids
