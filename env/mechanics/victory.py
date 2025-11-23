"""
Victory condition checking for the Grid Combat Environment.

This module provides pure logic for determining game outcomes:
- AWACS destruction (primary win condition)
- Enemy elimination (all enemy entities destroyed)
- Stalemate detection (no combat activity)
- Stagnation detection (no movement)
- Resource exhaustion (out of missiles)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from ..core.types import Team, GameResult, EntityKind

if TYPE_CHECKING:
    from ..world.world import WorldState
    from ..entities.base import Entity


@dataclass
class VictoryResult:
    """
    Result of a victory condition check.
    
    Attributes:
        result: Game outcome (IN_PROGRESS, BLUE_WINS, RED_WINS, DRAW)
        reason: Human-readable explanation of the outcome
        winner: Winning team (None if draw or in progress)
    """
    result: GameResult
    reason: str
    winner: Optional[Team] = None
    
    @property
    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self.result != GameResult.IN_PROGRESS
    
    def __str__(self) -> str:
        """Human-readable representation."""
        if self.result == GameResult.IN_PROGRESS:
            return "Game in progress"
        return f"{self.result}: {self.reason}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize victory result to a plain dict."""
        return {
            "result": self.result.name,
            "reason": self.reason,
            "winner": self.winner.name if self.winner else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VictoryResult":
        """Deserialize a victory result from a dict."""
        winner_name = data.get("winner")
        winner = Team[winner_name] if winner_name else None
        return cls(
            result=GameResult[data["result"]],
            reason=data.get("reason", ""),
            winner=winner,
        )


class VictoryConditions:
    """
    Stateless checker for game victory conditions.
    
    This class provides methods to check various end-game conditions
    without maintaining state. All state is passed in via the WorldState
    and tracking counters.
    
    Usage:
        checker = VictoryConditions(
            max_stalemate_turns=60,
            max_no_move_turns=15,
            check_missile_exhaustion=True
        )
        result = checker.check_all(world)
        
        if result.is_game_over:
            print(f"Game Over: {result.reason}")
            if result.winner:
                print(f"Winner: {result.winner.name}")
    """
    
    def __init__(
        self,
        max_stalemate_turns: int = 60,
        max_no_move_turns: int = 15,
        max_turns: Optional[int] = None,
        check_missile_exhaustion: bool = True
    ):
        """
        Initialize victory condition checker.
        
        Args:
            max_stalemate_turns: Max turns without shooting before draw (default: 60)
            max_no_move_turns: Max turns without movement before draw (default: 15)
            check_missile_exhaustion: Check if all missiles depleted (default: True)
        """
        self._max_stalemate_turns = max_stalemate_turns
        self._max_no_move_turns = max_no_move_turns
        self._max_turns = max_turns
        self._check_missile_exhaustion = check_missile_exhaustion
    
    def check_all(self, world: WorldState) -> VictoryResult:
        """
        Check all victory conditions in priority order.
        
        Checks are performed in this order:
        1. AWACS destruction (primary win condition)
        2. Enemy elimination (all enemy entities destroyed)
        3. Missile exhaustion
        4. Turn limit reached
        5. Combat stalemate (no shooting)
        6. Movement stagnation (no movement)
        
        Args:
            world: Current world state (contains turns_without_shooting and turns_without_movement)
        
        Returns:
            VictoryResult indicating game outcome
        """
        # Priority 1: AWACS destruction (most important)
        result = self.check_awacs_destruction(world)
        if result.is_game_over:
            return result
        
        # Priority 2: Enemy elimination
        result = self.check_all_enemies_destroyed(world)
        if result.is_game_over:
            return result
        
        # Priority 3: Resource exhaustion
        if self._check_missile_exhaustion:
            result = self.check_missile_exhaustion(world)
            if result.is_game_over:
                return result
        
        # Priority 4: Turn cap
        result = self.check_turn_limit(world.turn)
        if result.is_game_over:
            return result
        
        # Priority 5: Combat stalemate
        result = self.check_combat_stalemate(world.turns_without_shooting)
        if result.is_game_over:
            return result
        
        # Priority 6: Movement stagnation
        result = self.check_movement_stagnation(world.turns_without_movement)
        if result.is_game_over:
            return result
        
        # No victory condition met
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason="Game ongoing",
            winner=None
        )
    
    def check_awacs_destruction(self, world: WorldState) -> VictoryResult:
        """
        Check if any AWACS have been destroyed.
        
        This is the primary win condition:
        - If one team's AWACS is destroyed, the other team wins
        - If both AWACS are destroyed, it's a draw
        - If no AWACS exist in scenario, skip this check
        
        Args:
            world: Current world state
        
        Returns:
            VictoryResult indicating AWACS-based outcome
        """
        # Check if AWACS exist in the scenario at all
        blue_awacs_exists = any(
            e.kind == EntityKind.AWACS and e.team == Team.BLUE
            for e in world.get_all_entities()
        )
        
        red_awacs_exists = any(
            e.kind == EntityKind.AWACS and e.team == Team.RED
            for e in world.get_all_entities()
        )
        
        # If no AWACS in scenario, skip this victory condition
        if not blue_awacs_exists and not red_awacs_exists:
            return VictoryResult(
                result=GameResult.IN_PROGRESS,
                reason="No AWACS in scenario",
                winner=None
            )
        
        # Check if AWACS are alive
        blue_awacs_alive = any(
            e.kind == EntityKind.AWACS and e.team == Team.BLUE and e.alive
            for e in world.get_all_entities()
        )
        
        red_awacs_alive = any(
            e.kind == EntityKind.AWACS and e.team == Team.RED and e.alive
            for e in world.get_all_entities()
        )
        
        # Both AWACS destroyed -> Draw (only if both exist in scenario)
        if blue_awacs_exists and red_awacs_exists and not blue_awacs_alive and not red_awacs_alive:
            return VictoryResult(
                result=GameResult.DRAW,
                reason="Both AWACS destroyed - DRAW",
                winner=None
            )
        
        # Blue AWACS destroyed -> Red wins (only if Blue AWACS exists in scenario)
        if blue_awacs_exists and not blue_awacs_alive:
            return VictoryResult(
                result=GameResult.RED_WINS,
                reason="BLUE AWACS destroyed - RED WINS",
                winner=Team.RED
            )
        
        # Red AWACS destroyed -> Blue wins (only if Red AWACS exists in scenario)
        if red_awacs_exists and not red_awacs_alive:
            return VictoryResult(
                result=GameResult.BLUE_WINS,
                reason="RED AWACS destroyed - BLUE WINS",
                winner=Team.BLUE
            )
        
        # Both AWACS alive -> game continues
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason="Both AWACS alive",
            winner=None
        )
    
    def check_all_enemies_destroyed(self, world: WorldState) -> VictoryResult:
        """
        Check if all enemy entities have been destroyed.
        
        If all entities of one team are destroyed, the other team wins.
        If all entities of both teams are destroyed, it's a draw.
        
        Args:
            world: Current world state
        
        Returns:
            VictoryResult indicating enemy elimination outcome
        """
        # Get alive entities for each team
        blue_alive = world.get_team_entities(Team.BLUE, alive_only=True)
        red_alive = world.get_team_entities(Team.RED, alive_only=True)
        
        # Both teams eliminated -> Draw
        if len(blue_alive) == 0 and len(red_alive) == 0:
            return VictoryResult(
                result=GameResult.DRAW,
                reason="All entities destroyed - DRAW",
                winner=None
            )
        
        # All Blue entities destroyed -> Red wins
        if len(blue_alive) == 0:
            return VictoryResult(
                result=GameResult.RED_WINS,
                reason="All BLUE entities destroyed - RED WINS",
                winner=Team.RED
            )
        
        # All Red entities destroyed -> Blue wins
        if len(red_alive) == 0:
            return VictoryResult(
                result=GameResult.BLUE_WINS,
                reason="All RED entities destroyed - BLUE WINS",
                winner=Team.BLUE
            )
        
        # Both teams still have entities -> game continues
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason="Both teams have surviving entities",
            winner=None
        )
    
    def check_missile_exhaustion(self, world: WorldState) -> VictoryResult:
        """
        Check if all missiles have been depleted.
        
        If no entity has any missiles left, the game ends in a draw
        since no further combat is possible.
        
        Args:
            world: Current world state
        
        Returns:
            VictoryResult indicating missile exhaustion outcome
        """
        total_missiles = 0
        
        for entity in world.get_alive_entities():
            # Check if entity has missiles attribute (Shooter entities)
            if hasattr(entity, 'missiles'):
                total_missiles += entity.missiles
        
        if total_missiles == 0:
            return VictoryResult(
                result=GameResult.DRAW,
                reason="No missiles remaining - DRAW",
                winner=None
            )
        
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason=f"{total_missiles} missiles remaining",
            winner=None
        )
    
    def check_turn_limit(self, current_turn: int) -> VictoryResult:
        """
        Check if the global turn limit has been reached.
        
        Args:
            current_turn: The turn counter from the world
        
        Returns:
            VictoryResult indicating outcome when hitting the turn cap
        """
        if self._max_turns is not None and current_turn >= self._max_turns:
            return VictoryResult(
                result=GameResult.DRAW,
                reason=f"Turn limit reached ({self._max_turns}) - DRAW",
                winner=None
            )
        
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason="Turn limit not reached",
            winner=None
        )
    
    def check_combat_stalemate(self, turns_without_shooting: int) -> VictoryResult:
        """
        Check for combat stalemate.
        
        If no entity has fired a weapon for too many consecutive turns,
        the game ends in a draw (assumed stalemate).
        
        Args:
            turns_without_shooting: Number of consecutive turns without shooting
        
        Returns:
            VictoryResult indicating stalemate outcome
        """
        if turns_without_shooting >= self._max_stalemate_turns:
            return VictoryResult(
                result=GameResult.DRAW,
                reason=f"Stalemate - No missiles fired for {self._max_stalemate_turns} turns - DRAW",
                winner=None
            )
        
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason=f"Combat active ({turns_without_shooting}/{self._max_stalemate_turns} idle turns)",
            winner=None
        )
    
    def check_movement_stagnation(self, turns_without_movement: int) -> VictoryResult:
        """
        Check for movement stagnation.
        
        If no entity has moved for too many consecutive turns,
        the game ends in a draw (assumed stagnation/defensive stalemate).
        
        Args:
            turns_without_movement: Number of consecutive turns without movement
        
        Returns:
            VictoryResult indicating stagnation outcome
        """
        if turns_without_movement >= self._max_no_move_turns:
            return VictoryResult(
                result=GameResult.DRAW,
                reason=f"Stagnation - No movement for {self._max_no_move_turns} turns - DRAW",
                winner=None
            )
        
        return VictoryResult(
            result=GameResult.IN_PROGRESS,
            reason=f"Movement active ({turns_without_movement}/{self._max_no_move_turns} idle turns)",
            winner=None
        )
    
    def get_quick_stats(self, world: WorldState) -> dict:
        """
        Get quick statistics about game state for debugging/display.
        
        Args:
            world: Current world state
        
        Returns:
            Dictionary with statistics about alive entities and resources
        """
        stats = {
            "blue": {
                "total": 0,
                "alive": 0,
                "awacs_alive": False,
                "missiles": 0,
            },
            "red": {
                "total": 0,
                "alive": 0,
                "awacs_alive": False,
                "missiles": 0,
            },
            "total_missiles": 0,
        }
        
        for entity in world.get_all_entities():
            team_key = "blue" if entity.team == Team.BLUE else "red"
            stats[team_key]["total"] += 1
            
            if entity.alive:
                stats[team_key]["alive"] += 1
                
                if entity.kind == EntityKind.AWACS:
                    stats[team_key]["awacs_alive"] = True
                
                if hasattr(entity, 'missiles'):
                    missiles = entity.missiles
                    stats[team_key]["missiles"] += missiles
                    stats["total_missiles"] += missiles
        
        return stats
