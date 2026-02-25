

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import Counter
import math

from env.core.types import Team, EntityKind
from env.world import WorldState
from env.world.team_view import TeamView
from env.mechanics.combat import hit_probability


@dataclass
class UnitObservation:
    """Single unit observation with tactical intelligence."""
    id: int
    type: str
    dist: Optional[int] = None  # Manhattan distance (tactical)
    pos: Optional[Tuple[int, int]] = None
    rel_pos: Optional[Tuple[int, int]] = None
    bearing: Optional[str] = None  # NEW: "N", "NE", "E", "SE", "S", "SW", "W", "NW"
    threat: Optional[str] = None  # "CRITICAL|HIGH|MEDIUM|LOW"
    hit_prob: Optional[float] = None  # Probability I can hit this target
    can_they_hit_me: Optional[bool] = None  # Am I in their weapon range?
    hit_prob_from_enemy: Optional[float] = None  # NEW: Probability THEY can hit ME
    missiles: Optional[int] = None


@dataclass
class LocalState:
    """Local tactical view for a single unit."""
    id: int
    type: str
    pos: Tuple[int, int]
    missiles: int
    radar: bool
    range: int
    allies: List[UnitObservation]
    enemies: List[UnitObservation]


@dataclass
class GlobalState: 
    """Global strategic view for commander - enhanced with counts and enemy intent."""
    turn: int
    grid: Tuple[int, int]
    allied: List[UnitObservation]
    enemies: List[UnitObservation]
    counts: dict
    hvts: List[int] = None
    # NEW: Enhanced state
    ally_counts: Dict[str, int] = field(default_factory=dict)
    enemy_counts: Dict[str, int] = field(default_factory=dict)
    enemy_intent: str = "UNKNOWN"  # Inferred enemy formation/strategy
    our_awacs_pos: Optional[Tuple[int, int]] = None  # For threat calculations
    
    def __post_init__(self):
        if self.hvts is None:
            self.hvts = []


def calc_bearing(rel_pos: Tuple[int, int]) -> str:
    """Convert relative position to 8-direction bearing."""
    dx, dy = rel_pos
    
    if dx == 0 and dy == 0:
        return "HERE"
    
    # Calculate angle in degrees (0 = East, 90 = North)
    angle = math.atan2(-dy, dx) * 180 / math.pi  # -dy because y increases downward
    
    # Normalize to 0-360
    angle = (angle + 360) % 360
    
    # Map to 8 directions (each covers 45 degrees)
    if angle < 22.5 or angle >= 337.5:
        return "E"
    elif angle < 67.5:
        return "NE"
    elif angle < 112.5:
        return "N"
    elif angle < 157.5:
        return "NW"
    elif angle < 202.5:
        return "W"
    elif angle < 247.5:
        return "SW"
    elif angle < 292.5:
        return "S"
    else:
        return "SE"


def infer_enemy_intent(enemies: List[UnitObservation], 
                       awacs_pos: Optional[Tuple[int, int]], 
                       grid: Tuple[int, int]) -> str:
    """Infer enemy formation/strategy from spatial distribution."""
    if not enemies:
        return "NO_CONTACT"
    
    if len(enemies) == 1:
        return "SINGLE_CONTACT"
    
    # Calculate centroid
    positions = [e.pos for e in enemies if e.pos]
    if not positions:
        return "UNKNOWN"
    
    avg_x = sum(p[0] for p in positions) / len(positions)
    avg_y = sum(p[1] for p in positions) / len(positions)
    
    # Calculate spread (standard deviation)
    spread_x = math.sqrt(sum((p[0] - avg_x)**2 for p in positions) / len(positions))
    spread_y = math.sqrt(sum((p[1] - avg_y)**2 for p in positions) / len(positions))
    total_spread = math.sqrt(spread_x**2 + spread_y**2)
    
    # Check if clustering toward our AWACS
    if awacs_pos:
        distances_to_awacs = [
            abs(p[0] - awacs_pos[0]) + abs(p[1] - awacs_pos[1]) 
            for p in positions
        ]
        avg_dist_to_awacs = sum(distances_to_awacs) / len(distances_to_awacs)
        
        if avg_dist_to_awacs < 8 and total_spread < 4:
            return "AWACS_HUNT"
    
    # Check formation type
    grid_center_x = grid[0] / 2
    
    if total_spread > 6:
        return "FLANKING"
    elif total_spread < 3:
        if avg_x < grid_center_x * 0.6:
            return "DEFENSIVE_HOLD"
        elif avg_x > grid_center_x * 1.4:
            return "AGGRESSIVE_PUSH"
        else:
            return "CONCENTRATED_CENTER"
    else:
        return "SCATTERED"


class StateBuilder:
    """Builds optimized state representations with combat intelligence."""
    
    # Enemy weapon ranges (for threat calculation)
    ENEMY_RANGES = {
        "AIRCRAFT": 10,  
        "SAM": 8,
        "AWACS": 0,
        "DECOY": 0
    }
    
    # Entity range defaults
    ENTITY_DEFAULTS = {
        EntityKind.AIRCRAFT: 10,  
        EntityKind.SAM: 8,
        EntityKind.AWACS: 15,
        EntityKind.DECOY: 0
    }
    
    @staticmethod
    def build_global_state(world: WorldState, team: Team) -> GlobalState:
        """Build strategic commander view using TeamView - enhanced with counts and enemy intent."""
        view: TeamView = world.get_team_view(team)
        
        # Build allied units list
        allied_units = []
        ally_counts = Counter()
        our_awacs_pos = None
        
        for entity_id in view.get_friendly_ids():
            entity = world.get_entity(entity_id)
            if entity and entity.alive:
                unit_type = entity.kind.name
                ally_counts[unit_type] += 1
                
                allied_units.append(UnitObservation(
                    id=entity.id,
                    type=unit_type,
                    pos=entity.pos,
                    missiles=getattr(entity, 'missiles', 0)
                ))
                
                # Track AWACS position
                if entity.kind == EntityKind.AWACS:
                    our_awacs_pos = entity.pos
        
        # Build enemy units list (only visible via sensors)
        enemy_units = []
        enemy_counts = Counter()
        hvts = []
        
        for obs in view.get_enemy_observations():
            unit_type = obs.kind.name
            enemy_counts[unit_type] += 1
            
            enemy_units.append(UnitObservation(
                id=obs.entity_id,
                type=unit_type,
                pos=obs.position
            ))
            
            # Mark HVTs
            if obs.kind in (EntityKind.AWACS, EntityKind.SAM):
                hvts.append(obs.entity_id)
        
        # Unified counts (legacy format)
        cnt = Counter([u.type for u in allied_units] + [f"EN_{u.type}" for u in enemy_units])
        
        # Infer enemy intent
        enemy_intent = infer_enemy_intent(
            enemy_units, 
            our_awacs_pos, 
            (world.grid.width, world.grid.height)
        )
        
        return GlobalState(
            turn=world.turn,
            grid=(world.grid.width, world.grid.height),
            allied=allied_units,
            enemies=enemy_units,
            counts=dict(cnt),
            hvts=hvts,
            ally_counts=dict(ally_counts),
            enemy_counts=dict(enemy_counts),
            enemy_intent=enemy_intent,
            our_awacs_pos=our_awacs_pos
        )
    
    @staticmethod
    def build_local_state(
        entity_id: int,
        world: WorldState,
        team: Team
    ) -> LocalState:
        """Build tactical unit-specific view with enhanced threat logic."""
        entity = world.get_entity(entity_id)
        view: TeamView = world.get_team_view(team)
        
        my_pos = entity.pos
        my_type = entity.kind.name

        my_range = getattr(entity, 'missile_max_range', 
                        getattr(entity, 'radar_range', 
                                StateBuilder.ENTITY_DEFAULTS.get(entity.kind, 10)))
        
        my_base_hit = getattr(entity, 'base_hit_prob', 0.7)
        my_min_hit = getattr(entity, 'min_hit_prob', 0.1)
        
        # Helper functions
        def get_rel(target_pos):
            """Calculate relative position (dx, dy)."""
            return (target_pos[0] - my_pos[0], target_pos[1] - my_pos[1])
        
        def get_dist_manhattan(target_pos) -> int:
            """Manhattan distance for threat assessment (grid movement)."""
            return world.grid.manhattan_distance(my_pos, target_pos)
        
        def get_dist_euclidean(target_pos) -> float:
            """Euclidean distance for hit probability (straight line)."""
            return world.grid.distance(my_pos, target_pos)

        def calc_threat_v2(dist_manhattan: int, enemy_type: str, in_their_range: bool) -> str:
            """Enhanced threat calculation with enemy range consideration."""
            enemy_type_upper = enemy_type.upper()
            
            # Non-combat units (lower threat)
            if enemy_type_upper == "DECOY":
                if dist_manhattan <= 4:
                    return "MEDIUM"  # Distraction risk
                else:
                    return "LOW"
            
            if enemy_type_upper == "AWACS":
                if dist_manhattan <= 4:
                    return "HIGH"  # Intel advantage
                elif dist_manhattan <= 8:
                    return "MEDIUM"
                else:
                    return "LOW"
            
            # Combat units: AIRCRAFT, SAM
            # Factor in whether we're in their weapon range
            if in_their_range:
                # We're in danger - escalate threat
                if dist_manhattan <= 3:
                    return "CRITICAL"
                elif dist_manhattan <= 6:
                    return "HIGH"
                else:
                    return "MEDIUM"
            else:
                # Out of their range - lower threat
                if dist_manhattan <= 3:
                    return "HIGH"  # Still close
                elif dist_manhattan <= 6:
                    return "MEDIUM"
                elif dist_manhattan <= 10:
                    return "LOW"
                else:
                    return "LOW"

        def calc_hit_prob(dist_euclidean: float) -> float:
            """Calculate hit probability using Euclidean distance."""
            if dist_euclidean > my_range:
                return 0.0
            return hit_probability(
                distance=dist_euclidean,
                max_range=my_range,
                base=my_base_hit,
                min_p=my_min_hit
            )
        
        def calc_enemy_hit_prob(dist_euclidean: float, enemy_type: str) -> float:
            """Calculate probability enemy can hit us."""
            enemy_range = StateBuilder.ENEMY_RANGES.get(enemy_type.upper(), 0)
            if enemy_range == 0 or dist_euclidean > enemy_range:
                return 0.0
            # Assume enemy has similar hit probability curve
            return hit_probability(
                distance=dist_euclidean,
                max_range=enemy_range,
                base=0.7,
                min_p=0.1
            )

        # Build allies list
        allies = []
        for ally_id in view.get_friendly_ids():
            if ally_id == entity_id:
                continue
            ally = world.get_entity(ally_id)
            if ally and ally.alive:
                dist_manhattan = get_dist_manhattan(ally.pos)
                rel = get_rel(ally.pos)
                allies.append(UnitObservation(
                    id=ally.id,
                    pos=ally.pos,
                    type=ally.kind.name,
                    rel_pos=rel,
                    bearing=calc_bearing(rel),
                    dist=dist_manhattan,
                    missiles=getattr(ally, 'missiles', 0)
                ))
        
        # Build enemies list with enhanced threat
        enemies = []
        for obs in view.get_enemy_observations():  
            dist_euclidean = get_dist_euclidean(obs.position)
            dist_manhattan = get_dist_manhattan(obs.position)
            
            enemy_type = obs.kind.name
            enemy_range = StateBuilder.ENEMY_RANGES.get(enemy_type.upper(), 0)
            
            # Am I in their weapon range?
            can_they_hit_me = dist_euclidean <= enemy_range
            
            # Calculate threat with enhanced logic
            threat = calc_threat_v2(dist_manhattan, enemy_type, can_they_hit_me)
            hit_prob = calc_hit_prob(dist_euclidean)
            hit_prob_from_enemy = calc_enemy_hit_prob(dist_euclidean, enemy_type)
            
            rel = get_rel(obs.position)
            
            enemies.append(UnitObservation(
                id=obs.entity_id,
                type=enemy_type,
                rel_pos=rel,
                bearing=calc_bearing(rel),
                dist=dist_manhattan,
                threat=threat,
                hit_prob=round(hit_prob, 2),
                can_they_hit_me=can_they_hit_me,
                hit_prob_from_enemy=round(hit_prob_from_enemy, 2)
            ))
                        
        return LocalState(
            id=entity_id,
            type=my_type,
            pos=my_pos,
            missiles=getattr(entity, 'missiles', 0),
            radar=getattr(entity, 'radar_on', False),
            range=my_range,
            allies=allies,
            enemies=enemies
        )