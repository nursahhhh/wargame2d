# Grid Combat Environment - Architecture Guide

A comprehensive guide to the modular architecture of the Grid Combat Environment 2.0.

---

## 📦 Module Overview

The codebase is organized into **7 clean, modular components**, each with a single, clear responsibility:

```
env/
├── core/           ✅ Fundamental types and data structures
├── entities/       ✅ Game entities (Aircraft, SAM, AWACS, Decoy)
├── world/          ✅ Spatial logic and state management
├── mechanics/      ✅ Action resolution and game rules
├── utils/          ✅ Helper utilities
├── scenario.py     ✅ Scenario system for creating game setups
└── environment.py  ✅ Main gym-like interface
```

---

## 🔍 Detailed Module Documentation


## 1. `core/` - Fundamental Types ✅

**Purpose:** Pure data structures with NO game logic. These are the building blocks used by all other modules.

### Files

#### `types.py` - Core Type Definitions
**What:** Enums and type aliases (pure types, no values)
**Why:** Shared types prevent duplication and circular dependencies
**Exports:**
- `GridPos` - Type alias for `Tuple[int, int]` (x, y coordinates)
- `Team` - Enum for BLUE/RED teams
- `ActionType` - Enum for WAIT/MOVE/SHOOT/TOGGLE
- `MoveDir` - Enum for UP/DOWN/LEFT/RIGHT with delta tuples
- `EntityKind` - Enum for AIRCRAFT/AWACS/SAM/DECOY
- `GameResult` - Enum for game outcomes

**Example Usage:**
```python
from env.core import Team, MoveDir, GridPos

# Teams
team = Team.BLUE
enemy = team.opponent  # Returns Team.RED

# Movement directions with deltas
direction = MoveDir.UP
dx, dy = direction.delta  # (0, 1) - Y increases upward

# Type-safe positions
pos: GridPos = (5, 10)
```

#### `actions.py` - Action System
**What:** Action dataclass with validation, serialization, and factory methods
**Why:** Centralized action handling ensures consistency
**Exports:**
- `Action` - Main action dataclass

**Example Usage:**
```python
from env.core.actions import Action
from env.core import MoveDir

# Factory methods (preferred)
action = Action.wait()
action = Action.move(MoveDir.UP)
action = Action.shoot(target_id=5)
action = Action.toggle(on=True)

# Direct construction (also valid)
action = Action(ActionType.MOVE, {"dir": MoveDir.LEFT})

# Serialization
json_str = action.to_json()
action = Action.from_json(json_str)

# Validation happens automatically
action = Action.move("invalid")  # Raises ValueError
```

#### `observations.py` - Observation System
**What:** Observation dataclass and ObservationSet helper
**Why:** Manages what entities know about others (fog of war)
**Exports:**
- `Observation` - Single observation dataclass
- `ObservationSet` - Collection with filtering

**Example Usage:**
```python
from env.core.observations import Observation, ObservationSet
from env.core import Team, EntityKind

# Create observation
obs = Observation(
    entity_id=5,
    kind=EntityKind.AIRCRAFT,
    team=Team.RED,
    position=(10, 15),
    distance=3.5,
    seen_by={1, 2}  # IDs of observers
)

# Check team
if obs.is_enemy(Team.BLUE):
    print("Enemy spotted!")

# Collection with filtering
obs_set = ObservationSet()
obs_set.add(obs)
enemies = obs_set.filter_by_team(Team.RED)
aircraft = obs_set.filter_by_kind(EntityKind.AIRCRAFT)
closest = obs_set.get_closest_enemy(Team.BLUE)
```

---

## 2. `entities/` - Game Entities ✅

**Purpose:** Self-contained entity definitions. Each entity knows what it CAN do, but NOT how actions are resolved.

### Design Principle
**Entities define capabilities, Mechanics execute actions.**

This separation allows:
- Easy testing of entity logic
- Flexible action resolution (swap mechanics implementations)
- Clear contracts via abstract methods

### Files

#### `base.py` - Entity Base Class
**What:** Abstract base class defining the entity contract
**Why:** Ensures all entities have consistent interfaces
**Key Features:**
- Abstract `get_allowed_actions()` method
- Common properties (id, team, pos, alive, radar_range)
- Serialization framework
- Capability flags (can_move, can_shoot)

**Example Usage:**
```python
from env.entities import Entity

# All entities share common interface
entity.get_allowed_actions(world)  # What can I do?
entity.get_active_radar_range()    # Current radar range
entity.label()                     # "Aircraft#1(BLUE)"
entity.to_dict()                   # Serialize to dict
```

#### `aircraft.py` - Fighter Aircraft
**What:** Mobile fighter with radar and missiles
**Why:** Primary offensive unit

**Required Stats (must be specified as keyword arguments):**
- `radar_range`: Detection range (e.g., 5.0)
- `missiles`: Number of missiles available (e.g., 4)
- `missile_max_range`: Maximum firing range (e.g., 4.0)
- `base_hit_prob`: Hit probability at close range (e.g., 0.8)
- `min_hit_prob`: Minimum hit probability at max range (e.g., 0.1)

**Capabilities:**
- Can move: Yes
- Can shoot: Yes

**Example Usage:**
```python
from env.entities import Aircraft
from env.core import Team

aircraft = Aircraft(
    team=Team.BLUE,
    pos=(5, 5),
    name="Blue-Fighter-1",
    radar_range=5.0,
    missiles=4,
    missile_max_range=4.0,
    base_hit_prob=0.8,
    min_hit_prob=0.1
)

# Get available actions (movement + shooting at visible enemies)
actions = aircraft.get_allowed_actions(world)

# Serialization includes combat stats
data = aircraft.to_dict()  # Includes missiles, ranges, etc.
```

#### `awacs.py` - Surveillance Aircraft
**What:** Airborne early warning and control system
**Why:** Extended radar for team, high-value target

**Required Stats (must be specified as keyword arguments):**
- `radar_range`: Detection range (typically 9.0 - much larger than aircraft!)

**Capabilities:**
- Missiles: 0 (unarmed)
- Can move: Yes
- Can shoot: No

**Special:** Game ends when AWACS destroyed (primary victory condition)

**Example Usage:**
```python
from env.entities import AWACS

awacs = AWACS(
    team=Team.BLUE,
    pos=(3, 3),
    name="Blue-Eye",
    radar_range=9.0
)

# Can only move and wait (no weapons)
actions = awacs.get_allowed_actions(world)
# Returns: [Action.wait(), Action.move(UP), Action.move(DOWN), ...]
```

#### `sam.py` - Surface-to-Air Missile System
**What:** Stationary air defense with toggle-able radar
**Why:** Defensive unit with stealth mechanics

**Required Stats (must be specified as keyword arguments):**
- `radar_range`: Detection range when ON (e.g., 8.0)
- `missiles`: Number of missiles available (e.g., 6)
- `missile_max_range`: Maximum firing range (e.g., 6.0)
- `base_hit_prob`: Hit probability at close range (e.g., 0.8)
- `min_hit_prob`: Minimum hit probability at max range (e.g., 0.1)
- `cooldown_steps`: Turns to wait after firing (e.g., 5)

**Optional Parameters:**
- `on`: Whether radar starts ON or OFF (default: False)

**Capabilities:**
- Can move: No (stationary)
- Can shoot: Yes

**Special Mechanics:**
- Radar can be toggled ON/OFF
- Only visible to enemies when radar is ON
- Must cool down after firing

**Example Usage:**
```python
from env.entities import SAM

sam = SAM(
    team=Team.RED,
    pos=(18, 10),
    radar_range=8.0,
    missiles=6,
    missile_max_range=6.0,
    base_hit_prob=0.8,
    min_hit_prob=0.1,
    cooldown_steps=5,
    on=False  # Optional: radar starts OFF
)

# Get active radar range (respects ON/OFF state)
radar = sam.get_active_radar_range()  # 8.0 if ON, 0.0 if OFF

# Toggle radar
action = Action.toggle(on=True)

# Manage cooldown (called by environment)
sam.tick_cooldown()      # Decrease counter
sam.start_cooldown()     # Start after firing
```

#### `decoy.py` - Deceptive Unit
**What:** Unit that appears as aircraft to enemies
**Why:** Misdirection and tactical deception

**Stats:**
- Radar range: 0.0 (no sensors, hardcoded)
- Can move: Yes
- Can shoot: No

**Special:** Appears as `EntityKind.AIRCRAFT` to enemy observations

**Example Usage:**
```python
from env.entities import Decoy

decoy = Decoy(
    team=Team.BLUE,
    pos=(8, 8),
    name="Bait-1"
)

# Decoys can only move
actions = decoy.get_allowed_actions(world)
# Returns: [Action.wait(), Action.move(UP), ...]

# Enemies see it as aircraft (handled by SensorSystem)
```

### Serialization Notes

**When to override `to_dict()`:**
- ✅ Aircraft and SAM - Have extra fields (missiles, ranges)
- ❌ AWACS and Decoy - Only use base Entity fields

```python
# Aircraft overrides to include combat stats
class Aircraft(Entity):
    def to_dict(self):
        data = super().to_dict()
        data["missiles"] = self.missiles
        data["missile_max_range"] = self.missile_max_range
        # etc...
        return data

# AWACS doesn't need override
class AWACS(Entity):
    pass  # Uses base Entity.to_dict()
```

---

## 3. `world/` - Spatial Logic & State ✅

**Purpose:** Manages where entities are and tracks game state.

### Files

#### `grid.py` - Spatial Geometry
**What:** Pure spatial calculations (no game logic)
**Why:** Separates geometry from game rules

**Coordinate System:**
- X increases to the RIGHT
- Y increases UPWARD (mathematical convention)
- Origin (0, 0) is at BOTTOM-LEFT

**Key Methods:**
```python
from env.world import Grid

grid = Grid(width=20, height=20)

# Bounds checking
grid.in_bounds((5, 5))  # True
grid.in_bounds((-1, 0))  # False

# Distance calculations
dist = grid.distance((0, 0), (3, 4))  # 5.0 (Euclidean)
dist = grid.manhattan_distance((0, 0), (3, 4))  # 7 (taxicab)

# Neighbors
neighbors = grid.get_neighbors((5, 5))  # 4 cardinal directions
neighbors = grid.get_neighbors((5, 5), include_diagonals=True)  # 8 dirs

# Range queries
positions = grid.positions_in_range(center=(10, 10), max_range=5.0)

# Coordinate conversion (for rendering)
screen_y = grid.to_screen_y(math_y=15)  # Y=0 at top for rendering
math_y = grid.to_math_y(screen_y=5)     # Y=0 at bottom for math
```

**When to use Grid:**
- ✅ Position validation
- ✅ Distance calculations
- ✅ Finding neighbors
- ✅ Range queries
- ✅ Coordinate conversions

**When NOT to use Grid:**
- ❌ Hit probability (that's `rules/physics.py`)
- ❌ Victory checking (that's `rules/victory.py`)

#### `world.py` - World State Management
**What:** Central game state manager
**Why:** Single source of truth for all entities and game status

**Responsibilities:**
- Entity registry (add, remove, query)
- Kill tracking
- Team view access
- Turn counter
- Game over state
- Serialization/deserialization

**Example Usage:**
```python
from env.world import WorldState
from env.entities import Aircraft
from env.core import Team

# Create world
world = WorldState(width=20, height=20, seed=42)

# Add entities
aircraft_id = world.add_entity(Aircraft(team=Team.BLUE, pos=(5, 5)))

# Query entities
entity = world.get_entity(aircraft_id)
all_entities = world.get_all_entities()
alive_entities = world.get_alive_entities()
blue_entities = world.get_team_entities(Team.BLUE, alive_only=True)

# Check positions
occupied = world.is_position_occupied((5, 5))

# Kill tracking (state storage only - CombatResolver applies deaths)
world.mark_for_kill(entity_id)
pending = world.get_pending_kills()  # Get pending deaths
world.clear_pending_kills()  # Clear after applying

# Access team intelligence
blue_view = world.get_team_view(Team.BLUE)

# Serialization (includes TeamViews with enemy firing history)
json_str = world.to_json()
world = WorldState.from_json(json_str)
clone = world.clone()  # Deep copy for simulation
```

#### `team_view.py` - Per-Team Intelligence
**What:** Aggregated observations and intelligence for one team
**Why:** Implements fog of war - teams only see what they observe

**Responsibilities:**
- Collect observations from all friendly entities
- Track visible enemies (for targeting)
- Record enemy firing history (persistent state - serialized)
- Provide querying interface
- Serialization support (to_dict/from_dict)

**Example Usage:**
```python
# Get team view
blue_view = world.get_team_view(Team.BLUE)

# Check what's visible
enemy_ids = blue_view.get_enemy_ids(Team.BLUE)
can_target = blue_view.can_target(enemy_id=5)

# Get observations
all_obs = blue_view.get_all_observations()
enemy_obs = blue_view.get_enemy_observations()
friendly_obs = blue_view.get_friendly_observations()
specific_obs = blue_view.get_observation(entity_id=5)

# Track enemy behavior
blue_view.record_enemy_fired(enemy_id=5)
has_fired = blue_view.has_enemy_fired(enemy_id=5)  # Strategic intel
```

---

## 4. `mechanics/` - Action Resolution & Game Rules ✅

**Purpose:** Stateless resolvers that execute game actions, calculate game rules, and check victory conditions.

**Design Philosophy:** Mechanics don't store state - they operate on `WorldState` and return detailed result objects.

### Files

#### `combat.py` - Combat Resolution & Hit Probability
**What:** Validates and executes shooting actions; calculates hit probabilities
**Why:** Centralized combat logic with detailed tracking

**Hit Probability Function:**
```python
from env.mechanics import hit_probability

# Calculate hit probability based on distance
prob = hit_probability(
    distance=3.5,
    max_range=5.0,
    base=0.8,      # 80% at distance 0
    min_p=0.1      # 10% at max range
)
# Returns probability between min_p and base, linear falloff
```

**Formula:**
```
At distance 0:         probability = base (e.g., 0.8)
At distance=max_range: probability = min_p (e.g., 0.1)
Beyond max_range:      probability = min_p (clamped)
Linear interpolation in between
```

**Combat Resolution Features:**
- Target validation
- Range checking
- Hit probability calculation
- Ammunition management
- SAM cooldown management
- Kill tracking (marks entities for death)
- Death application (applies pending kills to entities)
- Randomized execution order

**Example Usage:**
```python
from env.mechanics import CombatResolver

combat = CombatResolver()

# Resolve all combat actions
combat_result = combat.resolve_combat(world, actions, randomize_order=True)

# Check results
for result in combat_result.combat_results:
    if result.success:
        print(result.log)
        print(f"  Distance: {result.distance:.2f}")
        print(f"  Hit prob: {result.hit_probability:.2%}")
        print(f"  Result: {'HIT' if result.hit else 'MISS'}")
        if result.target_killed:
            print("  TARGET DESTROYED!")
    else:
        print(f"Shot blocked: {result.log}")

# Check if any combat occurred (for stalemate)
if combat_result.combat_occurred:
    print("Combat happened this turn")

# Apply pending deaths (marked during combat)
death_logs, killed_ids = combat_result.death_logs, combat_result.killed_entity_ids
print(f"Entities destroyed: {killed_ids}")

# Utility - check remaining ammunition
total_missiles = combat.get_total_missiles_remaining(world)
```

**CombatResult Structure:**
```python
@dataclass
class CombatResult:
    attacker_id: int
    target_id: int | None
    success: bool  # Was shot fired?
    hit: bool | None  # Did it hit?
    distance: float | None
    hit_probability: float | None
    target_killed: bool
    log: str
```

#### `sensors.py` - Observation System
**What:** Computes what entities observe based on radar
**Why:** Implements fog of war and intelligence gathering

**Features:**
- Radar-based observation
- Decoy deception (decoys appear as aircraft to enemies)
- SAM visibility (SAMs only visible when radar ON)
- Team view aggregation

**Example Usage:**
```python
from env.mechanics import SensorSystem

sensors = SensorSystem()

# Main entry point - refreshes all observations
sensors.refresh_all_observations(world)

# Now team views are updated
blue_view = world.get_team_view(Team.BLUE)
observations = blue_view.get_all_observations()

# Single entity observation (for testing/debugging)
obs_list = sensors.compute_entity_observations(world, entity)

# Check visibility
can_see = sensors.can_observe(world, observer=aircraft, target=enemy)

# Get entities in range (utility)
entities = sensors.get_entities_in_radar_range(world, observer=awacs)
```

**How Decoy Deception Works:**
```python
# Friendlies see decoys as decoys
friendly_obs.kind == EntityKind.DECOY  # True

# Enemies see decoys as aircraft
enemy_obs.kind == EntityKind.AIRCRAFT  # True (deception!)
```

#### `movement.py` - Movement Resolution
**What:** Validates and executes movement, toggle, and wait actions
**Why:** Centralized movement logic with detailed result tracking

**Features:**
- Bounds checking
- Collision detection
- SAM radar toggle
- Movement stagnation tracking
- Randomized execution order (prevents ID bias)

**Example Usage:**
```python
from env.mechanics import MovementResolver

movement = MovementResolver()

# Resolve all movement actions
result = movement.resolve_actions(world, actions, randomize_order=True)

# Check results
for move in result.movement_results:
    if move.success:
        print(f"Moved from {move.old_pos} to {move.new_pos}")
    else:
        print(f"Failed movement for {move.entity_id}: {move.failure_reason}")

# Check if any movement occurred (for stagnation)
if result.movement_occurred:
    print("Someone moved this turn")

# Combined logs (toggle/wait/invalid/movement) in execution order
for log in result.logs:
    print(log)
```

**MovementResult Structure:**
```python
@dataclass
class MovementResult:
    entity_id: int
    success: bool
    old_pos: tuple[int, int]
    new_pos: tuple[int, int]
    failure_reason: str | None  # Optional failure code
```

#### `victory.py` - Victory Conditions
**What:** Stateless checker for all win/loss/draw conditions
**Why:** Separates game logic from state management

**Victory Conditions (Priority Order):**
1. **AWACS Destruction** - Destroy enemy AWACS to win
2. **Missile Exhaustion** - Draw if no missiles left
3. **Combat Stalemate** - Draw after N turns without shooting
4. **Movement Stagnation** - Draw after N turns without movement

**Example Usage:**
```python
from env.mechanics import VictoryConditions

# Create checker with victory rules (stateless, reusable)
checker = VictoryConditions(
    max_stalemate_turns=60,      # 60 turns without combat
    max_no_move_turns=15,         # 15 turns without movement
    check_missile_exhaustion=True # Check ammo
)

# Check victory each turn (world includes turn counters)
result = checker.check_all(world)

# Handle result
if result.is_game_over:
    print(f"Game Over: {result.reason}")
    if result.winner:
        print(f"Winner: {result.winner.name}")
    else:
        print("Draw!")

# Individual checks also available (not typically needed)
awacs_result = checker.check_awacs_destruction(world)
missile_result = checker.check_missile_exhaustion(world)
```

---

## 5. `utils/` - Helper Utilities ✅

**Purpose:** Generic utilities used across modules.

### Files

#### `id_generator.py` - Entity ID Generation
**What:** Thread-safe, monotonic ID generation
**Why:** Ensures unique entity IDs across the simulation

**Example Usage:**
```python
from env.utils import get_next_entity_id, reset_entity_ids

# Get next ID (used automatically by Entity base class)
entity_id = get_next_entity_id()  # 1, 2, 3, ...

# Reset for testing (ensures reproducible IDs)
reset_entity_ids(start=1)
entity_id = get_next_entity_id()  # 1 again
```

---

## 6. `scenario.py` - Scenario System ✅

**Purpose:** Create and manage complete game setups with type-safe Python definitions and JSON serialization.

### Classes

#### `Scenario` - Complete Game Setup
**What:** Self-contained scenario definition including configuration and all entities
**Why:** Makes scenarios reproducible, portable, and easy to share

**What's Included in a Scenario:**
- Grid dimensions
- Game rules (stalemate thresholds, victory conditions)
- Random seed for reproducibility
- All entities with explicit stats

**Example Usage:**
```python
from env.scenario import Scenario
from env.entities import Aircraft, AWACS, SAM
from env.core import Team

# Option 1: Build incrementally
scenario = Scenario(
    grid_width=20,
    grid_height=20,
    max_stalemate_turns=60,
    max_no_move_turns=15,
    check_missile_exhaustion=True,
    seed=42
)

# Add Blue team entities
scenario.add_blue(Aircraft(
    team=Team.BLUE,
    pos=(5, 10),
    radar_range=5.0,
    missiles=4,
    missile_max_range=4.0,
    base_hit_prob=0.8,
    min_hit_prob=0.1
))
scenario.add_blue(AWACS(
    team=Team.BLUE,
    pos=(3, 3),
    radar_range=9.0
))

# Add Red team entities
scenario.add_red(SAM(
    team=Team.RED,
    pos=(18, 10),
    radar_range=8.0,
    missiles=6,
    missile_max_range=6.0,
    base_hit_prob=0.8,
    min_hit_prob=0.1,
    cooldown_steps=5,
    on=False
))

# Save to JSON file
scenario.save_json("my_scenario.json")

# Load from JSON file
scenario = Scenario.load_json("my_scenario.json")

# Use in environment
from env import GridCombatEnv
env = GridCombatEnv()
state = env.reset(scenario=scenario.to_dict())
```

**Configuration Options:**
```python
# Grid settings
grid_width: int = 20
grid_height: int = 20

# Random seed
seed: Optional[int] = None

# Victory conditions
max_stalemate_turns: int = 60
max_no_move_turns: int = 15
check_missile_exhaustion: bool = True
```

**Built-in Scenario Builders:**

The module includes template functions for creating common scenarios:

```python
from env.scenario import create_mixed_scenario

# Create a pre-configured scenario
scenario = create_mixed_scenario()
# Returns: Scenario with both teams, multiple entity types, balanced setup
```

---

## 7. `environment.py` - Main Interface ✅

**Purpose:** Gym-like environment wrapper that orchestrates all subsystems.

### Classes

#### `GridCombatEnv` - Main Environment
**What:** The primary API for using the simulation
**Why:** Provides clean, standardized interface

**Complete Example:**
```python
from env import GridCombatEnv
from env.scenario import Scenario
from env.entities import Aircraft, AWACS, SAM
from env.core import Team, MoveDir
from env.core.actions import Action

# 1. Create scenario
scenario = Scenario(
    grid_width=20,
    grid_height=20,
    max_stalemate_turns=60,
    max_no_move_turns=15,
    check_missile_exhaustion=True,
    seed=42
)

# 2. Add entities to scenario
blue_aircraft = Aircraft(
    team=Team.BLUE, pos=(5, 5),
    radar_range=5.0, missiles=4,
    missile_max_range=4.0,
    base_hit_prob=0.8, min_hit_prob=0.1
)
blue_awacs = AWACS(
    team=Team.BLUE, pos=(3, 3),
    radar_range=9.0
)
red_aircraft = Aircraft(
    team=Team.RED, pos=(15, 15),
    radar_range=5.0, missiles=4,
    missile_max_range=4.0,
    base_hit_prob=0.8, min_hit_prob=0.1
)
red_sam = SAM(
    team=Team.RED, pos=(18, 10),
    radar_range=8.0, missiles=6,
    missile_max_range=6.0,
    base_hit_prob=0.8, min_hit_prob=0.1,
    cooldown_steps=5, on=False
)

scenario.add_blue(blue_aircraft)
scenario.add_blue(blue_awacs)
scenario.add_red(red_aircraft)
scenario.add_red(red_sam)

# 3. Create environment and reset with scenario
env = GridCombatEnv(verbose=False)
state = env.reset(scenario=scenario.to_dict())

# 4. Run game loop
done = False
turn = 0

while not done and turn < 100:
    # Get actions for each entity (your AI here)
    actions = {
        blue_aircraft.id: Action.move(MoveDir.RIGHT),
        blue_awacs.id: Action.wait(),
        red_aircraft.id: Action.wait(),
        red_sam.id: Action.toggle(on=True)
    }
    
    # Execute turn
    state, rewards, done, info = env.step(actions)
    
    # Check results
    world = state['world']
    print(f"Turn {world.turn}")
    
    if done:
        print(f"Game over: {world.game_over_reason}")
        print(f"Winner: {world.winner}")
    
    turn += 1
```

**Main API Methods:**
```python
# Initialize with scenario
env.reset(scenario: Scenario | Dict[str, Any]) -> Dict[str, Any]

# Execute turn (main game loop)
env.step(actions: Dict[int, Action]) -> Tuple[state, rewards, done, info]
# Returns:
#   state: Dict with 'world', 'tracking', 'config' keys
#   rewards: Dict[Team, float]
#   done: bool
#   info: StepInfo

# Properties
env.world -> WorldState  # Current world state
```

#### `StepInfo` - Turn Information
**What:** Metadata returned after each step
**Why:** Provides raw resolver outputs for logging/analytics

**Structure:**
```python
@dataclass
class StepInfo:
    movement: ActionResolutionResult  # MovementResolver output
    combat: CombatResolutionResult    # CombatResolver output
    victory: VictoryResult            # Result of victory check
```

**Accessing Game State:**

All game state is in the returned `state` dictionary:

```python
state, rewards, done, info = env.step(actions)

# Access world state
world = state['world']  # WorldState object
print(f"Turn {world.turn}")
print(f"Game over: {world.game_over}")
print(f"Winner: {world.winner}")

# Access tracking counters
tracking = state['tracking']
print(f"Turns without shooting: {tracking['turns_without_shooting']}")
print(f"Turns without movement: {tracking['turns_without_movement']}")

# Access configuration
config = state['config']
print(f"Max stalemate turns: {config['max_stalemate_turns']}")

# Access team views (fog of war)
blue_view = world.get_team_view(Team.BLUE)
observations = blue_view.get_all_observations()
for obs in observations:
    print(f"Observed: {obs.kind} at {obs.position}")

# Check entities
blue_entities = world.get_team_entities(Team.BLUE, alive_only=True)
print(f"Blue alive: {len(blue_entities)}")
```

### Game Loop (Inside `step()`)

The environment executes this sequence each turn:

```
1. Pre-step Housekeeping
   - Tick SAM cooldowns

2. Movement Phase
   - MovementResolver.resolve_actions()
   - Handles movement, toggles, waits
   - Counter updates handled internally

3. Sensing (Post-movement)
   - SensorSystem.refresh_all_observations()
   - Update team views

4. Combat Phase
   - CombatResolver.resolve_combat()
   - Includes death application
   - Counter updates handled internally

5. Victory Check
   - VictoryConditions.check_all()

6. Return Results
   - (state, rewards, done, info)
```

---

## 💡 Design Patterns & Best Practices

### Pattern 1: When to Use Grid vs Hit Probability

```python
# ✅ GRID - Spatial operations
from env.world import Grid

grid = world.grid
if grid.in_bounds(pos):
    distance = grid.distance(a, b)
    neighbors = grid.get_neighbors(pos)

# ✅ HIT PROBABILITY - Combat calculations
from env.mechanics import hit_probability

prob = hit_probability(distance=dist, max_range=5.0, base=0.8, min_p=0.1)
```

**Remember:**
- **Grid** = "Where is it? How far? What's nearby?"
- **Hit Probability** = "Did it hit? What's the chance?"

### Pattern 2: Stateless Resolvers

All mechanics are stateless - they don't store game state:

```python
# ✅ GOOD - Stateless resolver
class CombatResolver:
    def resolve(self, world, actions):
        # Operates on world, returns results
        # Does NOT store world reference
        return results

# ❌ BAD - Stateful resolver
class BadCombatResolver:
    def __init__(self, world):
        self.world = world  # Storing state!
```

**Benefits:**
- Easy to test
- No side effects
- Can reuse same resolver instance
- Clear data flow

### Pattern 3: Observable Information (Fog of War)

Teams only see entities they can observe through their team view:

```python
# Each team has its own view of the battlefield
state, rewards, done, info = env.step(actions)
world = state['world']

# Blue only sees:
# - All friendly entities (full info)
# - Enemy entities within radar range
blue_view = world.get_team_view(Team.BLUE)
observations = blue_view.get_all_observations()
enemy_obs = blue_view.get_enemy_observations()

for obs in enemy_obs:
    print(f"Observed enemy: {obs.kind} at {obs.position}, distance: {obs.distance:.2f}")
```

### Pattern 4: Entity Serialization

Override `to_dict()` only when you add fields:

```python
# ✅ Override when adding fields
class Aircraft(Entity):
    missiles: int = 2  # NEW field
    
    def to_dict(self):
        data = super().to_dict()
        data["missiles"] = self.missiles
        return data

# ✅ No override needed
class AWACS(Entity):
    radar_range: float = 9.0  # Already in base Entity
    # Uses Entity.to_dict() automatically
```

### Pattern 5: Scenario-Driven Design

Use `Scenario` for all game setup:

```python
# ✅ GOOD - Use Scenario for all configuration
from env.scenario import Scenario
from env import GridCombatEnv

scenario = Scenario(
    grid_width=30,
    grid_height=30,
    max_stalemate_turns=100,
    seed=42
)
# Add entities to scenario...

env = GridCombatEnv()
state = env.reset(scenario=scenario.to_dict())

# ❌ BAD - Hard-coded values or direct entity creation
class BadEnv:
    def __init__(self):
        self.width = 20  # Hard-coded!
```

---

## 🎯 Dependency Graph

```
core/  ← No dependencies
  ↓
entities/  ← Uses core types
  ↓
world/  ← Uses core types and entities
  ↓
mechanics/  ← Uses core, entities, world
  ↓
scenario.py  ← Uses entities and core types
  ↓
environment.py  ← Orchestrates everything (uses world, mechanics, scenario)
```

**Key Rule:** Lower modules NEVER import from higher modules. This prevents circular dependencies and keeps architecture clean.

---

## 📊 Module Summary Table

| Module | Lines | Files | Status | Dependencies |
|--------|-------|-------|--------|--------------|
| `core/` | ~618 | 3 | ✅ | None |
| `entities/` | ~807 | 5 | ✅ | core |
| `world/` | ~758 | 3 | ✅ | core, entities |
| `mechanics/` | ~1,493 | 4 | ✅ | core, entities, world |
| `utils/` | ~75 | 1 | ✅ | None |
| `scenario.py` | ~336 | 1 | ✅ | core, entities |
| `environment.py` | ~347 | 1 | ✅ | All above |
| **Total** | **~4,493** | **24** | ✅ | Clean hierarchy |

---

## 🚀 Getting Started

### Quick Start (5 minutes)

```python
# 1. Import
from env import GridCombatEnv
from env.scenario import Scenario
from env.entities import Aircraft, AWACS
from env.core import Team
from env.core.actions import Action

# 2. Create scenario
scenario = Scenario(grid_width=20, grid_height=20, seed=42)
scenario.add_blue(Aircraft(
    team=Team.BLUE, pos=(5, 5),
    radar_range=5.0, missiles=4,
    missile_max_range=4.0,
    base_hit_prob=0.8, min_hit_prob=0.1
))
scenario.add_blue(AWACS(team=Team.BLUE, pos=(3, 3), radar_range=9.0))

# 3. Create environment and reset
env = GridCombatEnv()
state = env.reset(scenario=scenario.to_dict())

# 4. Run
actions = {}  # Your AI here
state, rewards, done, info = env.step(actions)

# 5. Done!
world = state['world']
print(f"Turn {world.turn}, Game over: {done}")
```

### Next Steps

1. **Read `README.md`** - Project overview and examples
2. **Read `test_environment.py`** - Full working example
3. **Experiment** - Create custom scenarios
4. **Extend** - Add new entity types or mechanics

---

## 📚 Additional Resources

- **README.md** - Project overview, quick start, API reference
- **IMPLEMENTATION_COMPLETE.md** - Implementation status and achievements
- **test_mechanics.py** - Mechanics module testing
- **test_environment.py** - Full environment demonstration

---

## 🎊 Status: Production Ready

All core modules are **complete, tested, and documented**. The environment is ready for:
- ✅ Research and experimentation
- ✅ AI agent training
- ✅ Custom scenario development
- ✅ Extension with new features

**Enjoy building!** 🚀
