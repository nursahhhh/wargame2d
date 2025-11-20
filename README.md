# Grid Combat Environment 2.0

A modular, extensible 2D turn-based air combat simulation environment designed for reinforcement learning and game AI research.

## Features

- ✅ **Modular Architecture** - Clean separation of concerns across 8 modules
- ✅ **Gym-like Interface** - Easy-to-use `reset()` and `step()` API
- ✅ **Multiple Entity Types** - Aircraft, AWACS, SAMs, and Decoys
- ✅ **Realistic Mechanics** - Range-based radar, hit probability, cooldowns
- ✅ **Fog of War** - Per-team observations with decoy deception
- ✅ **Fully Tested** - All modules working end-to-end
- ✅ **Configurable** - Extensive configuration options
- ✅ **Stateless Resolvers** - Easy to test and extend

## Quick Start

```python
from env import GridCombatEnv, Scenario, create_basic_battle
from env.entities import Aircraft, AWACS
from env.core import Team
from env.core.actions import Action

# Option 1: Use a pre-built scenario
env = GridCombatEnv()
scenario = create_basic_battle()
state = env.reset(scenario=scenario.to_dict())

# Option 2: Build a custom scenario
scenario = Scenario(grid_width=20, grid_height=20, seed=42)
scenario.add_blue(Aircraft(team=Team.BLUE, pos=(5, 5)))
scenario.add_blue(AWACS(team=Team.BLUE, pos=(3, 3)))
scenario.add_red(Aircraft(team=Team.RED, pos=(15, 15)))
scenario.add_red(AWACS(team=Team.RED, pos=(17, 17)))

env = GridCombatEnv()
state = env.reset(scenario=scenario.to_dict())

# Run simulation
world = state["world"]
all_entities = world.get_alive_entities()

actions = {entity.id: Action.wait() for entity in all_entities}
state, rewards, done, info = env.step(actions)

# Check game state
print(f"Turn: {env.turn}")
print(f"Game over: {done}")
print(f"Blue reward: {rewards[Team.BLUE]}")
```

## Architecture

The codebase is organized into clean, modular components:

```
env/
├── core/           # Fundamental types (Team, Action, Observation, etc.)
├── entities/       # Entity classes (Aircraft, AWACS, SAM, Decoy)
├── world/          # Spatial logic (Grid, WorldState, TeamView)
├── mechanics/      # Action resolvers (sensors, movement, combat)
├── utils/          # Utilities (ID generation)
├── scenario.py     # Scenario definitions and builders
└── environment.py  # Main gym-like interface
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for detailed module documentation.

## Entity Types

### Aircraft
- **Mobile fighter** with radar and missiles
- Radar range: 5.0
- Missiles: 2
- Missile range: 4.0

### AWACS (Airborne Warning and Control System)
- **Surveillance aircraft** with extended radar
- Radar range: 9.0
- No weapons (observation only)
- **Primary objective** - game ends when destroyed

### SAM (Surface-to-Air Missile)
- **Stationary air defense**
- Can toggle radar on/off
- Only visible to enemies when radar is ON
- Cooldown after firing
- Missiles: 4
- Missile range: 6.0

### Decoy
- **Deceptive unit**
- Appears as aircraft to enemies
- No weapons or radar
- Used for misdirection

## Game Rules

### Victory Conditions
1. **AWACS Destruction** - Destroy enemy AWACS to win
2. **Missile Exhaustion** - Draw if no missiles remain
3. **Combat Stalemate** - Draw after 60 turns without shooting
4. **Movement Stagnation** - Draw after 15 turns without movement

### Combat Mechanics
- Hit probability decreases linearly with distance
- Base hit chance: 80% at range 0
- Minimum hit chance: 10% at max range
- Randomized action order prevents ID bias

### Observation System
- Entities observe others within radar range
- Decoys appear as aircraft to enemies
- SAMs invisible when radar is OFF
- Per-team fog of war

## Scenarios

Scenarios are the **single source of truth** for game configuration. They contain everything needed to initialize an environment:

### Pre-built Scenarios

```python
from env import create_basic_battle, create_sam_defense, create_awacs_support

# Basic 2v2 aircraft battle
scenario = create_basic_battle()

# Blue aircraft vs Red SAM network
scenario = create_sam_defense()

# Both sides with AWACS support
scenario = create_awacs_support()
```

### Custom Scenarios

```python
from env import Scenario
from env.entities import Aircraft, AWACS

scenario = Scenario(
    grid_width=30,              # Grid dimensions
    grid_height=30,
    seed=42,                    # Random seed
    max_stalemate_turns=100,    # Stalemate threshold
    max_no_move_turns=20,       # Stagnation threshold
    check_missile_exhaustion=True  # Check for missile depletion
)

# Add entities with custom stats
scenario.add_blue(Aircraft(
    team=Team.BLUE, pos=(5, 5),
    missiles=4,
    radar_range=5.0,
    missile_max_range=4.0,
    base_hit_prob=0.9,  # Custom hit probability
    min_hit_prob=0.2
))

# Environment only needs logging settings
env = GridCombatEnv(verbose=True, max_log_history=5)
states = env.reset(scenario=scenario.to_dict())
```

### Saving and Loading

```python
# Save scenario to JSON
scenario.save_json("my_scenario.json")

# Load scenario from JSON
scenario = Scenario.load_json("my_scenario.json")

# Use loaded scenario
env = GridCombatEnv()
states = env.reset(scenario=scenario.to_dict())
```

## Testing

Run the test suite:

```bash
# Test individual mechanics modules
python3 test_mechanics.py

# Test full environment
python3 test_environment.py
```

## Example: Simple AI

```python
def simple_ai(env, team):
    """Move toward center and shoot at visible enemies."""
    actions = {}
    
    for entity in env.world.get_team_entities(team):
        team_view = env.world.get_team_view(team)
        enemy_ids = team_view.get_enemy_ids(team)
        
        # Shoot at closest enemy if in range
        if entity.can_shoot and enemy_ids:
            for enemy_id in enemy_ids:
                actions[entity.id] = Action.shoot(enemy_id)
                break
        # Otherwise move toward center
        elif entity.can_move:
            cx = env.world.grid.width // 2
            cy = env.world.grid.height // 2
            dx = cx - entity.pos[0]
            
            if dx > 0:
                actions[entity.id] = Action.move(MoveDir.RIGHT)
            else:
                actions[entity.id] = Action.move(MoveDir.LEFT)
        else:
            actions[entity.id] = Action.wait()
    
    return actions
```

## Development Status

| Component | Status |
|-----------|--------|
| Core types | ✅ Complete |
| Entities | ✅ Complete |
| World state | ✅ Complete |
| Rules | ✅ Complete |
| Mechanics | ✅ Complete |
| Environment | ✅ Complete |
| Rendering | ⏸️ Stub only |
| Tests | ✅ Working |
| Gym wrapper | ⏸️ Future |

## Version History

- **2.0.0** - Complete modular redesign
  - Separated monolithic code into 8 modules
  - Added gym-like interface
  - Stateless mechanics resolvers
  - Comprehensive configuration
  - Full test coverage

- **1.0.0** - Original monolithic implementation
  - Single 800+ line file
  - Mixed concerns
  - Difficult to test and extend

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see [`ARCHITECTURE.md`](ARCHITECTURE.md) for design principles and module boundaries.

## Next Steps

Potential enhancements:

1. **Rendering Module** - ASCII grid visualization
2. **Gym Wrapper** - OpenAI Gym compatibility
3. **Replay System** - Save/load game states
4. **Advanced AI** - Neural network integration examples
5. **Multi-agent RL** - Training frameworks
6. **Performance** - Cython optimization for batch simulation
7. **More Scenarios** - Additional pre-built scenario templates

## Contact

For questions or issues, please open a GitHub issue.

