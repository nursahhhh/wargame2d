GAME_INFO = """
### GAME OVERVIEW
- Turn-based tactical combat between two teams
- **Objective:** Control your team's units each turn to achieve victory.
- **Victory Condition:** Destroy the enemy's AWACS unit
- **Defeat Condition:** Lose your AWACS unit
- **Resource Management:** 
  - Each turn incurs operational costs
  - Losing your units is disadvantageous
  - Destroying enemy units is advantageous
  - Prolonged stalemates are costly; progress toward objectives is necessary
- **Map Layout:** Typically ~13×20 grid; exact dimensions, and starting positions vary by scenario
- **Initial Setup:** Starting positions and unit placements are predetermined by scenario (not player-controlled)

### CORE MECHANICS
- **One Action Per Turn:** Each unit performs exactly one action per turn
- **Radar & Targeting:**
  - Enemy units are invisible until detected by any friendly unit's radar
  - Detection is shared across the entire team. This means:
    - If ANY friendly unit detects an enemy, ALL friendly units can see it
    - There is no benefit to having multiple units detect the same enemy
    - Radar positioning should focus on COVERAGE (seeing more area) not REDUNDANCY"
  - Any unit can target any enemy that any teammate has detected and is within weapon range
  - Position information comes ONLY from radar detection, not from observing enemy actions (e.g., seeing missiles fired)
- **Grid-Based Movement:** Units occupy discrete grid cells
- **Collision Rule:** Only one entity per cell; attempting to move into an occupied cell will be blocked
- **Radar Detection:** Each unit type has a circular detection range (measured in grid cells from unit's position)

### COMBAT MECHANICS
- **Limited Ammunition:** Armed units carry finite missiles
- **Range-Based Hit Probability:** 
  - Hit chance depends solely on straight-line distance between attacker and target
  - Closer proximity increases hit chance (applies to both teams' weapons)
  - Angle of attack is irrelevant; only distance matters
- **Range Limit:** Weapons have maximum effective range
- **Coordinated Strikes:** Since each unit fires only one missile per turn, destroying high-value targets may require multiple units targeting the same enemy across one or more turns. When multiple units fire in the same turn, shots resolve in random order (but can be treated as simultaneous for planning purposes)

### ENTITY TYPES
**AWACS**
- Long-range radar, unarmed, mobile
- Mission-critical: losing it means immediate defeat
- Can only MOVE or WAIT

**Aircraft**
- Armed, mobile units with medium radar range
- Limited missile ammunition
- Can MOVE, SHOOT, or WAIT

**Decoy**
- Unarmed, mobile
- Always appears as an aircraft to enemies (never reveals itself as a decoy)
- Can be targeted and shot at like any aircraft
- Can MOVE or WAIT

**SAM (Surface-to-Air Missile)**
- Stationary armed defense unit that always occupies its cell (even when stealthed)
- Can TOGGLE between states:
  - Active (ON): Detectable, can shoot
  - Stealth (OFF): Immediately invisible to enemy radar, cannot shoot or be targeted
- Has cooldown period between shots (typically 5 turns, may vary by scenario)
- Can SHOOT, TOGGLE, or WAIT

### AVAILABLE ACTIONS
- **MOVE:** Relocate to an adjacent cell (UP, DOWN, LEFT, RIGHT)
- **SHOOT:** Fire missile at a detected, in-range target
- **WAIT:** Skip turn without action
- **TOGGLE:** Switch SAM between active and stealth modes (SAM only)"""