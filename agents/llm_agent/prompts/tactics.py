TACTICAL_GUIDE = f"""
### TACTICAL PRINCIPLES & CONSIDERATIONS FOR 2D COMBAT GRID GAME
**Purpose:** This guide presents core tactical concepts and strategic patterns observed in 2D combat grid scenarios. 
It is NOT a prescriptive rulebook—treat it as a menu of ideas to inform your own tactical decisions based on specific battlefield conditions.


### VICTORY CONDITION
Destroy enemy AWACS while protecting yours.

### CORE PRINCIPLES

#### 1. AWACS PROTECTION (TOP PRIORITY)
- Keep AWACS at safe distance from closest armed enemy (>= 5 cells typically sufficient)
- Maintain unblocked retreat paths at all times
- **Movement Freedom:** As long as AWACS maintains threshold distance from ALL threats with open escape routes, it can move anywhere on the map to:
  - Provide radar coverage for the team
  - Scout and spot enemy units
  - Reposition for better battlefield awareness
- Don't keep AWACS static if it's already safe—use its mobility and radar actively

#### 2. TARGET PRIORITY
1. Enemy AWACS (if accessible - game-winning)
2. Enemy aircraft threatening your AWACS
3. Enemy SAMs blocking your advance path
4. Enemy decoys (LOWEST - avoid wasting ammunition)

#### 3. DISTANCE CONTROL & AMMUNITION ECONOMY
- **Control engagement distance:** Stay outside enemy weapon range when not attacking; close distance to engage
- Fire at closer range for higher hit probability (distance determines hit chance)
- Coordinate multi-unit strikes when possible
- Each shot counts especially when we have low ammo left, take closer shots when ammo is limited.

#### 4. SAM TACTICS
**Using Your SAMs:**
- **Range Advantage:** SAMs have longer range than aircraft—use this! Keep them ON to support allies and control territory
- **When to Stay ON:**
  - Supporting allied aircraft in combat (numerical advantage)
  - Denying area to enemy advance
  - Protecting other valuable assets with threat of fire
- **When to Go Stealth (OFF):**
  - During ~5-turn cooldown period (hiding vulnerability)
  - When isolated and about to be overwhelmed
  - Setting up a specific ambush trap
- **Combat Pattern:** Toggle ON → Shoot → Stay ON to support team OR Toggle OFF if entering cooldown and vulnerable
- **Key Insight:** Don't hide just for surprise—use your range to create numerical advantages (2v1, 3v1) in fights

**Countering Enemy SAMs:**
- **Cooldown Exploitation:** Enemy SAMs also have ~5-turn reload periods
- **Baiting Strategy:** 
  - Send most expendable unit (decoy preferred, or low-ammo aircraft) forward first
  - Keep valuable units behind, outside SAM range
  - If enemy SAM shoots the bait, it enters cooldown
  - Immediately rush in with your main force while SAM is reloading
  - Destroy the SAM or nearby high-value targets before it can fire again
- **IMPORTANT:** SAMs (yours and enemy's) are stationary and critical—use ambush tactics with yours, exploit cooldowns on theirs

#### 5. DECOY OPERATIONS
- **Value Preservation:** Decoys are more expendable than aircraft/AWACS, but each one lost reduces your tactical options. Preserve them when possible.
- **Strategic Expenditure:** Use decoys deliberately, not recklessly:
  - Scout unknown areas when intelligence is needed
  - Trade for high-value kills (enemy aircraft, SAM or AWACS elimination)
  - Sacrifice to protect more valuable assets (aircraft/SAM/AWACS) from immediate threats
- **Protective Screening:** Position decoys closer (than other allies) to threats than your aircraft ONLY when:
  - An allied aircraft/AWACS/SAM is at risk of being shot
  - The decoy absorbing fire allows valuable units to attack safely or escape
  - Example: Decoy in front draws enemy targeting while your aircraft fires from behind
- **When Alone, Retreat:** If a decoy is caught in enemy range with no allies to protect, retreat instead of absorbing fire for nothing
- **Avoid Suicide Scouting:** Don't blindly rush into suspected enemy positions; use radar coverage and careful advances instead
- **Enemy Deception:** Enemy cannot distinguish decoys from aircraft—use this for misdirection and tactical positioning
- **General Rule:** Spend decoys to save aircraft/AWACS or enable kills, not just because they're "expendable"

#### ENGAGEMENT TIPS
- **Numerical Advantage is Key:** Always seek 2v1, 3v1, or better situations
  - Attack before enemy can engage (initiative)
  - Coordinate SAM + aircraft strikes simultaneously
  - Use decoy to absorb enemy shot while allies attack
- Use coordinated multi-unit strikes to increase overall kill probability
- **SAM Support:** Position SAMs ON to support aircraft engagements with their superior range, creating unfavorable trades for enemy
- Use decoy-aircraft combinations: Position decoy closer to enemy to draw targeting priority while aircraft attacks from safer position
- Only sacrifice decoys when it protects higher-value units or enables a valuable kill

#### 6. WINNING PATTERNS
- **Numerical Superiority:** Create and exploit 2v1, 3v1 situations
  - SAM (ON) + Aircraft vs single enemy
  - Decoy absorbs shot while 2+ allies attack
  - Strike first before enemy can respond
- **SAM Ambush (Situational):** 
  - Use when SAM is isolated or during cooldown
  - Keep SAM OFF, bait enemy close, Toggle ON + Shoot
  - More useful for comebacks than standard play
- **Hit-and-Run:** Strike with numerical advantage, retreat before enemy can concentrate forces
- **Decoy Screen:** Decoys lead, aircraft follow 2-3 cells back, exploit cleared path
- **Pincer Movement:** Attack from multiple directions to trap enemy AWACS
- **Breakthrough Timing:** Thin enemy defenses first, then commit to AWACS kill

### REMEMBER
- Control distance: It is critical for both offense and defense as long as you keep your distance you can play more freely.
- Decoys are disposable intelligence assets - use them
- SAMs are ambush weapons, not frontline fighters
- Protect AWACS > Everything else"""


# ### DECISION FRAMEWORK EACH TURN
# 1. Is my AWACS safe? (If no - prioritize defense immediately)
# 2. Can I damage/kill enemy AWACS? (If yes - consider aggressive strike)
# 3. What new enemy positions revealed? (Update threat assessment)
# 4. Which units can contribute to objective? (Advance those, position others)
# 5. Are SAMs optimally positioned for ambush? (Toggle timing critical)
# 6. Is ammunition being used efficiently? (High-value targets only)
# TACTICAL_GUIDE = """
# ### STRATEGIC IDEAS (NOT SET IN STONE)
# - **Decoys:** Lead and scout; they are relatively dispensable (but for a reason), enemy will think they are aircrafts, you can use them to scout, protect other entities, or to make the enemy forces follow (keeping a safe distance if possible) it to the ambush.
#
# - **Aircraft:**
#     - Use for discovery and/or to attack in coordinated groups; fire simultaneously for higher hit odds.
#     - Firing from distance makes sense to get safe shots despite low probabilities, but if you limited missiles taking a closer shot despite the risk could make sense.
#     - Fighting around your ON SAM will provide aircrafts a big advantage.
#     - Shooting behind your decoy or sam also makes sense to both keep you safe and have a hit chance.
#
# - **SAMs:**
#   - Use for defense and baiting
#   - Keep it OFF to lure enemies or hide cooldowns
#   - Turn ON when enemy is close for surprise attacks
#   - Fights around (in the shoot distance) your SAM when it's ON, provides your aircrafts a big advantage.
#   - For the enemy SAMs, it makes sense to turn around them if possible or accept a long shot from it (possibly by using a decoy), then suddenly getting closer to it and attack.
#
# - **AWACS:**
#   - Critical asset—protect at all costs
#   - Move away from any threat (>=4 is safe by keeping this distance to the closest armed enemy you can use the radar capabilities freely)
#
# - **Movement:**
#   - Can use map edges for stealth
#   - When faced enemies you can fight or flee depending on your (positioning, entitiy numbers, arming, etc...)
#   - Close distance for higher hit probability, but it is double-edged situation
#
# - **General Suggestions to Consider***
#     - It is a 2D game with limitations use them for your advantage.
#     - At each turn, every entity can do only one thing. E.g. can't move and shoot at the same turn.
#     - You can play defensive and use your SAM hidden for ambush to eliminate some of the enemy forces and then attack.
#     - You can play offensive by directly attacking and using element of surprise.
#     - You can walk on the shadow (edges) and set a surprise assasination to the awacs.
#     - There are many options considering the game rules and entity capabilities, it's up to you.
# """