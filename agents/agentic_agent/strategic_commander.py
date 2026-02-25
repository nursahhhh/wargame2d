

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import StrategicPlan, RoleDirective, StrategicMemory, BattlefieldStatus
from .observed_state import GlobalState
from .observability import trace_strategic_planning, logfire

logger = logging.getLogger(__name__)


# Compressed game info for system prompt
GAME_INFO_COMPRESSED = """
CORE RULES:
- Victory: Destroy enemy AWACS | Defeat: Lose your AWACS
- One action per unit per turn: MOVE, SHOOT, WAIT, or TOGGLE (SAM only)
- Range-based hit probability (closer = higher chance)
- Radar detection shared across team - if ANY unit detects enemy, ALL can target

UNIT TYPES:
- AWACS: Long radar, unarmed, mobile (CRITICAL - protect at all costs!)
- AIRCRAFT: Armed, mobile, medium radar, limited missiles
- SAM: Stationary, armed, can TOGGLE stealth (invisible when OFF, can't shoot)
- DECOY: Unarmed, appears as aircraft to enemy (use as bait/screen)

COMBAT: Closer shots = higher hit chance. Coordinate multi-unit strikes.
"""

# Compressed tactical guide
TACTICAL_GUIDE_COMPRESSED = """
KEY TACTICS:
1. AWACS PROTECTION: Keep >5 cells from armed enemies, use mobility for radar coverage
2. TARGET PRIORITY: AWACS > threatening aircraft > SAM > Decoy (avoid wasting ammo on decoys)
3. SAM TACTICS: Use range advantage (8 cells), toggle stealth during cooldown (~5 turns)
4. DECOY USE: Send forward as bait, screen valuable units, draw enemy fire
5. COORDINATION: Create 2v1 or 3v1 situations, coordinate multi-unit strikes
6. DISTANCE CONTROL: Stay outside enemy range when not attacking, close to engage
7. SAM BAITING: Enemy SAMs have cooldown - bait shot with expendable unit, then rush
"""


class StrategicCommander:
    """Generates strategic plans with momentum tracking and situation analysis."""
    
    def __init__(self, llm_client, team_name: str):
        self.llm = llm_client
        self.team_name = team_name
    
    def plan(self, global_state: GlobalState, memory: Optional[StrategicMemory] = None) -> StrategicPlan:
        """Generate strategic plan with momentum analysis and situation assessment."""
        with trace_strategic_planning():
            
            try:
                # CALCULATE METRICS WITH MOMENTUM
                metrics = self._calculate_metrics(global_state, memory)
                
                logger.info(f"\n TURN {global_state.turn} - STRATEGIC PLANNING -> ")
                logger.info(f"Forces: {len(global_state.allied)} Allied | {len(global_state.enemies)} Enemies")
                logger.info(f"Missiles: {metrics['total_missiles']} | Threat: {metrics['threat_score']}")
                logger.info(f"Status: {metrics['status'].value} | Momentum: {metrics['momentum']}")
                logger.info(f"Enemy Intent: {global_state.enemy_intent}")

                # Build prompts
                system_prompt = self._get_system_prompt()
                user_prompt = self._build_prompt(global_state, metrics, memory)
                
                # Create logs directory
                log_dir = Path("logs/prompts/strategic")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Build combined prompt for logging
                full_prompt = f"""{'='*80}
STRATEGIC PLANNING - TURN {global_state.turn}
{'='*80}

=== SYSTEM PROMPT ===
{system_prompt}

{'='*80}

=== USER PROMPT ===
{user_prompt}

{'='*80}
METADATA:
- Turn: {global_state.turn}
- Team: {self.team_name}
- Allies: {len(global_state.allied)} {global_state.ally_counts}
- Enemies: {len(global_state.enemies)} {global_state.enemy_counts}
- Status: {metrics['status'].value}
- Momentum: {metrics['momentum']}
- Enemy Intent: {global_state.enemy_intent}
{'='*80}
"""
                
                # Save to file
                prompt_file = log_dir / f"turn_{global_state.turn:03d}.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(full_prompt)
                
                # Log to Logfire
                logfire.info("strategic_full_prompt",
                            turn=global_state.turn,
                            team=self.team_name,
                            system_prompt_length=len(system_prompt),
                            user_prompt_length=len(user_prompt),
                            momentum=metrics['momentum'],
                            enemy_intent=global_state.enemy_intent,
                            file_path=str(prompt_file))
                
                logger.info(f" Strategic prompt saved: {prompt_file}")
                
                # Make LLM call
                response = self.llm.complete(
                    system_prompt,  
                    user_prompt,
                    temperature=0.5,
                    max_tokens=2000,  # Increased for situation_analysis
                    response_format={"type": "json_object"}
                )
                logfire.info("strategic_response", 
                            turn=global_state.turn, 
                            raw_response=response,
                            response_length=len(response))
                logger.info(f"\n STRATEGIC RAW RESPONSE:\n{response}\n")
    
                # Log response to file
                response_file = log_dir / f"turn_{global_state.turn:03d}_response.txt"
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(f"""{'='*80}
STRATEGIC RESPONSE - TURN {global_state.turn}
{'='*80}

{response}

{'='*80}
METADATA:
- Response Length: {len(response)}
- Model: {self.llm.model}
- Temperature: 0.5
- Max Tokens: 2000
{'='*80}
""")
                
                logfire.info("strategic_response_saved",
                            turn=global_state.turn,
                            response_length=len(response),
                            file_path=str(response_file))
                
                data = json.loads(response)
                
                # POST-LLM VALIDATION
                data["directives"] = [
                    d for d in data.get("directives", [])
                    if d.get("role") in metrics["active_roles"]
                ]
                
                # Validate we have at least one directive
                if not data["directives"]:
                    logfire.warning("strategic_no_directives", turn=global_state.turn)
                    return self._fallback_plan(global_state, metrics)
                
                # Add defaults for new fields if not present
                if "situation_analysis" not in data:
                    data["situation_analysis"] = ""
                if "enemy_formation" not in data:
                    data["enemy_formation"] = global_state.enemy_intent
                if "momentum_suggestion" not in data:
                    data["momentum_suggestion"] = metrics.get("momentum_suggestion", "")
                
                plan = StrategicPlan(**data)
                
                logger.info(f"\n DIRECTIVES: \n")
                logger.info(f"Situation: {plan.situation_analysis}")
                logger.info(f"Enemy Formation: {plan.enemy_formation}")

                for directive in plan.directives:
                    logfire.info("strategic_directives",
                                 turn=global_state.turn,
                                 role=directive.role,
                                 mode=directive.mode,
                                 priority=directive.priority)
                
                    logger.info(f"Turn {global_state.turn} | Role: {directive.role} | Mode: {directive.mode} | Priority: {directive.priority}")
                return plan
                
            except json.JSONDecodeError as e:
                logfire.error("strategic_json_failed", turn=global_state.turn, error=str(e))
                logger.error(f" JSON PARSING FAILED -> -> {e}")
                return self._fallback_plan(global_state, self._calculate_metrics(global_state, memory))
            except Exception as e:
                logfire.error("strategic_failed", turn=global_state.turn, error=str(e))
                return self._fallback_plan(global_state, self._calculate_metrics(global_state, memory))
    
    def _calculate_metrics(self, state: GlobalState, memory: Optional[StrategicMemory] = None) -> Dict[str, Any]:
        """Calculate strategic metrics with momentum tracking."""
        allies = state.allied
        
        # 1. Ammo totals
        total_missiles = sum(u.missiles or 0 for u in allies)
        
        # 2. Per-role ammo status
        ammo_status = {}
        for role in ["AIRCRAFT", "SAM", "AWACS", "DECOY"]:
            units = [u for u in allies if u.type == role]
            ammo_status[role] = round(sum(u.missiles or 0 for u in units) / max(len(units), 1), 1)
        
        # 3. Weighted threat assessment
        weights = {"SAM": 3.0, "AIRCRAFT": 2.0, "AWACS": 1.5, "DECOY": 0.5}
        threat_score = sum(weights.get(e.type, 1.0) for e in state.enemies)

        # 4. Enemy spatial distribution
        grid_w = state.grid[0]
        sectors = {"left": 0, "center": 0, "right": 0}
        
        if grid_w >= 9:
            left_boundary = grid_w // 3
            right_boundary = (grid_w * 2) // 3
            
            for e in state.enemies:
                if e.pos:
                    x = e.pos[0]
                    if x < left_boundary:
                        sectors["left"] += 1
                    elif x > right_boundary:
                        sectors["right"] += 1
                    else:
                        sectors["center"] += 1
        else:
            sectors["center"] = len(state.enemies)
        
        # 5. Active roles
        active_roles = [k for k, v in state.counts.items() if not k.startswith('EN_') and v > 0]
        
        # 6. Strategic status
        status = BattlefieldStatus.BALANCED
        if total_missiles < threat_score:
            status = BattlefieldStatus.AMMO_SHORTAGE
        elif total_missiles > threat_score * 2 and len(allies) > len(state.enemies):
            status = BattlefieldStatus.AIR_SUPERIORITY
        elif len(allies) < len(state.enemies) * 0.5:
            status = BattlefieldStatus.FORCE_DEFICIT
        elif len(allies) == 0:
            status = BattlefieldStatus.ELIMINATED
        
        # 7. Momentum calculation
        momentum = 0.0
        momentum_suggestion = "SITUATION_NEUTRAL"
        if memory:
            momentum = memory.get_momentum()
            momentum_suggestion = memory.get_momentum_suggestion()
        
        return {
            "total_missiles": total_missiles,
            "threat_score": round(threat_score, 1),
            "sectors": sectors,
            "ammo_status": ammo_status,
            "active_roles": active_roles,
            "status": status,
            "momentum": round(momentum, 2),
            "momentum_suggestion": momentum_suggestion,
            "ally_counts": state.ally_counts,
            "enemy_counts": state.enemy_counts
        }

    def _build_prompt(
        self, 
        state: GlobalState, 
        metrics: Dict[str, Any],
        memory: Optional[StrategicMemory] = None
    ) -> str:
        """Enhanced prompt with situation analysis and momentum."""
        nl = "\n"  
        m = metrics
        
        # --- ADVISORY (Data only, no commands) ---
        advisor_notes = []
        low_ammo_roles = [r for r, count in m['ammo_status'].items() 
                          if count < 1.0 and r in m['active_roles']]
        
        if low_ammo_roles:
            advisor_notes.append(f"⚠️ CRITICAL: {', '.join(low_ammo_roles)} <1 missile avg")
        elif m['total_missiles'] < m['threat_score']:
            advisor_notes.append(f"⚠️ Ammo shortage: {m['total_missiles']} vs {m['threat_score']} threat")
        
        if m['threat_score'] == 0:
            advisor_notes.append("✓ No visible enemies (Consider RECON to find them)")
        elif m['total_missiles'] > m['threat_score'] * 2:
            advisor_notes.append(f"✓ Air superiority (Consider AGGRESSIVE push)")
        
        heavy_sector = max(m['sectors'], key=m['sectors'].get)
        if m['sectors'][heavy_sector] > 0:
            advisor_notes.append(f"📍 Enemy concentration: {heavy_sector} sector")
        
        # Momentum-based advice
        if m['momentum'] > 0.3:
            advisor_notes.append(f"📈 MOMENTUM: +{m['momentum']:.1f} (Winning - maintain pressure)")
        elif m['momentum'] < -0.3:
            advisor_notes.append(f"📉 MOMENTUM: {m['momentum']:.1f} (Losing - consider defensive)")
        
        # Enemy intent advice
        if state.enemy_intent == "AWACS_HUNT":
            advisor_notes.append("🎯 ALERT: Enemy appears to be hunting our AWACS!")
        elif state.enemy_intent == "FLANKING":
            advisor_notes.append("↔️ Enemy is flanking - watch multiple directions")
        
        # Memory summary
        mem_section = ""
        if memory:
            s = memory.get_summary()
            if s.get("status") != "No history":
                mem_section = f"""
**RECENT HISTORY:**
- Last 3 turns: {s['last_3_turns']['kills']}K / {s['last_3_turns']['losses']}L ({s['last_3_turns']['accuracy']*100:.0f}% Acc)
- Momentum: {s['momentum']} ({s['momentum_suggestion']})
- Missing enemies: {s['missing_enemy_count']}
- Recent events: {', '.join(s.get('recent_events', [])) or 'None'}
"""

        # Directives template (Cloze Completion)
        directives_template = []
        for role in m['active_roles']:
            directives_template.append(
                f'    {{"role": "{role}", "mode": "<MODE>", "priority": "<PRIORITY>"}}'  
            )

        prompt = f"""**TURN {state.turn} STRATEGIC COMMAND**

**FORCE COMPOSITION**
Allied: {json.dumps(m['ally_counts'])} | Enemy: {json.dumps(m['enemy_counts'])}
Enemy Intent: {state.enemy_intent}

**INTELLIGENCE REPORT**
Forces:    {len(state.allied)} Allied | {len(state.enemies)} Visible Enemies
Missiles:  {m['total_missiles']} total | {json.dumps(m['ammo_status'])} by role
Threat:    {m['threat_score']} weighted (SAM×3, AIRCRAFT×2, AWACS×1.5)
Position:  Left:{m['sectors']['left']} | Center:{m['sectors']['center']} | Right:{m['sectors']['right']}
HVTs:      {state.hvts if state.hvts else "None"}
Status:    {m['status'].value}
{mem_section}

**ADVISORY ANALYSIS**
{nl.join(f"• {note}" for note in advisor_notes) if advisor_notes else "• Situation Nominal"}

**ACTIVE ROLES:** {', '.join(m['active_roles'])}

**MISSION ORDERS**

You are the Strategic Commander. First analyze the situation, then issue directives.

OUTPUT FORMAT (JSON only):
{{
  "turn": {state.turn},
  "situation_analysis": "<BRIEF_ANALYSIS>",
  "enemy_formation": "<ENEMY_INTENT>",
  "momentum_suggestion": "{m['momentum_suggestion']}",
  "priorities": ["<PRIORITY_1>", "<PRIORITY_2>"],
  "directives": [
{nl.join(directives_template)}
  ],
  "hvt_targets": {state.hvts}
}}

**FIELD DESCRIPTIONS:**
- situation_analysis: 1-2 sentence battlefield assessment (what's happening, key threats)
- enemy_formation: One of: AWACS_HUNT, FLANKING, AGGRESSIVE_PUSH, DEFENSIVE_HOLD, SCATTERED, NO_CONTACT
- momentum_suggestion: MAINTAIN_AGGRESSIVE, CONSIDER_DEFENSIVE, or SITUATION_NEUTRAL

**DECISION GUIDELINES:**
1. Momentum > +0.3: MAINTAIN current winning strategy
2. Momentum < -0.3: Consider DEFENSIVE to stabilize
3. Enemy AWACS_HUNT: Protect AWACS, consider DEFENSIVE for AWACS
4. Low ammo (<1 avg): DEFENSIVE or RECON
5. Air superiority: AGGRESSIVE push
6. No enemies: RECON to find them
7. Default bias: AGGRESSIVE (maximize pressure)

**ROLE RULES:**
- SAM: Cannot RECON (static), prefer AGGRESSIVE/DEFENSIVE
- AWACS: Prefer DEFENSIVE/SUPPORT, AGGRESSIVE only if critical HVT opportunity
- DECOY: Prefer AGGRESSIVE (bait) or RECON (scout)
- AIRCRAFT: All modes allowed, prefer AGGRESSIVE when ammo available

**OUTPUT RULES:**
1. Valid JSON only (no markdown, no code blocks)
2. Replace ALL placeholders with actual values
3. situation_analysis should be concise but informative

**OUTPUT:** Think step by step, then output valid JSON only.
"""
        return prompt
    
    def _get_system_prompt(self) -> str:
        """System prompt with game info and tactical guide."""
        return f"""You are the Strategic Commander for {self.team_name}.
Analyze intelligence, assess the battlefield, and issue tactical directives.

{GAME_INFO_COMPRESSED}

{TACTICAL_GUIDE_COMPRESSED}

**YOUR OUTPUT MUST INCLUDE:**
1. situation_analysis: Brief battlefield assessment
2. enemy_formation: Inferred enemy strategy
3. priorities: Team-wide objectives
4. directives: Per-role mode and priority

**MODES:**
- AGGRESSIVE: Seek and destroy, press advantage
- DEFENSIVE: Protect assets, preserve forces
- RECON: Gather intel, avoid combat, search for enemies
- SUPPORT: Enable allies, provide radar coverage

**CONSISTENCY:**
- If last mode was AGGRESSIVE and yielded kills, CONTINUE unless ammo critical
- If DEFENSIVE preserved forces, CONTINUE unless opportunity arises
- Change mode ONLY if situation drastically changed

**PHILOSOPHY:**
- Advisory notes are recommendations, not orders
- Balance risk vs. strategic value
- Protect AWACS above all else

**OUTPUT:** Valid JSON only"""
        
    def _fallback_plan(self, state: GlobalState, metrics: Dict[str, Any]) -> StrategicPlan:
        """Generate safe fallback plan with situation analysis."""
        active = [k for k, v in state.counts.items() if not k.startswith('EN_') and v > 0]  
        
        if not active:
            return StrategicPlan(
                turn=state.turn,
                priorities=["Eliminated"],
                directives=[],
                hvt_targets=[],
                situation_analysis="All forces eliminated",
                enemy_formation=state.enemy_intent
            )
        
        directives = [
            RoleDirective(role=r, mode="AGGRESSIVE", priority="ATTACK")
            for r in active
        ]
        
        return StrategicPlan(
            turn=state.turn,
            priorities=["Survival", "Engage enemies"],
            directives=directives,
            hvt_targets=state.hvts or [],
            situation_analysis="Fallback plan - maintaining aggressive posture",
            enemy_formation=state.enemy_intent,
            momentum_suggestion=metrics.get("momentum_suggestion", "SITUATION_NEUTRAL")
        )