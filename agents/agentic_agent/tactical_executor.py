
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import os
from pathlib import Path

from env.core.actions import Action
from env.core.types import ActionType

from .schemas import (
    StrategicPlan, TacticalDecision, CoordinationState, RoleTacticalPlan
)
from .observed_state import LocalState, calc_bearing
from .observability import trace_tactical_execution, logfire, trace_fallback, trace_role_execution

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


# Game info for tactical prompts
TACTICAL_GAME_CONTEXT = """
CORE RULES:
- One action per unit per turn
- Closer = higher hit probability
- Protect AWACS at all costs (losing it = defeat)

UNIT CAPABILITIES:
- AIRCRAFT: Mobile, armed (missiles), medium radar
- SAM: Stationary, armed, can toggle stealth
- AWACS: Mobile, unarmed, long radar (HVT)
- DECOY: Mobile, unarmed, appears as aircraft
"""


class TacticalExecutor:
    
    def __init__(self, llm_client, team_name: str):
        self.llm = llm_client
        self.team_name = team_name
        self.role_order = ["AWACS", "SAM", "AIRCRAFT", "DECOY"]
        # Track decisions for UI/metadata
        self.last_decisions: Dict[int, Dict[str, Any]] = {}
    
    def execute(
        self,
        units_by_role: Dict[str, List[int]],
        local_states: Dict[int, LocalState],
        allowed_actions: Dict[int, List[Action]],
        strategic_plan: StrategicPlan
    ) -> Dict[int, Action]:
        """Execute tactical decisions in role-based order with skip logic."""
        coord_state = CoordinationState()
        all_actions = {}
        self.last_decisions = {}
        
        for role in self.role_order:
            unit_ids = units_by_role.get(role, [])
            if not unit_ids:
                continue
            
            with trace_role_execution(role):
                # Check if we can skip LLM for this role
                if self._should_skip_tactical_llm(role, unit_ids, local_states, allowed_actions):
                    role_actions = self._fallback_by_role(
                        role, unit_ids, allowed_actions, local_states, coord_state
                    )
                    logfire.info(f"{role}_skipped_llm",
                                role=role,
                                reason="no_enemies_in_range",
                                units=len(unit_ids))
                    logger.info(f"⏭️ {role}: Skipped LLM (no enemies in range) - using fallback")
                else:
                    role_actions = self._execute_role(
                        role=role,
                        unit_ids=unit_ids,
                        local_states=local_states,
                        allowed_actions=allowed_actions,
                        plan=strategic_plan,
                        coord=coord_state
                    )
                
                all_actions.update(role_actions)
                
                # Share intel from this role for next roles
                self._share_role_intel(role, unit_ids, local_states, coord_state)
        
        # Final coordination state (PER TURN)
        action_types = [a.type.name for a in all_actions.values()]
        logfire.info("turn_coordination",
                claimed_targets=list(coord_state.claimed_targets),
                occupied_positions=list(coord_state.occupied_positions),
                blocked_reasons=coord_state.blocked_reasons,
                shared_intel_keys=list(coord_state.shared_intel.keys()),
                total_actions=len(all_actions),
                shoot_count=action_types.count("SHOOT"),
                move_count=action_types.count("MOVE"),
                toggle_count=action_types.count("TOGGLE"))
 
        return all_actions
    
    def _should_skip_tactical_llm(
        self,
        role: str,
        unit_ids: List[int],
        local_states: Dict[int, LocalState],
        allowed_actions: Dict[int, List[Action]]
    ) -> bool:
        """Determine if we can skip LLM call for this role (cost savings)."""
        
        # AWACS: Skip if no enemies anywhere AND no threats
        if role == "AWACS":
            for uid in unit_ids:
                state = local_states[uid]
                # Don't skip if any enemies visible or any can hit us
                if state.enemies:
                    return False
            return True  # No enemies at all - use fallback
        
        # SAM: Skip if no enemies in weapon range
        if role == "SAM":
            for uid in unit_ids:
                state = local_states[uid]
                # Check if any enemy has non-zero hit probability
                if any(e.hit_prob > 0 for e in state.enemies):
                    return False  # Has valid targets - use LLM
            return True  # No valid targets - use fallback
        
        # AIRCRAFT/DECOY: ALWAYS call LLM - even with no enemies visible they
        # need intelligent decisions: repositioning, search patterns, screening,
        # and approach vectors are all complex enough to warrant LLM reasoning.
        return False
    
    def _share_role_intel(
        self,
        role: str,
        unit_ids: List[int],
        local_states: Dict[int, LocalState],
        coord: CoordinationState
    ):
        """Share intel from this role for other roles to use."""
        
        if role == "AWACS":
            # AWACS shares all visible enemy positions
            for uid in unit_ids:
                state = local_states[uid]
                for enemy in state.enemies:
                    coord.share_intel("AWACS", "enemy_sighting", {
                        "enemy_id": enemy.id,
                        "enemy_type": enemy.type,
                        "bearing": enemy.bearing,
                        "distance": enemy.dist,
                        "threat": enemy.threat,
                        "source_unit": uid
                    })
        
        elif role == "SAM":
            # SAM shares its coverage zone
            for uid in unit_ids:
                state = local_states[uid]
                coord.share_intel("SAM", "coverage_zone", {
                    "sam_id": uid,
                    "position": state.pos,
                    "range": state.range,
                    "missiles": state.missiles
                })
    
    def _execute_role(
        self,
        role: str,
        unit_ids: List[int],
        local_states: Dict[int, LocalState],
        allowed_actions: Dict[int, List[Action]],
        plan: StrategicPlan,
        coord: CoordinationState
    ) -> Dict[int, Action]:
        """Execute decisions for a single role with enhanced coordination."""
        with trace_tactical_execution(role):
            try:
                # Pre-populate coordination with CURRENT positions
                for uid in unit_ids:
                    current_pos = local_states[uid].pos
                    has_move_option = any(
                        a.type == ActionType.MOVE 
                        for a in allowed_actions.get(uid, [])
                    )
                    if has_move_option:
                        coord.occupy_position(current_pos)
                
                logfire.info(f"{role}_pre_coordination",
                            role=role,
                            unit_positions={uid: local_states[uid].pos for uid in unit_ids},
                            initial_occupied=list(coord.occupied_positions))

                # Find directive for this role
                directive = next((d for d in plan.directives if d.role == role), None)
                if not directive:
                    return self._fallback_by_role(role, unit_ids, allowed_actions, local_states, coord)
                
                # Format actions with bearing
                formatted_actions = {
                    uid: self._format_actions(
                        allowed_actions[uid],
                        local_states[uid],
                        directive=directive,
                        coord=coord
                    )
                    for uid in unit_ids
                }
                
                # Build unit data with bearing
                units_data = []
                for uid in unit_ids:
                    state = local_states[uid]
                    units_data.append({
                        "id": uid,
                        "pos": state.pos,
                        "missiles": state.missiles,
                        "enemies": [
                            {
                                "id": e.id,
                                "type": e.type,
                                "bearing": e.bearing,  # NEW: Use bearing instead of rel_pos
                                "dist": e.dist,
                                "threat": e.threat,
                                "hit_prob": e.hit_prob,
                                "hit_prob_from_enemy": e.hit_prob_from_enemy,  # NEW
                                "can_they_hit_me": e.can_they_hit_me,
                                "is_hvt": e.id in plan.hvt_targets
                            }
                            for e in state.enemies
                        ],
                        "opts": formatted_actions[uid]
                    })
                
                # Get shared intel for this role
                shared_intel = self._get_relevant_intel_for_role(role, coord)
                
                # Get constraints
                occupied_positions = coord.occupied_positions.copy()
                for uid in unit_ids:
                    occupied_positions.discard(local_states[uid].pos)
                
                constraints = self._get_role_constraints(role, occupied_positions, shared_intel)
                
                # Build prompt
                system_prompt = self._get_system_prompt(role)
                user_prompt = self._build_prompt(role, directive, units_data, coord, constraints, plan)
                
                # Create logs directory
                log_dir = Path(f"logs/prompts/tactical/{role.lower()}")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Build combined prompt for logging
                turn_num = plan.turn if hasattr(plan, 'turn') else 0
                full_prompt = f"""{'='*80}
TACTICAL EXECUTION - {role} - TURN {turn_num}
{'='*80}

=== SYSTEM PROMPT ===
{system_prompt}

{'='*80}

=== USER PROMPT ===
{user_prompt}

{'='*80}
METADATA:
- Role: {role}
- Turn: {turn_num}
- Units: {unit_ids}
- Directive Mode: {directive.mode}
- Directive Priority: {directive.priority}
- Shared Intel: {list(shared_intel.keys())}
{'='*80}
"""
                
                # Save to file
                prompt_file = log_dir / f"turn_{turn_num:03d}.txt"
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(full_prompt)
                
                logfire.info(f"{role}_full_prompt",
                            role=role,
                            turn=turn_num,
                            units_count=len(units_data),
                            directive_mode=directive.mode,
                            file_path=str(prompt_file))
                
                logger.info(f"💾 {role} tactical prompt saved: {prompt_file}")
                
                # LLM call
                response = self.llm.complete(
                    system_prompt,
                    user_prompt,
                    temperature=0.3,
                    max_tokens=3000,
                    response_format={"type": "json_object"}
                )
                
                logger.info(f"[{role}] LLM call complete")
                logfire.info(f"{role}_Tactical_Raw_response",
                            role=role,
                            response_length=len(response))

                logger.info(f"\n{'-'*60}\n[{role}] TACTICAL RAW RESPONSE:\n{response}\n{'-'*60}")
                
                # Log response to file
                response_file = log_dir / f"turn_{turn_num:03d}_response.txt"
                with open(response_file, "w", encoding="utf-8") as f:
                    f.write(f"""{'='*80}
TACTICAL RESPONSE - {role} - TURN {turn_num}
{'='*80}

{response}

{'='*80}
""")

                data = json.loads(response)
                plan_obj = RoleTacticalPlan(**data)

                # Check for missing units
                decided_units = {d.id for d in plan_obj.decisions}
                expected_units = set(unit_ids)
                missing_units = expected_units - decided_units

                if missing_units:
                    logfire.warning(f"{role}_missing_units",
                                role=role,
                                missing_units=list(missing_units))
                    logger.warning(f"⚠️ {role}: LLM didn't decide for units {missing_units}")
                
                # Log decisions and store for UI
                for decision in plan_obj.decisions:
                    self.last_decisions[decision.id] = {
                        "role": role,
                        "action_idx": decision.idx,
                        "action_type": decision.action_type,
                        "reasoning": decision.why
                    }
                    logfire.info(f"{role}_decision",
                                role=role, 
                                unit_id=decision.id, 
                                action_idx=decision.idx,
                                action_type=decision.action_type,
                                reasoning=decision.why)
                    logger.info(f" Role: {role} - Unit {decision.id} - {decision.action_type} idx {decision.idx} - {decision.why}")
                
                # Map decisions to actions with validation
                actions = self._map_decisions(
                    plan_obj.decisions,
                    allowed_actions,
                    local_states,
                    coord,
                    formatted_actions
                )

                return actions
            
            except json.JSONDecodeError as e:
                logfire.error(f"{role}_json_parse_failed", role=role, error=str(e))
                logger.error(f"❌ {role}: JSON parsing failed - {e}")
                with trace_fallback(f"{role}_json_failed"):
                    return self._fallback_by_role(role, unit_ids, allowed_actions, local_states, coord)

            except Exception as e:
                logfire.error(f"{role}_failed", role=role, error=str(e), exc_info=True)
                logger.error(f"❌ {role}: Tactical execution failed - {e}")
                with trace_fallback(f"{role}_failed"):  
                    return self._fallback_by_role(role, unit_ids, allowed_actions, local_states, coord)
    
    def _get_relevant_intel_for_role(self, role: str, coord: CoordinationState) -> Dict[str, List]:
        """Get intel relevant to this role from previous roles."""
        intel = {}
        
        if role == "AIRCRAFT":
            # Aircraft benefits from AWACS sightings and SAM coverage
            intel["awacs_sightings"] = coord.get_shared_intel("enemy_sighting")
            intel["sam_coverage"] = coord.get_shared_intel("coverage_zone")
        
        elif role == "DECOY":
            # Decoys benefit from knowing enemy positions
            intel["enemy_positions"] = coord.get_shared_intel("enemy_sighting")
        
        return intel
    
    def _format_actions(
        self, 
        actions: List[Action], 
        state: LocalState,
        directive=None,
        coord: CoordinationState = None
    ) -> List[Dict]:
        """Format actions with bearing and conflict detection."""
        formatted = []
        
        for idx, action in enumerate(actions):
            entry = {
                "idx": idx,
                "act": action.type.name
            }
            
            if action.type == ActionType.SHOOT:
                tgt_id = action.params.get("target_id")
                entry["tgt"] = tgt_id
                
                target = next((e for e in state.enemies if e.id == tgt_id), None)
                if target:
                    entry["hit_prob"] = target.hit_prob
                    entry["tgt_threat"] = target.threat
                    entry["tgt_type"] = target.type
                    entry["tgt_bearing"] = target.bearing  # NEW: bearing
                    entry["description"] = f"SHOOT {target.type} #{tgt_id} ({target.bearing}, {target.dist} tiles) - {target.hit_prob:.0%} hit"
            
            elif action.type == ActionType.MOVE:
                direction = action.params.get("dir")
                entry["dir"] = direction.name
                
                # Check for conflicts
                dx, dy = direction.delta
                new_pos = (state.pos[0] + dx, state.pos[1] + dy)
                is_conflict = coord and coord.is_position_occupied(new_pos)
                entry["conflict"] = is_conflict  # NEW: conflict detection
                
                if not state.enemies:
                    entry["effect"] = "SEARCH"
                    entry["intent"] = "Find targets"
                    entry["bias"] = self._get_search_bias_by_mode(state, directive)
                    entry["description"] = f"MOVE {direction.name} to search ({entry['bias']})"
                elif all(e.hit_prob == 0.0 for e in state.enemies):
                    entry["effect"] = "APPROACH"
                    entry["intent"] = "Close to engage"
                    entry["description"] = f"MOVE {direction.name} to APPROACH enemies"
                else:
                    nearest = min(state.enemies, key=lambda e: e.dist)
                    entry["effect"] = self._calculate_movement_effect(state.pos, direction, nearest)
                    entry["description"] = f"MOVE {direction.name} ({entry['effect']} nearest at {nearest.bearing})"
                
                if is_conflict:
                    entry["description"] += " ⚠️CONFLICT"
            
            elif action.type == ActionType.TOGGLE:
                entry["current"] = "ON" if state.radar else "OFF"
                new_state = "OFF" if state.radar else "ON"
                entry["description"] = f"TOGGLE radar → {new_state}"
            
            elif action.type == ActionType.WAIT:
                entry["description"] = "WAIT (hold position)"
            
            formatted.append(entry)
        
        return formatted
    
    def _get_search_bias_by_mode(self, state: LocalState, directive) -> str:
        """Get search bias based on strategic mode."""
        if not directive:
            return "PATROL_CENTER"
        
        mode = directive.mode
        x, y = state.pos
        
        if mode == "AGGRESSIVE":
            if x < 10:
                return "MOVE_EAST"
            elif x > 10:
                return "MOVE_WEST"
            else:
                return "PATROL_CENTER"
        elif mode == "DEFENSIVE":
            if y < 6:
                return "PATROL_TOP"
            else:
                return "PATROL_CENTER"
        elif mode == "RECON":
            if x < 5 or x > 15:
                return "PATROL_EDGES"
            else:
                return "MOVE_TO_EDGES"
        else:
            return "STAY_NEAR_ALLIES"

    def _calculate_movement_effect(
        self,
        my_pos: Tuple[int, int],
        direction,
        nearest_enemy
    ) -> str:
        """Calculate movement effect (APPROACH/RETREAT/FLANK)."""
        dx, dy = direction.delta
        new_pos = (my_pos[0] + dx, my_pos[1] + dy)
        
        enemy_pos = (
            my_pos[0] + nearest_enemy.rel_pos[0],
            my_pos[1] + nearest_enemy.rel_pos[1]
        )
        
        current_dist = nearest_enemy.dist
        new_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
        
        if new_dist < current_dist:
            return "APPROACH"
        elif new_dist > current_dist:
            return "RETREAT"
        else:
            return "FLANK"
    
    def _build_prompt(
        self,
        role: str,
        directive,
        units_data: List[Dict],
        coord: CoordinationState,
        constraints: str,
        plan: StrategicPlan
    ) -> str:
        """Build tactical prompt with situation context."""
        
        # Include situation analysis if available
        situation_context = ""
        if plan.situation_analysis:
            situation_context = f"\n**SITUATION:** {plan.situation_analysis}"
        if plan.enemy_formation and plan.enemy_formation != "UNKNOWN":
            situation_context += f"\n**ENEMY FORMATION:** {plan.enemy_formation}"
        
        return f"""**{role} TACTICAL - Mode: {directive.mode}**
Objective: {directive.priority}
{situation_context}

{constraints}

**Units:**
{json.dumps(units_data, indent=1)}

**Claimed Targets:** {list(coord.claimed_targets)}

**DECISION RULES:**
1. **CRITICAL threats (dist ≤3):** SHOOT if hit_prob >0.05 (survival first)
2. **HVTs (is_hvt: true):** SHOOT if hit_prob >0.1
3. **Mode-specific:**
   - AGGRESSIVE: Shoot if hit_prob >0.1, prefer APPROACH moves
   - DEFENSIVE: Shoot only CRITICAL, prefer RETREAT moves
   - RECON: Avoid combat unless CRITICAL/HVT, use FLANK moves
4. **Low ammo (<2 missiles):** Conserve for HVTs/CRITICAL only
5. **Conflicts:** Avoid moves marked with conflict: true

**MOVEMENT GUIDANCE:**
- "bearing" shows enemy direction (N, NE, E, SE, S, SW, W, NW)
- "effect" shows result: APPROACH (closer), RETREAT (further), FLANK (same dist)
- When no enemies: follow "bias" field for patrol direction

**OUTPUT FORMAT:**
{{
  "decisions": [
    {{"id": <unit_id>, "idx": <action_index>, "action_type": "<SHOOT|MOVE|WAIT|TOGGLE>", "why": "<brief reasoning>"}}
  ],
  "claimed_targets": [<target_ids_if_shooting>]
}}

**CRITICAL:** Output exactly {len(units_data)} decisions. Include action_type for validation.
"""
    
    def _get_system_prompt(self, role: str) -> str:
        """Get system prompt with game context."""
        return f"""You are a {role} tactical operator for {self.team_name}.

{TACTICAL_GAME_CONTEXT}

**YOUR ROLE: {role}**
{"- AWACS: Unarmed scout - SURVIVE and provide radar. Retreat from ALL threats." if role == "AWACS" else ""}
{"- SAM: Stationary defense - Area denial. Shoot high-probability targets." if role == "SAM" else ""}
{"- AIRCRAFT: Primary striker - Balance aggression with survival." if role == "AIRCRAFT" else ""}
{"- DECOY: Bait/screen - Draw enemy fire away from valuable units." if role == "DECOY" else ""}

**THREAT INTERPRETATION:**
- CRITICAL (dist ≤3): Immediate danger - shoot or retreat NOW
- HIGH (4-6): Can engage next turn
- MEDIUM (7-10): Positioning threat
- LOW (>10): No immediate concern

**BEARING:** N=north, NE=northeast, E=east, etc. Use to understand enemy positions.

**hit_prob_from_enemy:** If >0, enemy CAN hit you. Prioritize defense if high.

**OUTPUT:** Valid JSON with action_type field for each decision."""
    
    def _get_role_constraints(self, role: str, occupied_positions: set, shared_intel: Dict = None) -> str:
        """Get role-specific constraints with shared intel."""
        base = f"""**CONSTRAINTS:**
- NO moving to occupied positions: {list(occupied_positions)}
- NO shooting claimed targets
- Choose idx from opts array
- Include action_type in each decision for validation

**ACTION FORMAT:**
- idx: Action index (0, 1, 2, etc.)
- act: Action type
- description: What the action does
- conflict: true if move would collide (avoid these!)"""
        
        # Add shared intel if available
        intel_text = ""
        if shared_intel:
            if shared_intel.get("sam_coverage"):
                coverage = shared_intel["sam_coverage"]
                if coverage:
                    intel_text += f"\n\n**ALLIED SAM COVERAGE:** {len(coverage)} SAM(s) providing support"
            if shared_intel.get("awacs_sightings"):
                sightings = shared_intel["awacs_sightings"]
                if sightings:
                    intel_text += f"\n**AWACS INTEL:** {len(sightings)} enemy contacts reported"
    
        role_specific = {
            "AIRCRAFT": """

**AIRCRAFT TACTICS:**
- HVT Priority: Shoot if is_hvt=true AND hit_prob >0.15
- Ammo Check: If missiles <2, conserve for HVTs
- AGGRESSIVE: Shoot hit_prob >0.1, APPROACH if no shots
- DEFENSIVE: Only CRITICAL threats, prefer RETREAT""",
            
            "SAM": """

**SAM TACTICS:**
- Stationary: Cannot move, choose SHOOT, TOGGLE, or WAIT
- AGGRESSIVE: Shoot all with hit_prob >0.25
- DEFENSIVE: Only CRITICAL threats
- Toggle to stealth if vulnerable and no good shots""",
            
            "AWACS": """

**AWACS TACTICS:**
- SURVIVAL IS EVERYTHING: Any threat = RETREAT
- Check hit_prob_from_enemy: if >0, you're in danger
- Stay >5 tiles from armed enemies
- WAIT only if completely safe""",
            
            "DECOY": """

**DECOY TACTICS:**
- Draw fire: Move toward enemies (4-6 tiles ideal)
- Screen allies: Position between enemies and AWACS/SAM
- Expendable but valuable: Don't waste, trade for kills
- Spread out: Don't cluster with other decoys"""
        }
        
        return base + intel_text + role_specific.get(role, "")
    
    def _map_decisions(
        self,
        decisions: List[TacticalDecision],
        allowed_actions: Dict[int, List[Action]],
        local_states: Dict[int, LocalState],
        coord: CoordinationState,
        formatted_actions: Dict[int, List[Dict]] = None
    ) -> Dict[int, Action]:
        """Map LLM decisions to actual actions with validation."""
        actions = {}
        
        for dec in decisions:
            if dec.id not in allowed_actions:
                logfire.warning("unit_not_in_allowed_actions", unit_id=dec.id)
                coord.add_blocked(dec.id, "unit_not_found")
                continue
            
            opts = allowed_actions[dec.id]
            if not (0 <= dec.idx < len(opts)):
                logfire.error("invalid_action_index", unit_id=dec.id, idx=dec.idx, max_idx=len(opts)-1)
                coord.add_blocked(dec.id, f"invalid_idx_{dec.idx}")
                continue
            
            action = opts[dec.idx]
            
            # NEW: Validate action_type if provided
            if dec.action_type and dec.action_type != action.type.name:
                logfire.warning("action_type_mismatch",
                            unit_id=dec.id,
                            expected=dec.action_type,
                            actual=action.type.name)
                logger.warning(f"⚠️ Unit {dec.id}: action_type mismatch - expected {dec.action_type}, got {action.type.name}")
                # Don't block - the idx is authoritative, just log the mismatch
            
            # Log action selected
            logfire.info("action_selected",
                    unit_id=dec.id,
                    action_type=action.type.name,
                    action_idx=dec.idx,
                    reasoning=dec.why)
            
            # Validate coordination
            if action.type == ActionType.SHOOT:
                tgt = action.params.get("target_id")
                
                if coord.is_target_claimed(tgt):
                    logfire.warning("target_already_claimed", unit_id=dec.id, target_id=tgt)
                    logger.warning(f"⚠️ Unit {dec.id} tried to shoot claimed target {tgt} - SKIPPED")
                    coord.add_blocked(dec.id, f"target_{tgt}_claimed")
                    continue
                
                coord.claim_target(tgt)
                logfire.info("target_claimed", unit_id=dec.id, target_id=tgt)
            
            elif action.type == ActionType.MOVE:
                new_pos = self._calculate_new_pos(
                    local_states[dec.id].pos,
                    action.params.get("dir")
                )
                
                if coord.is_position_occupied(new_pos):
                    logfire.warning("position_occupied", unit_id=dec.id, position=new_pos)
                    logger.warning(f"⚠️ Unit {dec.id} tried to move to occupied position {new_pos} - SKIPPED")
                    coord.add_blocked(dec.id, f"pos_{new_pos}_occupied")
                    continue
                
                coord.occupy_position(new_pos)
                logfire.info("position_occupied", unit_id=dec.id, position=new_pos)
        
            actions[dec.id] = action
    
        return actions
    
    def _fallback_by_role(
        self,
        role: str,
        unit_ids: List[int],
        allowed_actions: Dict[int, List[Action]],
        local_states: Dict[int, LocalState],
        coord: CoordinationState = None
    ) -> Dict[int, Action]:
        """Role-aware fallback with threat-based prioritization."""
        actions = {}
        
        for uid in unit_ids:
            opts = allowed_actions.get(uid, [])
            if not opts:
                continue
            
            state = local_states[uid]
            
            if role in ["AIRCRAFT", "SAM"]:
                # Shoot any enemy with hit_prob > 0.1
                shootable_enemies = [e for e in state.enemies if e.hit_prob > 0.1]
                
                if shootable_enemies:
                    def priority_score(enemy):
                        threat_scores = {"CRITICAL": 40, "HIGH": 30, "MEDIUM": 20, "LOW": 10}
                        hvt_bonus = 20 if enemy.type in ["AWACS", "SAM"] else 0
                        return threat_scores.get(enemy.threat, 0) + (enemy.hit_prob * 10) + hvt_bonus
                    
                    best_target = max(shootable_enemies, key=priority_score)
                    
                    # Check if target is claimed
                    if coord and coord.is_target_claimed(best_target.id):
                        # Find next best unclaimed target
                        unclaimed = [e for e in shootable_enemies if not coord.is_target_claimed(e.id)]
                        if unclaimed:
                            best_target = max(unclaimed, key=priority_score)
                        else:
                            best_target = None
                    
                    if best_target:
                        shoot_action = next(
                            (a for a in opts 
                             if a.type == ActionType.SHOOT 
                             and a.params.get("target_id") == best_target.id),
                            None
                        )
                        if shoot_action:
                            if coord:
                                coord.claim_target(best_target.id)
                            actions[uid] = shoot_action
                            self.last_decisions[uid] = {
                                "role": role,
                                "action_idx": opts.index(shoot_action),
                                "action_type": "SHOOT",
                                "reasoning": f"Fallback: shoot {best_target.type} (threat: {best_target.threat})"
                            }
                            continue
                
                # Move toward enemies or center
                if state.enemies and all(e.hit_prob == 0 for e in state.enemies):
                    move_action = self._get_best_move(opts, state, "APPROACH")
                    
                    # Check if we actually got a move action
                    if move_action.type == ActionType.MOVE:
                        actions[uid] = move_action
                        self.last_decisions[uid] = {
                            "role": role,
                            "action_idx": opts.index(move_action),
                            "action_type": "MOVE",
                            "reasoning": "Fallback: approach enemies"
                        }
                    else:
                        # Stationary or no useful move - Wait/Toggle
                        actions[uid] = move_action  # This is non-move (likely Wait)
                        self.last_decisions[uid] = {
                            "role": role,
                            "action_idx": opts.index(move_action),
                            "action_type": move_action.type.name,
                            "reasoning": "Fallback: holding position (stationary/safe)"
                        }
                    continue
                
                if not state.enemies:
                    move_action = self._get_centering_move(opts, state)
                    
                    if move_action.type == ActionType.MOVE:
                        actions[uid] = move_action
                        self.last_decisions[uid] = {
                            "role": role,
                            "action_idx": opts.index(move_action),
                            "action_type": "MOVE",
                            "reasoning": "Fallback: search for enemies"
                        }
                    else:
                        actions[uid] = move_action
                        self.last_decisions[uid] = {
                            "role": role,
                            "action_idx": opts.index(move_action),
                            "action_type": move_action.type.name,
                            "reasoning": "Fallback: holding position (stationary/safe)"
                        }
                    continue
                
                actions[uid] = opts[0]
            
            elif role == "AWACS":
                has_threats = any(e.can_they_hit_me for e in state.enemies)
                
                if has_threats:
                    move_action = self._get_best_move(opts, state, "RETREAT")
                    actions[uid] = move_action
                    self.last_decisions[uid] = {
                        "role": role,
                        "action_idx": opts.index(move_action) if move_action in opts else 0,
                        "action_type": "MOVE",
                        "reasoning": "Fallback: retreat from threat"
                    }
                else:
                    wait = next((a for a in opts if a.type == ActionType.WAIT), None)
                    actions[uid] = wait if wait else opts[0]
                    self.last_decisions[uid] = {
                        "role": role,
                        "action_idx": opts.index(actions[uid]),
                        "action_type": actions[uid].type.name,
                        "reasoning": "Fallback: hold position (safe)"
                    }
            
            elif role == "DECOY":
                if state.enemies:
                    move_action = self._get_best_move(opts, state, "APPROACH")
                    actions[uid] = move_action
                    self.last_decisions[uid] = {
                        "role": role,
                        "action_idx": opts.index(move_action) if move_action in opts else 0,
                        "action_type": "MOVE",
                        "reasoning": "Fallback: draw enemy attention"
                    }
                else:
                    move = next((a for a in opts if a.type == ActionType.MOVE), None)
                    actions[uid] = move if move else opts[0]
                    self.last_decisions[uid] = {
                        "role": role,
                        "action_idx": opts.index(actions[uid]),
                        "action_type": actions[uid].type.name,
                        "reasoning": "Fallback: patrol"
                    }
            
            else:
                actions[uid] = opts[0]
        
        logfire.info(f"{role}_fallback", units=len(unit_ids), actions=len(actions))
        return actions

    def _get_best_move(
        self,
        opts: List[Action],
        state: LocalState,
        intent: str
    ) -> Action:
        """Get best movement action based on intent."""
        move_actions = [a for a in opts if a.type == ActionType.MOVE]
        if not move_actions:
            return opts[0]
        
        if not state.enemies:
            return self._get_centering_move(opts, state)
        
        nearest = min(state.enemies, key=lambda e: e.dist)
        enemy_pos = (state.pos[0] + nearest.rel_pos[0], state.pos[1] + nearest.rel_pos[1])
        
        best_move = None
        best_score = float('-inf') if intent == "APPROACH" else float('inf')
        
        for action in move_actions:
            direction = action.params.get("dir")
            dx, dy = direction.delta
            new_pos = (state.pos[0] + dx, state.pos[1] + dy)
            
            new_dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
            
            if intent == "APPROACH":
                score = -new_dist
                if score > best_score:
                    best_score = score
                    best_move = action
            else:
                score = new_dist
                if score > best_score:
                    best_score = score
                    best_move = action
        
        return best_move if best_move else move_actions[0]
    
    def _get_centering_move(self, opts: List[Action], state: LocalState) -> Action:
        """Move toward grid center when no enemies visible."""
        move_actions = [a for a in opts if a.type == ActionType.MOVE]
        if not move_actions:
            return opts[0]
        
        center_x, center_y = 10, 6
        
        best_move = None
        best_dist = float('inf')
        
        for action in move_actions:
            direction = action.params.get("dir")
            dx, dy = direction.delta
            new_pos = (state.pos[0] + dx, state.pos[1] + dy)
            
            dist_to_center = abs(new_pos[0] - center_x) + abs(new_pos[1] - center_y)
            
            if dist_to_center < best_dist:
                best_dist = dist_to_center
                best_move = action
        
        return best_move if best_move else move_actions[0]
    
    def _calculate_new_pos(
        self,
        current_pos: Tuple[int, int],
        direction
    ) -> Tuple[int, int]:
        """Calculate new position after moving in a direction."""
        dx, dy = direction.delta
        return (current_pos[0] + dx, current_pos[1] + dy)
    
    def get_decision_reasoning(self) -> Dict[int, Dict[str, Any]]:
        """Get last decisions for UI display."""
        return self.last_decisions