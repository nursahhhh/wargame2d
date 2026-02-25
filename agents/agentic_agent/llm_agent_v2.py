
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set
from collections import defaultdict

from env.core.types import Team, EntityKind
from env.core.actions import Action
from env.world import WorldState
from ..base_agent import BaseAgent
from agents.registry import register_agent
from .schemas import TurnOutcome, StrategicMemory
from .observed_state import StateBuilder, GlobalState, LocalState
from .strategic_commander import StrategicCommander
from .tactical_executor import TacticalExecutor
from .llm_clients import OpenRouterClient  
from .observability import logfire
from env.mechanics.combat import CombatResolutionResult

import os
from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    from env.environment import StepInfo

logger = logging.getLogger(__name__)


@register_agent("llm_agent_v2")
class LLMAgentV2(BaseAgent):
    """Enhanced LLM Agent with memory tracking and UI metadata."""

    def __init__(
        self,
        team: Team,
        name: str = None,
        strategic_model: str = "google/gemini-2.5-flash",  
        tactical_model: str = "google/gemini-2.5-flash",
        openrouter_key: Optional[str] = None,
        enable_memory: bool = True
    ):
        """Initialize V2 agent with enhanced memory tracking."""
        super().__init__(team, name or f"LLMAgentV2-{team.name}")
        
        logfire.instrument_openai()

        # API Key
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self._llm_disabled = False
        if not self.openrouter_key:
            logger.error(
                "OPENROUTER_API_KEY is not set. LLM agent '%s' will return empty actions. "
                "Set the env var on Render (Dashboard → Environment).",
                self.name
            )
            self._llm_disabled = True

        if not self._llm_disabled:
            self.strategic_llm = OpenRouterClient(
                api_key=self.openrouter_key,
                model=strategic_model
            )
            self.tactical_llm = OpenRouterClient(
                api_key=self.openrouter_key,
                model=tactical_model
            )
            self.strategic_commander = StrategicCommander(
                llm_client=self.strategic_llm,
                team_name=self.team.name
            )
            self.tactical_executor = TacticalExecutor(
                llm_client=self.tactical_llm,
                team_name=self.team.name
            )
        else:
            self.strategic_llm = None
            self.tactical_llm = None
            self.strategic_commander = None
            self.tactical_executor = None

        # Strategic memory with enhanced tracking
        self.enable_memory = enable_memory
        self.strategic_memory = StrategicMemory() if enable_memory else None

        # Enemy memory - track last seen positions
        self._enemy_memory: Dict[int, Dict[str, Any]] = {}
        self._recorded_kill_ids: Set[int] = set()

        logfire.info("agent_initialized",
                    name=self.name,
                    strategic_model=strategic_model,
                    tactical_model=tactical_model,
                    memory_enabled=enable_memory,
                    llm_disabled=self._llm_disabled,
                    provider="openrouter_only")

    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """Main decision-making pipeline with enhanced memory and UI metadata."""
        # Graceful degradation: if no API key, return empty actions
        if self._llm_disabled:
            logger.warning("LLM disabled (no API key) — %s returning empty actions.", self.name)
            return {}, {"error": "OPENROUTER_API_KEY not configured. Set it in Render Dashboard → Environment."}

        world: WorldState = state["world"]

        with logfire.span("agent_turn", turn=world.turn, team=self.team.name):
            try:
                # === STAGE 1: Build States ===
                global_state = StateBuilder.build_global_state(world, self.team)
                
                # Update enemy memory BEFORE building local states
                visible_enemy_ids = self._update_enemy_memory(world, global_state)
                missing_enemies = self._collect_missing_enemies(visible_enemy_ids, world.turn)
                
                # Update casualties from previous turn
                self._update_casualties(step_info, world)
                
                # Collect alive entities and allowed actions
                allowed_actions = {}
                local_states = {}
                units_by_role = defaultdict(list)

                for entity in world.get_team_entities(self.team, alive_only=True):
                    allowed = entity.get_allowed_actions(world)
                    if allowed:
                        local_state = StateBuilder.build_local_state(entity.id, world, self.team)
                        allowed_actions[entity.id] = allowed
                        local_states[entity.id] = local_state
                        units_by_role[local_state.type].append(entity.id)

                # === STAGE 2: Strategic Planning (with memory) ===
                strategic_plan = self.strategic_commander.plan(
                    global_state, 
                    memory=self.strategic_memory
                )
                
                # === STAGE 3: Tactical Execution ===
                actions = self.tactical_executor.execute(
                    units_by_role=units_by_role,
                    local_states=local_states,
                    allowed_actions=allowed_actions,
                    strategic_plan=strategic_plan
                )
                
                # === STAGE 4: Update Memory (after turn) ===
                if self.strategic_memory and step_info:
                    outcome = self._extract_turn_outcome(world, step_info, units_by_role)
                    self.strategic_memory.add_plan(strategic_plan)
                    self.strategic_memory.add_outcome(outcome)
                    
                    # Update missing enemies in strategic memory
                    for enemy in missing_enemies:
                        self.strategic_memory.update_missing_enemy(enemy["id"], enemy)
                
                # === STAGE 5: Build Enhanced Metadata for UI ===
                decision_reasoning = self.tactical_executor.get_decision_reasoning()
                memory_summary = self.strategic_memory.get_summary() if self.strategic_memory else {}
                
                metadata = {
                    "status": "success",
                    "version": "v2_hybrid",
                    "turn": world.turn,
                    "units_by_role": {k: len(v) for k, v in units_by_role.items()},
                    "actions_count": len(actions),
                    # Strategic info for UI
                    "strategic": {
                        "priorities": strategic_plan.priorities,
                        "hvt_targets": strategic_plan.hvt_targets,
                        "situation_analysis": strategic_plan.situation_analysis,
                        "enemy_formation": strategic_plan.enemy_formation,
                        "momentum_suggestion": strategic_plan.momentum_suggestion,
                        "directives": [
                            {"role": d.role, "mode": d.mode, "priority": d.priority}
                            for d in strategic_plan.directives
                        ]
                    },
                    # Tactical reasoning for UI (per-unit)
                    "tactical_reasoning": decision_reasoning,
                    # Memory summary for UI
                    "memory": memory_summary,
                    # Force composition
                    "forces": {
                        "ally_counts": global_state.ally_counts,
                        "enemy_counts": global_state.enemy_counts,
                        "enemy_intent": global_state.enemy_intent
                    },
                    # Missing enemies
                    "missing_enemies_count": len(missing_enemies),
                    # Legacy
                    "reasoning": f"Turn {world.turn}: {strategic_plan.situation_analysis or 'Executed ' + str(len(actions)) + ' actions'}"
                }
                
                logfire.info("turn_complete",
                             actions_count=len(actions),
                             units_by_role=metadata["units_by_role"],
                             enemy_intent=global_state.enemy_intent,
                             momentum=memory_summary.get("momentum", 0))
                
                return actions, metadata
            
            except Exception as e:
                logfire.error("agent_fatal_error", team=self.team.name, error=str(e), exc_info=True)
                return self._emergency_fallback(world)

    def _update_enemy_memory(self, world: WorldState, global_state: GlobalState) -> Set[int]:
        """Track enemy positions in memory. Returns set of currently visible enemy IDs."""
        visible_ids: Set[int] = set()
        
        for enemy in global_state.enemies:
            visible_ids.add(enemy.id)
            self._enemy_memory[enemy.id] = {
                "id": enemy.id,
                "type": enemy.type,
                "last_seen_position": enemy.pos,
                "last_seen_turn": world.turn,
            }
        
        return visible_ids
    
    def _collect_missing_enemies(self, visible_ids: Set[int], turn: int) -> List[Dict[str, Any]]:
        """Collect enemies not currently visible but in memory."""
        missing: List[Dict[str, Any]] = []
        
        for enemy_id, entry in self._enemy_memory.items():
            if enemy_id in visible_ids:
                continue
            if enemy_id in self._recorded_kill_ids:
                continue  # Enemy was killed, don't report as missing
            
            last_seen_turn = entry.get("last_seen_turn", turn)
            turns_since_seen = max(turn - last_seen_turn, 0)
            
            missing.append({
                **entry, 
                "turns_since_seen": turns_since_seen
            })
        
        return missing
    
    def _update_casualties(self, step_info: Optional["StepInfo"], world: WorldState) -> None:
        """Record deaths for narrative and remove from enemy memory."""
        if step_info is None or not self.strategic_memory:
            return
        
        combat = getattr(step_info, "combat", None)
        if combat is None:
            return
        
        killed_ids = getattr(combat, "killed_entity_ids", []) or []
        if not killed_ids:
            return
        
        killed_on_turn = max(world.turn - 1, 0)
        
        # Build killer lookup
        killers: Dict[int, int] = {}
        for result in getattr(combat, "combat_results", []) or []:
            if getattr(result, "target_killed", False) and getattr(result, "target_id", None) is not None:
                killers[result.target_id] = getattr(result, "attacker_id", None)
        
        for entity_id in killed_ids:
            if entity_id in self._recorded_kill_ids:
                continue
            
            entity = world.get_entity(entity_id)
            if entity is None:
                continue
            
            is_friendly = entity.team == self.team
            killer_id = killers.get(entity_id)
            
            entry = {
                "id": entity.id,
                "team": entity.team.name,
                "type": entity.kind.name,
                "killed_on_turn": killed_on_turn,
            }
            
            if killer_id is not None:
                killer_ent = world.get_entity(killer_id)
                if killer_ent:
                    entry["killed_by"] = {
                        "id": killer_ent.id,
                        "team": killer_ent.team.name,
                        "type": killer_ent.kind.name,
                    }
            
            # Add to strategic memory
            if is_friendly:
                self.strategic_memory.add_casualty("friendly", entry)
                self.strategic_memory.add_narrative(f"T{killed_on_turn}: Lost {entity.kind.name}#{entity.id}")
            else:
                self.strategic_memory.add_casualty("enemy", entry)
                self.strategic_memory.add_narrative(f"T{killed_on_turn}: Eliminated enemy {entity.kind.name}#{entity.id}")
                # Remove from enemy memory
                self._enemy_memory.pop(entity_id, None)
                self.strategic_memory.remove_missing_enemy(entity_id)
            
            self._recorded_kill_ids.add(entity_id)

    def _emergency_fallback(self, world: WorldState) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """Ultra-safe fallback if all else fails."""
        actions = {}
        logfire.error("emergency_fallback", team=self.team.name, turn=world.turn)
        
        for entity in world.get_team_entities(self.team, alive_only=True):
            allowed = entity.get_allowed_actions(world)
            if allowed:
                shoot = next((a for a in allowed if a.type.name == "SHOOT"), None)
                actions[entity.id] = shoot if shoot else allowed[0]
        
        return actions, {  
            "status": "emergency_fallback",
            "turn": world.turn,
            "actions_count": len(actions),
            "reasoning": "Emergency fallback - using basic attack logic"
        }
    
    def _extract_turn_outcome(
        self, 
        world: WorldState, 
        step_info: "StepInfo",
        units_by_role: Dict[str, List[int]]
    ) -> TurnOutcome:
        """Extract turn outcome with enhanced tracking for memory."""
        combat_result = getattr(step_info, 'combat_result', None) or getattr(step_info, 'combat', None)
        
        kills_count = 0
        kills_by_role: Dict[str, int] = {}
        losses_count = 0
        shots_fired = 0
        shots_hit = 0
        key_events: List[str] = []
        
        if combat_result:
            combat_results = getattr(combat_result, 'combat_results', []) or []
            
            # Track shots
            for r in combat_results:
                if getattr(r, 'success', False):
                    shots_fired += 1
                    attacker_id = getattr(r, 'attacker_id', None)
                    
                    # Find attacker role
                    attacker_role = None
                    for role, ids in units_by_role.items():
                        if attacker_id in ids:
                            attacker_role = role
                            break
                    
                    if getattr(r, 'hit', False):
                        shots_hit += 1
                    
                    if getattr(r, 'target_killed', False):
                        kills_count += 1
                        if attacker_role:
                            kills_by_role[attacker_role] = kills_by_role.get(attacker_role, 0) + 1
                        
                        target_id = getattr(r, 'target_id', None)
                        target = world.get_entity(target_id) if target_id else None
                        target_type = target.kind.name if target else "UNKNOWN"
                        key_events.append(f"Eliminated {target_type}#{target_id}")
            
            # Track losses
            killed_ids = getattr(combat_result, 'killed_entity_ids', []) or []
            for killed_id in killed_ids:
                killed = world.get_entity(killed_id)
                if killed and killed.team == self.team:
                    losses_count += 1
                    key_events.append(f"Lost {killed.kind.name}#{killed_id}")
        
        return TurnOutcome(
            turn=world.turn,
            kills_count=kills_count,
            losses_count=losses_count,
            kills_by_role=kills_by_role,
            shots_fired=shots_fired,
            shots_hit=shots_hit,
            key_events=key_events
        )