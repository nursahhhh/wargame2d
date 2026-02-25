import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any

from pydantic import BaseModel
from pydantic_ai import Agent,Tool
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from env.core.types import MoveDir, ActionType
from env.entities.base import Entity
from env.entities import SAM
from ..team_intel import TeamIntel, VisibleEnemy

# ============================================================
# Load environment
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ============================================================
# Data models
# ============================================================
# ============================================================
# Data models
# ============================================================
class StrategyUpdateRequest(BaseModel):
    reason: str
    observed_shift: str
    urgency_level: int  # 1-5 scale

    model_config = {
        "arbitrary_types_allowed": True
    }


class UnitAction(BaseModel):
    entity_id: int
    action: str
    direction: Optional[str] = None  # Previously 'dir'
    reason_tag: str
    note: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None

    model_config = {
        "arbitrary_types_allowed": True
    }


class TeamTurnPlan(BaseModel):
    actions: List[UnitAction]

    model_config = {
        "arbitrary_types_allowed": True
    }


class CommanderIntent(BaseModel):
    intent: str
    justification: str
    suggested_patterns: List[str]

    model_config = {
        "arbitrary_types_allowed": True
    }


class CommanderDeps(BaseModel):
    intel: TeamIntel
    step: int
    escalation_reason: Optional[str] = None
    observed_shift: Optional[str] = None
    urgency: Optional[int] = 3  # default normal

    model_config = {
        "arbitrary_types_allowed": True
    }

    
# Game state wrapper
# ============================================================
class GameDeps:
    """
    Wrapper providing structured game state to LLM agents.
    """

    def __init__(self, intel: TeamIntel, step: int):
        self.intel = intel
        self.step = step

    # ----------------------------
    # Grid info
    # ----------------------------
    @property
    def grid(self) -> Dict[str, int]:
        return {"width": self.intel.grid.width, "height": self.intel.grid.height}

    # ----------------------------
    # Global state
    # ----------------------------
    @property
    def global_state(self) -> Dict[str, Any]:
        return {"aggression": round(self.intel.aggression_level(turn=self.step), 2)}

    # ----------------------------
    # Friendlies
    # ----------------------------
    def friendlies(self) -> List[Dict[str, Any]]:
        units = []
        for entity in self.intel.friendlies:
            if not entity.alive:
                continue
            summary = {
                "id": entity.id,
                "team": getattr(entity.team, "name", str(entity.team)),
                "kind": getattr(entity.kind, "name", str(entity.kind)),
                "position": entity.pos,
                "capabilities": self._capabilities(entity),
                "nearby_allies": self._nearby_allies(entity),
                "nearby_enemies": self._nearby_enemies(entity),
                "grouped_with_allies": self._grouped_with_allies(entity),
                "pressure": {
                    "value": round(self.intel.pressure_around(entity), 2),
                    "level": self.intel.pressure_level(entity),
                },
                "allowed_actions": self._allowed_actions(entity),
            }
            units.append(summary)
        return units
    
    def allowed_actions_map(self) -> Dict[int, List[Any]]:
        """
        Returns a mapping: entity_id -> list of allowed Action objects for all alive friendlies.
        """
        mapping = {}
        for entity in self.intel.friendlies:
            if not entity.alive:
                continue
            mapping[entity.id] = getattr(entity, "allowed_actions", [])
        return mapping

    # ----------------------------
    # Unit capabilities
    # ----------------------------
    def _capabilities(self, entity: Entity) -> Dict[str, Any]:
        caps = {
            "mobile": entity.can_move,
            "armed": bool(getattr(entity, "missiles", 0)) or entity.can_shoot,
            "missiles": getattr(entity, "missiles", None),
            "missile_max_range": getattr(entity, "missile_max_range", None),
            "base_hit_prob": getattr(entity, "base_hit_prob", None),
            "min_hit_prob": getattr(entity, "min_hit_prob", None),
            "radar_range": entity.get_active_radar_range(),
        }
        if isinstance(entity, SAM) or hasattr(entity, "is_toggled"):
            caps["is_radar_active"] = getattr(entity, "is_toggled", False)
            caps["activation_range"] = getattr(entity, "activation_range", None)
            caps["can_shoot_when_active"] = getattr(entity, "can_shoot", False)
        return caps

    # ----------------------------
    # Nearby allies
    # ----------------------------
    def _nearby_allies(self, entity: Entity, radius: float = 3.0) -> List[Dict[str, Any]]:
        allies = []
        for other in self.intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue
            distance = self.intel.grid.distance(entity.pos, other.pos)
            if distance <= radius:
                allies.append({
                    "id": other.id,
                    "kind": getattr(other.kind, "name", str(other.kind)),
                    "position": other.pos,
                    "distance": distance,
                    "armed": bool(getattr(other, "missiles", 0)) or other.can_shoot,
                })
        return allies

    # ----------------------------
    # Nearby enemies
    # ----------------------------
    def _nearby_enemies(self, entity: Entity, radius: float = 5.0) -> List[Dict[str, Any]]:
        enemies = []
        for enemy in self.intel.visible_enemies:
            distance = self.intel.grid.distance(entity.pos, enemy.position)
            if distance > radius:
                continue
            entry = {
                "id": enemy.id,
                "team": getattr(enemy.team, "name", str(enemy.team)),
                "kind": getattr(enemy.kind, "name", str(enemy.kind)),
                "position": enemy.position,
                "distance": distance,
                "threat_score": round(self.intel.enemy_threat_score(enemy, entity.pos), 2),
                "fire_behavior": {
                    "total_shots": getattr(enemy, "fire_count_total", None),
                    "recent_shots": getattr(enemy, "fire_count_last_k", None),
                    "last_fired_steps_ago": getattr(enemy, "last_fire_step_delta", None),
                },
                "grouped": self.intel._enemy_is_grouped(enemy, self.intel.visible_enemies, radius),
                "our_hit_prob": float(self.intel.estimate_hit_probability(entity, enemy)),
            }
            if entity.name == "AWACS":
                entry["enemy_proximity_trend"] = self.intel.radar_threat_trend(entity, enemy)
            enemies.append(entry)
        return enemies

    # ----------------------------
    # Grouped with allies
    # ----------------------------
    def _grouped_with_allies(self, entity: Entity, radius: float = 2.5) -> bool:
        return any(a["distance"] <= radius for a in self._nearby_allies(entity, radius))

    # ----------------------------
    # Allowed actions
    # ----------------------------
    def _allowed_actions(self, entity: Entity) -> List[Dict[str, Any]]:
        actions = getattr(entity, "allowed_actions", [])
        results = []
        for a in actions:
            action_dict = {
                "type": a.type.name,
                "params": {"direction": getattr(a.params, "dir", None)}
                if a.type == ActionType.MOVE else a.params,
            }
            results.append(action_dict)
        return results
# Define the executor function


def select_actions(actions: list[UnitAction]) -> list[Dict[str, Any]]:
    chosen_actions: Dict[int, Dict[str, Any]] = {}
    for action in actions:
        unit_id = getattr(action, "entity_id", None)
        if unit_id is None:
            continue
        chosen_actions[unit_id] = {
            "entity_id": unit_id,
            "action": getattr(action, "action", "HOLD_POSITION"),
            "reason_tag": getattr(action, "reason_tag", "HOLD_POSITION"),
            "direction": getattr(action, "direction", None),
            "note": getattr(action, "note", None),
            "reasoning": getattr(action, "reasoning", None),
        }
    return list(chosen_actions.values())

ACTION_TOOL = Tool(
    select_actions,  
    description="Select at most one valid action per entity. Only choose from allowed actions.",
)    

# ============================================================
# Tools
# ============================================================


def request_strategy_update(
    reason: str,
    observed_shift: str,
    urgency_level: int
) -> Dict[str, Any]:
    """
    Request a new high-level strategy from the Commander agent.
    """
    # This is a placeholder. You can add logging or validation here if needed.
    return {
        "reason": reason,
        "observed_shift": observed_shift,
        "urgency_level": urgency_level
    }

STRATEGY_UPDATE_TOOL = Tool(
    request_strategy_update,
    description="Request a new high-level strategy from Commander when current strategy is invalidated."
) 

def set_commander_intent(
    intent: str,
    justification: str,
    suggested_patterns: list[str]
) -> Dict[str, Any]:
    """
    Set high-level strategic intent for the team.
    """
    return {
        "intent": intent,
        "justification": justification,
        "suggested_patterns": suggested_patterns
    }

COMMANDER_TOOL = Tool(
    set_commander_intent,
    description="Set high-level strategic intent for the team."
)
# ============================================================
# Prompts
# ============================================================

EXECUTOR_PROMPT = """
You are a tactical AI executor controlling friendly units in a 2D combat grid: AWACS, Aircraft, Decoys, and SAM sites.
All cells within range of friendly AWACS or active SAM radar are instantly shared among all friendly units.

MISSION OBJECTIVES (ordered by priority):
1. PROTECT FRIENDLY AWACS — Survival is absolute.
2. DESTROY ENEMY AWACS — Terminal win condition.
3. AVOID DETECTION — Stay outside enemy radar.
4. GAIN INFORMATION — Expand radar coverage into unobserved areas.
5. ACHIEVE NUMERICAL ADVANTAGE — Coordinate SAM + aircraft.
6. ENGAGE IN COMBAT — Prefer engagements that improve force advantage.

STRATEGIC EXECUTION:
- You must follow the current Commander intent exactly.
- Evaluate strategic triggers each turn:
  ally_unit_destroyed, new_enemy_detected, base_under_attack, heavy_casualties,
  objective_captured, sudden_loss_of_air_superiority, unexpected_enemy_maneuver,
  strategic_asset_threatened.
- If any trigger occurs, call the Commander for a strategy update.
- Otherwise, execute actions according to the current strategy.

UNIT CAPABILITIES:
- AWACS: MOVE, WAIT; provides wide-area radar; apply HARD constraints strictly.
- Aircraft: MOVE, SHOOT (if armed), WAIT; escort, exploration, interception.
- Decoy: MOVE, WAIT; expendable scouting or deception.
- SAM: Shoot, Toggle, Wait; static area denial, default ON.

HARD CONSTRAINTS:
- AWACS must never enter known enemy radar.
- AWACS must never end turn with zero safe escape routes.
- Actions that guarantee AWACS destruction next turn are forbidden.
- Exploration inside friendly radar coverage is invalid.

OUTPUT FORMAT:
- Respond ONLY with JSON matching the provided function schema.
- Call exactly ONE function: select_actions (if strategy valid) or request_strategy_update (if replanning needed).
- Include a reason_tag and reasoning object for every action:
  { "threat_level": 0-10, "exposure_risk": 0-10, "advantage_gain": 0-10,
    "coordination_value": 0-10, "justification": "Short tactical explanation" }
- Omit units that should WAIT unless explicitly included.

RECENT HISTORY, COMMANDER INTENT, GAME STATE, and EXPERIENCE HEURISTICS
will be passed as structured input via the GameDeps object.
"""
COMMANDER_PROMPT ="""
    You are a tactical commander AI responsible for dynamically inferring the **high-level strategic patterns** for your team.
    Do NOT select low-level moves — focus on overarching strategy that can guide the Captain agent.

    Your goal is to interpret the current game state and suggest strategic patterns that:
    - Maximize force effectiveness
    - Reduce risk to critical units (AWACS, SAM)
    - Exploit enemy weaknesses dynamically
    - Adapt to emerging situations and threats

    Consider for each unit type:
    - Position and capabilities (armed, mobile, radar)
    - Nearby allies and enemies
    - Current pressure, threat scores, and grouping
    - Potential synergies and combined maneuvers

    Respond ONLY in valid JSON with this format:
    {
        "intent": "...",                  # Short identifier for overall strategy
        "justification": "...",           # Explain why this strategy is chosen
        "suggested_patterns": ["..."]     # Tactical guidance for units (dynamic suggestions)
    }

    Suggested patterns can include, but are not limited to:
    - "Split aircraft to explore high-value areas while minimizing exposure"
    - "Keep decoys trailing lead aircraft for distraction"
    - "Converge units for coordinated ambush or multi-angle attack"
    - "Hold key positions to deny enemy access or force chokepoints"
    - "Force enemy to split, then exploit 2v1 advantage"
    - "Reposition SAMs to maximize overlapping radar coverage"
    - "Adapt unit distribution based on real-time enemy behavior"
    """

# ============================================================
# Agents
# ============================================================
executor_agent: Agent[GameDeps, TeamTurnPlan] = Agent[
    GameDeps, TeamTurnPlan
](
    model="openrouter:x-ai/grok-4.1-fast",
    deps_type=GameDeps,
    output_type=TeamTurnPlan,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 16,
        logfire_reasoning={"enabled": True, "effort": "medium"},
        api_key=API_KEY,
    ),
    instructions=EXECUTOR_PROMPT,
    tools=[ACTION_TOOL, STRATEGY_UPDATE_TOOL],
    output_retries=3,
)

commander_agent: Agent[GameDeps, CommanderIntent] = Agent[
    GameDeps, CommanderIntent
](
    model="openrouter:x-ai/grok-4.1-fast",
    deps_type=GameDeps,
    output_type=CommanderIntent,
    model_settings=OpenRouterModelSettings(
        max_tokens=1024 * 16,
        logfire_reasoning={"enabled": True, "effort": "high"},
        api_key=API_KEY,
    ),
    instructions=COMMANDER_PROMPT,
    tools=[COMMANDER_TOOL],
    output_retries=3,
)

# ============================================================
# Utility functions
# ============================================================
def get_executor_plan(intel: GameDeps) -> TeamTurnPlan:
    return executor_agent.run(intel)

def get_commander_strategy(intel: GameDeps) -> CommanderIntent:
    return commander_agent.run(intel)