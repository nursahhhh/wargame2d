
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass, field
from enum import Enum

class RoleDirective(BaseModel):
    """Strategic directive for specific unit role."""
    role: Literal["AIRCRAFT", "SAM", "AWACS", "DECOY"]
    mode: Literal["AGGRESSIVE", "DEFENSIVE", "SUPPORT", "RECON"]
    priority: str = Field(..., description="Primary objective for this role")


class StrategicPlan(BaseModel):
    """Commander's strategic assessment and directives."""
    model_config = ConfigDict(validate_assignment=True)  
    
    turn: int
    priorities: List[str] = Field(..., max_length=3)
    directives: List[RoleDirective]
    hvt_targets: List[int] = Field(default_factory=list)
    # NEW: Enhanced strategic output
    situation_analysis: str = Field(default="", description="Battlefield assessment")
    enemy_formation: str = Field(default="UNKNOWN", description="Inferred enemy intent")
    momentum_suggestion: str = Field(default="", description="Strategy recommendation based on momentum")
    
    @field_validator('priorities')  
    @classmethod
    def validate_priorities(cls, v):
        if len(v) < 1:
            raise ValueError("Must have at least 1 priority")
        return v


class TacticalDecision(BaseModel):
    """Single unit tactical decision."""
    id: int = Field(..., description="Entity ID")
    idx: int = Field(..., description="Action index from allowed actions")
    action_type: str = Field(default="", description="Action type for validation (SHOOT/MOVE/WAIT/TOGGLE)")
    why: str = Field(..., max_length=500, description="Brief reasoning")
    
    @field_validator('why')
    @classmethod
    def validate_reasoning(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Reasoning too short")
        return v.strip()


@dataclass
class CoordinationState:
    """Shared coordination state with intel sharing (NO Pydantic overhead)."""
    claimed_targets: set[int] = field(default_factory=set)
    occupied_positions: set[tuple[int, int]] = field(default_factory=set)
    # NEW: Enhanced coordination
    shared_intel: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    blocked_reasons: Dict[int, str] = field(default_factory=dict)
    
    def claim_target(self, target_id: int):
        """Mark target as claimed."""
        self.claimed_targets.add(target_id)
    
    def occupy_position(self, pos: tuple[int, int]):
        """Mark position as occupied."""
        self.occupied_positions.add(pos)
    
    def is_target_claimed(self, target_id: int) -> bool:
        """Check if target already claimed."""
        return target_id in self.claimed_targets
    
    def is_position_occupied(self, pos: tuple[int, int]) -> bool:
        """Check if position already occupied."""
        return pos in self.occupied_positions
    
    def add_blocked(self, unit_id: int, reason: str):
        """Track why a unit's action was blocked."""
        self.blocked_reasons[unit_id] = reason
    
    def share_intel(self, from_role: str, intel_type: str, data: Dict[str, Any]):
        """Share intelligence between roles (e.g., AWACS -> AIRCRAFT)."""
        key = f"{from_role}_{intel_type}"
        if key not in self.shared_intel:
            self.shared_intel[key] = []
        self.shared_intel[key].append(data)
    
    def get_shared_intel(self, intel_type: str) -> List[Dict[str, Any]]:
        """Get all shared intel of a type."""
        result = []
        for key, data in self.shared_intel.items():
            if intel_type in key:
                result.extend(data)
        return result


class RoleTacticalPlan(BaseModel):
    """Tactical plan for a role (batch of units)."""
    decisions: List[TacticalDecision]
    claimed_targets: List[int] = Field(default_factory=list)
    
    @field_validator("decisions")
    @classmethod
    def validate_decisions(cls, v):
        """Single validator: Check duplicates + negative indices."""
        ids = []
        duplicates = set()
        invalid_idx = []
        
        for d in v:
            if d.id in ids:
                duplicates.add(d.id)
            ids.append(d.id)
            
            if d.idx < 0:
                invalid_idx.append(d.id)
        
        errors = []
        if duplicates:
            errors.append(f"Duplicate decisions for units: {duplicates}")
        if invalid_idx:
            errors.append(f"Negative indices for units: {invalid_idx}")
        
        if errors:
            raise ValueError(" | ".join(errors))
        
        return v


class TurnOutcome(BaseModel):
    """Outcome of a single turn for memory - enhanced with narrative."""
    turn: int
    kills_count: int = 0
    losses_count: int = 0
    kills_by_role: Dict[str, int] = Field(default_factory=dict)  # NEW: Track kills per role
    shots_fired: int = 0 
    shots_hit: int = 0
    key_events: List[str] = Field(default_factory=list)  # NEW: Narrative events


class StrategicMemory(BaseModel):
    """Rolling memory with momentum tracking and narrative history."""
    recent_plans: List[StrategicPlan] = Field(default_factory=list)
    recent_outcomes: List[TurnOutcome] = Field(default_factory=list)

    narrative_history: List[str] = Field(default_factory=list)  # Key events narrative
    missing_enemies: Dict[int, Dict[str, Any]] = Field(default_factory=dict)  # Last known positions
    casualties: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: {"friendly": [], "enemy": []})
    
    def add_plan(self, plan: StrategicPlan):
        """Add plan, keep last 3."""
        self.recent_plans.append(plan)
        if len(self.recent_plans) > 3:
            self.recent_plans.pop(0)
    
    def add_outcome(self, outcome: TurnOutcome):
        """Add outcome, keep last 3."""
        self.recent_outcomes.append(outcome)
        if len(self.recent_outcomes) > 3:
            self.recent_outcomes.pop(0)
        # Also add key events to narrative
        for event in outcome.key_events:
            self.add_narrative(f"T{outcome.turn}: {event}")
    
    def add_narrative(self, event: str):
        """Add narrative event, keep last 10."""
        self.narrative_history.append(event)
        if len(self.narrative_history) > 10:
            self.narrative_history.pop(0)
    
    def update_missing_enemy(self, enemy_id: int, data: Dict[str, Any]):
        """Track last known position of enemy."""
        self.missing_enemies[enemy_id] = data
    
    def remove_missing_enemy(self, enemy_id: int):
        """Remove enemy from missing (killed or visible again)."""
        self.missing_enemies.pop(enemy_id, None)
    
    def add_casualty(self, side: str, entry: Dict[str, Any]):
        """Record casualty (friendly or enemy)."""
        if side in self.casualties:
            self.casualties[side].append(entry)
    
    def get_momentum(self) -> float:
        """Calculate momentum score (-1.0 to +1.0)."""
        if not self.recent_outcomes:
            return 0.0
        
        total_kills = sum(o.kills_count for o in self.recent_outcomes)
        total_losses = sum(o.losses_count for o in self.recent_outcomes)
        total = total_kills + total_losses
        
        if total == 0:
            return 0.0
        
        return (total_kills - total_losses) / total
    
    def get_momentum_suggestion(self) -> str:
        """Get strategy suggestion based on momentum."""
        momentum = self.get_momentum()
        
        if momentum > 0.3:
            return "MAINTAIN_AGGRESSIVE"
        elif momentum < -0.3:
            return "CONSIDER_DEFENSIVE"
        else:
            return "SITUATION_NEUTRAL"
    
    def get_summary(self) -> Dict:
        """Get concise summary for LLM context."""
        if not self.recent_outcomes:
            return {"status": "No history"}
        
        total_kills = sum(o.kills_count for o in self.recent_outcomes)
        total_losses = sum(o.losses_count for o in self.recent_outcomes)
        shots_fired = sum(o.shots_fired for o in self.recent_outcomes)
        shots_hit = sum(o.shots_hit for o in self.recent_outcomes)
        accuracy = shots_hit / max(shots_fired, 1)
        
        return {
            "last_3_turns": {
                "kills": total_kills,
                "losses": total_losses,
                "accuracy": round(accuracy, 2)
            },
            "momentum": round(self.get_momentum(), 2),
            "momentum_suggestion": self.get_momentum_suggestion(),
            "last_priority": self.recent_plans[-1].priorities[0] if self.recent_plans else None,
            "recent_events": self.narrative_history[-3:] if self.narrative_history else [],
            "missing_enemy_count": len(self.missing_enemies)
        }


class BattlefieldStatus(str, Enum):
    """Strategic battlefield assessment."""
    BALANCED = "BALANCED"
    AMMO_SHORTAGE = "AMMO_SHORTAGE"
    AIR_SUPERIORITY = "AIR_SUPERIORITY"
    FORCE_DEFICIT = "FORCE_DEFICIT"
    ELIMINATED = "ELIMINATED"