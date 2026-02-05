from typing import Dict, Any, List
from env.core.types import Team
from env.entities.sam import SAM
from env.world import WorldState


def extract_events(
    *,
    prev_world: WorldState,
    world: WorldState,
    team: Team
) -> List[Dict[str, Any]]:
    """
    Extract ONLY negative, learning-relevant events.
    """
    events: List[Dict[str, Any]] = []

    prev_entities = {
        e.id: e
        for e in prev_world.get_team_entities(team, alive_only=False)
    }

    curr_entities = {
        e.id: e
        for e in world.get_alive_entities()
    }

    # ---------------------------------------------------------
    # 1. ALLY LOSS (irreversible)
    # ---------------------------------------------------------
    for entity_id, prev_entity in prev_entities.items():
        if prev_entity.alive and entity_id not in curr_entities:
            event = {
                "type": "ALLY_LOST",
                "turn": world.turn,
                "entity_id": entity_id,
                "entity_type": prev_entity.__class__.__name__,
                "last_position": prev_entity.pos,
                "irreversible": True,
                "severity": "HIGH"
            }

            # Escalate if SAM
            if isinstance(prev_entity, SAM):
                event["type"] = "SAM_LOST"
                event["severity"] = "CRITICAL"
                event["capability_lost"] = "AIR_DEFENSE_COVERAGE"

            events.append(event)

    # ---------------------------------------------------------
    # 2. TACTICAL DEGRADATION
    # ---------------------------------------------------------
    if world.turns_without_movement >= 5:
        events.append({
            "type": "TACTICAL_STALL",
            "turn": world.turn,
            "stall_type": "NO_MOVEMENT",
            "turns": world.turns_without_movement,
            "severity": "MEDIUM"
        })

    if world.turns_without_shooting >= 5:
        events.append({
            "type": "TACTICAL_STALL",
            "turn": world.turn,
            "stall_type": "NO_ENGAGEMENT",
            "turns": world.turns_without_shooting,
            "severity": "MEDIUM"
        })

    # ---------------------------------------------------------
    # 3. TERMINAL FAILURE
    # ---------------------------------------------------------
    if world.game_over and world.winner != team:
        events.append({
            "type": "MISSION_FAILURE",
            "turn": world.turn,
            "winner": world.winner.name if world.winner else None,
            "reason": world.game_over_reason,
            "severity": "CRITICAL",
            "irreversible": True
        })

    return events
