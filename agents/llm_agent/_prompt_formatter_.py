"""
Draft helpers for shaping game state into an LLM-friendly prompt.

This formatter is intentionally conservative:
- It NEVER mutates game state
- It only reads what exists
- Missing fields are handled safely (None)

Updated version:
- Replaces boolean `has_fired_before` with fire-behavior abstraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from env.core.actions import Action
from env.core.types import ActionType
from env.entities.base import Entity
from ..team_intel import TeamIntel, VisibleEnemy



@dataclass
class PromptConfig:
    nearby_ally_radius: float = 3.0
    nearby_enemy_radius: float = 5.0
    grouping_radius: float = 2.5
    include_hit_probabilities: bool = True
    awacs_stats = None


class PromptFormatter:

    """
    Convert TeamIntel + allowed actions into:
    - structured payload (dict)
    - human-readable prompt (string)
    """

    # Public API
    
    def build_prompt(
        self,
        *,
        intel: TeamIntel,
        allowed_actions: Dict[int, List[Action]],
        config: Optional[PromptConfig] = None,
    ) -> Tuple[str, Dict[str, Any]]:

        cfg = config or PromptConfig()

        payload: Dict[str, Any] = {
            "grid": {
                "width": intel.grid.width,
                "height": intel.grid.height,
            },
            "friendlies": [],
        }

        for entity in intel.friendlies:
            if not entity.alive:
                continue

            summary = self._summarize_entity(entity)
            summary["capabilities"] = self._capabilities(entity)
            summary["nearby_allies"] = self._nearby_allies(
                entity, intel, cfg.nearby_ally_radius
            )
            summary["nearby_enemies"] = self._nearby_enemies(
                entity, intel, cfg.nearby_enemy_radius, cfg.include_hit_probabilities
            )
            summary["grouped_with_allies"] = (
                any(a["distance"] <= cfg.grouping_radius for a in summary["nearby_allies"])
                if summary["nearby_allies"]
                else False
            )

            payload["friendlies"].append(summary)

        prompt = self._format_prompt(payload)
        return prompt, payload


    # Entity summaries
    def _summarize_entity(self, entity: Entity) -> Dict[str, Any]:
        return {
            "id": entity.id,
            "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
            "kind": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
            "position": entity.pos,
        }

    def _capabilities(self, entity: Entity) -> Dict[str, Any]:
        return {
            "mobile": entity.can_move,
            "armed": bool(getattr(entity, "missiles", 0)) or entity.can_shoot,
            "missiles": getattr(entity, "missiles", None),
            "missile_max_range": getattr(entity, "missile_max_range", None),
            "base_hit_prob": getattr(entity, "base_hit_prob", None),
            "min_hit_prob": getattr(entity, "min_hit_prob", None),
            "radar_range": entity.get_active_radar_range(),
        }

    # Nearby allies
  
    def _nearby_allies(
        self,
        entity: Entity,
        intel: TeamIntel,
        radius: float,
    ) -> List[Dict[str, Any]]:

        allies: List[Dict[str, Any]] = []

        for other in intel.friendlies:
            if other.id == entity.id or not other.alive:
                continue

            distance = intel.grid.distance(entity.pos, other.pos)
            if distance <= radius:
                allies.append(
                    {
                        "id": other.id,
                        "kind": other.kind.name if hasattr(other.kind, "name") else str(other.kind),
                        "position": other.pos,
                        "distance": distance,
                        "armed": bool(getattr(other, "missiles", 0)) or other.can_shoot,
                    }
                )
        return allies

    # Nearby enemies 
    def _nearby_enemies(
        self,
        entity: Entity,
        intel: TeamIntel,
        radius: float,
        include_hit_probs: bool,
    ) -> List[Dict[str, Any]]:

        enemies: List[Dict[str, Any]] = []

        for enemy in intel.visible_enemies:
            distance = intel.grid.distance(entity.pos, enemy.position)
            if distance > radius:
                continue

            fire_total = getattr(enemy, "fire_count_total", None)
            fire_recent = getattr(enemy, "fire_count_last_k", None)
            fire_delta = getattr(enemy, "last_fire_step_delta", None)

            entry: Dict[str, Any] = {
                "id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "kind": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "position": enemy.position,
                "distance": distance,
                "fire_behavior": {
                    "total_shots": fire_total,
                    "recent_shots": fire_recent,
                    "last_fired_steps_ago": fire_delta,
                    "aggression_level": self._infer_aggression(fire_recent, fire_delta),
                },
                "armed": None,
                "grouped": self._is_grouped(enemy, intel.visible_enemies, intel, radius),
            }

            if include_hit_probs:
                entry["our_hit_prob"] = intel.estimate_hit_probability(entity, enemy)
                entry["their_hit_prob"] = None

            enemies.append(entry)

        return enemies

    def _infer_aggression(
        self,
        recent_shots: Optional[int],
        last_fire_delta: Optional[int],
    ) -> Optional[str]:

        if recent_shots is None and last_fire_delta is None:
            return None

        if recent_shots is not None:
            if recent_shots >= 4:
                return "very_high"
            if recent_shots >= 2:
                return "high"
            if recent_shots == 1:
                return "medium"

        if last_fire_delta is not None:
            if last_fire_delta <= 1:
                return "recent"
            if last_fire_delta <= 3:
                return "cooling_down"
            return "inactive"

        return None

    def _is_grouped(
        self,
        enemy: VisibleEnemy,
        all_enemies: Iterable[VisibleEnemy],
        intel: TeamIntel,
        radius: float,
    ) -> bool:

        for other in all_enemies:
            if other.id == enemy.id:
                continue
            if intel.grid.distance(enemy.position, other.position) <= radius:
                return True
        return False
    def estimate_next_position(pos,action):
        x,y = pos 
        if action.type != ActionType.MOVE:
            return pos
        else : 
            dx,dy = DIRECTION_DELTAS.get(action.params["dir"],(0,0))
            return (x+dx,y+dy)

        
    def _summarize_actions(world, awacs_id, candidate_actions):
        awacs = world.entities[awacs_id]
        awacs_pos = awacs.position

        enemy_awacs = world.enemy_awacs
        enemy_pos = enemy_awacs.position
        enemy_radar_range = enemy_awacs.radar_range

        summaries = []

        for action in candidate_actions:
            next_pos = estimate_next_position(awacs_pos, action)

            enters_enemy_radar = in_radar_range(
                next_pos,
                enemy_pos,
                enemy_radar_range
            )

            summaries.append({
                "action_type": action.type,
                "direction": getattr(action, "direction", None),
                "current_pos": awacs_pos,
                "estimated_next_pos": next_pos,
                "enters_enemy_radar": enters_enemy_radar,
                "distance_to_enemy_awacs": round(
                    ((next_pos[0] - enemy_pos[0]) ** 2 +
                    (next_pos[1] - enemy_pos[1]) ** 2) ** 0.5,
                    2
                )
            })

        return summaries
        

    # ===============================================================
    # Prompt formatting
    # ===============================================================
    def _format_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Strategic prompt for LLM.
        Emphasizes win condition, long-term safety and role-based reasoning.
        """

        lines: List[str] = []

        # ==================================================
        # GLOBAL STRATEGIC CONTEXT
        # ==================================================

        lines.append("""
                ### ADVANCED TACTICAL DOCTRINES (STRICT & MANDATORY)

                GLOBAL PRIORITY ORDER (NON-NEGOTIABLE):
                1. AWACS SURVIVAL & STEALTH
                2. Enemy AWACS Destruction
                3. Radar Avoidance
                4. SAM Utilization
                5. Search & Coverage
                6. Air-to-Air Combat (LAST RESORT)

                AWACS STEALTH & SURVIVAL DOCTRINE (CRITICAL):
                - Friendly AWACS MUST NEVER ENTER enemy radar range
                - Radar entry is a mission-failure-level risk
                - AWACS behavior must be PROACTIVE, not reactive
                - If current position OR next planned move risks enemy radar detection, MOVE AWAY IMMEDIATELY
                - WAIT is allowed for AWACS ONLY IF:
                - AWACS is fully radar-safe
                - No predicted radar intersection exists in the next move
                - AWACS must avoid straight-line patterns and boundary-hugging behavior
                - Prefer diagonal or offset holding positions that preserve escape vectors

                TERMINAL OBJECTIVE RULE:
                - Destroying the enemy AWACS is the PRIMARY and OVERRIDING objective
                - Once enemy AWACS is detected:
                - ALL aircraft must prioritize closing distance
                - WAIT actions are FORBIDDEN
                - RETREAT is FORBIDDEN
                - Even unarmed aircraft must move to apply pressure, block escape routes, or constrain maneuver space

                SHARED SENSOR FUSION (TEAM INTELLIGENCE)
                - All friendly units share radar and intelligence information
                - Radar coverage from ANY friendly unit (AWACS, SAM, Aircraft)
                is considered TEAM-KNOWN AREA
                - Areas already seen by any friendly radar are NOT valuable for search

                EFFICIENT EXPLORATION & SEARCH
                - Search is performed ONLY in:
                - Unexplored
                - Unknown
                - Team-unseen regions
                - Aircraft MUST NOT search inside:
                - Friendly SAM radar coverage
                - Friendly AWACS radar coverage
                - Redundant exploration of known areas is FORBIDDEN

                Boundary awareness:
                - Be aware of environment boundaries and own radar radius
                - Do NOT overextend toward map edges
                - Preserve future maneuver space

                Search strategy:
                - Split into NON-parallel, NON-mirrored vectors
                - Corners and deep backline regions are high-probability
                - Horizontal-only sweeping is discouraged
                - Expand toward the farthest SAFE unexplored region

                AWACS ACTIVE SCOUTING (CONDITIONAL)
                - If AWACS is NOT under radar threat:
                - It SHOULD contribute to exploration
                - AWACS exploration rules:
                - Safety is ALWAYS dominant
                - Never trade safety for coverage
                - Prefer extending vision toward UNKNOWN regions
                - Must NOT revisit known areas unless repositioning for safety

                RADAR VISIBILITY AWARENESS
                - Explicitly reason about:
                - Am I detected?
                - Am I being tracked?
                - ONLY radar-detected aircraft may evade or lure
                - Radar-undetected aircraft must preserve stealth and continue objectives


                SAM FUNNELING STRATEGY (ROLE-BASED):
                - Air-to-air combat is costly due to limited ammunition
                - SAM-assisted kills are PREFERRED over direct engagement
                - If an aircraft is DETECTED by enemy radar and is near a friendly SAM:
                - ONLY that detected aircraft may retreat to lure the enemy into SAM range
                - Aircraft that are NOT detected by enemy radar must:
                - Maintain stealth
                - Continue AWACS search or terminal objectives
                - NOT follow the SAM lure path
                - SAM luring is NOT a global retreat order

                RADAR VISIBILITY AWARENESS:
                - Aircraft must explicitly reason about whether they are currently detected or tracked
                - ONLY radar-detected aircraft may alter behavior to evade or lure
                - Radar-undetected aircraft must preserve stealth and continue mission objectives

                WAIT ACTION CONSTRAINT:
                - WAIT is allowed ONLY IF:
                - No enemy is detected
                - Radar exposure is zero
                - Position is already optimal
                - Moving would increase future risk
                - Using WAIT in presence of search objectives, SAM opportunities, or AWACS destruction paths is a strategic failure
               

                TARGET DECONFLICTION & ENGAGEMENT SATURATION RULES:
                - Aircraft must avoid redundant targeting of the same enemy
                - Each aircraft must evaluate how many friendly units are already engaging a given enemy

                Engagement sufficiency rules:
                - An enemy is considered SUFFICIENTLY ENGAGED if:
                - It is within friendly SAM engagement range, OR
                - It is already engaged by at least ONE friendly aircraft in a stable firing position

                Rules:
                - If an enemy is sufficiently engaged, additional aircraft MUST NOT target it
                - Aircraft must search for:
                - Unengaged enemies
                - Under-engaged enemies
                - If multiple enemies are present:
                - Priority is given to enemies with ZERO current engagements
                - Then to enemies threatening friendly units or AWACS

                Coordination constraint:
                - Aircraft should implicitly coordinate by observing ongoing engagements
                - Over-concentration of multiple aircraft on a single enemy while other enemies remain unengaged is a tactical failure
               
                ### ENTITY TYPES

                **AWACS**
                - Long-range radar, unarmed, mobile
                - Mission-critical: losing it means immediate defeat
                - Allowed actions: MOVE, WAIT
                - Mission: provide intelligence & survive
                - Safety rules:
                - Never trade safety for short-term coverage
                - Reposition only if future safety improves
                - Evaluate enemy next-step movements when deciding

                **Aircraft**
                - Armed, mobile, medium radar range
                - Limited missiles
                - Allowed actions: MOVE, SHOOT, WAIT
                - Mission: protect AWACS, attack enemy AWACS opportunistically

                **Decoy**
                - Unarmed, mobile
                - Always appears as aircraft to enemies
                - Allowed actions: MOVE, WAIT

                **SAM**
                - Stationary, armed
                - Allowed actions: SHOOT, TOGGLE, WAIT
                - TOGGLE: switch between active and stealth modes
                - Can be used for area denial and attrition

                ### ADVERSARIAL REASONING (MANDATORY)
                - Assume detected enemies act intelligently
                - Predict at least ONE possible enemy move next turn
                - Evaluate worst-case outcomes for AWACS and high-value units
                - Do not base safety on current positions only
                - Any move that allows enemy to close distance next turn is HIGH RISK

                ### SCOUTING AND MOVEMENT
                - Movement itself is scouting
                - Prefer moves that:
                - Expand radar coverage into unseen areas
                - Reduce blind spots
                - Maintain escape paths
                - Avoid circular/repetitive movement (UP → RIGHT → DOWN → LEFT)
                - Lateral or WAIT actions are valid if risk/coverage balance is better

                ### ENGAGEMENT PRINCIPLES
                - Avoid unnecessary air-to-air combat
                - Prefer SAM usage for pressure and attrition
                - Only SHOOT if it meaningfully reduces AWACS risk
                - Multiple units targeting same high-value enemy across turns is acceptable

                ### DECISION FORMAT
                - Respond ONLY with valid JSON using provided function
                - Select at most ONE action per unit
                - Use only allowed actions
                - Include a reason_tag reflecting strategic intent

                ### EXAMPLE REASON_TAGS
                - HIGH_PRESSURE_AVOIDENCE
                - LOW_PRESSURE_ADVANCE
                - ENEMY_DETECTED_ATTACK
                - SUPPORT_AWACS
                - DEFEND_AWACS
                - HOLD_POSITION""")




        # ==================================================
        # FRIENDLY UNITS
        # ==================================================
        
        lines.append("\n=== FRIENDLY UNITS ===")

        for friendly in payload["friendlies"]:
            kind = friendly["kind"]
            fid = friendly["id"]
            pos = friendly["position"]

            lines.append(f"\nUnit {kind}#{fid} at {pos}")

            # Capabilities
            cap = friendly["capabilities"]
            lines.append(
                f"- Capabilities: mobile={cap['mobile']}, armed={cap['armed']}, radar_range={cap['radar_range']}"
            )

            # Role emphasis
            if kind.upper() == "AWACS":
                lines.append(
                    "- ROLE: Strategic sensor and LOSS-CONDITION entity.\n"
                    "  Survival is more important than information gain or positioning."
                )
            elif kind.upper() == "SAM":
                lines.append(
                    "- ROLE: Area denial and attrition.\n"
                    "  Prefer using this unit to reduce enemy pressure."
                )
            else:
                lines.append(
                    "- ROLE: Force projection.\n"
                    "  Protect AWACS and enable enemy AWACS elimination."
                )

            # Nearby enemies
            enemies = friendly.get("nearby_enemies", [])
            if enemies:
                lines.append("- Nearby enemy context:")
                for e in enemies:
                    tags = []

                    fb = e.get("fire_behavior", {})
                    if fb:
                        aggr = fb.get("aggression_level")
                        if aggr:
                            tags.append(f"aggr={aggr}")

                    if e.get("grouped"):
                        tags.append("grouped")

                    if "distance" in e:
                        tags.append(f"d={e['distance']:.1f}")

                    tag_txt = ", ".join(tags)
                    lines.append(
                        f"  • {e['kind']}#{e['id']} at {e['position']} [{tag_txt}]"
                    )
            else:
                lines.append("- Nearby enemy context: none")

            # Allowed actions
            actions = friendly.get("allowed_actions", [])
            if actions:
                lines.append("- Available actions:")
                for a in actions:
                    if a.type == "MOVE":
                        d=a["params"]["dir"]
                        lines.append(f"  • MOVE {d} (risk unknown) ")
                    else:
                        lines.append(f"  •  {a['type']}")



        return "\n".join(lines)
