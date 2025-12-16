import json
import random
from copy import deepcopy
from enum import Enum
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set
from .actors.analyst import analyst_agent, GameAnalysis, ActionAnalysis
from env.core.actions import Action
from env.core.types import Team, ActionType, MoveDir
from env.world import WorldState
from .actors.executer import (
    player,
    MoveAction,
    ShootAction,
    WaitAction,
    ToggleAction,
    EntityAction as LLMEntityAction,
)
from .actors.game_deps import GameDeps
from .actors.strategist import strategist_agent
from .actors.watchdog import watchdog_agent, CallbackAssessment
from .prompts.analyst import ANALYST_USER_PROMPT_TEMPLATE
from .prompts.game_info import GAME_INFO
from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from agents.llm_agent.helpers.prompt_formatter import PromptFormatter, PromptConfig

if TYPE_CHECKING:
    from env.environment import StepInfo


@register_agent("llm_basic")
class LLMAgent(BaseAgent):
    """
    """

    def __init__(
            self,
            team: Team,
            name: str = None,
            seed: Optional[int] = None,
            **_: Any,
    ):
        """
        Initialize random agent.

        Args:
            team: Team to control
            name: Agent name (default: "LLMAgent")
            seed: Random seed for reproducibility (None = random)
        """
        super().__init__(team, name)
        self.rng = random.Random(seed)
        self.state_formatter = PromptFormatter()
        self.prompt_config = PromptConfig()
        # Track last-seen info for enemies that drop out of visibility.
        self._enemy_memory: Dict[int, Dict[str, Any]] = {}
        # Track confirmed casualties with turn metadata.
        self._casualties: Dict[str, list[Dict[str, Any]]] = {"friendly": [], "enemy": []}
        self._recorded_kill_ids: set[int] = set()

        self.game_deps = GameDeps()

    def get_actions(
            self,
            state: Dict[str, Any],
            step_info: Optional["StepInfo"] = None,
            **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        """
        Generate random actions for all entities by sampling allowed actions.

        Args:
            state: Current game state
            step_info: Optional previous step resolution info (unused)

        Returns:
            Tuple of (actions, metadata)
        """
        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)
        actions: Dict[int, Action] = {}
        metadata: Dict[str, Any] = {}
        allowed_actions: Dict[int, list[Action]] = {}

        # Drop enemies that are known-dead from memory so they don't appear as "missing".
        self._prune_dead_enemies(world)

        visible_enemy_ids = self._update_enemy_memory(intel, world.turn)
        missing_enemies = self._collect_missing_enemies(visible_enemy_ids, world.turn)
        self._update_casualties(step_info, world)

        for entity in intel.friendlies:
            if not entity.alive:
                continue
            allowed = entity.get_allowed_actions(world)
            if not allowed:
                continue
            allowed_actions[entity.id] = allowed

        state_dict = self.state_formatter.build_prompt(
            intel=intel,
            allowed_actions=allowed_actions,
            config=self.prompt_config,
            turn_number=world.turn,
            missing_enemies=missing_enemies,
            casualties=self._casualties,
        )

        self.game_deps.current_state_dict = state_dict
        if step_info is not None:
            # Keep only the most recent step info for watchdog decisions.
            self.game_deps.step_info_list = [step_info]
        initial_strategy = self.game_deps.current_turn_number == 0
        actions, metadata = self._play_turn(state_dict, allowed_actions, initial_strategy, intel, world)
        self.game_deps.current_turn_number += 1



        # prompt_dict = self.prompt_formatter.build_prompt(
        #     intel=intel,
        #     allowed_actions=allowed_actions,
        #     config=self.prompt_config,
        #     turn_number=world.turn,
        #     missing_enemies=missing_enemies,
        # )
        #
        # commands = kwargs.get("commands") or ""
        # prompt_text = f"{commands}. Here is the game state:\n{json.dumps(prompt_dict, indent=2, ensure_ascii=False)}"
        #
        # res = player.run_sync(user_prompt=prompt_text)
        # llm_actions = res.output.entity_actions

        # actions, conversion_errors = self._convert_entity_actions(llm_actions, allowed_actions)
        # metadata = {"llm_prompt_dict": prompt_dict}
        # if conversion_errors: metadata["llm_action_errors"] = conversion_errors

        # Action formatting
        # Action responses? like collision warnings, invalid actions, etc.
        #

        return actions, metadata


    def _play_turn(
            self,
            state_dict: dict[str, Any],
            allowed_actions: Dict[int, list[Action]],
            initial_strategy: bool,
            intel: TeamIntel,
            world: WorldState,
    ):
        # # Analyse current state
        # last_step_info = self._format_last_step_info()
        # last_step_logs = ""
        # if last_step_info:
        #     last_step_logs = json.dumps(last_step_info, default=str, indent=2, ensure_ascii=False)

        user_prompt = ANALYST_USER_PROMPT_TEMPLATE.format(
            game_info=GAME_INFO,
            game_state_json=json.dumps(state_dict, default=str, indent=2, ensure_ascii=False),
            last_step_logs=None,
        )

        analyst_res = analyst_agent.run_sync(user_prompt=user_prompt)
        analysis: Optional[GameAnalysis] = getattr(analyst_res, "output", None)
        analysed_state_dict = self._merge_analysis(state_dict, analysis) if analysis else None
        self.game_deps.analysed_state_dict = analysed_state_dict

        state_for_watchdog = analysed_state_dict or state_dict
        callback_decision: Optional[CallbackAssessment] = None
        if self.game_deps.callback_conditions:
            callback_decision = self._run_watchdog(state_for_watchdog, intel, world)

        include_strategy = initial_strategy or (
            callback_decision is not None and callback_decision.needs_callback
        )
        self.game_deps.just_replanned = include_strategy and not initial_strategy

        # Optionally (first turn) create strategy
        strategy_output = None
        if include_strategy:
            prompt_sections = [
                "Use the current state below to issue a concise multi-phase plan. Respond back with final_result tool call with GamePlan schema only.",
            ]

            if callback_decision and callback_decision.needs_callback:
                prompt_sections.append("CALLBACK TRIGGERED: Replan based on the reason and recent changes.")
                if callback_decision.reason:
                    prompt_sections.append(f"CALLBACK REASON: {callback_decision.reason}")

            last_plan = self._format_last_plan()
            if last_plan:
                prompt_sections.append("LAST STRATEGY (for context):")
                prompt_sections.append(json.dumps(last_plan, indent=2, ensure_ascii=False))

            prompt_sections.append("CURRENT STATE:")
            prompt_sections.append(json.dumps(state_for_watchdog, indent=2, ensure_ascii=False))

            if callback_decision and callback_decision.needs_callback and callback_decision.reason:
                prompt_sections.append("FOCUS: Adjust the plan specifically to the callback trigger above.")

            strategist_res = strategist_agent.run_sync(
                user_prompt="\n".join(prompt_sections),
                deps=self.game_deps,
            )
            strategy_output = getattr(strategist_res, "output", None)
            if strategy_output:
                self.game_deps.multi_phase_strategy = self._safe_model_dump(strategy_output.multi_phase_plan)
                self.game_deps.current_phase_strategy = self._safe_model_dump(strategy_output.current_phase_plan)
                self.game_deps.entity_roles = {role.entity_id: role.role for role in strategy_output.roles}
                self.game_deps.callback_conditions = self._safe_model_dump(strategy_output.callbacks)
                self.game_deps.callback_conditions_set_turn = self.game_deps.current_turn_number

        # Execute based on strategy and analysed state
        exec_user_prompt = (
            "Use the current state JSON to pick one action per friendly unit. "
            "Follow the current phase guidance and assigned roles. "
            "Return the TeamAction schema only.\n"
            f"{json.dumps(analysed_state_dict or state_dict, indent=2, ensure_ascii=False)}"
        )
        executer_res = player.run_sync(user_prompt=exec_user_prompt, deps=self.game_deps)
        executor_output: Optional[Any] = getattr(executer_res, "output", None)

        llm_actions: List[LLMEntityAction] = executor_output.entity_actions if executor_output else []
        actions, conversion_errors = self._convert_entity_actions(llm_actions, allowed_actions)

        metadata: Dict[str, Any] = {
            "analyst_raw_response": self._safe_model_dump(analysis),
            "analysed_state_dict": analysed_state_dict,
            "strategy_raw_response": self._safe_model_dump(strategy_output),
            "executor_raw_response": self._safe_model_dump(executor_output),
            "executor_action_conversion_errors": conversion_errors if conversion_errors else None,
            "watchdog_raw_response": self._safe_model_dump(callback_decision),
            "strategy_callback_triggered": bool(callback_decision.needs_callback) if callback_decision else False,
        }
        return actions, metadata



    def _run_watchdog(
            self,
            state_dict: dict[str, Any],
            intel: TeamIntel,
            world: WorldState,
    ) -> Optional[CallbackAssessment]:
        """Run the watchdog agent to see if strategist callbacks should trigger."""
        conditions = self._safe_model_dump(self.game_deps.callback_conditions) or []
        last_step_summary = self._format_last_step_info(intel, world)
        prompt = (
            "Determine whether to trigger a strategist callback using the conditions and current state."
            f"\nCALLBACK CONDITIONS SET ON TURN: {self.game_deps.callback_conditions_set_turn}"
            f"\nCALLBACK CONDITIONS:\n{json.dumps(conditions, default=str, indent=2, ensure_ascii=False)}"
            f"\nCURRENT TURN: {self.game_deps.current_turn_number}"
            "\n\nLAST STEP (visible-only summary):\n"
            f"{json.dumps(last_step_summary, default=str, indent=2, ensure_ascii=False)}"
            "\n\nCURRENT STATE:\n"
            f"{json.dumps(state_dict, default=str, indent=2, ensure_ascii=False)}"
        )
        watchdog_res = watchdog_agent.run_sync(user_prompt=prompt, deps=self.game_deps)
        return getattr(watchdog_res, "output", None)

    def _format_last_plan(self) -> Optional[dict[str, Any]]:
        """Return the last strategist outputs we cached, if any."""
        if not (self.game_deps.multi_phase_strategy or self.game_deps.current_phase_strategy):
            return None
        return {
            "multi_phase_plan": self.game_deps.multi_phase_strategy,
            "current_phase_plan": self.game_deps.current_phase_strategy,
            "roles": self.game_deps.entity_roles,
            "callbacks": self.game_deps.callback_conditions,
        }


    def _format_last_step_info(self, intel: TeamIntel, world: WorldState) -> Optional[dict[str, Any]]:
        """
        Return a visibility-filtered summary of the most recent step_info, if present.
        Includes only our units and currently visible enemies to avoid information leaks.
        """
        if not self.game_deps.step_info_list:
            return None
        last_step = self.game_deps.step_info_list[-1]
        visible_ids = self._visible_entity_ids(intel)

        movement_summary = self._summarize_movement(last_step.movement, visible_ids, world)
        combat_summary = self._summarize_combat(last_step.combat, visible_ids, world)

        summary: dict[str, Any] = {}
        if movement_summary:
            summary["movement"] = movement_summary
        if combat_summary:
            summary["combat"] = combat_summary
        return summary or None

    def _visible_entity_ids(self, intel: TeamIntel) -> Set[int]:
        friendly_ids = {e.id for e in intel.friendlies}
        visible_enemy_ids = {e.id for e in intel.visible_enemies}
        return friendly_ids | visible_enemy_ids

    def _summarize_movement(
            self,
            movement: Any,
            visible_ids: Set[int],
            world: WorldState,
    ) -> Optional[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for result in getattr(movement, "movement_results", []) or []:
            entity_id = getattr(result, "entity_id", None)
            if entity_id is None or entity_id not in visible_ids:
                continue
            entity = world.get_entity(entity_id)
            label = entity.label() if entity else f"id={entity_id}"
            entry = {
                "entity": label,
                "entity_id": entity_id,
                "from": {"x": result.old_pos[0], "y": result.old_pos[1]},
                "to": {"x": result.new_pos[0], "y": result.new_pos[1]},
                "status": "MOVED" if result.success else "BLOCKED",
            }
            if result.failure_reason:
                entry["reason"] = result.failure_reason
            events.append(entry)

        return {"events": events} if events else None

    def _summarize_combat(
            self,
            combat: Any,
            visible_ids: Set[int],
            world: WorldState,
    ) -> Optional[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for result in getattr(combat, "combat_results", []) or []:
            attacker_id = getattr(result, "attacker_id", None)
            target_id = getattr(result, "target_id", None)
            if attacker_id not in visible_ids and (target_id not in visible_ids):
                continue

            attacker = world.get_entity(attacker_id) if attacker_id is not None else None
            target = world.get_entity(target_id) if target_id is not None else None
            outcome = "HIT" if getattr(result, "hit", False) else "MISS"
            events.append(
                {
                    "attacker": attacker.label() if attacker else f"id={attacker_id}",
                    "attacker_id": attacker_id,
                    "target": target.label() if target else f"id={target_id}",
                    "target_id": target_id,
                    "outcome": outcome if getattr(result, "success", False) else "FAILED",
                    "target_killed": bool(getattr(result, "target_killed", False)),
                }
            )

        visible_deaths = [
            eid for eid in getattr(combat, "killed_entity_ids", []) or [] if eid in visible_ids
        ]

        if not events and not visible_deaths:
            return None

        summary: dict[str, Any] = {}
        if events:
            summary["events"] = events
        if visible_deaths:
            summary["death_ids"] = visible_deaths
        return summary


    def _convert_entity_actions(
            self,
            entity_actions: list[LLMEntityAction],
            allowed_actions: Dict[int, list[Action]],
    ) -> tuple[Dict[int, Action], list[str]]:
        """Convert LLM entity actions into env Action mapping."""
        converted: Dict[int, Action] = {}
        errors: list[str] = []

        for entity_action in entity_actions:
            llm_action = entity_action.action
            entity_id = getattr(llm_action, "entity_id", None)
            if entity_id is None:
                errors.append("LLM action missing entity_id")
                continue

            env_action, err = self._convert_single_action(llm_action)
            if err:
                errors.append(err)
                continue

            allowed = allowed_actions.get(entity_id, [])
            if allowed and env_action not in allowed:
                errors.append(
                    f"Action {env_action} not in allowed set for entity {entity_id}; using fallback."
                )
                env_action = self._pick_fallback_action(allowed)

            converted[entity_id] = env_action

        # Ensure every controllable unit gets an action; default to WAIT/fallback if missing.
        for entity_id, allowed in allowed_actions.items():
            if entity_id not in converted and allowed:
                converted[entity_id] = self._pick_fallback_action(allowed)

        return converted, errors

    @staticmethod
    def _convert_single_action(
            llm_action: MoveAction | ShootAction | WaitAction | ToggleAction,
    ) -> tuple[Optional[Action], Optional[str]]:
        """Translate a single LLM action into an env Action."""
        try:
            if isinstance(llm_action, MoveAction):
                direction = MoveDir[llm_action.direction]
                return Action(ActionType.MOVE, {"dir": direction}), None
            if isinstance(llm_action, ShootAction):
                return Action(ActionType.SHOOT, {"target_id": llm_action.target_id}), None
            if isinstance(llm_action, WaitAction):
                return Action(ActionType.WAIT), None
            if isinstance(llm_action, ToggleAction):
                return Action(ActionType.TOGGLE, {"on": llm_action.on}), None
        except (KeyError, ValueError) as exc:
            return None, f"Failed to convert action {llm_action}: {exc}"

        return None, f"Unsupported action type from LLM: {type(llm_action).__name__}"

    @staticmethod
    def _pick_fallback_action(allowed: list[Action]) -> Action:
        """Choose a safe fallback action from allowed list, preferring WAIT."""
        wait_action = next((a for a in allowed if a.type == ActionType.WAIT), None)
        return wait_action or allowed[0]

    def _update_enemy_memory(self, intel: TeamIntel, turn: int) -> Set[int]:
        """
        Refresh last-seen data for visible enemies and return currently visible IDs.
        """
        visible_ids: Set[int] = set()
        for enemy in intel.visible_enemies:
            visible_ids.add(enemy.id)
            self._enemy_memory[enemy.id] = {
                "enemy_id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "last_seen_position": {"x": enemy.position[0], "y": enemy.position[1]},
                "last_seen_turn": turn,
            }
        return visible_ids

    def _prune_dead_enemies(self, world: WorldState) -> None:
        """
        Remove enemies from memory if the world shows they are dead or gone.
        Prevents dead enemies from being treated as "missing".
        """
        stale_ids = []
        for enemy_id in self._enemy_memory:
            entity = world.get_entity(enemy_id)
            if entity is None or not entity.alive:
                stale_ids.append(enemy_id)

        for enemy_id in stale_ids:
            self._enemy_memory.pop(enemy_id, None)

    def _collect_missing_enemies(self, visible_ids: Set[int], turn: int) -> List[Dict[str, Any]]:
        """
        Build a list of enemies that were seen before but are not currently visible.
        """
        missing: List[Dict[str, Any]] = []
        for enemy_id, entry in self._enemy_memory.items():
            if enemy_id in visible_ids:
                continue
            last_seen_turn = entry.get("last_seen_turn", turn)
            turns_since_seen = max(turn - last_seen_turn, 0)
            missing.append(
                {
                    **entry,
                    "turns_since_seen": turns_since_seen,
                }
            )
        return missing

    def _update_casualties(self, step_info: Optional["StepInfo"], world: WorldState) -> None:
        """
        Capture confirmed kills from the previous turn using StepInfo and store
        them with turn metadata for prompt formatting.
        """
        if step_info is None:
            return

        combat = getattr(step_info, "combat", None)
        killed_ids = getattr(combat, "killed_entity_ids", []) if combat is not None else []
        if not killed_ids:
            return

        killed_on_turn = max(world.turn - 1, 0)
        for entity_id in killed_ids:
            if entity_id in self._recorded_kill_ids:
                continue
            entity = world.get_entity(entity_id)
            if entity is None:
                continue
            entry = {
                "unit_id": entity.id,
                "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
                "type": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
                "last_position": {"x": entity.pos[0], "y": entity.pos[1]},
                "killed_on_turn": killed_on_turn,
            }
            if entity.team == self.team:
                self._casualties["friendly"].append(entry)
            else:
                self._casualties["enemy"].append(entry)
            self._recorded_kill_ids.add(entity_id)

    def _merge_analysis(self, state_dict: dict[str, Any], analysis: GameAnalysis) -> dict[str, Any]:
        """
        Overlay analyst insights onto the raw state dict while marking the source.
        """
        merged = deepcopy(state_dict)

        merged["situation"] = merged.get("situation", {})
        merged["situation"]["analyst_overlay"] = {
            "source": "analyst_agent",
            "spatial_status": analysis.spatial_status,
            "critical_alerts": analysis.critical_alerts,
            "opportunities": analysis.opportunities,
            "constraints": analysis.constraints,
            "situation_summary": analysis.situation_summary,
        }

        merged_units: list[dict[str, Any]] = merged.get("units", [])
        insights_by_unit = {ins.unit_id: ins for ins in analysis.unit_insights}
        merged["units"] = [
            self._merge_unit_analysis(unit, insights_by_unit.get(unit.get("unit_id"))) for unit in merged_units
        ]

        return merged

    def _merge_unit_analysis(
        self, unit: dict[str, Any], insight: Optional[Any]
    ) -> dict[str, Any]:
        if insight is None:
            return unit

        unit_overlay = {
            "source": "analyst_agent",
            "role": insight.role,
            "key_considerations": insight.key_considerations,
        }

        actions = unit.get("actions", [])
        merged_actions, unmatched = self._merge_action_implications(actions, insight.action_analysis)
        unit["actions"] = merged_actions
        if unmatched:
            unit_overlay["unmatched_action_analysis"] = unmatched

        unit["analyst_overlay"] = unit_overlay
        return unit

    def _merge_action_implications(
        self, actions: list[dict[str, Any]], action_analysis: list[ActionAnalysis]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        merged_actions: list[dict[str, Any]] = []
        matched_indices: set[int] = set()

        for action in actions:
            match_idx, implication = self._find_action_match(action, action_analysis, matched_indices)
            if implication is not None:
                action["implication_analysis"] = implication.implication
            merged_actions.append(action)

        unmatched = [
            self._safe_model_dump(action_analysis[idx])
            for idx in range(len(action_analysis))
            if idx not in matched_indices
        ]
        return merged_actions, unmatched

    def _find_action_match(
        self,
        action: dict[str, Any],
        analyses: list[ActionAnalysis],
        matched_indices: set[int],
    ) -> tuple[Optional[int], Optional[ActionAnalysis]]:
        action_sig = self._action_signature(action)

        for idx, analysis in enumerate(analyses):
            if idx in matched_indices:
                continue
            analysis_sig = self._analysis_action_signature(analysis.action)
            if self._action_signatures_match(action_sig, analysis_sig):
                matched_indices.add(idx)
                return idx, analysis
        return None, None

    @staticmethod
    def _action_signature(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": action.get("type"),
            "direction": str(action.get("direction")).upper() if action.get("direction") else None,
            "destination": action.get("destination"),
            "target": action.get("target_id"),
            "on": action.get("on"),
        }

    @staticmethod
    def _analysis_action_signature(action: Any) -> dict[str, Any]:
        destination = None
        if getattr(action, "destination", None):
            destination = {"x": action.destination.x, "y": action.destination.y}
        return {
            "type": action.type,
            "direction": action.direction,
            "destination": destination,
            "target": action.target,
            "on": action.on,
        }

    @staticmethod
    def _action_signatures_match(prompt_action: dict[str, Any], analysis_action: dict[str, Any]) -> bool:
        if prompt_action.get("type") != analysis_action.get("type"):
            return False
        for key in ("direction", "destination", "target", "on"):
            expected = analysis_action.get(key)
            if expected is None:
                continue
            if prompt_action.get(key) != expected:
                return False
        return True

    @staticmethod
    def _safe_model_dump(model_obj: Any) -> Any:
        if model_obj is None:
            return None
        if isinstance(model_obj, Enum):
            return model_obj.value if hasattr(model_obj, "value") else model_obj.name
        if isinstance(model_obj, dict):
            return {k: LLMAgent._safe_model_dump(v) for k, v in model_obj.items()}
        if isinstance(model_obj, (list, tuple, set)):
            return [LLMAgent._safe_model_dump(v) for v in model_obj]
        dump = getattr(model_obj, "model_dump", None)
        if callable(dump):
            return LLMAgent._safe_model_dump(dump())
        to_dict = getattr(model_obj, "dict", None)
        if callable(to_dict):
            return LLMAgent._safe_model_dump(to_dict())
        return model_obj
