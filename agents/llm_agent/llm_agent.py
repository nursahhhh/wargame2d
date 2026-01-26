import random
import json
from typing import Dict, Any, Optional, TYPE_CHECKING, List, Set

from pydantic_ai import AgentRunResult

from env.core.actions import Action
from env.core.types import ActionType, MoveDir, Team
from env.world import WorldState

from agents.base_agent import BaseAgent
from agents.team_intel import TeamIntel
from agents.registry import register_agent
from agents.llm_agent.helpers.state_formatter import StateFormatter
from agents.llm_agent.actors.game_deps import GameDeps
from agents.llm_agent.actors.strategist import strategist_agent, StrategyOutput
from agents.llm_agent.actors.analyst import analyst_agent, AnalystOutput
from agents.llm_agent.actors.executer import executer_agent, TeamTurnPlan

if TYPE_CHECKING:
    from env.environment import StepInfo


@register_agent("llm_basic")
class LLMAgent(BaseAgent):
    """
    Lightweight agent focused on producing a concise, human-readable state view.

    For now it issues safe fallback actions (WAIT when possible) while emitting
    the state in metadata so the LLM flow can be built incrementally.
    """

    def __init__(
        self,
        team: Team,
        name: str = None,
        seed: Optional[int] = None,
        **_: Any,
    ):
        super().__init__(team, name)
        self.rng = random.Random(seed)
        self.state_formatter = StateFormatter()
        self._enemy_memory: Dict[int, Dict[str, Any]] = {}
        self._casualties: Dict[str, List[Dict[str, Any]]] = {"friendly": [], "enemy": []}
        self._recorded_kill_ids: Set[int] = set()
        self.game_deps = GameDeps()
        self.game_deps.team_name = self.team.name

    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs: Any,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:
        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)
        actions: Dict[int, Action] = {}
        allowed_actions: Dict[int, List[Action]] = {}

        visible_enemy_ids = self._update_enemy_memory(intel, world.turn)
        missing_enemies = self._collect_missing_enemies(visible_enemy_ids, world.turn)
        self._update_casualties(step_info, world, intel)
        visible_step_log = self._distill_step_info(step_info, intel, world)
        if visible_step_log is not None:
            self.game_deps.visible_history[world.turn - 1] = visible_step_log

        for entity in intel.friendlies:
            if not entity.alive:
                continue
            allowed = entity.get_allowed_actions(world)
            if not allowed:
                continue
            allowed_actions[entity.id] = allowed
            actions[entity.id] = self._pick_fallback_action(allowed)

        state_dict = self.state_formatter.build_state(
            world=world,
            intel=intel,
            allowed_actions=allowed_actions,
            turn=world.turn,
            team=self.team,
            missing_enemies=missing_enemies,
            casualties=self._casualties,
        )
        state_text = self.state_formatter.build_state_string(
            world=world,
            intel=intel,
            allowed_actions=allowed_actions,
            turn=world.turn,
            team=self.team,
            missing_enemies=missing_enemies,
            casualties=self._casualties,
        )
        self.game_deps.current_turn_number = world.turn
        self.game_deps.current_state = state_text
        self.game_deps.current_state_dict = state_dict
        # Reset replan flag each turn; it will be set when strategist runs.
        self.game_deps.just_replanned = False

        # Pseudo-flow for multi-agent pipeline (Strategist -> Analyst -> Executor).
        # 1) Strategist: produce initial plan on first turn.
        strategy_error = self._maybe_get_initial_strategy()
        strategy_plan = self.game_deps.strategy_plan

        # 2) Analyst: assess current state + history, decide whether to re-strategize, produce notes.
        analyst_output, analyst_error = self._run_analyst(store=False)
        if analyst_output and not analyst_output.needs_replan:
            self._store_analyst_output(analyst_output)

        # 3) If analyst wants a replan and we have not just replanned, call strategist again.
        if analyst_output and analyst_output.needs_replan and not self.game_deps.just_replanned:
            strategy_plan, strategy_error, _ = self._restrategize(
                replan_reason=analyst_output.replan_reason,
                latest_analyst=analyst_output,
            )
            analyst_output, analyst_error = self._run_analyst(store=True)

        # Ensure the analyst output is recorded if it hasn't been stored yet (e.g., replan skipped).
        if (
            analyst_output
            and analyst_error is None
            and self.game_deps.current_turn_number not in self.game_deps.analyst_history
        ):
            self._store_analyst_output(analyst_output)

        # 4) Executor: call executor and map its actions.
        exec_result = self._run_executor(allowed_actions)
        actions.update(exec_result.get("actions", {}))


        metadata = {
            "state": state_dict,
            "state_text": state_text,
            "note": "State-only pass; actions are safe fallbacks.",
            "visible_step_log": visible_step_log,
            "visible_history": self.game_deps.visible_history,
            "strategy_plan": strategy_plan,
            "strategy_error": strategy_error,
            "executor_plan": exec_result.get("plan"),
            "executor_error": exec_result.get("error"),
            "analyst_notes": analyst_output.model_dump() if analyst_output else None,
            "analyst_error": analyst_error,
        }

        #print(state_text)
        print("\n" + "=" * 80 + "\n")
        return actions, metadata

    def _maybe_get_initial_strategy(self) -> Optional[str]:
        """
        Fetch the initial strategy on the first turn if not already cached.
        """
        if self.game_deps.strategy_plan is not None or self.game_deps.current_turn_number != 0:
            return None

        try:
            user_prompt = self._build_strategy_user_prompt(is_replan=False, replan_reason=None)
            result: AgentRunResult[StrategyOutput] = strategist_agent.run_sync(
                user_prompt=user_prompt, deps=self.game_deps
            )
            self.game_deps.strategy_plan = result.output
            self.game_deps.just_replanned = True
            self.game_deps.strategy_set_turn = self.game_deps.current_turn_number
            return None
        except Exception as exc:
            return str(exc)


    def _run_analyst(self, store: bool = True) -> tuple[Optional[AnalystOutput], Optional[str]]:
        """
        Run the analyst agent to summarize and decide on re-strategizing.
        """
        try:
            result: AgentRunResult[AnalystOutput] = analyst_agent.run_sync(
                user_prompt="Provide the analyst view for this turn.",
                deps=self.game_deps,
            )
            if store:
                self._store_analyst_output(result.output)
            return result.output, None
        except Exception as exc:
            return (
                None,
                str(exc),
            )

    def _store_analyst_output(self, output: AnalystOutput) -> None:
        """
        Persist analyst output for the current turn.
        """
        self.game_deps.analyst_history[self.game_deps.current_turn_number] = output



    def _restrategize(
        self,
        replan_reason: str = "",
        latest_analyst: Optional[AnalystOutput] = None,
    ) -> tuple[Optional[StrategyOutput], Optional[str], bool]:
        """
        Run the strategist to re-plan.
        """
        try:
            user_prompt = self._build_strategy_user_prompt(
                is_replan=True,
                replan_reason=replan_reason,
                latest_analyst=latest_analyst,
            )
            result: AgentRunResult[StrategyOutput] = strategist_agent.run_sync(
                user_prompt=user_prompt, deps=self.game_deps
            )
            self.game_deps.strategy_plan = result.output
            self.game_deps.just_replanned = True
            self.game_deps.strategy_set_turn = self.game_deps.current_turn_number
            return result.output, None, True
        except Exception as exc:
            return None, str(exc), False

    def _run_executor(
        self,
        allowed_actions: Dict[int, List[Action]],
    ) -> Dict[str, Any]:
        """
        Run the executor to propose actions and map them to env actions.
        """
        try:
            result: AgentRunResult[TeamTurnPlan] = executer_agent.run_sync(
                user_prompt="Analyse and propose the best actions for this turn.",
                deps=self.game_deps,
            )
            mapped = self._map_team_plan_to_actions(result.output, allowed_actions)
            return {"actions": mapped, "plan": result.output, "error": None}
        except Exception as exc:
            return {
                "actions": {},
                "plan": None,
                "error": str(exc),
            }

    def _build_strategy_user_prompt(
        self,
        is_replan: bool,
        replan_reason: Optional[str],
        latest_analyst: Optional[AnalystOutput] = None,
    ) -> str:
        """
        Build the strategist user prompt for initial or re-plan calls.
        """
        deps = self.game_deps
        current_state = deps.current_state or "No current state available."

        if not is_replan:
            return (
                "Analyse the game state carefully and come up with winning strategy for the team.\n"
                f"{current_state}\n"
            )

        last_strategy_turn = deps.strategy_set_turn
        last_strategy_text = (
            deps.strategy_plan.to_text(include_analysis=False, include_callbacks=True)
            if deps.strategy_plan
            else "No previous strategy recorded."
        )
        callback_text = (
            "\n".join(f"- {c}" for c in deps.strategy_plan.call_me_back_if)
            if deps.strategy_plan and deps.strategy_plan.call_me_back_if
            else "None provided."
        )

        since_last_notes = self._summarize_key_facts_since_last_strategy()
        recent_events = self._summarize_events_since_last_strategy()
        last_analysis = self._latest_analysis(latest_analyst=latest_analyst)

        reason = replan_reason.strip() if replan_reason else "Not specified."
        last_turn_label = last_strategy_turn if last_strategy_turn is not None else "Unknown"

        return (
            f"# PURPOSE\n"
            f"Re-strategize because: {reason}\n"
            f"Last strategized on turn: {last_turn_label}\n"
            "\n"
            "# LAST STRATEGY\n"
            f"{last_strategy_text}\n"
            "\n"
            "# WHAT HAPPENED SINCE LAST STRATEGY\n"
            f"{since_last_notes}\n"
            "# LATEST ANALYSIS\n"
            f"{last_analysis}\n"
            "\n"
            "# CURRENT STATE\n"
            f"{current_state}\n"
            "\n"
        )

    def _summarize_key_facts_since_last_strategy(self) -> str:
        """
        Collect analyst key facts from turns after the last strategy was produced.
        """
        history = self.game_deps.analyst_history or {}
        if not history:
            return "- No analyst key facts recorded."

        start_turn = (self.game_deps.strategy_set_turn or -1) + 1
        turns = [t for t in history.keys() if t >= start_turn and t < self.game_deps.current_turn_number]
        if not turns:
            return "- No new key facts since last strategy."

        lines: List[str] = []
        for turn in sorted(turns):
            entry = history[turn]
            if not entry.key_facts:
                continue
            lines.append(f"- Turn {turn}:")
            for fact in entry.key_facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines) if lines else "- No new key facts since last strategy."

    def _latest_analysis(self, latest_analyst: Optional[AnalystOutput]) -> str:
        """
        Get the most recent analysis (prefer provided latest_analyst, otherwise from history).
        """
        if latest_analyst:
            return latest_analyst.analysis

        history = self.game_deps.analyst_history or {}
        if not history:
            return "None recorded."

        last_turn = max(history.keys())
        return f"Turn {last_turn}:\n{history[last_turn].analysis}"

    def _summarize_events_since_last_strategy(self) -> str:
        """
        Summarize observed step logs since the last strategy turn.
        """
        history = self.game_deps.visible_history or {}
        if not history:
            return "- No observed events recorded."

        start_turn = (self.game_deps.strategy_set_turn or -1) + 1
        turns = [t for t in history.keys() if t >= start_turn and t < self.game_deps.current_turn_number]
        if not turns:
            return "- No observed events since last strategy."

        lines: List[str] = []
        for turn in sorted(turns):
            try:
                summary = json.dumps(history[turn], ensure_ascii=False, indent=2)
            except Exception:
                summary = str(history[turn])
            lines.append(f"- Turn {turn}:\n{summary}")
        return "\n".join(lines)

    def _map_team_plan_to_actions(
        self,
        plan: TeamTurnPlan,
        allowed_actions: Dict[int, List[Action]],
    ) -> Dict[int, Action]:
        """
        Map TeamTurnPlan into environment Action objects, falling back to WAIT if invalid.
        """
        mapped: Dict[int, Action] = {}
        for entity_action in plan.entity_actions:
            entity_id = getattr(entity_action.action, "entity_id", None)
            if entity_id is None:
                continue
            allowed = allowed_actions.get(entity_id, [])
            env_action = self._first_matching_action(entity_action.action, allowed)
            if env_action:
                mapped[entity_id] = env_action
            elif allowed:
                mapped[entity_id] = self._pick_fallback_action(allowed)
        return mapped

    @staticmethod
    def _first_matching_action(llm_action: Any, allowed: List[Action]) -> Optional[Action]:
        """
        Try to find an environment action matching the LLM-described action.
        """
        for candidate in allowed:
            if llm_action.type != candidate.type.value and llm_action.type != candidate.type.name:
                continue
            if candidate.type == ActionType.WAIT:
                return candidate
            if candidate.type == ActionType.MOVE and hasattr(llm_action, "direction"):
                dir_param = candidate.params.get("dir")
                desired = llm_action.direction
                if isinstance(desired, str):
                    try:
                        desired_dir = MoveDir[desired]
                    except KeyError:
                        desired_dir = None
                else:
                    desired_dir = desired
                if desired_dir and dir_param == desired_dir:
                    return candidate
            if candidate.type == ActionType.SHOOT and hasattr(llm_action, "target_id"):
                if llm_action.target_id == candidate.params.get("target_id"):
                    return candidate
            if candidate.type == ActionType.TOGGLE and hasattr(llm_action, "on"):
                if bool(llm_action.on) == candidate.params.get("on"):
                    return candidate
        return None

    def _update_enemy_memory(self, intel: TeamIntel, turn: int) -> Set[int]:
        visible_ids: Set[int] = set()
        for enemy in intel.visible_enemies:
            visible_ids.add(enemy.id)
            self._enemy_memory[enemy.id] = {
                "id": enemy.id,
                "team": enemy.team.name if hasattr(enemy.team, "name") else str(enemy.team),
                "type": enemy.kind.name if hasattr(enemy.kind, "name") else str(enemy.kind),
                "last_seen_position": {"x": enemy.position[0], "y": enemy.position[1]},
                "last_seen_turn": turn,
            }
        return visible_ids

    def _collect_missing_enemies(self, visible_ids: Set[int], turn: int) -> List[Dict[str, Any]]:
        missing: List[Dict[str, Any]] = []
        for enemy_id, entry in self._enemy_memory.items():
            if enemy_id in visible_ids:
                continue
            last_seen_turn = entry.get("last_seen_turn", turn)
            turns_since_seen = max(turn - last_seen_turn, 0)
            missing.append({**entry, "turns_since_seen": turns_since_seen})
        return missing

    def _update_casualties(self, step_info: Optional["StepInfo"], world: WorldState, intel: TeamIntel) -> None:
        """
        Record deaths with killer info and death location/turn.
        """
        if step_info is None:
            return

        combat = getattr(step_info, "combat", None)
        if combat is None:
            return

        killed_ids = getattr(combat, "killed_entity_ids", []) or []
        if not killed_ids:
            return

        killed_on_turn = max(world.turn - 1, 0)
        # Build lookup for killer -> target from combat_results
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

            killer_id = killers.get(entity_id)
            killer_is_friendly = killer_id in intel.friendly_ids if killer_id is not None else False
            enemy_was_visible = entity_id in intel.visible_enemy_ids
            is_friendly = entity.team == self.team

            # Only log enemy losses if we plausibly observed them (visible or killed by us).
            if not is_friendly and not enemy_was_visible and not killer_is_friendly:
                continue

            entry: Dict[str, Any] = {
                "id": entity.id,
                "team": entity.team.name if hasattr(entity.team, "name") else str(entity.team),
                "type": entity.kind.name if hasattr(entity.kind, "name") else str(entity.kind),
                "death_position": {"x": entity.pos[0], "y": entity.pos[1]},
                "killed_on_turn": killed_on_turn,
            }
            if killer_id is not None:
                killer_ent = world.get_entity(killer_id)
                if killer_ent:
                    entry["killed_by"] = {
                        "id": killer_ent.id,
                        "team": killer_ent.team.name if hasattr(killer_ent.team, "name") else str(killer_ent.team),
                        "type": killer_ent.kind.name if hasattr(killer_ent.kind, "name") else str(killer_ent.kind),
                    }
                else:
                    entry["killed_by"] = {"id": killer_id}

            if is_friendly:
                self._casualties["friendly"].append(entry)
            else:
                self._casualties["enemy"].append(entry)
                # Clear enemy from memory so it stops appearing as "missing".
                self._enemy_memory.pop(entity_id, None)

            self._recorded_kill_ids.add(entity_id)

    @staticmethod
    def _pick_fallback_action(allowed: List[Action]) -> Action:
        wait_action = next((a for a in allowed if a.type == ActionType.WAIT), None)
        if wait_action:
            return wait_action
        # Prefer a MOVE with minimal displacement for safety if no WAIT.
        move_action = next((a for a in allowed if a.type == ActionType.MOVE), None)
        if move_action:
            return move_action
        return allowed[0]

    def _distill_step_info(
        self,
        step_info: Optional["StepInfo"],
        intel: TeamIntel,
        world: WorldState,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a fog-safe log of last turn's visible movement and combat.
        """
        if step_info is None:
            return None

        movement_entries: List[Dict[str, Any]] = []
        for result in getattr(step_info.movement, "movement_results", []) or []:
            entity_id = result.entity_id
            friendly = intel.get_friendly(entity_id)
            enemy_visible = entity_id in intel.visible_enemy_ids
            if not friendly and not enemy_visible:
                continue
            mover = friendly or intel.get_enemy(entity_id)
            entry: Dict[str, Any] = {
                "entity_id": entity_id,
                "team": mover.team.name if mover and hasattr(mover, "team") else None,
                "type": mover.kind.name if mover and hasattr(mover, "kind") else None,
                "success": result.success,
                "from": {"x": result.old_pos[0], "y": result.old_pos[1]},
                "to": {"x": result.new_pos[0], "y": result.new_pos[1]},
                "direction": self._infer_direction(result.old_pos, result.new_pos),
                "failure_reason": result.failure_reason,
            }
            movement_entries.append(entry)

        combat_entries: List[Dict[str, Any]] = []
        for result in getattr(step_info.combat, "combat_results", []) or []:
            attacker_friend = intel.get_friendly(result.attacker_id)
            target_friend = intel.get_friendly(result.target_id) if result.target_id is not None else None
            attacker_visible_enemy = result.attacker_id in intel.visible_enemy_ids
            target_visible_enemy = (
                result.target_id in intel.visible_enemy_ids if result.target_id is not None else False
            )
            attacker_memory = self._enemy_memory.get(result.attacker_id)
            target_memory = self._enemy_memory.get(result.target_id) if result.target_id is not None else None

            # Only include if any party is friendly or currently visible enemy.
            if not (attacker_friend or target_friend or attacker_visible_enemy or target_visible_enemy):
                continue

            attacker_info: Dict[str, Any] = {}
            if attacker_friend or attacker_visible_enemy:
                attacker_ent = attacker_friend or intel.get_enemy(result.attacker_id)
                attacker_info = {
                    "id": result.attacker_id,
                    "team": attacker_ent.team.name if attacker_ent and hasattr(attacker_ent, "team") else None,
                    "type": attacker_ent.kind.name if attacker_ent and hasattr(attacker_ent, "kind") else None,
                }
            else:
                attacker_info = {
                    "id": result.attacker_id,
                    "team": (attacker_memory or {}).get("team"),
                    "type": (attacker_memory or {}).get("type") or "UNKNOWN",
                }

            target_info: Dict[str, Any] = {}
            if target_friend or target_visible_enemy:
                target_ent = target_friend or intel.get_enemy(result.target_id) if result.target_id is not None else None
                target_info = {
                    "id": result.target_id,
                    "team": target_ent.team.name if target_ent and hasattr(target_ent, "team") else None,
                    "type": target_ent.kind.name if target_ent and hasattr(target_ent, "kind") else None,
                }
            else:
                target_info = {
                    "id": result.target_id,
                    "team": (target_memory or {}).get("team"),
                    "type": (target_memory or {}).get("type") or "UNKNOWN",
                }
            combat_entries.append(
                {
                    "attacker": attacker_info,
                    "target": target_info,
                    "fired": bool(result.success),
                    "hit": result.hit if result.success else None,
                    "target_killed": result.target_killed if result.success else False,
                }
            )

        return {
            "turn": max(world.turn - 1, 0),
            "movement": movement_entries,
            "combat": combat_entries,
        }



    @staticmethod
    def _infer_direction(old_pos: tuple[int, int], new_pos: tuple[int, int]) -> Optional[str]:
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        if dx == 0 and dy == 0:
            return None
        for move_dir in [MoveDir.UP, MoveDir.DOWN, MoveDir.LEFT, MoveDir.RIGHT]:
            if (dx, dy) == move_dir.delta:
                return move_dir.name
        return f"dx={dx},dy={dy}"
