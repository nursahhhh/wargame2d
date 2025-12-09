"""
Random agent implementation for testing and baseline comparison.

This agent makes random valid decisions for all its entities.
"""
import json
import random
from typing import Dict, Any, Optional, TYPE_CHECKING
from env.core.actions import Action
from env.core.types import Team, ActionType, MoveDir
from env.world import WorldState
from .agent_details import (
    player,
    MoveAction,
    ShootAction,
    WaitAction,
    ToggleAction,
    EntityAction as LLMEntityAction,
)
from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from .prompt_formatter import PromptFormatter, PromptConfig

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
            name: Agent name (default: "RandomAgent")
            seed: Random seed for reproducibility (None = random)
        """
        super().__init__(team, name)
        self.rng = random.Random(seed)
        self.prompt_formatter = PromptFormatter()
        self.prompt_config = PromptConfig()


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
        allowed_actions: Dict[int, list[Action]] = {}

        for entity in intel.friendlies:
            if not entity.alive:
                continue
            allowed = entity.get_allowed_actions(world)
            if not allowed:
                continue
            allowed_actions[entity.id] = allowed
            actions[entity.id] = self.rng.choice(allowed)

        prompt_dict = self.prompt_formatter.build_prompt(
            intel=intel,
            allowed_actions=allowed_actions,
            config=self.prompt_config,
            turn_number=world.turn,
        )

        commands = kwargs.get("commands") or ""
        prompt_text = f"{commands}. Here is the game state:\n{json.dumps(prompt_dict, indent=2, ensure_ascii=False)}"
        res = player.run_sync(user_prompt=prompt_text)
        llm_actions = res.output.entity_actions
        actions, conversion_errors = self._convert_entity_actions(llm_actions, allowed_actions)
        metadata = {"llm_prompt_dict": prompt_dict}
        if conversion_errors: metadata["llm_action_errors"] = conversion_errors

        # Action formatting
        # Action responses? like collision warnings, invalid actions, etc.
        #

        return actions, metadata

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
