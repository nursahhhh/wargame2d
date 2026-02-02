from __future__ import annotations

from typing import Any, Dict, Optional

from agents import BaseAgent, create_agent_from_spec
from env import GridCombatEnv
from env.core.types import Team
from env.environment import StepInfo
from env.mechanics.sensors import SensorSystem
from env.scenario import Scenario
from env.world import WorldState

from agents.memory_agent.episode_recorder import EpisodeRecorder
from agents.memory_agent.memory_store import MemoryStore

from infra.logger import get_logger
from .frame import Frame

log = get_logger(__name__)


class GameRunner:
    """
    Step-by-step game runner that returns UI-friendly frames.
    """

    def __init__(
        self,
        scenario: Scenario,
        world: WorldState | Dict[str, Any] | None = None,
        verbose: bool = False,
        episode_id: int = 0,
    ):
        self.scenario = scenario.clone()
        self.verbose = verbose

        self.env = GridCombatEnv(verbose=verbose)
        self._state = self.env.reset(scenario=self.scenario, world=world)

        self._blue_agent = self._agent_from_scenario(self.scenario, Team.BLUE)
        self._red_agent = self._agent_from_scenario(self.scenario, Team.RED)

        self._done = False
        self._last_info: StepInfo | None = None
        self._final_world: WorldState | None = None

        # 🧠 Memory components
        self.recorder = EpisodeRecorder(max_window=12)
        self.memory_store = MemoryStore(episode_id=episode_id)

        log.info("GameRunner initialized for scenario seed=%s", self.scenario.seed)

    # ------------------------------------------------------------------#
    # Core API
    # ------------------------------------------------------------------#
    def step(
        self,
        injections: Optional[Dict[str, Any]] = None,
    ) -> Frame:
        """
        Execute one turn, record episode data,
        flush memory on critical events,
        and return a UI-friendly frame.
        """

        if self._done:
            if self._final_world is not None:
                final_world = self._final_world
                self._final_world = None
                return Frame(world=final_world, done=True)
            raise RuntimeError("Game is already finished")

        injections = injections or {}

        # --------------------------------------------------
        # 1. Snapshot world BEFORE actions
        # --------------------------------------------------
        world_before = self._clone_world_with_observations(
            self._state["world"]
        )

        # --------------------------------------------------
        # 2. Get actions from agents
        # --------------------------------------------------
        blue_actions, blue_meta = self._blue_agent.get_actions(
            self._state,
            step_info=self._last_info,
            **injections.get("blue", {}),
        )

        red_actions, red_meta = self._red_agent.get_actions(
            self._state,
            step_info=self._last_info,
            **injections.get("red", {}),
        )

        merged_actions = {**blue_actions, **red_actions}

        # --------------------------------------------------
        # 3. Apply actions
        # --------------------------------------------------
        self._state, _rewards, self._done, self._last_info = self.env.step(
            merged_actions
        )

        # --------------------------------------------------
        # 4. Record step (RAM)
        # --------------------------------------------------
        self.recorder.record(
            step=self.turn,
            world_before=world_before,
            actions=merged_actions,
            action_metadata={
                "blue": blue_meta,
                "red": red_meta,
            },
            step_info=self._last_info,
        )




        # --------------------------------------------------
        # 5. Flush on critical events
        # --------------------------------------------------

## this part is problematic . the events must be defined . İt is not defined in stepinfo class fix this first of all 

        if self._last_info and self._last_info.events:
            for event in self._last_info.events:
                if getattr(event, "is_critical", False):
                    log.info("Critical event detected: %s", event)

                    self.memory_store.flush_segment(
                        trigger_event=event.type,
                        outcome=event.to_dict() if hasattr(event, "to_dict") else {},
                    )

        # --------------------------------------------------
        # 6. Handle terminal state
        # --------------------------------------------------
        if self._done:
            self._final_world = self._clone_world_with_observations(
                self._state["world"]
            )

            # Final flush (end of episode)
            self.memory_store.flush_segment(
                trigger_event="EPISODE_END",
                outcome={"result": "done"},
            )

        # --------------------------------------------------
        # 7. Return UI frame
        # --------------------------------------------------
        return Frame(
            world=world_before,
            actions=merged_actions,
            action_metadata={
                "blue": blue_meta,
                "red": red_meta,
            },
            step_info=self._last_info,
            done=self._done,
        )

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    @property
    def turn(self) -> int:
        world: WorldState = self._state["world"]
        return world.turn

    def _agent_from_scenario(self, scenario: Scenario, team: Team) -> BaseAgent:
        matches = [spec for spec in scenario.agents if spec.team == team]
        if not matches:
            raise ValueError(f"No AgentSpec found for team {team}")
        return create_agent_from_spec(matches[0])

    def _clone_world_with_observations(self, world: WorldState) -> WorldState:
        clone = world.clone()
        SensorSystem().refresh_all_observations(clone)
        return clone
