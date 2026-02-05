from __future__ import annotations

from typing import Any, Dict, Optional

from agents import BaseAgent, create_agent_from_spec
from env import GridCombatEnv
from env.core.types import Team
from env.environment import StepInfo
from env.mechanics.sensors import SensorSystem
from env.scenario import Scenario
from env.world import WorldState
from .events import extract_events
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
        # 5. Extract NEGATIVE events (semantic layer)
        # --------------------------------------------------
        current_world: WorldState = self._state["world"]

        events = extract_events(
            prev_world=world_before,
            world=current_world,
            team=Team.BLUE,
        )

        self.memory_store.record_step({
                    "step": self.turn,

                    "context": {
                        "threat_level": blue_meta.get("threat_level"),
                        "visible_enemies": red_meta.get("visible_enemies"),
                        "distance_to_objective": blue_meta.get("distance_to_objective"),
                    },

                    "decision": {
                        "actor": "blue",
                        "source": "llm",
                        "actions": blue_actions,
                    }
                })

        for event in events:
            log.warning("Negative event detected: %s", event)
            # Flush memory immediately for irreversible failures
            IRREVERSIBLE_SEVERITIES = {"HIGH", "CRITICAL"}

            if event.get("severity") in IRREVERSIBLE_SEVERITIES:
               
               
             
                self.memory_store.flush_segment(
                    trigger_event=event["type"],
                    outcome=event,
                )
                log.info("Memory flushed due to irreversible event")

                print("negatice events saved into memory file ")
       
     
        # --------------------------------------------------
        # 5. Handle terminal state
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
        #6. Return UI frame
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
