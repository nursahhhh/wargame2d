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

def abort_episode(self):
    """
    Call this when UI closes the game manually.
    """

    if self._done:
        return

    early_outcome = self._evaluate_early_termination()

    if not early_outcome:
        early_outcome = {
            "type": "MANUAL_ABORT",
            "result": "INCOMPLETE",
            "reason": "USER_TERMINATED"
        }

    self._done = True

    self.memory_store.record_step({
        "step": self.turn,
        "event": "EPISODE_ABORTED",
        "end_of_episode": True,
        "result": early_outcome["result"],
        "reason": early_outcome["reason"],
    })

    self.memory_store.flush_segment(
        trigger_event=early_outcome["type"],
        outcome=early_outcome,
    )

    log.info("Episode manually aborted")

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
        self.recorder = EpisodeRecorder(max_window=20)
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
        
            self._final_world = self._clone_world_with_observations(
                self._state["world"]
            )

            self.memory_store.record_step({
                "step": self.turn,
                "event": "END_OF_EPISODE",
                "end_of_episode": True,
                "result": "ENV_TERMINAL"
            })

            self.memory_store.flush_segment(
                trigger_event="EPISODE_END",
                outcome={
                    "result": "ENV_TERMINAL",
                    "terminal": True,
                },
            )

            log.info("Environment terminal reached")

           

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

        blue_payload = blue_meta.get("prompt_payload", {})

        self.memory_store.record_step({
            "step": self.turn,
            "context": blue_payload,   # full snapshot
            "decision": {
                "actor": "blue",
                "source": "llm",
                "actions": blue_actions,
            }
        })


        for event in events:
            log.warning("Negative event detected: %s", event)
            if not self._done:
                early_outcome = self._evaluate_early_termination()

                if early_outcome:
                    self._done = True

                    self.memory_store.record_step({
                        "step": self.turn,
                        "event": "EARLY_TERMINATION",
                        "end_of_episode": True,
                        "result": early_outcome["result"],
                        "reason": early_outcome["reason"],
                    })

                    self.memory_store.flush_segment(
                        trigger_event=early_outcome["type"],
                        outcome=early_outcome,
                    )

                    log.info("Early termination triggered: %s", early_outcome)




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
    # --------------------------------------------------
    # Early termination evaluation
    # --------------------------------------------------
    def _evaluate_early_termination(self) -> dict | None:
        world: WorldState = self._state["world"]

        blue_units = [u for u in world._entities if u.team == Team.BLUE and u.alive]
        red_units = [u for u in world._entities if u.team == Team.RED and u.alive]
        
        blue_armed = [u for u in blue_units if getattr(u, "can_shoot", False)]
        red_armed = [u for u in red_units if getattr(u, "can_shoot", False)]
            # Case 1: both exhausted
        if not blue_armed and not red_armed:
            return {
                "type": "EARLY_TIE",
                "result": "TIE",
                "reason": "NO_ARMED_UNITS_REMAIN"
            }

        # Case 2: blue collapsed
        if not blue_armed:
            return {
                "type": "STRATEGIC_COLLAPSE",
                "result": "LOSS",
                "reason": "BLUE_NO_OFFENSIVE_CAPABILITY"
            }

        # Case 3: red collapsed
        if not red_armed:
            return {
                "type": "STRATEGIC_COLLAPSE",
                "result": "WIN",
                "reason": "RED_NO_OFFENSIVE_CAPABILITY"
            }

        return None
