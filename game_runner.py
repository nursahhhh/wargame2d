from typing import Any, Dict, Optional, List

from env import GridCombatEnv
from env.environment import StepInfo
from env.scenario import Scenario
from env.core.types import Team
from env.world import WorldState
from agents import AgentSpec, PreparedAgent, create_agent_from_spec


class GameRunner:
    """
    Barebones game runner: take a scenario (and optional world) and play until done.
    """

    def __init__(
        self,
        scenario: Scenario,
        world: WorldState | Dict[str, Any] | None = None,
        verbose: bool = True,
    ):
        self.scenario = scenario
        self.world = world
        self.verbose = verbose
        self.env = GridCombatEnv(verbose=verbose)

    def run_episode(self) -> Dict[str, Any]:
        scenario = self.scenario.clone()
        state = self.env.reset(scenario=scenario, world=self.world)

        blue_agent = self._agent_from_scenario(scenario, Team.BLUE)
        red_agent = self._agent_from_scenario(scenario, Team.RED)

        done = False
        last_info: StepInfo | None = None
        while not done:
            blue_actions, blue_action_metadata = blue_agent.agent.get_actions(
                state,
                step_info=last_info,
                **blue_agent.act_params,
            )
            red_actions, red_action_metadata = red_agent.agent.get_actions(
                state,
                step_info=last_info,
                **red_agent.act_params,
            )

            state, _rewards, done, last_info = self.env.step({**blue_actions, **red_actions})

        return state

    def _agent_from_scenario(self, scenario: Scenario, team: Team) -> PreparedAgent:
        if not scenario.agents:
            raise ValueError("Scenario is missing agent specs.")
        matches = [spec for spec in scenario.agents if spec.team == team]
        if not matches:
            raise ValueError(f"No AgentSpec found for team {team}")
        if len(matches) > 1:
            raise ValueError(f"Multiple AgentSpecs found for team {team}; expected exactly one.")
        return create_agent_from_spec(matches[0])
