from pathlib import Path
import sys

# Add repository root to sys.path so tests can import local modules without installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import AgentSpec, create_agent_from_spec
from env.core.types import Team
from env.entities import AWACS, Aircraft, SAM, Decoy
from env.environment import GridCombatEnv
from env.scenario import Scenario
from env.world import WorldState
from paths import SCENARIO_STORAGE_DIR


def test_scenario_roundtrip_persist_and_load():
    scenario = Scenario(
        grid_width=12,
        grid_height=10,
        max_stalemate_turns=30,
        max_no_move_turns=20,
        max_turns=40,
        check_missile_exhaustion=False,
        seed=123,
        entities=[
            AWACS(team=Team.BLUE, pos=(0, 0), radar_range=8.0),
            Aircraft(
                team=Team.BLUE,
                pos=(2, 2),
                radar_range=5.0,
                missiles=3,
                missile_max_range=3.5,
                base_hit_prob=0.75,
                min_hit_prob=0.2,
            ),
            SAM(
                team=Team.BLUE,
                pos=(1, 3),
                radar_range=6.0,
                missiles=4,
                missile_max_range=5.0,
                base_hit_prob=0.8,
                min_hit_prob=0.1,
                cooldown_steps=4,
                on=True,
            ),
            Decoy(team=Team.RED, pos=(8, 6)),
            Aircraft(
                team=Team.RED,
                pos=(9, 4),
                radar_range=4.5,
                missiles=2,
                missile_max_range=3.0,
                base_hit_prob=0.7,
                min_hit_prob=0.15,
            ),
        ],
        agents=[
            AgentSpec(type="random", team=Team.BLUE, name="Blue Random", init_params={"seed": 1}),
            AgentSpec(type="random", team=Team.RED, name="Red Random", init_params={"seed": 2}),
        ],
    )

    path = SCENARIO_STORAGE_DIR / "test_scenario_roundtrip.json"
    try:
        scenario.save_json(path)
        loaded = Scenario.load_json(path)
    finally:
        ...
        #if path.exists():
        #    path.unlink()

    # Validate config fields
    assert loaded.grid_width == scenario.grid_width
    assert loaded.grid_height == scenario.grid_height
    assert loaded.max_stalemate_turns == scenario.max_stalemate_turns
    assert loaded.max_no_move_turns == scenario.max_no_move_turns
    assert loaded.max_turns == scenario.max_turns
    assert loaded.check_missile_exhaustion == scenario.check_missile_exhaustion
    assert loaded.seed == scenario.seed

    # Validate entities (order preserved)
    assert len(loaded.entities) == len(scenario.entities)
    for original, restored in zip(scenario.entities, loaded.entities):
        assert original.to_dict() == restored.to_dict()

    # Validate agents
    assert len(loaded.agents) == 2
    by_team = {spec.team: spec for spec in loaded.agents}
    expected = {spec.team: spec for spec in scenario.agents}
    assert set(by_team.keys()) == set(expected.keys())
    for team, spec in expected.items():
        assert by_team[team].to_dict() == spec.to_dict()

test_scenario_roundtrip_persist_and_load()


def test_environment_roundtrip_with_world_resume():
    scenario = Scenario(
        grid_width=14,
        grid_height=14,
        max_stalemate_turns=80,
        max_no_move_turns=80,
        max_turns=120,
        check_missile_exhaustion=True,
        seed=999,
        entities=[
            AWACS(team=Team.BLUE, pos=(1, 1), radar_range=7.0),
            Aircraft(
                team=Team.BLUE,
                pos=(2, 2),
                radar_range=5.0,
                missiles=2,
                missile_max_range=3.0,
                base_hit_prob=0.75,
                min_hit_prob=0.25,
            ),
            AWACS(team=Team.RED, pos=(12, 12), radar_range=7.0),
            Aircraft(
                team=Team.RED,
                pos=(11, 12),
                radar_range=5.0,
                missiles=2,
                missile_max_range=3.0,
                base_hit_prob=0.75,
                min_hit_prob=0.25,
            ),
        ],
        agents=[
            AgentSpec(type="random", team=Team.BLUE, name="Blue Random", init_params={"seed": 7}),
            AgentSpec(type="random", team=Team.RED, name="Red Random", init_params={"seed": 13}),
        ],
    )

    env = GridCombatEnv()
    prepared_agents = [create_agent_from_spec(spec) for spec in scenario.agents]  # type: ignore[arg-type]

    def collect_actions(state, agents, step_info=None):
        actions = {}
        for prepared in agents:
            team_actions, _ = prepared.agent.get_actions(
                state,
                step_info=step_info,
                **prepared.act_params,
            )
            actions.update(team_actions)
        return actions

    state = env.reset(scenario=scenario)

    last_info = None
    for _ in range(2):
        state, _rewards, done, last_info = env.step(collect_actions(state, prepared_agents, last_info))
        assert not done

    saved_world_dict = env.world.to_dict()  # type: ignore[union-attr]

    scenario_path = SCENARIO_STORAGE_DIR / "test_environment_roundtrip_scenario.json"
    world_path = SCENARIO_STORAGE_DIR / "test_environment_roundtrip_world.json"
    try:
        scenario.save_json(scenario_path)
        env.world.to_json(filepath=world_path)  # type: ignore[union-attr]

        loaded_scenario = Scenario.load_json(scenario_path)
        loaded_world = WorldState.from_json(filepath=world_path)

        reloaded_env = GridCombatEnv()
        reloaded_state = reloaded_env.reset(scenario=loaded_scenario, world=loaded_world)

        assert reloaded_env.world.to_dict() == saved_world_dict  # type: ignore[union-attr]

        expected_env = GridCombatEnv()
        expected_state = expected_env.reset(
            scenario=scenario.clone(),
            world=WorldState.from_dict(saved_world_dict),
        )
        reloaded_info = None
        expected_info = None

        rng_states = {prepared.agent.team: prepared.agent.rng.getstate() for prepared in prepared_agents}
        comparison_agents = []
        for spec in scenario.agents:  # type: ignore[union-attr]
            clone = create_agent_from_spec(spec)
            clone.agent.rng.setstate(rng_states[spec.team])
            comparison_agents.append(clone)

        for _ in range(2):
            reloaded_actions = collect_actions(reloaded_state, prepared_agents, reloaded_info)
            expected_actions = collect_actions(expected_state, comparison_agents, expected_info)
            assert reloaded_actions == expected_actions

            reloaded_state, _, reloaded_done, reloaded_info = reloaded_env.step(reloaded_actions)
            expected_state, _, expected_done, expected_info = expected_env.step(expected_actions)

            assert reloaded_env.world.to_dict() == expected_env.world.to_dict()  # type: ignore[union-attr]
            assert reloaded_done == expected_done
            if reloaded_done:
                break
    finally:
        if scenario_path.exists():
            scenario_path.unlink()
        if world_path.exists():
            world_path.unlink()


test_environment_roundtrip_with_world_resume()
