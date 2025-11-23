"""
End-to-end rendering demo with random agents.

Running normally (``python test_random_agents_renderer.py``) will launch the
live viewer, pit two RandomAgents against each other, and stream turns to the
UI in real time. The unittest entrypoint runs a short headless smoke test to
ensure rendering keeps up with gameplay.
"""

import time
from agents.random_agent import RandomAgent
from env import GridCombatEnv, WebRenderer, create_mixed_scenario
from env.core import Team


if __name__ == "__main__":
    print("Starting live random-agent demo at http://localhost:5056 ...")
    env = GridCombatEnv(verbose=False)
    scenario = create_mixed_scenario()
    # Need to contain agent types and metadata inside scenario and create team agents accordingly?
    state = env.reset(scenario=scenario.to_dict())
    blue = RandomAgent(Team.BLUE, name="Blue RNG", seed=7)
    red = RandomAgent(Team.RED, name="Red RNG", seed=11)
    renderer = WebRenderer(port=5056, live=True, auto_open=True)

    done = False
    while not done:
        # Agents should also accepts commands?
        actions = {**blue.get_actions(state), **red.get_actions(state)}
        renderer.capture(state, actions) # We should also capture action metadata derived by agents?
        state, _, done, _ = env.step(actions)
        time.sleep(0.35)
        if done:
            break

    print("Finished streaming random-agent turns.")
