"""
Test script for the new state structure.

Verifies:
1. State contains world object
2. State contains tracking counters
3. State contains config
4. Scenario system works
5. JSON save/load works
"""

from env import (
    GridCombatEnv, 
    Team, 
    Scenario,
    create_basic_battle,
    create_mixed_scenario
)
from env.entities import Aircraft, AWACS, SAM, Decoy
from env.core.actions import Action


def test_state_structure():
    """Test that state has the correct structure."""
    print("="*70)
    print("TEST 1: State Structure")
    print("="*70)
    
    env = GridCombatEnv()
    
    # Create scenario with explicit stats (config is in scenario now!)
    scenario = Scenario(grid_width=20, grid_height=20, seed=42)
    scenario.add_blue(Aircraft(
        team=Team.BLUE, pos=(2, 10),
        missiles=4, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    scenario.add_red(Aircraft(
        team=Team.RED, pos=(18, 10),
        missiles=4, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    
    # Reset with scenario
    state = env.reset(scenario=scenario.to_dict())
    
    # Verify state structure
    print(f"\n✓ State keys: {list(state.keys())}")
    assert "world" in state, "State missing 'world'"
    assert "tracking" in state, "State missing 'tracking'"
    assert "config" in state, "State missing 'config'"
    
    # Verify world
    world = state["world"]
    print(f"✓ World type: {type(world).__name__}")
    print(f"✓ Grid: {world.grid.width}x{world.grid.height}")
    print(f"✓ Entities: {len(world.get_alive_entities())}")
    
    # Verify tracking
    tracking = state["tracking"]
    print(f"✓ Tracking: {tracking}")
    assert "turns_without_shooting" in tracking
    assert "turns_without_movement" in tracking
    
    # Verify config
    config_data = state["config"]
    print(f"✓ Config: {config_data}")
    assert "max_stalemate_turns" in config_data
    assert "max_no_move_turns" in config_data
    assert "check_missile_exhaustion" in config_data
    
    print("\n✅ State structure test PASSED")


def test_world_access():
    """Test accessing world data."""
    print("\n" + "="*70)
    print("TEST 2: World Access")
    print("="*70)
    
    env = GridCombatEnv()
    scenario = create_basic_battle()  # Scenario already has seed=42
    state = env.reset(scenario=scenario.to_dict())
    
    # Access world
    world = state["world"]
    
    print(f"\n✓ Total entities: {len(world.get_all_entities())}")
    print(f"✓ Blue entities: {len(world.get_team_entities(Team.BLUE))}")
    print(f"✓ Red entities: {len(world.get_team_entities(Team.RED))}")
    
    # Test fog-of-war via team views
    blue_view = world.get_team_view(Team.BLUE)
    blue_obs = blue_view.get_all_observations()
    print(f"✓ Blue observations: {len(blue_obs)}")
    
    for obs in blue_obs:
        print(f"  - Entity #{obs.entity_id}: {obs.kind.value} at {obs.position}, "
              f"distance={obs.distance:.1f}")
    
    print("\n✅ World access test PASSED")


def test_step_state():
    """Test state after step."""
    print("\n" + "="*70)
    print("TEST 3: Step State")
    print("="*70)
    
    env = GridCombatEnv()
    scenario = create_basic_battle()  # Scenario already has seed=42
    state = env.reset(scenario=scenario.to_dict())
    
    # Get entity IDs for actions
    world = state["world"]
    all_entities = world.get_alive_entities()
    
    # All wait
    actions = {entity.id: Action.wait() for entity in all_entities}
    
    # Step
    state, rewards, done, info = env.step(actions)
    
    # Verify state after step
    print(f"\n✓ Turn: {state['world'].turn}")
    print(f"✓ Tracking: {state['tracking']}")
    print(f"✓ Rewards: {rewards}")
    print(f"✓ Done: {done}")
    print(f"✓ Info type: {type(info).__name__}")
    
    # Verify turn advanced
    assert state["world"].turn == 1, "Turn should be 1 after first step"
    
    # Verify tracking counters
    assert state["tracking"]["turns_without_shooting"] == 1
    assert state["tracking"]["turns_without_movement"] == 1
    
    print("\n✅ Step state test PASSED")


def test_scenario_system():
    """Test scenario creation and usage."""
    print("\n" + "="*70)
    print("TEST 4: Scenario System")
    print("="*70)
    
    # Test manual scenario creation
    scenario = Scenario()
    scenario.add_blue(Aircraft(
        team=Team.BLUE, pos=(2, 10),
        missiles=4, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    scenario.add_blue(AWACS(team=Team.BLUE, pos=(1, 10), radar_range=9.0))
    scenario.add_red(Aircraft(
        team=Team.RED, pos=(18, 10),
        missiles=4, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    scenario.add_red(SAM(
        team=Team.RED, pos=(18, 18),
        missiles=6, radar_range=8.0,
        missile_max_range=6.0,
        base_hit_prob=0.8, min_hit_prob=0.1,
        cooldown_steps=5, on=True
    ))
    
    print(f"✓ Manual scenario: {scenario}")
    
    # Test predefined scenarios
    basic = create_basic_battle()
    print(f"✓ Basic battle: {basic}")
    
    mixed = create_mixed_scenario()
    print(f"✓ Mixed scenario: {mixed}")
    
    # Test with environment
    env = GridCombatEnv()
    state = env.reset(scenario=basic.to_dict())
    world = state["world"]
    print(f"✓ Entities from scenario: {len(world.get_alive_entities())}")
    
    print("\n✅ Scenario system test PASSED")


def test_json_save_load():
    """Test JSON save and load."""
    print("\n" + "="*70)
    print("TEST 5: JSON Save/Load")
    print("="*70)
    
    # Create scenario
    scenario = Scenario()
    scenario.add_blue(Aircraft(
        team=Team.BLUE, pos=(2, 10),
        missiles=4, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    scenario.add_blue(AWACS(team=Team.BLUE, pos=(1, 10), radar_range=9.0))
    scenario.add_red(Aircraft(
        team=Team.RED, pos=(18, 10),
        missiles=2, radar_range=5.0,
        missile_max_range=4.0,
        base_hit_prob=0.8, min_hit_prob=0.1
    ))
    scenario.add_red(Decoy(team=Team.RED, pos=(17, 10)))
    
    # Save to JSON
    filepath = "test_scenario.json"
    scenario.save_json(filepath)
    print(f"✓ Saved to {filepath}")
    
    # Load from JSON
    loaded_scenario = Scenario.load_json(filepath)
    print(f"✓ Loaded from {filepath}")
    
    # Verify
    assert len(loaded_scenario.blue_entities) == 2
    assert len(loaded_scenario.red_entities) == 2
    print(f"✓ Blue entities: {len(loaded_scenario.blue_entities)}")
    print(f"✓ Red entities: {len(loaded_scenario.red_entities)}")
    
    # Use loaded scenario
    env = GridCombatEnv()
    state = env.reset(scenario=loaded_scenario.to_dict())
    world = state["world"]
    print(f"✓ Entities in world: {len(world.get_alive_entities())}")
    
    # Clean up
    import os
    os.remove(filepath)
    print(f"✓ Cleaned up {filepath}")
    
    print("\n✅ JSON save/load test PASSED")


def test_stalemate_tracking():
    """Test stalemate counter tracking."""
    print("\n" + "="*70)
    print("TEST 6: Stalemate Tracking")
    print("="*70)
    
    env = GridCombatEnv()
    scenario = create_basic_battle()
    # Override scenario config for shorter test
    scenario.max_no_move_turns = 5
    state = env.reset(scenario=scenario.to_dict())
    
    world = state["world"]
    all_entities = world.get_alive_entities()
    
    # Step multiple times with all wait
    for i in range(3):
        actions = {entity.id: Action.wait() for entity in all_entities}
        state, rewards, done, info = env.step(actions)
        
        tracking = state["tracking"]
        print(f"Turn {i+1}: {tracking}")
        assert tracking["turns_without_movement"] == i + 1
        assert tracking["turns_without_shooting"] == i + 1
    
    print("\n✅ Stalemate tracking test PASSED")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING NEW STATE STRUCTURE")
    print("="*70)
    
    try:
        test_state_structure()
        test_world_access()
        test_step_state()
        test_scenario_system()
        test_json_save_load()
        test_stalemate_tracking()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        
        print("\nSummary of changes:")
        print("1. ✓ State now returns world object directly")
        print("2. ✓ Tracking counters included in state")
        print("3. ✓ Config included in state")
        print("4. ✓ All logs removed")
        print("5. ✓ Scenario system with type-safe Python")
        print("6. ✓ JSON save/load support")
        print("7. ✓ reset() accepts scenario and returns full state")
        print("\nAgents now have complete information to make decisions!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

