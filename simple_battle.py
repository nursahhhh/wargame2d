"""
Simple battle scenario - entities close together for quick action.

This demonstrates a fast-paced game with entities that can quickly
engage each other.
"""

from env.scenario import Scenario
from env.entities import Aircraft
from env.core.types import Team
from agents import RandomAgent
from game_runner import run_single_game, run_multiple_games


def create_simple_battle() -> Scenario:
    """
    Create a simple 2v2 aircraft battle.
    
    Small grid, aircraft close together, will engage quickly.
    """
    scenario = Scenario(
        grid_width=12,
        grid_height=12,
        max_stalemate_turns=30,
        max_no_move_turns=10,
        seed=None  # Random each time
    )
    
    # Blue team - left side
    scenario.add_blue(Aircraft(
        team=Team.BLUE,
        pos=(2, 5),
        name="Blue-1",
        radar_range=6.0,
        missiles=3,
        missile_max_range=5.0,
        base_hit_prob=0.7,
        min_hit_prob=0.2
    ))
    
    scenario.add_blue(Aircraft(
        team=Team.BLUE,
        pos=(2, 7),
        name="Blue-2",
        radar_range=6.0,
        missiles=3,
        missile_max_range=5.0,
        base_hit_prob=0.7,
        min_hit_prob=0.2
    ))
    
    # Red team - right side
    scenario.add_red(Aircraft(
        team=Team.RED,
        pos=(10, 5),
        name="Red-1",
        radar_range=6.0,
        missiles=3,
        missile_max_range=5.0,
        base_hit_prob=0.7,
        min_hit_prob=0.2
    ))
    
    scenario.add_red(Aircraft(
        team=Team.RED,
        pos=(10, 7),
        name="Red-2",
        radar_range=6.0,
        missiles=3,
        missile_max_range=5.0,
        base_hit_prob=0.7,
        min_hit_prob=0.2
    ))
    
    return scenario


def main():
    """Run simple battle scenarios."""
    
    print("=" * 80)
    print("SIMPLE BATTLE - 2v2 Aircraft Engagement")
    print("=" * 80)
    
    # Create agents
    blue_agent = RandomAgent(
        team=Team.BLUE, 
        name="Blue Random",
        shoot_probability=0.6,  # More aggressive
        move_probability=0.4    # Less movement
    )
    red_agent = RandomAgent(
        team=Team.RED, 
        name="Red Random",
        shoot_probability=0.6,
        move_probability=0.4
    )
    
    # Single game with verbose output
    print("\n--- Single Game (Verbose) ---\n")
    scenario = create_simple_battle()
    result = run_single_game(
        scenario=scenario,
        blue_agent=blue_agent,
        red_agent=red_agent,
        verbose=True,
        max_turns=50
    )
    
    # Multiple games for statistics
    print("\n\n" + "=" * 80)
    print("Running 20 games for statistics...")
    print("=" * 80)
    
    results = run_multiple_games(
        scenario=create_simple_battle(),
        blue_agent=blue_agent,
        red_agent=red_agent,
        num_games=20,
        verbose=False
    )
    
    # Analyze results
    blue_wins = sum(1 for r in results if r.winner == Team.BLUE)
    red_wins = sum(1 for r in results if r.winner == Team.RED)
    draws = sum(1 for r in results if r.winner is None)
    
    avg_turns = sum(r.total_turns for r in results) / len(results)
    avg_blue_alive = sum(r.blue_final_count for r in results) / len(results)
    avg_red_alive = sum(r.red_final_count for r in results) / len(results)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (20 games)")
    print("=" * 80)
    print(f"\nWin Distribution:")
    print(f"  Blue wins:  {blue_wins:2d} ({blue_wins/len(results)*100:5.1f}%)")
    print(f"  Red wins:   {red_wins:2d} ({red_wins/len(results)*100:5.1f}%)")
    print(f"  Draws:      {draws:2d} ({draws/len(results)*100:5.1f}%)")
    
    print(f"\nGame Statistics:")
    print(f"  Average turns:        {avg_turns:.1f}")
    print(f"  Average Blue alive:   {avg_blue_alive:.1f}")
    print(f"  Average Red alive:    {avg_red_alive:.1f}")
    
    # Game outcome distribution
    print(f"\nGame Outcomes:")
    outcome_counts = {}
    for r in results:
        outcome_counts[r.reason] = outcome_counts.get(r.reason, 0) + 1
    
    for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {count}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

