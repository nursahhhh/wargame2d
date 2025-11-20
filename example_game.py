"""
Example script demonstrating the game runner with random agents.

This shows how to:
1. Create a scenario
2. Initialize agents
3. Run a single game
4. Run multiple games
"""

from env.scenario import create_mixed_scenario
from env.core.types import Team
from agents import RandomAgent
from game_runner import run_single_game, run_multiple_games


def main():
    """Run example games."""
    
    print("Grid Combat Environment - Example Game Runner")
    print("=" * 80)
    
    # Create scenario
    scenario = create_mixed_scenario()
    print(f"Loaded scenario: {scenario}")
    print(f"  Blue entities: {len(scenario.blue_entities)}")
    print(f"  Red entities: {len(scenario.red_entities)}")
    print()
    
    # Create agents
    blue_agent = RandomAgent(
        team=Team.BLUE, 
        name="Blue Random", 
        seed=42,
        shoot_probability=0.4,
        move_probability=0.6
    )
    red_agent = RandomAgent(
        team=Team.RED, 
        name="Red Random", 
        seed=43,
        shoot_probability=0.4,
        move_probability=0.6
    )
    
    print(f"Blue Agent: {blue_agent}")
    print(f"Red Agent:  {red_agent}")
    print()
    
    # =========================================================================
    # Example 1: Run a single game with verbose output
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Game (Verbose)")
    print("=" * 80)
    
    result = run_single_game(
        scenario=scenario,
        blue_agent=blue_agent,
        red_agent=red_agent,
        verbose=True,
        max_turns=100  # Limit turns for demonstration
    )
    
    # =========================================================================
    # Example 2: Run multiple games and analyze statistics
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Games (10 episodes)")
    print("=" * 80)
    
    results = run_multiple_games(
        scenario=scenario,
        blue_agent=blue_agent,
        red_agent=red_agent,
        num_games=10,
        verbose=False  # Don't print each turn
    )
    
    # Analyze results
    blue_wins = sum(1 for r in results if r.winner == Team.BLUE)
    red_wins = sum(1 for r in results if r.winner == Team.RED)
    draws = sum(1 for r in results if r.winner is None)
    
    avg_turns = sum(r.total_turns for r in results) / len(results)
    avg_blue_reward = sum(r.blue_total_reward for r in results) / len(results)
    avg_red_reward = sum(r.red_total_reward for r in results) / len(results)
    
    print("\nStatistics across 10 games:")
    print(f"  Blue wins: {blue_wins} ({blue_wins/len(results)*100:.1f}%)")
    print(f"  Red wins:  {red_wins} ({red_wins/len(results)*100:.1f}%)")
    print(f"  Draws:     {draws} ({draws/len(results)*100:.1f}%)")
    print(f"\n  Average turns: {avg_turns:.1f}")
    print(f"  Average Blue reward: {avg_blue_reward:+.2f}")
    print(f"  Average Red reward:  {avg_red_reward:+.2f}")
    
    # Print individual results
    print("\nIndividual Game Results:")
    print(f"{'Game':<6} {'Winner':<10} {'Turns':<8} {'Blue Alive':<12} {'Red Alive':<12} {'Reason':<30}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        winner_str = r.winner.name if r.winner else "DRAW"
        print(f"{i:<6} {winner_str:<10} {r.total_turns:<8} "
              f"{r.blue_final_count:<12} {r.red_final_count:<12} "
              f"{r.reason[:30]:<30}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

