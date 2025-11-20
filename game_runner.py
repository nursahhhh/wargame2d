"""
Game runner for the Grid Combat Environment.

This module provides the main game loop that:
1. Initializes the environment
2. Creates agents for each team
3. Runs the simulation until completion
4. Collects statistics and results
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from env import GridCombatEnv
from env.scenario import Scenario
from env.core.types import Team, GameResult
from agents import BaseAgent


@dataclass
class GameResult:
    """
    Complete results from a finished game.
    
    Contains all relevant statistics and outcome information.
    """
    winner: Optional[Team]
    game_result: str  # "BLUE_WINS", "RED_WINS", "DRAW"
    reason: str
    total_turns: int
    blue_final_count: int
    red_final_count: int
    blue_total_reward: float
    red_total_reward: float
    blue_entities_lost: int
    red_entities_lost: int


@dataclass
class TurnInfo:
    """Information about a single turn."""
    turn: int
    blue_entities_alive: int
    red_entities_alive: int
    blue_actions_count: int
    red_actions_count: int
    blue_reward: float
    red_reward: float


class GameRunner:
    """
    Orchestrates the game loop between environment and agents.
    
    This class handles:
    - Environment initialization
    - Agent coordination
    - Turn-by-turn execution
    - Statistics tracking
    - Logging and visualization
    
    Example usage:
        from env.scenario import create_mixed_scenario
        from agents import RandomAgent
        
        # Create agents
        blue_agent = RandomAgent(Team.BLUE, name="Blue Random")
        red_agent = RandomAgent(Team.RED, name="Red Random")
        
        # Create runner
        runner = GameRunner(
            blue_agent=blue_agent,
            red_agent=red_agent,
            verbose=True
        )
        
        # Run game
        scenario = create_mixed_scenario()
        result = runner.run_episode(scenario)
        
        print(f"Winner: {result.winner}")
        print(f"Total turns: {result.total_turns}")
    """
    
    def __init__(
        self,
        blue_agent: BaseAgent,
        red_agent: BaseAgent,
        verbose: bool = True,
        max_turns: Optional[int] = None
    ):
        """
        Initialize the game runner.
        
        Args:
            blue_agent: Agent controlling Blue team
            red_agent: Agent controlling Red team
            verbose: Print detailed turn-by-turn information
            max_turns: Maximum turns per episode (None = no limit,
                      will rely on environment victory conditions)
        """
        self.blue_agent = blue_agent
        self.red_agent = red_agent
        self.verbose = verbose
        self.max_turns = max_turns
        
        # Environment (created in run_episode)
        self.env: Optional[GridCombatEnv] = None
        
        # Statistics tracking
        self.turn_history: List[TurnInfo] = []
        self.cumulative_rewards: Dict[Team, float] = {Team.BLUE: 0.0, Team.RED: 0.0}
    
    def run_episode(self, scenario: Scenario) -> GameResult:
        """
        Run a complete game episode.
        
        Args:
            scenario: Scenario to run
        
        Returns:
            GameResult with complete statistics
        """
        # Initialize environment
        # Use JSON serialization to get fresh entity instances each time
        # This prevents entity mutation between episodes
        scenario_json = scenario.to_json_dict()
        fresh_scenario = Scenario.from_json_dict(scenario_json)
        scenario_dict = fresh_scenario.to_dict()
        
        self.env = GridCombatEnv(verbose=self.verbose)
        state = self.env.reset(scenario=scenario_dict)
        
        # Reset agents
        self.blue_agent.reset()
        self.red_agent.reset()
        
        # Reset statistics
        self.turn_history = []
        self.cumulative_rewards = {Team.BLUE: 0.0, Team.RED: 0.0}
        
        # Get initial entity counts
        world = state["world"]
        initial_blue_count = len(world.get_team_entities(Team.BLUE))
        initial_red_count = len(world.get_team_entities(Team.RED))
        
        if self.verbose:
            print("=" * 80)
            print(f"Starting Game: {self.blue_agent} vs {self.red_agent}")
            print(f"Grid: {world.grid.width}x{world.grid.height}")
            print(f"Initial entities: Blue={initial_blue_count}, Red={initial_red_count}")
            print("=" * 80)
        
        # Main game loop
        done = False
        turn = 0
        
        while not done:
            turn += 1
            
            # Check max turns
            if self.max_turns and turn > self.max_turns:
                if self.verbose:
                    print(f"\nMax turns ({self.max_turns}) reached. Ending game.")
                break
            
            # Get actions from both agents
            blue_actions = self.blue_agent.get_actions(state)
            red_actions = self.red_agent.get_actions(state)
            
            # Combine actions
            all_actions = {**blue_actions, **red_actions}
            
            # Step environment
            state, rewards, done, info = self.env.step(all_actions)
            
            # Update cumulative rewards
            self.cumulative_rewards[Team.BLUE] += rewards[Team.BLUE]
            self.cumulative_rewards[Team.RED] += rewards[Team.RED]
            
            # Track turn info
            world = state["world"]
            turn_info = TurnInfo(
                turn=turn,
                blue_entities_alive=len(world.get_team_entities(Team.BLUE)),
                red_entities_alive=len(world.get_team_entities(Team.RED)),
                blue_actions_count=len(blue_actions),
                red_actions_count=len(red_actions),
                blue_reward=rewards[Team.BLUE],
                red_reward=rewards[Team.RED]
            )
            self.turn_history.append(turn_info)
            
            # Log turn
            if self.verbose:
                self._log_turn(turn_info, state)
        
        # Build final result
        result = self._build_result(state, initial_blue_count, initial_red_count)
        
        if self.verbose:
            self._print_summary(result)
        
        return result
    
    def _log_turn(self, turn_info: TurnInfo, state: Dict) -> None:
        """
        Log information about the current turn.
        
        Args:
            turn_info: Turn statistics
            state: Current game state
        """
        world = state["world"]
        
        print(f"\n--- Turn {turn_info.turn} ---")
        print(f"  Alive: Blue={turn_info.blue_entities_alive}, "
              f"Red={turn_info.red_entities_alive}")
        print(f"  Actions: Blue={turn_info.blue_actions_count}, "
              f"Red={turn_info.red_actions_count}")
        
        if turn_info.blue_reward != 0 or turn_info.red_reward != 0:
            print(f"  Rewards: Blue={turn_info.blue_reward:+.1f}, "
                  f"Red={turn_info.red_reward:+.1f}")
        
        # Log stalemate tracking
        if world.turns_without_shooting > 0:
            print(f"  Turns without shooting: {world.turns_without_shooting}")
        if world.turns_without_movement > 0:
            print(f"  Turns without movement: {world.turns_without_movement}")
    
    def _build_result(
        self, 
        final_state: Dict, 
        initial_blue: int, 
        initial_red: int
    ) -> GameResult:
        """
        Build final game result from state.
        
        Args:
            final_state: Final game state
            initial_blue: Initial Blue entity count
            initial_red: Initial Red entity count
        
        Returns:
            Complete GameResult
        """
        world = final_state["world"]
        
        final_blue = len(world.get_team_entities(Team.BLUE))
        final_red = len(world.get_team_entities(Team.RED))
        
        # Determine game result string
        if world.winner == Team.BLUE:
            game_result_str = "BLUE_WINS"
        elif world.winner == Team.RED:
            game_result_str = "RED_WINS"
        else:
            game_result_str = "DRAW"
        
        return GameResult(
            winner=world.winner,
            game_result=game_result_str,
            reason=world.game_over_reason or "Unknown",
            total_turns=world.turn,
            blue_final_count=final_blue,
            red_final_count=final_red,
            blue_total_reward=self.cumulative_rewards[Team.BLUE],
            red_total_reward=self.cumulative_rewards[Team.RED],
            blue_entities_lost=initial_blue - final_blue,
            red_entities_lost=initial_red - final_red
        )
    
    def _print_summary(self, result: GameResult) -> None:
        """
        Print game summary.
        
        Args:
            result: Game result to summarize
        """
        print("\n" + "=" * 80)
        print("GAME OVER")
        print("=" * 80)
        print(f"Result: {result.game_result}")
        print(f"Winner: {result.winner.name if result.winner else 'DRAW'}")
        print(f"Reason: {result.reason}")
        print(f"Total turns: {result.total_turns}")
        print(f"\nFinal Status:")
        print(f"  Blue: {result.blue_final_count} alive "
              f"({result.blue_entities_lost} lost)")
        print(f"  Red:  {result.red_final_count} alive "
              f"({result.red_entities_lost} lost)")
        print(f"\nCumulative Rewards:")
        print(f"  Blue: {result.blue_total_reward:+.1f}")
        print(f"  Red:  {result.red_total_reward:+.1f}")
        print("=" * 80)


def run_single_game(
    scenario: Scenario,
    blue_agent: BaseAgent,
    red_agent: BaseAgent,
    verbose: bool = True,
    max_turns: Optional[int] = None
) -> GameResult:
    """
    Convenience function to run a single game.
    
    Args:
        scenario: Scenario to run
        blue_agent: Blue team agent
        red_agent: Red team agent
        verbose: Print detailed information
        max_turns: Maximum turns (None = no limit)
    
    Returns:
        GameResult with statistics
    
    Example:
        from env.scenario import create_mixed_scenario
        from agents import RandomAgent
        from env.core.types import Team
        
        result = run_single_game(
            scenario=create_mixed_scenario(),
            blue_agent=RandomAgent(Team.BLUE),
            red_agent=RandomAgent(Team.RED),
            verbose=True
        )
    """
    runner = GameRunner(
        blue_agent=blue_agent,
        red_agent=red_agent,
        verbose=verbose,
        max_turns=max_turns
    )
    return runner.run_episode(scenario)


def run_multiple_games(
    scenario: Scenario,
    blue_agent: BaseAgent,
    red_agent: BaseAgent,
    num_games: int = 10,
    verbose: bool = False
) -> List[GameResult]:
    """
    Run multiple games and collect results.
    
    Args:
        scenario: Scenario to run
        blue_agent: Blue team agent
        red_agent: Red team agent
        num_games: Number of games to run
        verbose: Print detailed information per game
    
    Returns:
        List of GameResults
    
    Example:
        results = run_multiple_games(
            scenario=create_mixed_scenario(),
            blue_agent=RandomAgent(Team.BLUE, seed=42),
            red_agent=RandomAgent(Team.RED, seed=43),
            num_games=100
        )
        
        blue_wins = sum(1 for r in results if r.winner == Team.BLUE)
        print(f"Blue win rate: {blue_wins / len(results) * 100:.1f}%")
    """
    runner = GameRunner(
        blue_agent=blue_agent,
        red_agent=red_agent,
        verbose=verbose
    )
    
    results = []
    for i in range(num_games):
        if not verbose:
            print(f"Running game {i+1}/{num_games}...", end="\r")
        result = runner.run_episode(scenario)
        results.append(result)
    
    if not verbose:
        print(f"\nCompleted {num_games} games.")
    
    return results

