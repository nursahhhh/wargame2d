"""
Full environment test - demonstrates complete game simulation.

This script shows how to use the GridCombatEnv to run a complete game.
"""

from env import GridCombatEnv, Scenario
from env.entities import Aircraft, AWACS, SAM, Decoy
from env.core import Team, MoveDir
from env.core.actions import Action


def create_simple_scenario():
    """Create a simple test scenario."""
    # Create scenario with all config
    scenario = Scenario(
        grid_width=20,
        grid_height=20,
        seed=42,
        max_stalemate_turns=30,
        max_no_move_turns=10
    )
    
    # Add Blue team
    scenario.add_blue(Aircraft(team=Team.BLUE, pos=(5, 5), name="Blue-Fighter-1"))
    scenario.add_blue(Aircraft(team=Team.BLUE, pos=(6, 5), name="Blue-Fighter-2"))
    scenario.add_blue(AWACS(team=Team.BLUE, pos=(3, 3), name="Blue-AWACS"))
    
    # Add Red team
    scenario.add_red(Aircraft(team=Team.RED, pos=(15, 15), name="Red-Fighter"))
    scenario.add_red(AWACS(team=Team.RED, pos=(17, 17), name="Red-AWACS"))
    scenario.add_red(SAM(team=Team.RED, pos=(18, 10), name="Red-SAM"))
    
    # Create environment and reset with scenario
    env = GridCombatEnv(verbose=True)
    env.reset(scenario=scenario.to_dict())
    
    return env


def simple_ai(env: GridCombatEnv, team: Team) -> dict:
    """
    Simple AI that moves entities and shoots at visible enemies.
    
    Strategy:
    - Aircraft move toward center
    - If enemies in range, shoot at closest
    - SAMs toggle radar on and shoot
    """
    actions = {}
    
    team_entities = env.world.get_team_entities(team)
    team_view = env.world.get_team_view(team)
    
    for entity in team_entities:
        # Get visible enemies
        enemy_ids = team_view.get_enemy_ids(team)
        
        # Try to shoot if possible
        if entity.can_shoot and enemy_ids:
            # Get closest enemy
            closest_enemy_id = None
            closest_dist = float('inf')
            
            for enemy_id in enemy_ids:
                obs = team_view.get_observation(enemy_id)
                if obs and obs.distance < closest_dist:
                    closest_dist = obs.distance
                    closest_enemy_id = enemy_id
            
            if closest_enemy_id:
                # Check if in range
                from env.entities.aircraft import Aircraft
                from env.entities.sam import SAM
                
                if isinstance(entity, (Aircraft, SAM)):
                    if closest_dist <= entity.missile_max_range:
                        actions[entity.id] = Action.shoot(closest_enemy_id)
                        continue
        
        # SAM-specific behavior
        from env.entities.sam import SAM
        if isinstance(entity, SAM):
            # Turn radar on if off
            if not entity.on:
                actions[entity.id] = Action.toggle(on=True)
                continue
            # If on and can't shoot, wait
            actions[entity.id] = Action.wait()
            continue
        
        # Default: move toward center
        if entity.can_move:
            cx, cy = env.world.grid.width // 2, env.world.grid.height // 2
            dx = cx - entity.pos[0]
            dy = cy - entity.pos[1]
            
            # Move toward center
            if abs(dx) > abs(dy):
                direction = MoveDir.RIGHT if dx > 0 else MoveDir.LEFT
            else:
                direction = MoveDir.UP if dy > 0 else MoveDir.DOWN
            
            actions[entity.id] = Action.move(direction)
        else:
            actions[entity.id] = Action.wait()
    
    return actions


def main():
    print("="*70)
    print("GRID COMBAT ENVIRONMENT - FULL TEST")
    print("="*70)
    
    # Create scenario
    env = create_simple_scenario()
    
    print(f"\n✅ Environment created")
    print(f"   Grid: {env.world.grid}")
    print(f"   Blue entities: {len(env.world.get_team_entities(Team.BLUE))}")
    print(f"   Red entities: {len(env.world.get_team_entities(Team.RED))}")
    
    # Get initial observations
    obs, rewards, done, info = env.step({})
    
    # Run simulation
    max_turns = 50
    turn = 0
    
    print(f"\n{'='*70}")
    print("STARTING SIMULATION")
    print(f"{'='*70}")
    
    while not done and turn < max_turns:
        turn += 1
        
        # Get actions from simple AI
        blue_actions = simple_ai(env, Team.BLUE)
        red_actions = simple_ai(env, Team.RED)
        
        # Combine actions
        all_actions = {**blue_actions, **red_actions}
        
        # Execute turn
        obs, rewards, done, info = env.step(all_actions)
        
        # Note: verbose logging happens in step() when config.verbose=True
    
    # Print final results
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nFinal turn: {turn}")
    print(f"Game over: {done}")
    
    if info.game_result:
        print(f"\nResult: {info.game_result.result.name}")
        print(f"Reason: {info.game_result.reason}")
        if info.game_result.winner:
            print(f"Winner: {info.game_result.winner.name}")
        else:
            print(f"Winner: DRAW")
    
    print(f"\nFinal statistics:")
    print(f"  Blue alive: {info.entities_alive[Team.BLUE]}")
    print(f"  Red alive: {info.entities_alive[Team.RED]}")
    print(f"  Missiles remaining: {info.total_missiles_remaining}")
    print(f"  Kills this game: {len(info.kills)}")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

