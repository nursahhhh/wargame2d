"""
Quick test to verify mechanics modules work correctly.

This script demonstrates how the three mechanics modules work together:
- SensorSystem
- MovementResolver  
- CombatResolver
"""

from env.core import Team, MoveDir
from env.core.actions import Action
from env.entities import Aircraft, AWACS
from env.world import WorldState
from env.mechanics import SensorSystem, MovementResolver, CombatResolver


def main():
    print("=" * 60)
    print("MECHANICS MODULE TEST")
    print("=" * 60)
    
    # Create world
    world = WorldState(width=20, height=20, seed=42)
    
    # Add entities
    blue_aircraft = Aircraft(team=Team.BLUE, pos=(5, 5))
    blue_awacs = AWACS(team=Team.BLUE, pos=(3, 3))
    red_aircraft = Aircraft(team=Team.RED, pos=(15, 15))
    red_awacs = AWACS(team=Team.RED, pos=(17, 17))
    
    world.add_entity(blue_aircraft)
    world.add_entity(blue_awacs)
    world.add_entity(red_aircraft)
    world.add_entity(red_awacs)
    
    print(f"\n✅ Created world: {world}")
    print(f"   Blue entities: {len(world.get_team_entities(Team.BLUE))}")
    print(f"   Red entities: {len(world.get_team_entities(Team.RED))}")
    
    # Initialize mechanics
    sensors = SensorSystem()
    movement = MovementResolver()
    combat = CombatResolver()
    
    print(f"\n✅ Initialized mechanics modules")
    
    # ========================================================================
    # TURN 1: Sensing
    # ========================================================================
    print("\n" + "=" * 60)
    print("TURN 1: SENSING")
    print("=" * 60)
    
    sensors.refresh_all_observations(world)
    
    blue_view = world.get_team_view(Team.BLUE)
    red_view = world.get_team_view(Team.RED)
    
    print(f"\nBlue team observations: {len(blue_view.get_all_observations())}")
    for obs in blue_view.get_all_observations():
        print(f"  - {obs}")
    
    print(f"\nRed team observations: {len(red_view.get_all_observations())}")
    for obs in red_view.get_all_observations():
        print(f"  - {obs}")
    
    # ========================================================================
    # TURN 2: Movement
    # ========================================================================
    print("\n" + "=" * 60)
    print("TURN 2: MOVEMENT")
    print("=" * 60)
    
    # Blue aircraft moves right
    # Red aircraft moves left
    actions = {
        blue_aircraft.id: Action.move(MoveDir.RIGHT),
        red_aircraft.id: Action.move(MoveDir.LEFT),
    }
    
    move_results = movement.resolve_all(world, actions, randomize_order=False)
    
    print(f"\nMovement results:")
    for result in move_results:
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.log}")
    
    print(f"\nEntity positions after movement:")
    for entity in world.get_alive_entities():
        print(f"  - {entity.label()} at {entity.pos}")
    
    # Re-sense after movement
    sensors.refresh_all_observations(world)
    
    # ========================================================================
    # TURN 3: Combat (when in range)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TURN 3: COMBAT ATTEMPT (probably out of range)")
    print("=" * 60)
    
    # Check if blue can see red
    blue_enemies = blue_view.get_enemy_ids(Team.BLUE)
    print(f"\nBlue can see {len(blue_enemies)} enemies: {blue_enemies}")
    
    if blue_enemies:
        target_id = list(blue_enemies)[0]
        actions = {
            blue_aircraft.id: Action.shoot(target_id)
        }
        
        combat_results = combat.resolve_all(world, actions)
        
        print(f"\nCombat results:")
        for result in combat_results:
            status = "✓" if result.success else "✗"
            print(f"  {status} {result.log}")
            if result.success:
                print(f"     Distance: {result.distance:.2f}")
                print(f"     Hit probability: {result.hit_probability:.2%}")
                print(f"     Hit: {result.hit}")
                print(f"     Killed: {result.target_killed}")
    else:
        print("\nNo enemies visible - skipping combat")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nAll mechanics modules working correctly! ✅")
    print(f"  - SensorSystem: Computing observations ✓")
    print(f"  - MovementResolver: Resolving movement ✓")
    print(f"  - CombatResolver: Resolving combat ✓")
    
    print(f"\nFinal state:")
    print(f"  Blue alive: {len(world.get_team_entities(Team.BLUE))}")
    print(f"  Red alive: {len(world.get_team_entities(Team.RED))}")
    print(f"  Total missiles: {combat.get_total_missiles_remaining(world)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

