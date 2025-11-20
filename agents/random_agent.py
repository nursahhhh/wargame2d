"""
Random agent implementation for testing and baseline comparison.

This agent makes random valid decisions for all its entities.
"""

import random
from typing import Dict, Any, List, Optional
from env.core.actions import Action
from env.core.types import Team, MoveDir, ActionType
from env.entities.base import Entity
from env.entities.sam import SAM
from env.world import WorldState
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that takes random actions.
    
    Decision process:
    1. For each entity, randomly choose an action type
    2. If MOVE: pick random valid direction
    3. If SHOOT: pick random visible enemy
    4. If TOGGLE (SAM only): randomly toggle on/off
    5. Otherwise: WAIT
    
    This serves as a baseline for comparing learning agents.
    """
    
    def __init__(
        self, 
        team: Team, 
        name: str = None,
        shoot_probability: float = 0.3,
        move_probability: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize random agent.
        
        Args:
            team: Team to control
            name: Agent name (default: "RandomAgent")
            shoot_probability: Probability of attempting to shoot when possible
            move_probability: Probability of moving vs waiting
            seed: Random seed for reproducibility (None = random)
        """
        super().__init__(team, name)
        self.shoot_probability = shoot_probability
        self.move_probability = move_probability
        self.rng = random.Random(seed)
    
    def get_actions(self, state: Dict[str, Any]) -> Dict[int, Action]:
        """
        Generate random actions for all entities.
        
        Args:
            state: Current game state
        
        Returns:
            Dict of entity_id -> Action
        """
        world: WorldState = state["world"]
        actions = {}
        
        # Get all our entities
        my_entities = world.get_team_entities(self.team)
        
        # Get our team view (fog-of-war)
        team_view = world.get_team_view(self.team)
        
        # Get visible enemies
        enemy_ids = list(team_view.get_enemy_ids(self.team))
        
        for entity in my_entities:
            if not entity.alive:
                continue
            
            # Special handling for SAM
            if isinstance(entity, SAM):
                actions[entity.id] = self._get_sam_action(entity, enemy_ids)
            else:
                actions[entity.id] = self._get_entity_action(entity, enemy_ids)
        
        return actions
    
    def _get_entity_action(self, entity: Entity, enemy_ids: List[int]) -> Action:
        """
        Get random action for a regular entity (Aircraft, AWACS, Decoy).
        
        Args:
            entity: The entity to get action for
            enemy_ids: List of visible enemy IDs
        
        Returns:
            Random action
        """
        # Determine what actions are possible
        can_shoot = entity.can_shoot and len(enemy_ids) > 0 and entity.missiles > 0
        can_move = entity.can_move
        
        # Randomly decide what to do
        if can_shoot and self.rng.random() < self.shoot_probability:
            # Shoot at random enemy
            target_id = self.rng.choice(enemy_ids)
            return Action.shoot(target_id)
        
        elif can_move and self.rng.random() < self.move_probability:
            # Move in random direction
            direction = self.rng.choice(list(MoveDir))
            return Action.move(direction)
        
        else:
            # Wait
            return Action.wait()
    
    def _get_sam_action(self, sam: SAM, enemy_ids: List[int]) -> Action:
        """
        Get random action for a SAM.
        
        SAMs can:
        - Toggle radar on/off
        - Shoot (if radar is on and not in cooldown)
        - Wait
        
        Args:
            sam: The SAM entity
            enemy_ids: List of visible enemy IDs
        
        Returns:
            Random SAM action
        """
        # If radar is off, maybe turn it on
        if not sam.on:
            if self.rng.random() < 0.3:  # 30% chance to turn on
                return Action.toggle(on=True)
            else:
                return Action.wait()
        
        # Radar is on - maybe shoot or toggle off
        can_shoot = len(enemy_ids) > 0 and sam.missiles > 0 and sam._cooldown == 0
        
        if can_shoot and self.rng.random() < self.shoot_probability:
            target_id = self.rng.choice(enemy_ids)
            return Action.shoot(target_id)
        
        elif self.rng.random() < 0.1:  # 10% chance to toggle radar
            return Action.toggle(on=False)
        
        else:
            return Action.wait()
    
    def reset(self) -> None:
        """Reset agent state (reinitialize RNG if needed)."""
        # Random agent is stateless, so nothing to reset
        pass

