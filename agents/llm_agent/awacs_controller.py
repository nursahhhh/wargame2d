from typing import List, Tuple
import heapq
import math
from .planner.astar import astar


class AWACSController:
    """Fully rule-based AWACS controller"""

    def __init__(self):
        self.state = "SEARCH"
        self.current_path: List[Tuple[int, int]] = []
        self.risk_threshold = 0.7
        self.last_summary = {
            "state":"SEARCH",
            "risk_level":0.0,
            "position":None,
            "under_threat":False
        }

    # --------------------------------------------------
    def decide(self, awacs_entity, world, team_intel):
        """
        returns (direction, reason_tag, note)
        """

        # --- Risk evaluation ---
        risk = self._compute_risk(awacs_entity, team_intel)

        # --- FSM transitions ---
        if risk > 0.7:
            self.state = "EVADE"
            self.current_path = []

        elif team_intel.visible_enemies:
            self.state = "HOLD"
            self.current_path = []

        else:
            self.state = "SEARCH"

        # --- STATE LOGIC ---
        if self.state == "HOLD":
            return None, "HOLD_POSITION", "Enemy detected, holding radar position"

        if self.state == "EVADE":
            target = self._furthest_safe_cell(
                awacs_entity.pos, world, team_intel
            )
            self.current_path = astar(
                start=tuple(awacs_entity.pos),
                goal=target,
                world=world,
            )

        if self.state == "SEARCH" and not self.current_path:
            target = self._furthest_unobserved_cell(
                awacs_entity.pos, world, team_intel
            )
            if target:
                self.current_path = astar(
                    start=tuple(awacs_entity.pos),
                    goal=target,
                    world=world,
                )

        # --- Execute movement ---
        if self.current_path:
            next_pos = self.current_path.pop(0)
            dx = next_pos[0] - awacs_entity.pos[0]
            dy = next_pos[1] - awacs_entity.pos[1]
            direction = self._delta_to_dir(dx, dy)
            return direction, self.state, f"AWACS {self.state} movement"

        self.last_summary ={
            "state":self.state,
            "risk_level":risk,
            "position":awacs_entity.pos,
            "under_threat": risk > self.risk_threshold
        }

        return None, "IDLE", "No movement required"

    # --------------------------------------------------
    def _compute_risk(self, awacs, intel) -> float:
        """Deterministic risk metric"""
        risk = 0.0
        for enemy in intel.visible_enemies:
            d = math.dist(awacs.pos, enemy.pos)
            risk += max(0, (10 - d) / 10)

        return min(risk, 1.0)

    # --------------------------------------------------
    def _furthest_unobserved_cell(self, pos, world, intel):
        best, best_d = None, -1
        for x in range(world.config.grid_width):
            for y in range(world.config.grid_height):
                if intel.is_cell_observed((x, y)):
                    continue
                d = math.dist(pos, (x, y))
                if d > best_d:
                    best_d = d
                    best = (x, y)
        return best

    # --------------------------------------------------
    def _furthest_safe_cell(self, pos, world, intel):
        best, best_score = None, -1
        for x in range(world.config.grid_width):
            for y in range(world.config.grid_height):
                cell = (x, y)
                score = 0
                for enemy in intel.visible_enemies:
                    score += math.dist(cell, enemy.pos)
                if score > best_score:
                    best_score = score
                    best = cell
        return best

    # --------------------------------------------------
    def _delta_to_dir(self, dx, dy):
        if dx == 1:
            return "RIGHT"
        if dx == -1:
            return "LEFT"
        if dy == 1:
            return "UP"
        if dy == -1:
            return "DOWN"
        return None


