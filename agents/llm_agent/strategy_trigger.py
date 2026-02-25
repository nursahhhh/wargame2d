from ..team_intel import TeamIntel

class StrategyTriggerEvaluator:
    """
    Centralized evaluator for determining whether the Commander
    should be re-invoked due to significant battlefield changes.
    """

    def __init__(self):
        self.last_enemy_grouping = None
        self.last_threat_level = None
        self.last_objective_score = None
        self.last_aggression = None
        self.last_strategy_step = -1
        self.known_enemy_ids = set()
        self.initialized = False

    # ==========================================================
    # PUBLIC ENTRY POINT
    # ==========================================================

    def should_call_commander(
        self,
        intel: TeamIntel,
        step: int,
        current_strategy: dict | None,
    ) -> bool:

        # First strategy ever
        if current_strategy is None:
            self._snapshot(intel, step)
            return True

        # Strategic event triggers
        if self._key_unit_destroyed(intel):
            return True

        if self._enemy_grouping_changed(intel):
            return True

        if self._threat_shifted(intel):
            return True

        if self._objective_feasibility_changed(intel):
            return True

        if self._new_enemy_detected(intel):
            return True

        # Safety periodic refresh (low frequency)
        if step - self.last_strategy_step >= 6:
            return True

        return False

    # ==========================================================
    # SNAPSHOT AFTER STRATEGY UPDATE
    # ==========================================================

    def update_snapshot(self, intel: TeamIntel, step: int):
        self._snapshot(intel, step)

    def _snapshot(self, intel: TeamIntel, step: int):
        self.last_enemy_grouping = intel.enemy_grouping_state()
        self.last_threat_level = intel.global_threat_level()
        self.last_objective_score = intel.objective_feasibility_score()
        self.last_aggression = intel.aggression_level(step)
        self.last_strategy_step = step
        self.known_enemy_ids = {e.id for e in intel.visible_enemies}
        self.initialized = True

    # ==========================================================
    # INDIVIDUAL TRIGGERS
    # ==========================================================

    def _key_unit_destroyed(self, intel: TeamIntel) -> bool:
        if intel.was_destroyed("AWACS"):
            return True
        if intel.was_destroyed("SAM"):
            return True
        if intel.friendly_loss_ratio() >= 0.4:
            return True
        return False

    def _enemy_grouping_changed(self, intel: TeamIntel) -> bool:
        current_grouping = intel.enemy_grouping_state()

        if self.last_enemy_grouping is None:
            return False

        if current_grouping != self.last_enemy_grouping:
            return True

        if intel.enemy_centroid_shift() > 3.0:
            return True

        return False

    def _threat_shifted(self, intel: TeamIntel) -> bool:
        current_threat = intel.global_threat_level()

        if self.last_threat_level is None:
            return False

        if abs(current_threat - self.last_threat_level) > 0.25:
            return True

        if intel.pressure_spike_near("AWACS"):
            return True

        return False

    def _objective_feasibility_changed(self, intel: TeamIntel) -> bool:
        current_score = intel.objective_feasibility_score()

        if self.last_objective_score is None:
            return False

        if abs(current_score - self.last_objective_score) > 0.3:
            return True

        return False

    def _new_enemy_detected(self, intel: TeamIntel) -> bool:
        current_enemy_ids = {e.id for e in intel.visible_enemies}

        # Brand new enemy detected
        new_ids = current_enemy_ids - self.known_enemy_ids
        if new_ids:
            return True

        # Flank detection: enemy far from known centroid
        for enemy in intel.visible_enemies:
            if intel.distance_to_enemy_centroid(enemy.position) > 4.0:
                return True

        return False