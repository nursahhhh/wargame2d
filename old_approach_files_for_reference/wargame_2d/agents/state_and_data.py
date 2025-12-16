from dataclasses import dataclass, field
from typing import Optional, List, Dict
from wargame_2d.env import World


@dataclass
class GameDeps:
    current_turn_number: int = 0
    restrategized: bool = False

    team_strategy: Optional[str] = None
    entity_strategies: List[Dict[int, str]] = field(default_factory=dict)
    re_strategize_when: Optional[str] = None

    # Always set before each run
    game_state: str = ""
    turn_summaries: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    max_history_turns: int = 10
    world: World = None


COMBAT_GAME_DEPS = GameDeps()