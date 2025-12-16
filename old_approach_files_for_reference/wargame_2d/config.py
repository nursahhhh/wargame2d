"""
Configuration file for Grid Air Combat
"""
from wargame_2d.controllers import ControllerType

# World Configuration
WORLD_CHOICE = 2  # 1 = empty world, 2 = demo scenario
WORLD_WIDTH = 20
WORLD_HEIGHT = 12
BASE_SEED = 7  # Starting seed for games

# Controller Configuration
# Options: ControllerType.HUMAN, ControllerType.RULE_BASED, ControllerType.LLM, ControllerType.RANDOM
BLUE_CONTROLLER = ControllerType.RULE_BASED
RED_CONTROLLER = ControllerType.LLM

# Controller Parameters (passed to the controller functions)
BLUE_CONTROLLER_PARAMS = {
    "min_shoot_prob": 0.30,
    "log_history_turns": 3  # Show last 3 turns
}

RED_CONTROLLER_PARAMS = {
    # "regenerate_code": False,  # Set to True to regenerate code each game
    # "save_code_path": "./generated_controllers/red_team_controller.py",  # Optional: save code
    # "additional_context": "",  # Optional: additional instructions

    "min_shoot_prob": 0.30,
    "log_history_turns": 3  # Show last 5 turns for more context
}

# Game Mode Configuration
AUTO_PLAY = True  # True = auto-play mode, False = manual control
AUTO_PLAY_DELAY = 0.5  # Delay between turns in seconds (0.1 to 5.0)

# Multiple Games Configuration
NUM_GAMES = 1  # Number of games to run (1 = single game, N = run N times)