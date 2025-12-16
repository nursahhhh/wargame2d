from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import logfire

from wargame_2d.controllers import ControllerType

"""
Improved Pygame Frontend for Grid Air Combat
--------------------------------------------
FIXES:
1. Toggle action ('O') now cancels itself if pressed again (smart toggle)
2. Mutual targeting visualization improved with better separation and labels
3. AI shoot actions can now be cancelled with 'C' or overridden
NEW FEATURES:
4. Auto-play mode for AI vs AI simulation
5. Complete game recording to JSON for replay analysis
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from wargame_2d.ai_controller import get_ai_actions

import pygame

try:
    from wargame_2d.env import (
        World, Team, Entity, Aircraft, AWACS, SAM, Decoy,
        Action, ActionType, MoveDir, Shooter
    )
except Exception as e:
    print("ERROR: Could not import grid_air_combat_env.py")
    raise

# ----------------------------- Config --------------------------------------
CELL = 50
MARGIN = 1
PANEL_W = 420
FPS = 60

RADAR_ALPHA = 40
PREVIEW_ALPHA = 220
TARGET_ALPHA = 200
ICON_SIZE = 32

COLORS = {
    'bg': (20, 22, 28),
    'grid': (45, 50, 60),
    'panel': (28, 30, 36),
    'text': (235, 237, 240),
    'muted': (165, 170, 180),
    'blue': (90, 165, 255),
    'red': (255, 95, 95),
    'green': (125, 205, 125),
    'yellow': (245, 225, 125),
    'orange': (255, 160, 80),
    'radar_blue': (85, 155, 255),
    'radar_red': (255, 85, 85),
    'dead': (100, 100, 100),
    'button': (42, 45, 52),
    'button_hover': (62, 67, 78),
    'button_active': (85, 92, 105),
    'button_disabled': (35, 37, 42),
    'preview_move': (125, 205, 125),
    'preview_shoot': (255, 165, 125),
    'preview_wait': (205, 205, 205),
    'preview_toggle_on': (125, 220, 125),
    'preview_toggle_off': (220, 125, 125),
    'highlight': (255, 200, 100),
    'cooldown_bg': (60, 60, 60),
    'cooldown_fill': (255, 160, 80),
    'cooldown_ready': (125, 255, 125),
    'ai_action': (180, 140, 255),  # Purple tint for AI actions
}

ICON_MAP = {
    'aircraft': 'A',
    'awacs': 'W',
    'decoy': 'D',
    'sam': 'S',
}


# ----------------------------- Game Recorder -------------------------------
class GameRecorder:
    """Records complete game history for replay and analysis - with incremental saving"""

    def __init__(self, game_id: str = None, save_directory: str = "game_recordings"):
        self.game_id = game_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_directory = save_directory

        # Create directory if it doesn't exist
        Path(self.save_directory).mkdir(exist_ok=True)

        # File path for this game
        self.filepath = Path(self.save_directory) / f"game_{self.game_id}.json"

        self.recording = {
            "game_id": self.game_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {},
            "initial_state": {},
            "turns": [],
            "final_state": {},
            "result": {}
        }
        self.current_turn = 0

        # Track if game is complete
        self.game_complete = False

    def set_configuration(self, config: Dict[str, Any]):
        """Record game configuration and save immediately"""
        self.recording["configuration"] = config
        self._save_incremental()

    def set_initial_state(self, world: World):
        """Record initial game state and save immediately"""
        self.recording["initial_state"] = self._capture_world_state(world)
        self._save_incremental()

    def record_turn(self, world: World, actions: Dict[int, Action], logs: List[str]):
        """Record a complete turn and save immediately"""
        turn_data = {
            "turn_number": self.current_turn,
            "pre_action_state": self._capture_world_state(world),
            "actions": self._serialize_actions(actions),
            "logs": logs,
            "post_action_state": None  # Will be filled after step
        }
        self.recording["turns"].append(turn_data)
        self.current_turn += 1

        # Save after adding turn
        self._save_incremental()

        return len(self.recording["turns"]) - 1  # Return index for updating

    def update_turn_post_state(self, turn_index: int, world: World):
        """Update turn with post-action state and save"""
        if turn_index < len(self.recording["turns"]):
            self.recording["turns"][turn_index]["post_action_state"] = \
                self._capture_world_state(world)
            self._save_incremental()

    def record_llm_output(self, turn_index: int, team: str, llm_output: Dict[str, Any]):
        """Record LLM controller output for a turn and save"""
        if turn_index >= len(self.recording["turns"]):
            return

        turn_data = self.recording["turns"][turn_index]

        # Initialize llm_outputs dict if not exists
        if "llm_outputs" not in turn_data:
            turn_data["llm_outputs"] = {}

        # Store LLM output for this team
        turn_data["llm_outputs"][team.lower()] = {
            "timestamp": datetime.now().isoformat(),
            "output": llm_output
        }

        self._save_incremental()

    def update_turn_logs(self, turn_index: int, logs: List[str]):
        """Update turn with execution logs and save"""
        if turn_index < len(self.recording["turns"]):
            self.recording["turns"][turn_index]["logs"] = logs
            self._save_incremental()

    def set_final_state(self, world: World):
        """Record final game state and results, then save"""
        self.recording["final_state"] = self._capture_world_state(world)
        self.recording["result"] = world.get_game_over_summary()
        self.game_complete = True
        self._save_incremental()

    def _save_incremental(self):
        """Save current recording state to file (incremental updates)"""
        try:
            # Write to temporary file first (safer)
            temp_filepath = self.filepath.with_suffix('.tmp')

            with open(temp_filepath, 'w') as f:
                json.dump(self.recording, f, indent=2)

            # Atomic rename (replaces existing file)
            temp_filepath.replace(self.filepath)

        except Exception as e:
            print(f"[RECORDER] Warning: Failed to save recording: {e}")

    def save_to_file(self, directory: str = None):
        """
        Legacy method for compatibility - now just returns the filepath.
        Actual saving happens incrementally via _save_incremental().
        """
        if directory and directory != self.save_directory:
            # If different directory specified, copy file there
            new_dir = Path(directory)
            new_dir.mkdir(exist_ok=True)
            new_filepath = new_dir / f"game_{self.game_id}.json"

            import shutil
            shutil.copy2(self.filepath, new_filepath)

            print(f"\n[RECORDER] Game copied to: {new_filepath}")
            return new_filepath

        print(f"\n[RECORDER] Game recorded at: {self.filepath}")
        print(f"[RECORDER] Status: {'COMPLETE' if self.game_complete else 'IN PROGRESS'}")
        return self.filepath

    def _capture_world_state(self, world: World) -> Dict[str, Any]:
        """Capture complete world state"""
        return {
            "turn_number": self.current_turn,
            "dimensions": {
                "width": world.width,
                "height": world.height
            },
            "game_status": {
                "game_over": world.game_over,
                "winner": world.winner.name if world.winner else None,
                "reason": world.game_over_reason,
                "total_turns": world.total_turns,
                "turns_without_shooting": world.turns_without_shooting
            },
            "entities": [self._serialize_entity(e) for e in world.entities],
            "statistics": world.get_game_stats()
        }

    def _serialize_entity(self, entity: Entity) -> Dict[str, Any]:
        """Serialize entity to dict"""
        data = {
            "id": entity.id,
            "kind": entity.kind,
            "team": entity.team.name,
            "position": {"x": entity.pos[0], "y": entity.pos[1]},
            "alive": entity.alive,
            "radar_range": entity.radar_range,
            "can_move": entity.can_move,
            "can_shoot": entity.can_shoot
        }

        # Add shooter-specific data
        if isinstance(entity, Shooter):
            data["shooter"] = {
                "missiles": entity.missiles,
                "missile_max_range": entity.missile_max_range,
                "base_hit_prob": entity.base_hit_prob,
                "min_hit_prob": entity.min_hit_prob
            }

        # Add SAM-specific data
        if isinstance(entity, SAM):
            data["sam"] = {
                "on": entity.on,
                "cooldown": entity._cooldown,
                "cooldown_steps": entity.cooldown_steps
            }

        # Add last action
        if entity.last_action:
            data["last_action"] = {
                "type": entity.last_action.type.name,
                "params": self._serialize_action_params(entity.last_action.params)
            }

        return data

    def _serialize_actions(self, actions: Dict[int, Action]) -> Dict[str, Any]:
        """Serialize actions dict"""
        return {
            str(entity_id): {
                "type": action.type.name,
                "params": self._serialize_action_params(action.params)
            }
            for entity_id, action in actions.items()
        }

    def _serialize_action_params(self, params: Dict) -> Dict[str, Any]:
        """Serialize action parameters"""
        from enum import Enum
        serialized = {}
        for key, value in params.items():
            if isinstance(value, MoveDir):
                serialized[key] = value.name
            elif isinstance(value, Enum):
                serialized[key] = value.name
            else:
                serialized[key] = value
        return serialized


# ----------------------------- Icon Drawing --------------------------------
def draw_aircraft_icon(surf, center, size, color, team):
    """Draw a fighter jet icon"""
    pts = [
        (center[0], center[1] - size // 2),
        (center[0] - size // 6, center[1] + size // 3),
        (center[0] + size // 6, center[1] + size // 3),
    ]
    pygame.draw.polygon(surf, color, pts)
    wing_pts = [
        (center[0] - size // 2, center[1]),
        (center[0] + size // 2, center[1]),
    ]
    pygame.draw.line(surf, color, wing_pts[0], wing_pts[1], 3)
    pygame.draw.circle(surf, color, (center[0], center[1] - size // 6), 3)


def draw_awacs_icon(surf, center, size, color, team):
    """Draw an AWACS icon (aircraft with radar dome)"""
    pts = [
        (center[0], center[1] - size // 3),
        (center[0] - size // 5, center[1] + size // 3),
        (center[0] + size // 5, center[1] + size // 3),
    ]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.line(surf, color, (center[0] - size // 2, center[1]),
                     (center[0] + size // 2, center[1]), 4)
    pygame.draw.ellipse(surf, color, (center[0] - size // 3, center[1] - size // 2,
                                      size * 2 // 3, size // 4), 2)


def draw_sam_icon(surf, center, size, color, team, is_on=True, cooldown=0, max_cooldown=5):
    """Draw a SAM site icon with animated cooldown indicator"""
    base_y = center[1] + size // 3

    pygame.draw.rect(surf, color, (center[0] - size // 3, base_y - 3,
                                   size * 2 // 3, 6))
    pygame.draw.rect(surf, color, (center[0] - 4, base_y - size // 2, 8, size // 2))

    missile_color = color if is_on else (100, 100, 100)
    pygame.draw.line(surf, missile_color,
                     (center[0] - 6, base_y - size // 3),
                     (center[0] - 6, base_y - size * 2 // 3), 3)
    pygame.draw.line(surf, missile_color,
                     (center[0] + 6, base_y - size // 3),
                     (center[0] + 6, base_y - size * 2 // 3), 3)

    if is_on:
        pygame.draw.circle(surf, (125, 255, 125), (center[0], base_y + 8), 3)
    else:
        pygame.draw.circle(surf, (255, 125, 125), (center[0], base_y + 8), 3)

    if cooldown > 0 and max_cooldown > 0:
        cooldown_center = (center[0], center[1])
        cooldown_radius = size // 2 + 6

        pygame.draw.circle(surf, COLORS['cooldown_bg'], cooldown_center, cooldown_radius, 3)

        progress = 1.0 - (cooldown / max_cooldown)
        start_angle = -math.pi / 2
        end_angle = start_angle + (2 * math.pi * progress)

        if progress > 0:
            points = [cooldown_center]
            num_segments = max(3, int(36 * progress))
            for i in range(num_segments + 1):
                angle = start_angle + (end_angle - start_angle) * (i / num_segments)
                x = cooldown_center[0] + cooldown_radius * math.cos(angle)
                y = cooldown_center[1] + cooldown_radius * math.sin(angle)
                points.append((x, y))

            if len(points) > 2:
                temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                temp_center = (size, size)
                temp_points = [(p[0] - cooldown_center[0] + temp_center[0],
                                p[1] - cooldown_center[1] + temp_center[1]) for p in points]
                pygame.draw.polygon(temp_surf, (*COLORS['cooldown_fill'], 100), temp_points)
                surf.blit(temp_surf, (cooldown_center[0] - size, cooldown_center[1] - size))

            arc_points = points[1:]
            if len(arc_points) > 1:
                pygame.draw.lines(surf, COLORS['cooldown_fill'], False, arc_points, 4)

        font = pygame.font.SysFont('consolas', 14, bold=True)
        cd_text = font.render(str(cooldown), True, COLORS['cooldown_fill'])
        text_rect = cd_text.get_rect(center=cooldown_center)

        bg_rect = text_rect.inflate(6, 4)
        pygame.draw.rect(surf, (0, 0, 0), bg_rect, border_radius=3)
        pygame.draw.rect(surf, COLORS['cooldown_fill'], bg_rect, 2, border_radius=3)
        surf.blit(cd_text, text_rect)

    elif is_on and cooldown == 0:
        cooldown_center = (center[0], center[1])
        cooldown_radius = size // 2 + 6
        pygame.draw.circle(surf, COLORS['cooldown_ready'], cooldown_center, cooldown_radius, 2)


def draw_decoy_icon(surf, center, size, color, team):
    """Draw a decoy icon (balloon/drone)"""
    pygame.draw.circle(surf, color, center, size // 3, 2)
    pygame.draw.line(surf, color,
                     (center[0] - size // 6, center[1] - size // 6),
                     (center[0] + size // 6, center[1] + size // 6), 1)
    pygame.draw.line(surf, color,
                     (center[0] + size // 6, center[1] - size // 6),
                     (center[0] - size // 6, center[1] + size // 6), 1)
    for angle in [0, 90, 180, 270]:
        rad = math.radians(angle)
        x1 = center[0] + math.cos(rad) * size // 3
        y1 = center[1] + math.sin(rad) * size // 3
        x2 = center[0] + math.cos(rad) * size // 2
        y2 = center[1] + math.sin(rad) * size // 2
        pygame.draw.line(surf, color, (x1, y1), (x2, y2), 2)


def draw_entity_icon(surf, entity, center, size, selected=False):
    """Draw the appropriate icon for an entity"""
    color = COLORS['blue'] if entity.team == Team.BLUE else COLORS['red']
    if not entity.alive:
        color = COLORS['dead']

    if selected:
        pygame.draw.circle(surf, COLORS['yellow'], center, size // 2 + 4, 3)

    if entity.kind == 'aircraft':
        draw_aircraft_icon(surf, center, size, color, entity.team)
    elif entity.kind == 'awacs':
        draw_awacs_icon(surf, center, size, color, entity.team)
    elif entity.kind == 'sam':
        is_on = entity.on if isinstance(entity, SAM) else True
        cooldown = entity._cooldown if isinstance(entity, SAM) else 0
        max_cooldown = entity.cooldown_steps if isinstance(entity, SAM) else 5
        draw_sam_icon(surf, center, size, color, entity.team, is_on, cooldown, max_cooldown)
    elif entity.kind == 'decoy':
        draw_decoy_icon(surf, center, size, color, entity.team)

    if isinstance(entity, Shooter) and entity.alive:
        font = pygame.font.SysFont('consolas', 11, bold=True)
        text = font.render(f"{entity.missiles}", True, COLORS['text'])
        bg_rect = text.get_rect(center=(center[0], center[1] + size // 2 + 8))
        pygame.draw.circle(surf, (0, 0, 0), bg_rect.center, 8)
        surf.blit(text, bg_rect)


# ----------------------------- UI Helpers ----------------------------------
@dataclass
class Button:
    rect: pygame.Rect
    label: str
    key_hint: Optional[str] = None
    enabled: bool = True
    visible: bool = True

    def draw(self, surf, font, active=False):
        if not self.visible:
            return

        color = COLORS['button_disabled'] if not self.enabled else COLORS['button']
        if self.enabled and self.rect.collidepoint(pygame.mouse.get_pos()):
            color = COLORS['button_hover']
        if active:
            color = COLORS['button_active']

        pygame.draw.rect(surf, color, self.rect, border_radius=6)
        txt = self.label if not self.key_hint else f"{self.label} [{self.key_hint}]"
        text_color = COLORS['muted'] if not self.enabled else COLORS['text']
        img = font.render(txt, True, text_color)
        surf.blit(img, img.get_rect(center=self.rect.center))

    def is_clicked(self, pos) -> bool:
        return self.visible and self.enabled and self.rect.collidepoint(pos)


# ----------------------------- Game UI -------------------------------------
class GameUI:
    def __init__(self, world: World,
                 blue_controller: 'ControllerType' = None,
                 blue_controller_params: Dict[str, Any] = None,
                 red_controller: 'ControllerType' = None,
                 red_controller_params: Dict[str, Any] = None,
                 auto_play: bool = False,
                 auto_play_delay: float = 0.5):
        from wargame_2d.controllers import ControllerType
        self.world = world
        self.mode = 'setup'
        self.view = 'god'
        self.selected: Optional[int] = None
        self.pending_actions: Dict[int, Action] = {}
        self.ai_suggested_actions: Dict[int, Action] = {}
        self.shoot_mode = False
        self.setup_selected: Optional[int] = None

        # Controller configuration
        self.blue_controller = blue_controller or ControllerType.HUMAN
        self.blue_controller_params = blue_controller_params or {}
        self.red_controller = red_controller or ControllerType.HUMAN
        self.red_controller_params = red_controller_params or {}

        # Auto-play configuration
        self.auto_play = auto_play
        self.auto_play_delay = auto_play_delay
        self.last_auto_turn_time = 0

        # Game recorder
        self.recorder = GameRecorder()

        self.setup_team = Team.BLUE
        self.setup_kind = 'aircraft'
        self.setup_missiles = 2
        self.setup_radar = 5.0
        self.setup_max_range = 4.0

        pygame.init()
        pygame.display.set_caption('Grid Air Combat')

        self.cols = self.world.width
        self.rows = self.world.height

        self.fullscreen = False
        self._create_display(fullscreen=False)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('consolas', 16)
        self.font_small = pygame.font.SysFont('consolas', 13)
        self.big_font = pygame.font.SysFont('consolas', 22, bold=True)
        self.title_font = pygame.font.SysFont('consolas', 20, bold=True)

        self._build_buttons()
        self.world.refresh_observations()

    def _create_display(self, fullscreen: bool):
        if fullscreen:
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN)
        else:
            width = self.cols * CELL + PANEL_W
            height = self.rows * CELL
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

        self.width, self.height = self.screen.get_size()
        needed_w = self.cols * CELL + PANEL_W
        self.grid_origin_x = max(0, (self.width - needed_w) // 2)
        self.panel_origin_x = self.grid_origin_x + self.cols * CELL

    def _build_buttons(self):
        """Build all UI buttons with proper spacing"""
        left = self.panel_origin_x + 15
        w = PANEL_W - 30
        half = (w - 10) // 2

        # Setup buttons
        y = 180
        self.btn_setup_aircraft = Button(pygame.Rect(left, y, half, 34), "Aircraft [1]")
        self.btn_setup_awacs = Button(pygame.Rect(left + half + 10, y, half, 34), "AWACS [2]")
        y += 38
        self.btn_setup_sam = Button(pygame.Rect(left, y, half, 34), "SAM [3]")
        self.btn_setup_decoy = Button(pygame.Rect(left + half + 10, y, half, 34), "Decoy [4]")
        y += 48
        self.btn_toggle_team = Button(pygame.Rect(left, y, w, 34), "Toggle Team [T]")
        y += 38
        self.btn_missiles_minus = Button(pygame.Rect(left, y, 50, 34), "[-]")
        self.btn_missiles_plus = Button(pygame.Rect(left + 55, y, 50, 34), "[+]")
        y += 38
        self.btn_radar_minus = Button(pygame.Rect(left, y, 50, 34), "[R-]")
        self.btn_radar_plus = Button(pygame.Rect(left + 55, y, 50, 34), "[R+]")
        y += 38
        self.btn_range_minus = Button(pygame.Rect(left, y, 50, 34), "[M-]")
        self.btn_range_plus = Button(pygame.Rect(left + 55, y, 50, 34), "[M+]")
        y += 48
        self.btn_delete = Button(pygame.Rect(left, y, w, 34), "Delete Selected [Del]")
        y += 38
        self.btn_start = Button(pygame.Rect(left, y, w, 42), "START GAME [S]")

        # Play buttons
        self.btn_wait = Button(pygame.Rect(left, 200, w, 36), "WAIT", "Space")
        self.btn_move_up = Button(pygame.Rect(left + half - 15, 250, 70, 36), "↑", "Up")
        self.btn_move_left = Button(pygame.Rect(left, 290, 70, 36), "←", "Left")
        self.btn_move_right = Button(pygame.Rect(left + w - 70, 290, 70, 36), "→", "Right")
        self.btn_move_down = Button(pygame.Rect(left + half - 15, 330, 70, 36), "↓", "Down")
        self.btn_shoot = Button(pygame.Rect(left, 380, w, 40), "SHOOT", "F")
        self.btn_toggle = Button(pygame.Rect(left, 430, w, 36), "TOGGLE ON/OFF", "O")
        self.btn_clear = Button(pygame.Rect(left, 480, w, 36), "CLEAR ACTION", "C")
        self.btn_end = Button(pygame.Rect(left, 530, w, 46), "END TURN", "Enter")

        # View buttons
        self.btn_view_god = Button(pygame.Rect(left, 150, (w - 20) // 3, 32), "God [G]")
        self.btn_view_blue = Button(pygame.Rect(left + (w - 20) // 3 + 10, 150, (w - 20) // 3, 32), "Blue [B]")
        self.btn_view_red = Button(pygame.Rect(left + 2 * (w - 20) // 3 + 20, 150, (w - 20) // 3, 32), "Red [R]")

    def _grid_rect(self) -> pygame.Rect:
        return pygame.Rect(self.grid_origin_x, 0, self.cols * CELL, self.rows * CELL)

    def _to_screen(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        """Convert mathematical coordinates to screen coordinates"""
        x, y = cell
        # Convert mathematical Y (0=bottom) to screen Y (0=top)
        screen_y = (self.rows - 1 - y) * CELL + CELL // 2
        screen_x = self.grid_origin_x + x * CELL + CELL // 2
        return (screen_x, screen_y)

    def cell_at_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert screen click to mathematical coordinates"""
        mx, my = pos
        r = self._grid_rect()
        if not r.collidepoint(mx, my):
            return None
        cx = (mx - self.grid_origin_x) // CELL
        screen_cy = my // CELL
        # Convert screen Y to mathematical Y
        cy = self.rows - 1 - screen_cy
        if 0 <= cx < self.cols and 0 <= cy < self.rows:
            return (int(cx), int(cy))
        return None

    # ----------------------------- Drawing ---------------------------------
    def draw_grid(self):
        self.screen.fill(COLORS['bg'])

        for x in range(self.cols + 1):
            X = self.grid_origin_x + x * CELL
            pygame.draw.line(self.screen, COLORS['grid'], (X, 0), (X, self.rows * CELL), MARGIN)
        for y in range(self.rows + 1):
            Y = y * CELL
            pygame.draw.line(self.screen, COLORS['grid'],
                             (self.grid_origin_x, Y),
                             (self.grid_origin_x + self.cols * CELL, Y), MARGIN)

        # ADD THIS: Y-axis labels to show mathematical coordinates
        label_font = pygame.font.SysFont('consolas', 10)
        for screen_y in range(0, self.rows, 2):  # Every 2 rows
            math_y = self.rows - 1 - screen_y
            label = label_font.render(f"Y={math_y}", True, COLORS['muted'])
            self.screen.blit(label, (self.grid_origin_x - 35, screen_y * CELL + CELL // 2 - 5))

    def draw_radar(self, e: Entity):
        if not e.alive or e.radar_range <= 0:
            return

        if isinstance(e, SAM) and not e.on:
            return

        if self.view in ('blue', 'red'):
            team = Team.BLUE if self.view == 'blue' else Team.RED
            if e.team != team:
                return

        color = COLORS['radar_blue'] if e.team == Team.BLUE else COLORS['radar_red']
        sx, sy = self._to_screen(e.pos)
        radius = int(e.radar_range * CELL)

        surf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*color, RADAR_ALPHA),
                           (sx - self.grid_origin_x, sy), radius)
        self.screen.blit(surf, (self.grid_origin_x, 0))

    def draw_entity(self, e: Entity):
        viewing_team = None
        if self.view == 'blue':
            viewing_team = Team.BLUE
        elif self.view == 'red':
            viewing_team = Team.RED

        if viewing_team is not None:
            visible_ids = set(self.world.command_center(viewing_team).observations.keys())

            if e.team != viewing_team and e.id not in visible_ids and e.alive:
                return

            if e.team != viewing_team and e.alive:
                obs = self.world.command_center(viewing_team).observations.get(e.id)
                if obs:
                    cx, cy = self._to_screen(e.pos)
                    is_selected = (self.mode == 'play' and self.selected == e.id)
                    self._draw_observed_entity(obs.kind, e.team, e.alive, (cx, cy), is_selected, e)
                    return

        cx, cy = self._to_screen(e.pos)

        is_selected = False
        if self.mode == 'setup' and self.setup_selected == e.id:
            is_selected = True
        elif self.mode == 'play' and self.selected == e.id:
            is_selected = True

        draw_entity_icon(self.screen, e, (cx, cy), ICON_SIZE, is_selected)

    def _draw_observed_entity(self, observed_kind, team, alive, center, selected, entity):
        """Draw entity as it appears to enemy (decoys appear as aircraft)"""
        color = COLORS['blue'] if team == Team.BLUE else COLORS['red']
        if not alive:
            color = COLORS['dead']

        if selected:
            pygame.draw.circle(self.screen, COLORS['yellow'], center, ICON_SIZE // 2 + 4, 3)

        if observed_kind == 'aircraft':
            draw_aircraft_icon(self.screen, center, ICON_SIZE, color, team)
            font = pygame.font.SysFont('consolas', 11, bold=True)
            text = font.render("?", True, COLORS['muted'])
            bg_rect = text.get_rect(center=(center[0], center[1] + ICON_SIZE // 2 + 8))
            pygame.draw.circle(self.screen, (0, 0, 0), bg_rect.center, 8)
            self.screen.blit(text, bg_rect)
        elif observed_kind == 'awacs':
            draw_awacs_icon(self.screen, center, ICON_SIZE, color, team)
        elif observed_kind == 'sam':
            is_on = entity.on if isinstance(entity, SAM) else True
            draw_sam_icon(self.screen, center, ICON_SIZE, color, team, is_on)
        elif observed_kind == 'decoy':
            draw_decoy_icon(self.screen, center, ICON_SIZE, color, team)

    def draw_action_previews(self):
        """Draw visual previews of queued actions with enhanced indicators"""
        limit_team = None
        if self.view in ('blue', 'red'):
            limit_team = Team.BLUE if self.view == 'blue' else Team.RED

        # Combine AI suggestions with user actions (user overrides AI)
        all_actions = dict(self.ai_suggested_actions)
        all_actions.update(self.pending_actions)

        # Collect shoot actions
        shoot_actions = []
        for eid, act in all_actions.items():
            e = self.world.entities_by_id.get(eid)
            if not e or not e.alive:
                continue
            if limit_team is not None and e.team != limit_team:
                continue
            if act.type == ActionType.SHOOT:
                tgt_id = act.params.get('target_id')
                tgt = self.world.entities_by_id.get(tgt_id)
                if tgt:
                    is_ai = eid in self.ai_suggested_actions and eid not in self.pending_actions
                    shoot_actions.append((e, tgt, is_ai))

        # Draw shoot actions with mutual targeting visualization
        mutual_pairs = {}
        for idx, (shooter, target, is_ai) in enumerate(shoot_actions):
            reciprocal = None
            for s, t, ai in shoot_actions:
                if s.id == target.id and t.id == shooter.id:
                    reciprocal = (s, t)
                    break

            pair_key = tuple(sorted([shooter.id, target.id]))

            if reciprocal and pair_key not in mutual_pairs:
                mutual_pairs[pair_key] = True
                self._draw_mutual_shots(shooter, target, is_ai)
            elif not reciprocal:
                sx, sy = self._to_screen(shooter.pos)
                tx, ty = self._to_screen(target.pos)

                if isinstance(shooter, Shooter):
                    d = self.world.distance(shooter.pos, target.pos)
                    p = self.world.hit_probability(
                        distance=d,
                        max_range=shooter.missile_max_range,
                        base=shooter.base_hit_prob,
                        min_p=shooter.min_hit_prob
                    )
                else:
                    p = 0.0

                self._draw_shot_line((sx, sy), (tx, ty), p, False, 0, is_ai)

        # Draw other actions
        for eid, act in all_actions.items():
            e = self.world.entities_by_id.get(eid)
            if not e or not e.alive:
                continue
            if limit_team is not None and e.team != limit_team:
                continue

            if act.type == ActionType.MOVE:
                d: MoveDir = act.params.get('dir')
                if not isinstance(d, MoveDir):
                    continue
                dx, dy = d.delta
                target = (e.pos[0] + dx, e.pos[1] + dy)
                if not self.world.in_bounds(target):
                    continue
                sx, sy = self._to_screen(e.pos)
                tx, ty = self._to_screen(target)
                self._draw_arrow((sx, sy), (tx, ty), COLORS['preview_move'])

            elif act.type == ActionType.WAIT:
                cx, cy = self._to_screen(e.pos)
                self._draw_wait_icon(cx, cy)

            elif act.type == ActionType.TOGGLE:
                cx, cy = self._to_screen(e.pos)
                desired_on = act.params.get('on', True)
                self._draw_toggle_icon(cx, cy, desired_on)

    def _draw_mutual_shots(self, e1: Entity, e2: Entity, is_ai: bool):
        """Draw two mutual shots with clear visual separation - color coded by team"""
        sx, sy = self._to_screen(e1.pos)
        tx, ty = self._to_screen(e2.pos)

        # Calculate probabilities for both shots
        if isinstance(e1, Shooter):
            d1 = self.world.distance(e1.pos, e2.pos)
            p1 = self.world.hit_probability(
                distance=d1,
                max_range=e1.missile_max_range,
                base=e1.base_hit_prob,
                min_p=e1.min_hit_prob
            )
        else:
            p1 = 0.0

        if isinstance(e2, Shooter):
            d2 = self.world.distance(e2.pos, e1.pos)
            p2 = self.world.hit_probability(
                distance=d2,
                max_range=e2.missile_max_range,
                base=e2.base_hit_prob,
                min_p=e2.min_hit_prob
            )
        else:
            p2 = 0.0

        # Team colors for shots
        color1 = COLORS['blue'] if e1.team == Team.BLUE else COLORS['red']
        color2 = COLORS['blue'] if e2.team == Team.BLUE else COLORS['red']

        # Draw two curved shots with better separation
        srf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        s_local = (sx - self.grid_origin_x, sy)
        t_local = (tx - self.grid_origin_x, ty)

        # Calculate perpendicular offset for curves
        dx = t_local[0] - s_local[0]
        dy = t_local[1] - s_local[1]
        length = math.hypot(dx, dy)

        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length

            offset = 25

            # Draw e1 -> e2 shot (curved upward/leftward) in e1's team color
            mid_x = (s_local[0] + t_local[0]) / 2
            mid_y = (s_local[1] + t_local[1]) / 2
            control1 = (mid_x + perp_x * offset, mid_y + perp_y * offset)

            # Draw quadratic bezier curve for shot 1
            points1 = []
            for i in range(21):
                t = i / 20.0
                x = (1 - t) ** 2 * s_local[0] + 2 * (1 - t) * t * control1[0] + t ** 2 * t_local[0]
                y = (1 - t) ** 2 * s_local[1] + 2 * (1 - t) * t * control1[1] + t ** 2 * t_local[1]
                points1.append((x, y))

            pygame.draw.lines(srf, (*color1, PREVIEW_ALPHA), False, points1, 5)

            # Arrowhead for shot 1
            end_angle1 = math.atan2(points1[-1][1] - points1[-2][1], points1[-1][0] - points1[-2][0])
            arrow_size = 12
            arrow1_left = (t_local[0] - arrow_size * math.cos(end_angle1 - 0.4),
                           t_local[1] - arrow_size * math.sin(end_angle1 - 0.4))
            arrow1_right = (t_local[0] - arrow_size * math.cos(end_angle1 + 0.4),
                            t_local[1] - arrow_size * math.sin(end_angle1 + 0.4))
            pygame.draw.polygon(srf, (*color1, PREVIEW_ALPHA), [t_local, arrow1_left, arrow1_right])

            # Label for shot 1 probability
            label1_pos = (control1[0] - 20, control1[1] - 10)
            self._draw_prob_label(srf, f"{p1 * 100:.0f}%", label1_pos, color1)

            # Draw e2 -> e1 shot (curved downward/rightward) in e2's team color
            control2 = (mid_x - perp_x * offset, mid_y - perp_y * offset)

            points2 = []
            for i in range(21):
                t = i / 20.0
                x = (1 - t) ** 2 * t_local[0] + 2 * (1 - t) * t * control2[0] + t ** 2 * s_local[0]
                y = (1 - t) ** 2 * t_local[1] + 2 * (1 - t) * t * control2[1] + t ** 2 * s_local[1]
                points2.append((x, y))

            pygame.draw.lines(srf, (*color2, PREVIEW_ALPHA), False, points2, 5)

            # Arrowhead for shot 2
            end_angle2 = math.atan2(points2[-1][1] - points2[-2][1], points2[-1][0] - points2[-2][0])
            arrow2_left = (s_local[0] - arrow_size * math.cos(end_angle2 - 0.4),
                           s_local[1] - arrow_size * math.sin(end_angle2 - 0.4))
            arrow2_right = (s_local[0] - arrow_size * math.cos(end_angle2 + 0.4),
                            s_local[1] - arrow_size * math.sin(end_angle2 + 0.4))
            pygame.draw.polygon(srf, (*color2, PREVIEW_ALPHA), [s_local, arrow2_left, arrow2_right])

            # Label for shot 2 probability
            label2_pos = (control2[0] - 20, control2[1] - 10)
            self._draw_prob_label(srf, f"{p2 * 100:.0f}%", label2_pos, color2)

        self.screen.blit(srf, (self.grid_origin_x, 0))

    def _draw_prob_label(self, surf, text, pos, color):
        """Draw probability label with background"""
        font = pygame.font.SysFont('consolas', 13, bold=True)
        label = font.render(text, True, COLORS['yellow'])
        label_rect = label.get_rect(topleft=pos)
        label_rect.inflate_ip(6, 4)
        pygame.draw.rect(surf, (0, 0, 0, 200), label_rect, border_radius=3)
        pygame.draw.rect(surf, (*color, 200), label_rect, 2, border_radius=3)
        surf.blit(label, pos)

    def draw_shoot_targets(self):
        """Highlight valid targets in shoot mode with enhanced visuals"""
        if not self.shoot_mode or not self.selected:
            return

        shooter = self.world.entities_by_id.get(self.selected)
        if not isinstance(shooter, Shooter) or not shooter.alive or shooter.missiles <= 0:
            return

        team_cc = self.world.command_center(shooter.team)
        surf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)

        for eid in team_cc.visible_enemy_ids:
            tgt = self.world.entities_by_id.get(eid)
            if not tgt or not tgt.alive:
                continue

            d = self.world.distance(shooter.pos, tgt.pos)
            if d > shooter.missile_max_range:
                continue

            p = self.world.hit_probability(
                distance=d,
                max_range=shooter.missile_max_range,
                base=shooter.base_hit_prob,
                min_p=shooter.min_hit_prob
            )

            tx, ty = self._to_screen(tgt.pos)
            tx_local = tx - self.grid_origin_x

            pygame.draw.circle(surf, (*COLORS['preview_shoot'], TARGET_ALPHA),
                               (tx_local, ty), CELL // 2 - 2, 4)
            pygame.draw.circle(surf, (*COLORS['yellow'], TARGET_ALPHA // 2),
                               (tx_local, ty), CELL // 2 + 4, 2)

            cross_size = 18
            pygame.draw.line(surf, (*COLORS['preview_shoot'], TARGET_ALPHA),
                             (tx_local - cross_size, ty), (tx_local + cross_size, ty), 2)
            pygame.draw.line(surf, (*COLORS['preview_shoot'], TARGET_ALPHA),
                             (tx_local, ty - cross_size), (tx_local, ty + cross_size), 2)

            label_text = f"HIT: {p * 100:.0f}%"
            font = pygame.font.SysFont('consolas', 13, bold=True)
            label = font.render(label_text, True, COLORS['yellow'])
            label_pos = (tx_local - label.get_width() // 2, ty - CELL // 2 - 24)

            bg_rect = label.get_rect(topleft=label_pos)
            bg_rect.inflate_ip(8, 4)
            pygame.draw.rect(surf, (0, 0, 0, 220), bg_rect, border_radius=4)
            pygame.draw.rect(surf, (*COLORS['preview_shoot'], 200), bg_rect, 2, border_radius=4)
            surf.blit(label, label_pos)

        self.screen.blit(surf, (self.grid_origin_x, 0))

    def _draw_arrow(self, start, end, color):
        """Draw an animated movement arrow"""
        srf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        s_local = (start[0] - self.grid_origin_x, start[1])
        e_local = (end[0] - self.grid_origin_x, end[1])

        pygame.draw.line(srf, (*color, PREVIEW_ALPHA), s_local, e_local, 5)

        angle = math.atan2(e_local[1] - s_local[1], e_local[0] - s_local[0])
        head_len = 14
        left = (e_local[0] - head_len * math.cos(angle - 0.4),
                e_local[1] - head_len * math.sin(angle - 0.4))
        right = (e_local[0] - head_len * math.cos(angle + 0.4),
                 e_local[1] - head_len * math.sin(angle + 0.4))
        pygame.draw.polygon(srf, (*color, PREVIEW_ALPHA), [e_local, left, right])

        self.screen.blit(srf, (self.grid_origin_x, 0))

    def _draw_shot_line(self, start, end, probability, is_mutual=False, side=0, is_ai=False):
        """Draw a shooting line with hit probability - colored by team"""
        srf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        s_local = (start[0] - self.grid_origin_x, start[1])
        e_local = (end[0] - self.grid_origin_x, end[1])

        # Find the shooter entity to get team color
        shooter = None
        for e in self.world.entities:
            if e.alive and self._to_screen(e.pos) == start:
                shooter = e
                break

        # Use team color if shooter found, otherwise default
        if shooter:
            color = COLORS['blue'] if shooter.team == Team.BLUE else COLORS['red']
        else:
            color = COLORS['preview_shoot']

        # Tint for AI actions (slightly desaturated)
        if is_ai and shooter:
            base = color
            ai_tint = COLORS['ai_action']
            color = tuple((base[i] + ai_tint[i]) // 2 for i in range(3))

        # Straight line for non-mutual shots
        pygame.draw.line(srf, (*color, PREVIEW_ALPHA), s_local, e_local, 5)

        # Target reticle
        pygame.draw.circle(srf, (*color, PREVIEW_ALPHA), e_local, 12, 3)
        pygame.draw.line(srf, (*color, PREVIEW_ALPHA),
                         (e_local[0] - 15, e_local[1]), (e_local[0] + 15, e_local[1]), 2)
        pygame.draw.line(srf, (*color, PREVIEW_ALPHA),
                         (e_local[0], e_local[1] - 15), (e_local[0], e_local[1] + 15), 2)

        # Hit probability label
        prob_text = f"{probability * 100:.0f}%"
        font = pygame.font.SysFont('consolas', 14, bold=True)
        label = font.render(prob_text, True, COLORS['yellow'])

        mid_x = (s_local[0] + e_local[0]) / 2
        mid_y = (s_local[1] + e_local[1]) / 2
        label_pos = (mid_x - label.get_width() // 2, mid_y - 20)

        label_rect = label.get_rect(topleft=label_pos)
        label_rect.inflate_ip(6, 4)
        pygame.draw.rect(srf, (0, 0, 0, 200), label_rect, border_radius=3)
        pygame.draw.rect(srf, (*color, 200), label_rect, 2, border_radius=3)
        srf.blit(label, label_pos)

        self.screen.blit(srf, (self.grid_origin_x, 0))

    def _draw_wait_icon(self, cx, cy):
        """Draw a wait/pause icon"""
        srf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        c_local = (cx - self.grid_origin_x, cy)

        pygame.draw.circle(srf, (*COLORS['preview_wait'], PREVIEW_ALPHA), c_local, 16, 3)

        angle = math.radians(-90)
        hand_end = (c_local[0] + math.cos(angle) * 10,
                    c_local[1] + math.sin(angle) * 10)
        pygame.draw.line(srf, (*COLORS['preview_wait'], PREVIEW_ALPHA),
                         c_local, hand_end, 2)

        self.screen.blit(srf, (self.grid_origin_x, 0))

    def _draw_toggle_icon(self, cx, cy, turning_on=True):
        """Draw a toggle ON/OFF indicator"""
        srf = pygame.Surface((self.cols * CELL, self.rows * CELL), pygame.SRCALPHA)
        c_local = (cx - self.grid_origin_x, cy)

        color = COLORS['preview_toggle_on'] if turning_on else COLORS['preview_toggle_off']

        pygame.draw.circle(srf, (*color, PREVIEW_ALPHA), c_local, 16, 3)

        arc_rect = pygame.Rect(c_local[0] - 16, c_local[1] - 16, 32, 32)
        if turning_on:
            pygame.draw.arc(srf, (*color, PREVIEW_ALPHA), arc_rect,
                            math.radians(135), math.radians(405), 3)
            pygame.draw.line(srf, (*color, PREVIEW_ALPHA),
                             c_local, (c_local[0], c_local[1] - 12), 3)
        else:
            pygame.draw.line(srf, (*color, PREVIEW_ALPHA),
                             (c_local[0] - 8, c_local[1] - 8),
                             (c_local[0] + 8, c_local[1] + 8), 3)
            pygame.draw.line(srf, (*color, PREVIEW_ALPHA),
                             (c_local[0] + 8, c_local[1] - 8),
                             (c_local[0] - 8, c_local[1] + 8), 3)

        self.screen.blit(srf, (self.grid_origin_x, 0))

    def draw_panel(self):
        """Draw the right-side control panel"""
        panel_rect = pygame.Rect(self.panel_origin_x, 0, PANEL_W, self.height)
        pygame.draw.rect(self.screen, COLORS['panel'], panel_rect)

        y = 15

        if self.mode == 'setup':
            self._draw_setup_panel(y)
        else:
            self._draw_play_panel(y)

    def _draw_setup_panel(self, y):
        """Draw setup mode UI"""
        title = self.title_font.render("SETUP MODE", True, COLORS['text'])
        self.screen.blit(title, (self.panel_origin_x + 15, y))
        y += 40

        instructions = [
            "Left-click grid: Place unit",
            "Right-click unit: Select it",
            "Then click DELETE button",
        ]
        for line in instructions:
            img = self.font_small.render(line, True, COLORS['muted'])
            self.screen.blit(img, (self.panel_origin_x + 15, y))
            y += 18

        y += 10

        team_color = COLORS['blue'] if self.setup_team == Team.BLUE else COLORS['red']
        team_text = f"Team: {self.setup_team.name}"
        img = self.font.render(team_text, True, team_color)
        self.screen.blit(img, (self.panel_origin_x + 15, y))
        y += 22

        kind_text = f"Type: {self.setup_kind.upper()}"
        img = self.font.render(kind_text, True, COLORS['text'])
        self.screen.blit(img, (self.panel_origin_x + 15, y))
        y += 22

        if self.setup_kind in ('aircraft', 'sam'):
            missiles_text = f"Missiles: {self.setup_missiles}"
            img = self.font_small.render(missiles_text, True, COLORS['text'])
            self.screen.blit(img, (self.panel_origin_x + 15, y))
            y += 18

            range_text = f"Max Range: {self.setup_max_range:.1f}"
            img = self.font_small.render(range_text, True, COLORS['text'])
            self.screen.blit(img, (self.panel_origin_x + 15, y))
            y += 18

        radar_text = f"Radar: {self.setup_radar:.1f}"
        img = self.font_small.render(radar_text, True, COLORS['text'])
        self.screen.blit(img, (self.panel_origin_x + 15, y))
        y += 18

        y += 15

        self.btn_setup_aircraft.draw(self.screen, self.font, self.setup_kind == 'aircraft')
        self.btn_setup_awacs.draw(self.screen, self.font, self.setup_kind == 'awacs')
        self.btn_setup_sam.draw(self.screen, self.font, self.setup_kind == 'sam')
        self.btn_setup_decoy.draw(self.screen, self.font, self.setup_kind == 'decoy')

        self.btn_toggle_team.draw(self.screen, self.font)

        self.btn_missiles_minus.visible = self.setup_kind in ('aircraft', 'sam')
        self.btn_missiles_plus.visible = self.setup_kind in ('aircraft', 'sam')
        if self.btn_missiles_minus.visible:
            self.btn_missiles_minus.draw(self.screen, self.font)
            self.btn_missiles_plus.draw(self.screen, self.font)
            label = self.font_small.render(f"Missiles: {self.setup_missiles}", True, COLORS['text'])
            self.screen.blit(label, (self.panel_origin_x + 120, self.btn_missiles_minus.rect.centery - 7))

        self.btn_radar_minus.visible = True
        self.btn_radar_plus.visible = True
        if self.btn_radar_minus.visible:
            self.btn_radar_minus.draw(self.screen, self.font)
            self.btn_radar_plus.draw(self.screen, self.font)
            label = self.font_small.render(f"Radar: {self.setup_radar:.1f}", True, COLORS['text'])
            self.screen.blit(label, (self.panel_origin_x + 120, self.btn_radar_minus.rect.centery - 7))

        self.btn_range_minus.visible = self.setup_kind in ('aircraft', 'sam')
        self.btn_range_plus.visible = self.setup_kind in ('aircraft', 'sam')
        if self.btn_range_minus.visible:
            self.btn_range_minus.draw(self.screen, self.font)
            self.btn_range_plus.draw(self.screen, self.font)
            label = self.font_small.render(f"Range: {self.setup_max_range:.1f}", True, COLORS['text'])
            self.screen.blit(label, (self.panel_origin_x + 120, self.btn_range_minus.rect.centery - 7))

        self.btn_delete.enabled = self.setup_selected is not None
        self.btn_delete.draw(self.screen, self.font)
        self.btn_start.draw(self.screen, self.font)

        if self.setup_selected:
            e = self.world.entities_by_id.get(self.setup_selected)
            if e:
                y = self.btn_start.rect.bottom + 15
                pygame.draw.rect(self.screen, COLORS['button'],
                                 pygame.Rect(self.panel_origin_x + 10, y, PANEL_W - 20, 70),
                                 border_radius=6)
                y += 10
                img = self.font.render("Selected Unit:", True, COLORS['yellow'])
                self.screen.blit(img, (self.panel_origin_x + 20, y))
                y += 22
                img = self.font_small.render(f"{e.kind.upper()} #{e.id}", True, COLORS['text'])
                self.screen.blit(img, (self.panel_origin_x + 20, y))
                y += 18
                img = self.font_small.render(f"Team: {e.team.name} | Pos: {e.pos}", True, COLORS['text'])
                self.screen.blit(img, (self.panel_origin_x + 20, y))

    def _draw_play_panel(self, y):
        """Draw play mode UI"""
        title = self.title_font.render(f"PLAY - {self.view.upper()}", True, COLORS['text'])
        self.screen.blit(title, (self.panel_origin_x + 15, y))
        y += 40

        # Show game over status if applicable
        if self.world.game_over:
            game_over_color = COLORS['yellow']
            if self.world.winner == Team.BLUE:
                game_over_color = COLORS['blue']
            elif self.world.winner == Team.RED:
                game_over_color = COLORS['red']

            game_over_text = self.big_font.render("GAME OVER", True, game_over_color)
            self.screen.blit(game_over_text, (self.panel_origin_x + 15, y))
            y += 30

            reason_lines = self.world.game_over_reason.split(' - ')
            for line in reason_lines:
                reason_text = self.font.render(line, True, COLORS['text'])
                self.screen.blit(reason_text, (self.panel_origin_x + 15, y))
                y += 22

            y += 10

        blue_alive = sum(1 for e in self.world.entities if e.team == Team.BLUE and e.alive)
        red_alive = sum(1 for e in self.world.entities if e.team == Team.RED and e.alive)

        img = self.font.render(f"BLUE: {blue_alive} units", True, COLORS['blue'])
        self.screen.blit(img, (self.panel_origin_x + 15, y))
        y += 20

        img = self.font.render(f"RED: {red_alive} units", True, COLORS['red'])
        self.screen.blit(img, (self.panel_origin_x + 15, y))
        y += 30

        # Show missile counts
        if self.world.game_over:
            stats = self.world.get_game_stats()
            blue_missiles = stats['missiles']['blue_remaining']
            red_missiles = stats['missiles']['red_remaining']

            img = self.font_small.render(f"Missiles - BLUE: {blue_missiles} | RED: {red_missiles}",
                                         True, COLORS['muted'])
            self.screen.blit(img, (self.panel_origin_x + 15, y))
            y += 30

        self.btn_view_god.draw(self.screen, self.font, self.view == 'god')
        self.btn_view_blue.draw(self.screen, self.font, self.view == 'blue')
        self.btn_view_red.draw(self.screen, self.font, self.view == 'red')

        y = self.btn_view_red.rect.bottom + 20

        if self.shoot_mode:
            instructions = [
                "SHOOT MODE:",
                "Click highlighted target",
                "ESC to cancel",
            ]
        else:
            instructions = [
                "Click to select entity",
                "Queue actions below",
                "Then END TURN",
            ]

        for line in instructions:
            img = self.font_small.render(line, True, COLORS['muted'])
            self.screen.blit(img, (self.panel_origin_x + 15, y))
            y += 18

        y += 10

        sel = self.world.entities_by_id.get(self.selected) if self.selected else None
        if sel and sel.alive:
            info_lines = [
                f"{sel.kind.upper()} #{sel.id} ({sel.team.name})",
                f"Position: {sel.pos}",
            ]
            if isinstance(sel, Shooter):
                info_lines.append(f"Missiles: {sel.missiles} (R:{sel.missile_max_range:.1f})")
            if isinstance(sel, SAM):
                info_lines.append(f"Status: {'ON' if sel.on else 'OFF'}")
                if sel._cooldown > 0:
                    info_lines.append(f"Cooldown: {sel._cooldown}/{sel.cooldown_steps}")
                else:
                    info_lines.append("Cooldown: READY")
            if sel.radar_range > 0:
                info_lines.append(f"Radar: {sel.radar_range:.1f}")

            for line in info_lines:
                img = self.font_small.render(line, True, COLORS['text'])
                self.screen.blit(img, (self.panel_origin_x + 15, y))
                y += 18

            y += 15

            self.btn_wait.rect.top = y
            self.btn_wait.visible = True
            self.btn_wait.draw(self.screen, self.font)
            y += 44

            if sel.can_move:
                self.btn_move_up.visible = True
                self.btn_move_left.visible = True
                self.btn_move_right.visible = True
                self.btn_move_down.visible = True

                center_x = self.panel_origin_x + PANEL_W // 2
                self.btn_move_up.rect.centerx = center_x
                self.btn_move_up.rect.top = y

                self.btn_move_left.rect.left = self.panel_origin_x + 15
                self.btn_move_left.rect.top = y + 44

                self.btn_move_right.rect.right = self.panel_origin_x + PANEL_W - 15
                self.btn_move_right.rect.top = y + 44

                self.btn_move_down.rect.centerx = center_x
                self.btn_move_down.rect.top = y + 88

                self.btn_move_up.draw(self.screen, self.font)
                self.btn_move_left.draw(self.screen, self.font)
                self.btn_move_right.draw(self.screen, self.font)
                self.btn_move_down.draw(self.screen, self.font)
                y += 132
            else:
                self.btn_move_up.visible = False
                self.btn_move_left.visible = False
                self.btn_move_right.visible = False
                self.btn_move_down.visible = False

            if isinstance(sel, Shooter) and sel.missiles > 0:
                can_shoot = True
                if isinstance(sel, SAM):
                    can_shoot = sel.on and sel._cooldown == 0

                if can_shoot:
                    self.btn_shoot.visible = True
                    self.btn_shoot.rect.top = y
                    self.btn_shoot.draw(self.screen, self.font, self.shoot_mode)
                    y += 44
                else:
                    self.btn_shoot.visible = False
            else:
                self.btn_shoot.visible = False

            if isinstance(sel, SAM):
                self.btn_toggle.visible = True
                self.btn_toggle.rect.top = y
                self.btn_toggle.draw(self.screen, self.font)
                y += 44
            else:
                self.btn_toggle.visible = False

            if sel.id in self.pending_actions or sel.id in self.ai_suggested_actions:
                self.btn_clear.visible = True
                self.btn_clear.rect.top = y
                self.btn_clear.draw(self.screen, self.font)
                y += 44
            else:
                self.btn_clear.visible = False
        else:
            self.btn_wait.visible = False
            self.btn_move_up.visible = False
            self.btn_move_left.visible = False
            self.btn_move_right.visible = False
            self.btn_move_down.visible = False
            self.btn_shoot.visible = False
            self.btn_toggle.visible = False
            self.btn_clear.visible = False

        self.btn_end.visible = True
        self.btn_end.rect.bottom = self.height - 20
        self.btn_end.draw(self.screen, self.font)

        if self.pending_actions:
            summary_y = self.btn_end.rect.top - 25
            count_text = f"Queued: {len(self.pending_actions)} action(s)"
            img = self.font.render(count_text, True, COLORS['green'])
            self.screen.blit(img, (self.panel_origin_x + 15, summary_y))

    def draw(self):
        """Main draw routine"""
        self.draw_grid()

        for e in self.world.entities:
            self.draw_radar(e)

        if self.mode == 'play':
            self.draw_action_previews()

        for e in self.world.entities:
            self.draw_entity(e)

        if self.mode == 'play':
            self.draw_shoot_targets()

        self.draw_panel()

        pygame.display.flip()

    # ----------------------------- Actions ---------------------------------
    def queue_action(self, eid: int, action: Action):
        """Queue an action for an entity"""
        e = self.world.entities_by_id.get(eid)
        if not e or not e.alive:
            return
        self.pending_actions[eid] = action

    def clear_action(self, eid: int):
        """Remove queued action for an entity"""
        self.pending_actions.pop(eid, None)
        self.ai_suggested_actions.pop(eid, None)

    def try_move(self, eid: int, direction: MoveDir):
        """Queue a move action"""
        e = self.world.entities_by_id.get(eid)
        if not e or not e.can_move:
            return
        self.queue_action(eid, Action(ActionType.MOVE, {"dir": direction}))

    def try_wait(self, eid: int):
        """Queue a wait action"""
        self.queue_action(eid, Action(ActionType.WAIT))

    def try_toggle(self, eid: int):
        """Queue a toggle action for SAM"""
        e = self.world.entities_by_id.get(eid)
        if not isinstance(e, SAM):
            return

        existing = self.pending_actions.get(eid)
        if existing and existing.type == ActionType.TOGGLE:
            self.clear_action(eid)
            print(f"Cancelled toggle for SAM #{eid}")
        else:
            self.queue_action(eid, Action(ActionType.TOGGLE, {"on": not e.on}))
            print(f"Toggled SAM #{eid} to {'ON' if not e.on else 'OFF'}")

    def begin_shoot_mode(self):
        """Enter shoot mode for selected entity"""
        if not self.selected:
            return
        shooter = self.world.entities_by_id.get(self.selected)
        if not isinstance(shooter, Shooter) or not shooter.alive or shooter.missiles <= 0:
            return

        if isinstance(shooter, SAM):
            if not shooter.on or shooter._cooldown > 0:
                return

        self.shoot_mode = True

    def end_shoot_mode(self):
        """Exit shoot mode"""
        self.shoot_mode = False

    def end_turn(self):
        """Execute all queued actions and advance game"""
        from wargame_2d.controllers import ControllerType

        print("\n" + "=" * 50)
        print(f"EXECUTING TURN {self.world.total_turns + 1}")
        print("=" * 50)

        final_actions = dict(self.ai_suggested_actions)
        final_actions.update(self.pending_actions)

        # Record turn before execution
        turn_index = self.recorder.record_turn(self.world, final_actions, [])

        # Record LLM outputs that generated THESE actions (stored from previous step)
        if hasattr(self, '_pending_llm_outputs'):
            for team_name, llm_output in self._pending_llm_outputs.items():
                self.recorder.record_llm_output(turn_index, team_name, llm_output)
            self._pending_llm_outputs.clear()

        logs = self.world.step(final_actions)

        # Track enemy firing events for intel
        for team in [Team.BLUE, Team.RED]:
            cc = self.world.command_center(team)

            # Check observable enemy actions for missile firing
            for log in self.world.get_team_observable_logs(team):
                # Parse logs for "fires at" pattern indicating missile launch
                if "fires at" in log:
                    # Extract entity ID from log (format: "Name#ID(...) fires at...")
                    import re
                    match = re.search(r'#(\d+)', log)
                    if match:
                        entity_id = int(match.group(1))
                        entity = self.world.entities_by_id.get(entity_id)

                        # If this is an enemy entity, record that it fired
                        if entity and entity.team != team:
                            cc.record_enemy_firing(entity_id)

        # Update turn recording with logs and post-state
        self.recorder.update_turn_logs(turn_index, logs)
        self.recorder.update_turn_post_state(turn_index, self.world)

        for log in logs:
            print(log)

        self.pending_actions.clear()
        self.world.refresh_observations()

        if self.world.game_over:
            self._handle_game_over()
        else:
            self.ai_suggested_actions.clear()

            # Initialize pending LLM outputs dict for next turn
            if not hasattr(self, '_pending_llm_outputs'):
                self._pending_llm_outputs = {}

            # Get next turn actions and store LLM outputs for recording in NEXT turn
            if self.blue_controller != ControllerType.HUMAN:
                blue_actions, blue_llm_output = get_ai_actions(
                    self.world,
                    Team.BLUE,
                    controller_type=self.blue_controller,
                    controller_params=self.blue_controller_params
                )
                self.ai_suggested_actions.update(blue_actions)

                # Store LLM output to be recorded when these actions execute
                if blue_llm_output is not None:
                    self._pending_llm_outputs["BLUE"] = blue_llm_output

            if self.red_controller != ControllerType.HUMAN:
                red_actions, red_llm_output = get_ai_actions(
                    self.world,
                    Team.RED,
                    controller_type=self.red_controller,
                    controller_params=self.red_controller_params
                )
                self.ai_suggested_actions.update(red_actions)

                # Store LLM output to be recorded when these actions execute
                if red_llm_output is not None:
                    self._pending_llm_outputs["RED"] = red_llm_output

        self.end_shoot_mode()

    def _handle_game_over(self):
        """Handle game over state"""
        summary = self.world.get_game_over_summary()

        self.recorder.set_final_state(self.world)

        print("\n" + "=" * 60)
        print("GAME OVER - JSON SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("=" * 60)

        filepath = self.recorder.save_to_file()

        self.ai_suggested_actions.clear()

        if self.auto_play:
            print("\n[AUTO-PLAY] Game complete. Waiting 3 seconds before closing...")
            pygame.time.wait(3000)
            pygame.quit()
            print(f"\nRecording saved to: {filepath}")
            import sys
            sys.exit(0)
        else:
            pygame.time.wait(2000)
            pygame.quit()
            print(f"\nGame window closed. Recording saved to: {filepath}")
            import sys
            sys.exit(0)

    # ----------------------------- Setup -----------------------------------
    def place_entity(self, cell: Tuple[int, int]):
        """Place a new entity during setup"""
        if not self.world.in_bounds(cell):
            return
        if self.world.is_occupied(cell):
            return

        kind = self.setup_kind
        team = self.setup_team
        e: Optional[Entity] = None

        if kind == 'aircraft':
            e = Aircraft(
                team=team,
                pos=cell,
                missiles=self.setup_missiles,
                radar_range=self.setup_radar,
                missile_max_range=self.setup_max_range
            )
        elif kind == 'awacs':
            e = AWACS(team=team, pos=cell, radar_range=self.setup_radar)
        elif kind == 'decoy':
            e = Decoy(team=team, pos=cell, radar_range=self.setup_radar)
        elif kind == 'sam':
            e = SAM(
                team=team,
                pos=cell,
                missiles=self.setup_missiles,
                missile_max_range=self.setup_max_range,
                radar_range=self.setup_radar
            )

        if e:
            self.world.add(e)
            self.world.refresh_observations()

    def select_entity_at(self, cell: Tuple[int, int]):
        """Select an entity at the given cell"""
        self.setup_selected = None
        for e in self.world.entities:
            if e.pos == cell:
                self.setup_selected = e.id
                print(f"Selected: {e.kind.upper()} #{e.id} at {e.pos}")
                return
        print(f"No entity at {cell}")

    def delete_selected_entity(self):
        """Delete the currently selected entity in setup"""
        if not self.setup_selected:
            print("No entity selected to delete")
            return

        e = self.world.entities_by_id.get(self.setup_selected)
        if e:
            print(f"Deleting: {e.kind.upper()} #{e.id}")
            self.world.entities.remove(e)
            del self.world.entities_by_id[e.id]
            self.setup_selected = None
            self.world.refresh_observations()
        else:
            print(f"Entity #{self.setup_selected} not found")
            self.setup_selected = None

    # ----------------------------- Events ----------------------------------
    def handle_click(self, pos: Tuple[int, int], button: int):
        """Handle mouse clicks"""
        if pos[0] >= self.panel_origin_x:
            self.handle_panel_click(pos, button)
            return

        cell = self.cell_at_mouse(pos)
        if not cell:
            return

        if self.mode == 'setup':
            if button == 1:
                self.place_entity(cell)
            elif button == 3:
                self.select_entity_at(cell)

        elif self.mode == 'play':
            if button == 1:
                if self.shoot_mode and self.selected:
                    self.handle_shoot_click(cell)
                else:
                    self.handle_entity_select(cell)
            elif button == 3:
                if self.shoot_mode:
                    self.end_shoot_mode()
                else:
                    self.selected = None

    def handle_panel_click(self, pos: Tuple[int, int], button: int):
        """Handle clicks in the UI panel"""
        if button != 1:
            return

        if self.mode == 'setup':
            if self.btn_setup_aircraft.is_clicked(pos):
                self.setup_kind = 'aircraft'
            elif self.btn_setup_awacs.is_clicked(pos):
                self.setup_kind = 'awacs'
            elif self.btn_setup_sam.is_clicked(pos):
                self.setup_kind = 'sam'
            elif self.btn_setup_decoy.is_clicked(pos):
                self.setup_kind = 'decoy'
            elif self.btn_toggle_team.is_clicked(pos):
                self.setup_team = Team.RED if self.setup_team == Team.BLUE else Team.BLUE
            elif self.btn_missiles_minus.is_clicked(pos):
                self.setup_missiles = max(0, self.setup_missiles - 1)
            elif self.btn_missiles_plus.is_clicked(pos):
                self.setup_missiles = min(20, self.setup_missiles + 1)
            elif self.btn_radar_minus.is_clicked(pos):
                self.setup_radar = max(0, self.setup_radar - 0.5)
            elif self.btn_radar_plus.is_clicked(pos):
                self.setup_radar = min(15, self.setup_radar + 0.5)
            elif self.btn_range_minus.is_clicked(pos):
                self.setup_max_range = max(1, self.setup_max_range - 0.5)
            elif self.btn_range_plus.is_clicked(pos):
                self.setup_max_range = min(15, self.setup_max_range + 0.5)
            elif self.btn_delete.is_clicked(pos):
                self.delete_selected_entity()
            elif self.btn_start.is_clicked(pos):
                self.start_game()

        elif self.mode == 'play':
            if self.btn_view_god.is_clicked(pos):
                self.view = 'god'
            elif self.btn_view_blue.is_clicked(pos):
                self.view = 'blue'
            elif self.btn_view_red.is_clicked(pos):
                self.view = 'red'
            elif self.world.game_over:
                return
            elif self.btn_wait.is_clicked(pos) and self.selected:
                self.try_wait(self.selected)
            elif self.btn_move_up.is_clicked(pos) and self.selected:
                self.try_move(self.selected, MoveDir.UP)
            elif self.btn_move_down.is_clicked(pos) and self.selected:
                self.try_move(self.selected, MoveDir.DOWN)
            elif self.btn_move_left.is_clicked(pos) and self.selected:
                self.try_move(self.selected, MoveDir.LEFT)
            elif self.btn_move_right.is_clicked(pos) and self.selected:
                self.try_move(self.selected, MoveDir.RIGHT)
            elif self.btn_shoot.is_clicked(pos) and self.selected:
                self.begin_shoot_mode()
            elif self.btn_toggle.is_clicked(pos) and self.selected:
                self.try_toggle(self.selected)
            elif self.btn_clear.is_clicked(pos) and self.selected:
                self.clear_action(self.selected)
            elif self.btn_end.is_clicked(pos):
                self.end_turn()

    def handle_entity_select(self, cell: Tuple[int, int]):
        """Select entity at cell in play mode"""
        for e in self.world.entities:
            if e.pos == cell and e.alive:
                self.selected = e.id
                return
        self.selected = None

    def handle_shoot_click(self, cell: Tuple[int, int]):
        """Handle shooting at a target"""
        if not self.selected:
            return

        shooter = self.world.entities_by_id.get(self.selected)
        if not isinstance(shooter, Shooter):
            return

        target = None
        for e in self.world.entities:
            if e.pos == cell and e.alive:
                target = e
                break

        if not target:
            return

        if shooter.team == target.team:
            return

        team_cc = self.world.command_center(shooter.team)
        if target.id not in team_cc.visible_enemy_ids:
            return

        d = self.world.distance(shooter.pos, target.pos)
        if d > shooter.missile_max_range:
            return

        self.queue_action(self.selected, Action(ActionType.SHOOT, {"target_id": target.id}))
        self.end_shoot_mode()

    def start_game(self):
        """Transition from setup to play mode"""
        self.mode = 'play'
        self.setup_selected = None
        self.selected = None
        self.world.refresh_observations()

        # Record game configuration
        config = {
            "world": {
                "width": self.world.width,
                "height": self.world.height,
                "max_stalemate_turns": self.world.max_stalemate_turns
            },
            "blue_team": {
                "controller": self.blue_controller.value,
                "controller_params": self.blue_controller_params
            },
            "red_team": {
                "controller": self.red_controller.value,
                "controller_params": self.red_controller_params
            },
            "auto_play": self.auto_play,
            "auto_play_delay": self.auto_play_delay
        }
        self.recorder.set_configuration(config)
        self.recorder.set_initial_state(self.world)

        # Generate initial AI actions
        self.ai_suggested_actions.clear()
        self._pending_llm_outputs = {}  # Initialize pending LLM outputs tracker

        if self.blue_controller != ControllerType.HUMAN:
            blue_actions, blue_llm_output = get_ai_actions(
                self.world,
                Team.BLUE,
                controller_type=self.blue_controller,
                controller_params=self.blue_controller_params
            )
            self.ai_suggested_actions.update(blue_actions)

            # Store initial LLM output for Turn 0
            if blue_llm_output is not None:
                self._pending_llm_outputs["BLUE"] = blue_llm_output

        if self.red_controller != ControllerType.HUMAN:
            red_actions, red_llm_output = get_ai_actions(
                self.world,
                Team.RED,
                controller_type=self.red_controller,
                controller_params=self.red_controller_params
            )
            self.ai_suggested_actions.update(red_actions)

            # Store initial LLM output for Turn 0
            if red_llm_output is not None:
                self._pending_llm_outputs["RED"] = red_llm_output

        print("\n" + "=" * 50)
        print("GAME STARTED")
        print(f"Blue Controller: {self.blue_controller.value.upper()}")
        print(f"Red Controller: {self.red_controller.value.upper()}")
        print(f"Auto-Play: {'ENABLED' if self.auto_play else 'DISABLED'}")
        if self.auto_play:
            print(f"Turn Delay: {self.auto_play_delay}s")
        print("=" * 50)

    def handle_key(self, event):
        """Handle keyboard input"""
        key = event.key

        if key == pygame.K_F11:
            self.fullscreen = not self.fullscreen
            self._create_display(self.fullscreen)
            self._build_buttons()
            return

        if key == pygame.K_ESCAPE:
            if self.shoot_mode:
                self.end_shoot_mode()
            elif self.mode == 'play':
                self.selected = None
            return

        if self.mode == 'setup':
            if key == pygame.K_t:
                self.setup_team = Team.RED if self.setup_team == Team.BLUE else Team.BLUE
            elif key == pygame.K_1:
                self.setup_kind = 'aircraft'
            elif key == pygame.K_2:
                self.setup_kind = 'awacs'
            elif key == pygame.K_3:
                self.setup_kind = 'sam'
            elif key == pygame.K_4:
                self.setup_kind = 'decoy'
            elif key == pygame.K_LEFTBRACKET:
                self.setup_missiles = max(0, self.setup_missiles - 1)
            elif key == pygame.K_RIGHTBRACKET:
                self.setup_missiles = min(20, self.setup_missiles + 1)
            elif key == pygame.K_DELETE and self.setup_selected:
                self.delete_selected_entity()
            elif key == pygame.K_s:
                self.start_game()

        elif self.mode == 'play':
            if key == pygame.K_g:
                self.view = 'god'
            elif key == pygame.K_b:
                self.view = 'blue'
            elif key == pygame.K_r:
                self.view = 'red'
            elif key == pygame.K_RETURN:
                self.end_turn()

            if self.selected:
                if key == pygame.K_SPACE:
                    self.try_wait(self.selected)
                elif key == pygame.K_c:
                    self.clear_action(self.selected)
                elif key == pygame.K_f:
                    self.begin_shoot_mode()
                elif key == pygame.K_o:
                    self.try_toggle(self.selected)
                elif key == pygame.K_UP:
                    self.try_move(self.selected, MoveDir.UP)
                elif key == pygame.K_DOWN:
                    self.try_move(self.selected, MoveDir.DOWN)
                elif key == pygame.K_LEFT:
                    self.try_move(self.selected, MoveDir.LEFT)
                elif key == pygame.K_RIGHT:
                    self.try_move(self.selected, MoveDir.RIGHT)

    # ----------------------------- Main Loop -------------------------------
    def run(self):
        """Main game loop with auto-play support"""
        running = True

        print("=" * 50)
        print("GRID AIR COMBAT - PYGAME UI")
        print("=" * 50)

        if not self.auto_play:
            print("Setup your scenario, then press 'S' or START GAME")
        else:
            print("AUTO-PLAY MODE: Game will run automatically")
            # Auto-start so auto-play loop can run (mode == 'play')
            if self.mode != 'play':
                self.start_game()
                # Start the delay timer for the first auto turn
                self.last_auto_turn_time = time.time()
        print()

        while running:

            current_time = time.time()

            # Auto-play logic
            if self.auto_play and self.mode == 'play' and not self.world.game_over:
                if current_time - self.last_auto_turn_time >= self.auto_play_delay:
                    self.end_turn()
                    self.last_auto_turn_time = current_time

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif not self.auto_play:
                        self.handle_key(event)

                elif event.type == pygame.MOUSEBUTTONDOWN and not self.auto_play:
                    self.handle_click(event.pos, event.button)

                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.size
                    needed_w = self.cols * CELL + PANEL_W
                    self.grid_origin_x = max(0, (self.width - needed_w) // 2)
                    self.panel_origin_x = self.grid_origin_x + self.cols * CELL
                    self._build_buttons()

            # Draw
            self.draw()

            # Control frame rate
            if self.auto_play:
                self.clock.tick(30)
            else:
                self.clock.tick(FPS)

        pygame.quit()
        print("\nGame ended.")
# ----------------------------- Entry ---------------------------------------

def create_empty_world(seed: int = 42) -> World:
    """Create an empty world for setup"""
    return World(20, 12, seed=seed)


def create_demo_world(seed: int = 42) -> World:
    """Create a pre-configured demo scenario"""
    w = World(20, 12, seed=seed)

    w.add(Aircraft(team=Team.BLUE, pos=(7, 3), missiles=7, radar_range=3.0, missile_max_range=3.0, name="Blue-F1"))
    w.add(Aircraft(team=Team.BLUE, pos=(7, 7), missiles=7, radar_range=3.0, missile_max_range=3.0, name="Blue-F2"))
    w.add(AWACS(team=Team.BLUE, pos=(1, 5), radar_range=7.0, name="Blue-AWACS"))
    w.add(SAM(team=Team.BLUE, pos=(3, 5), missiles=3, missile_max_range=4.0, radar_range=4.0, name="Blue-SAM"))
    w.add(Decoy(team=Team.BLUE, pos=(9, 5), radar_range=2.0, name="Blue-Decoy1"))

    w.add(Aircraft(team=Team.RED, pos=(16, 3), missiles=7, radar_range=3.0, missile_max_range=3.0, name="Red-F1"))
    w.add(Aircraft(team=Team.RED, pos=(16, 7), missiles=7, radar_range=3.0, missile_max_range=3.0, name="Red-F2"))
    w.add(AWACS(team=Team.RED, pos=(18, 5), radar_range=7.0, name="Red-AWACS"))
    w.add(SAM(team=Team.RED, pos=(16, 5), missiles=3, missile_max_range=4.0, radar_range=4.0, name="Red-SAM"))
    w.add(Decoy(team=Team.RED, pos=(15, 5), radar_range=2.0, name="Red-Decoy1"))

    return w


def main():
    from wargame_2d import config # Import configuration

    logfire.configure()
    logfire.instrument_pydantic_ai()

    print("=" * 60)
    print("GRID AIR COMBAT - CONFIGURATION")
    print("=" * 60)
    print("\nLoading configuration from config.py...")

    # Configuration summary
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"World: {'Empty' if config.WORLD_CHOICE == 1 else 'Demo Scenario'}")
    print(f"Blue Team: {config.BLUE_CONTROLLER.value.upper()}")
    print(f"Red Team: {config.RED_CONTROLLER.value.upper()}")
    print(f"Play Mode: {'AUTO-PLAY' if config.AUTO_PLAY else 'MANUAL'}")
    if config.AUTO_PLAY:
        print(f"  - Turn Delay: {config.AUTO_PLAY_DELAY}s")
    print(f"Number of Games: {config.NUM_GAMES}")
    if config.NUM_GAMES > 1:
        print(f"  - Base Seed: {config.BASE_SEED}")
        print(f"  - Seeds will be: {config.BASE_SEED} to {config.BASE_SEED + config.NUM_GAMES - 1}")
    print("=" * 60)

    if not config.AUTO_PLAY and config.NUM_GAMES > 1:
        print("\nWARNING: Running multiple games in manual mode requires user input for each game.")


    #input("\nPress Enter to start...")

    # Run N games
    for game_num in range(config.NUM_GAMES):
        current_seed = config.BASE_SEED + game_num

        print("\n" + "=" * 80)
        print(f"STARTING GAME {game_num + 1}/{config.NUM_GAMES} (Seed: {current_seed})")
        print("=" * 80)

        # Create world with current seed
        if config.WORLD_CHOICE == 1:
            world = create_empty_world(seed=current_seed)
        else:
            world = create_demo_world(seed=current_seed)

        # Override world seed
        world._rng = random.Random(current_seed)

        ui = GameUI(
            world,
            blue_controller=config.BLUE_CONTROLLER,
            blue_controller_params=config.BLUE_CONTROLLER_PARAMS,
            red_controller=config.RED_CONTROLLER,
            red_controller_params=config.RED_CONTROLLER_PARAMS,
            auto_play=config.AUTO_PLAY,
            auto_play_delay=config.AUTO_PLAY_DELAY
        )
        ui.run()

        if game_num < config.NUM_GAMES - 1:
            print(f"\n[MULTI-GAME] Game {game_num + 1} complete. Starting next game...")
            if not config.AUTO_PLAY:
                input("Press Enter to continue to next game...")

    print("\n" + "=" * 80)
    print(f"ALL GAMES COMPLETE - {config.NUM_GAMES} game(s) finished")
    print("=" * 80)


if __name__ == '__main__':
    main()