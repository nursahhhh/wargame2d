"""
Grid - Spatial logic for the combat environment.

The Grid handles:
- Coordinate validation
- Distance calculations
- Occupancy tracking
- Coordinate system conversions

Coordinate System:
- X increases to the RIGHT
- Y increases UPWARD (mathematical convention)
- Origin (0, 0) is at BOTTOM-LEFT
"""

from __future__ import annotations
import math
from typing import Set, Optional
from ..core.types import GridPos


class Grid:
    """
    A 2D grid with mathematical coordinates (Y+ = UP).

    Provides spatial queries and calculations without game logic and or state.

    Attributes:
        width: Grid width (X dimension)
        height: Grid height (Y dimension)
    """

    def __init__(self, width: int, height: int):
        """
        Initialize a grid.

        Args:
            width: Grid width (must be positive)
            height: Grid height (must be positive)

        Raises:
            ValueError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive: {width}x{height}")

        self.width = width
        self.height = height

    def in_bounds(self, pos: GridPos) -> bool:
        """
        Check if a position is within grid boundaries.

        Args:
            pos: Position to check (x, y)

        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def distance(self, a: GridPos, b: GridPos) -> float:
        """
        Calculate Euclidean distance between two positions.

        Args:
            a: First position (x, y)
            b: Second position (x, y)

        Returns:
            Euclidean distance as a float
        """
        return math.hypot(a[0] - b[0], a[1] - b[1]) # equivalent to sqrt((x2 - x1)^2 + (y2 - y1)^2)

    def manhattan_distance(self, a: GridPos, b: GridPos) -> int:
        """
        Calculate Manhattan (taxicab) distance between two positions.

        Args:
            a: First position (x, y)
            b: Second position (x, y)

        Returns:
            Manhattan distance as an integer
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


    def get_neighbors(self, pos: GridPos, include_diagonals: bool = False) -> list[GridPos]:
        """
        Get neighboring positions (4 or 8 directions).

        Args:
            pos: Center position
            include_diagonals: If True, include diagonal neighbors (8 total)
                               If False, only cardinal directions (4 total)

        Returns:
            List of valid neighboring positions
        """
        x, y = pos

        # Cardinal directions (up, down, left, right)
        candidates = [
            (x, y + 1),  # UP
            (x, y - 1),  # DOWN
            (x - 1, y),  # LEFT
            (x + 1, y),  # RIGHT
        ]

        if include_diagonals:
            # Diagonal directions
            candidates.extend([
                (x - 1, y + 1),  # UP-LEFT
                (x + 1, y + 1),  # UP-RIGHT
                (x - 1, y - 1),  # DOWN-LEFT
                (x + 1, y - 1),  # DOWN-RIGHT
            ])

        # Filter to only valid positions
        neighbors = [pos for pos in candidates if self.in_bounds(pos)]
        return neighbors

    def positions_in_range(self, center: GridPos, max_range: float) -> list[GridPos]:
        """
        Get all positions within a given range of a center point.

        Args:
            center: Center position
            max_range: Maximum distance (Euclidean)

        Returns:
            List of positions within range (including center)
        """
        cx, cy = center
        r = int(math.ceil(max_range))
        positions = []

        for y in range(max(0, cy - r), min(self.height, cy + r + 1)):
            for x in range(max(0, cx - r), min(self.width, cx + r + 1)):
                pos = (x, y)
                if self.distance(center, pos) <= max_range:
                    positions.append(pos)

        return positions

    def to_screen_y(self, math_y: int) -> int:
        """
        Convert mathematical Y coordinate to screen Y coordinate.

        Screen coordinates have Y=0 at top, mathematical has Y=0 at bottom.

        Args:
            math_y: Mathematical Y coordinate (0 = bottom)

        Returns:
            Screen Y coordinate (0 = top)
        """
        return self.height - 1 - math_y

    def to_math_y(self, screen_y: int) -> int:
        """
        Convert screen Y coordinate to mathematical Y coordinate.

        Args:
            screen_y: Screen Y coordinate (0 = top)

        Returns:
            Mathematical Y coordinate (0 = bottom)
        """
        return self.height - 1 - screen_y

    def __str__(self) -> str:
        """String representation."""
        return f"Grid({self.width}x{self.height})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Grid(width={self.width}, height={self.height})"