"""
ID generation utilities for entities.

Provides thread-safe, monotonic ID generation for game entities.
"""

import itertools
from typing import Iterator


class IDGenerator:
    """
    Generates unique, sequential IDs for entities.

    This is a simple wrapper around itertools.count that makes
    testing easier and provides a clear contract.
    """

    def __init__(self, start: int = 1):
        """
        Initialize the ID generator.

        Args:
            start: The first ID to generate (default: 1)
        """
        self._counter: Iterator[int] = itertools.count(start)

    def next_id(self) -> int:
        """Generate the next unique ID."""
        return next(self._counter)

    def reset(self, start: int = 1) -> None:
        """
        Reset the ID generator to a new starting value.

        Args:
            start: The new starting ID
        """
        self._counter = itertools.count(start)


# Global ID generator instance
# This can be reset for testing or replaced with a custom instance
_global_id_generator = IDGenerator()


def get_next_entity_id() -> int:
    """Get the next unique entity ID from the global generator."""
    return _global_id_generator.next_id()


def reset_entity_ids(start: int = 1) -> None:
    """
    Reset the global entity ID generator.

    Useful for testing to ensure reproducible IDs.

    Args:
        start: The new starting ID
    """
    _global_id_generator.reset(start)