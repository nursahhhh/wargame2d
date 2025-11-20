"""
Entity definitions for the Grid Combat Environment.

This module exports all entity types:
- Entity (abstract base)
- Aircraft (mobile fighter)
- AWACS (surveillance aircraft)
- SAM (stationary air defense)
- Decoy (deceptive unit)
"""

from .base import Entity
from .aircraft import Aircraft
from .awacs import AWACS
from .sam import SAM
from .decoy import Decoy

__all__ = [
    "Entity",
    "Aircraft",
    "AWACS",
    "SAM",
    "Decoy",
]