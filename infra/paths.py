from __future__ import annotations

from pathlib import Path

# Resolved project root (parent directory of this infra package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common storage locations.
STORAGE_DIR = PROJECT_ROOT / "storage"
SCENARIO_STORAGE_DIR = STORAGE_DIR / "scenarios"
UI_ENTRYPOINT = PROJECT_ROOT / "ui" / "ops_deck.html"
