# Grid Combat Environment - Architecture Map

A short guide for new contributors; focus on where to start and how data flows.

## Top-Level Pieces
- `env/`: Simulation engine (core types, entities, mechanics, `Scenario`, `GridCombatEnv`).
- `agents/`: Built-in agent implementations plus the factory used by scenarios.
- `runtime/frame.py`: Lightweight turn snapshot (`Frame`) with serialization helpers for the UI.
- `runtime/runner.py`: Orchestrates a game. Loads a `Scenario`, wires up agents, steps the `GridCombatEnv`, and emits `Frame` objects.
- `infra/paths.py`: Shared filesystem locations (project root, storage, UI entrypoint) used by env and API layers.
- `api/app.py`: FastAPI surface for the runner. Minimal stateful wrapper that exposes `/start`, `/step`, and `/status`.
- `ui/ops_deck.html`: Static control panel that calls the API to drive and visualize a match.

## How a Turn Moves Through the System
1. A client posts a scenario (and optional saved `world`) to `POST /start`. `api/app.py` builds a `Scenario` and instantiates a global `GameRunner`.
2. Each `POST /step` call forwards optional agent injections, asks the runner to advance one turn, and returns a serialized `Frame` (world snapshot + actions + observations).
3. The UI (`ui/ops_deck.html`) polls `/status` to show progress and hits `/step` to advance the game. It renders directly from the `Frame` payload.

## Game Runner at a Glance
- Entry point for the simulation layer (`runtime/runner.py`).
- Holds a `GridCombatEnv` instance and two prepared agents derived from the scenario.
- `step()` merges both teams' actions, calls `env.step()`, and returns the pre-step world as a `Frame` so the UI can show fog-of-war correct views.
- `run()` is a convenience loop for full episodes; `get_final_frame()` returns the terminal state without actions.

## API Surface
- `POST /start`: body `{ "scenario": {...}, "world": {...}|null }` — creates the runner.
- `POST /step`: body `{ "injections": {...}|null }` — advances one turn, returns `Frame`.
- `GET /status`: quick heartbeat (`active`, `turn`, `step`, `done`).
- Dev command: `uv run uvicorn api.app:app --reload --port 8000`.

## UI Touchpoints
- Open `ui/ops_deck.html` in a browser; it expects the API above to be running locally on port 8000.
- Renders from the `Frame` fields: `world` (canonical state), `entities` (UI-friendly shape), `observations` (fog-of-war per team), and optional `actions/step_info`.

## Core Engine Notes
- `GridCombatEnv` (in `env/environment.py`) owns turn sequencing: movement → sensing → combat → victory checks.
- `Scenario` objects (in `env/scenario.py`) are the single source of truth for grid size, rules, entities, seeds, and agent specs.
- `WorldState` holds mutable game state; cloning plus `SensorSystem.refresh_all_observations()` is used when preparing UI-ready snapshots to avoid stale visibility data.

## Engine Internals (env/)
- Flow inside `GridCombatEnv.step()`: tick cooldowns → movement resolver → sensor refresh → combat resolver → victory checks → return `(state, rewards, done, info)`.
- Key subsystems: `world/` (grid + world state), `core/` (types, actions), `entities/` (unit capabilities), `mechanics/` (movement/sensors/combat/victory, all stateless).
- `Scenario` builds everything up front; env takes the scenario dict and initializes `WorldState` plus rules with no hidden defaults.
- Observations are stored per-team on `WorldState`; always call the sensor system after movement to keep fog-of-war aligned with positions.
