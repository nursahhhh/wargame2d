"""HTTP API entrypoint for driving the game from a web UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from env.scenario import Scenario
from runtime.runner import GameRunner
from infra.paths import UI_ENTRYPOINT
from runtime.logfire_config import configure_logfire

# Configure observability before app/agent imports are used.
configure_logfire()

app = FastAPI()
runner: GameRunner | None = None


# Allow the browser-based control panel (served from file:// or other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    scenario: dict
    world: dict | None = None

class StepRequest(BaseModel):
    injections: dict | None = None


@app.post("/start")
def start(request: StartRequest):
    global runner
    scenario = Scenario.from_dict(request.scenario)
    runner = GameRunner(scenario, world=request.world)
    return {"success": True}


@app.post("/step")
def step(request: StepRequest):
    if runner is None:
        raise HTTPException(400, "No active game")
    try:
        return runner.step(request.injections).to_dict()
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.post("/stop")
def stop():
    global runner

    if runner is None:
        raise HTTPException(400, "No active game")

    try:
        runner.abort_episode()
        runner = None
        return {"success": True, "message": "Game aborted"}
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.get("/status")
def status():
    if runner is None:
        return {"active": False}
    return {"active": True, "turn": runner.turn, "step": runner.step_count, "done": runner.done}


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the bundled control panel so the app runs from a single origin."""
    if not UI_ENTRYPOINT.exists():
        raise HTTPException(500, "UI entrypoint not found")
    return FileResponse(UI_ENTRYPOINT)
