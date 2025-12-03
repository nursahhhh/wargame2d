"""Local launcher that runs the API and serves the UI from one command."""

import argparse
import threading
import webbrowser

import uvicorn

from infra.logger import STORAGE_DIR, configure_logging, get_logger


def _open_browser(url: str, delay: float = 1.0) -> None:
    """Open the UI in the default browser after the server spins up."""
    timer = threading.Timer(delay, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main():
    parser = argparse.ArgumentParser(description="Run the WG backend and UI together.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API/UI (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the UI in the browser")
    args = parser.parse_args()

    # Configure logging once at startup (console + file).
    configure_logging(level="INFO", json=True)
    log = get_logger(__name__)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        _open_browser(url)

    """
    FastAPI is just Python code describing endpoints.
    It does not:
        - Accept TCP connections
        - Listen on ports
        - Speak HTTP protocol
        - Manage event loops
        - Handle concurrency
    Those things are the job of an ASGI server, and uvicorn is one of the most commonly used servers for FastAPI.
    """
    log.info("Starting WG backend + UI at %s", url)
    uvicorn.run(
        "api.app:app", # Import the module api.app, grab the variable app from that module, start a web server that serves that FastAPI app
        host=args.host,
        port=args.port, # Bind the server to host:port
        reload=args.reload, # Reload automatically if code changes (all python files on project is tracked I guess?) (when --reload is used)
        log_level="info",
    )


if __name__ == "__main__":
    main()
