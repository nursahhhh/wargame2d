"""Centralized Logfire observability — gracefully disabled when LOGFIRE_TOKEN is not set."""
import os
import logging
from contextlib import contextmanager

_log = logging.getLogger(__name__)

# Only configure Logfire if a token is explicitly provided.
_LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN", "").strip()
_logfire_enabled = bool(_LOGFIRE_TOKEN)

if _logfire_enabled:
    try:
        import logfire as _logfire_mod
        _logfire_mod.configure(
            token=_LOGFIRE_TOKEN,
            environment=os.getenv("ENV", "production"),
            console=os.getenv("LOGFIRE_CONSOLE", "false").lower() == "true",
            send_to_logfire=True,
        )
        logfire = _logfire_mod
        _log.info("Logfire observability enabled.")
    except Exception as e:
        _log.warning("Logfire configure failed (%s) — falling back to no-op.", e)
        _logfire_enabled = False

if not _logfire_enabled:
    # Provide a no-op stub so the rest of the codebase never needs to branch.
    class _NoOpLogfire:
        """Minimal stub that silently swallows all logfire calls."""
        @contextmanager
        def span(self, name, **kwargs):
            yield

        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def debug(self, *a, **kw): pass
        def instrument_openai(self, *a, **kw): pass
        def instrument_pydantic_ai(self, *a, **kw): pass

    logfire = _NoOpLogfire()  # type: ignore[assignment]


# ── Public trace helpers ────────────────────────────────────────────────────

@contextmanager
def trace_strategic_planning():
    """Trace strategic planning phase."""
    with logfire.span("strategic_planning"):
        yield


@contextmanager
def trace_tactical_execution(role: str):
    """Trace tactical execution per role."""
    with logfire.span(f"tactical_{role.lower()}"):
        yield


@contextmanager
def trace_role_execution(role: str):
    """Trace single role execution."""
    with logfire.span(f"role_{role.lower()}", role=role):
        yield


@contextmanager
def trace_fallback(reason: str):
    """Trace fallback triggers."""
    with logfire.span("fallback", reason=reason):
        logfire.error("fallback_triggered", reason=reason, alert=True)
        yield
