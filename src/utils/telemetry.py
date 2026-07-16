from __future__ import annotations
import os
import atexit

from dotenv import load_dotenv
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from opentelemetry import trace

_initialized: bool = False


# ── Initialisation ────────────────────────────────────────────────────────────

def setup_telemetry() -> None:
    """
    Initialise Langfuse (which also sets up OTel under the hood). Idempotent.
    In Langfuse 4.x the SDK is OTel-native: calling Langfuse() registers an
    OTLP exporter on the global TracerProvider, so trace.get_tracer() works
    automatically after this call.
    """
    global _initialized
    if _initialized:
        return

    load_dotenv()   # ensure .env is loaded regardless of import order
    enabled = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"

    Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        tracing_enabled=enabled,
    )
    _initialized = True
    atexit.register(shutdown)   # backup flush on normal process exit


# ── Accessors ─────────────────────────────────────────────────────────────────

def get_langfuse() -> Langfuse:
    if not _initialized:
        setup_telemetry()
    return get_client()


def get_tracer(name: str = "evidencerank") -> trace.Tracer:
    if not _initialized:
        setup_telemetry()
    return trace.get_tracer(name)


# ── LangChain integration ─────────────────────────────────────────────────────

def make_callback() -> CallbackHandler:
    """
    Return a Langfuse CallbackHandler with no explicit trace_context.

    In Langfuse 4.x this automatically inherits the current OTel span as
    parent — as long as the caller is inside a tracer.start_as_current_span()
    block.
    """
    return CallbackHandler()


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def shutdown() -> None:
    """Flush all buffered spans/events. Call explicitly in CLI finally blocks."""
    get_client().flush()


# ── Trace ID extraction ───────────────────────────────────────────────────────

def current_otel_trace_id() -> str:
    """Return the current OTel trace_id as a 32-hex string (empty string if none)."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return ""
    return format(ctx.trace_id, "032x")
