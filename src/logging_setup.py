"""Process-wide logging configuration.

Replaces the bare `logging.basicConfig(...)` that used to live at the
top of `src/main.py`. Two production problems drove this:

1. **Railway classified every app log as an error.** Railway assigns
   log level by stream — stdout is `info`, stderr is `error` — and
   Python's default StreamHandler writes to stderr, so every INFO line
   showed up red in the dashboard and `--filter @level:error` was
   useless. The fix is a single stdout handler emitting one JSON
   object per line; Railway parses the `level` attribute of JSON logs
   and classifies them correctly.

2. **Multi-line tracebacks became 30 separate "error" entries.** In
   JSON mode the traceback travels inside the one entry's `message`,
   so a stack trace is a single searchable log line.

JSON is on when the process is running on Railway (detected via the
env Railway always injects) and off for local dev, where the plain
format is easier to read. `NEMARA_LOG_JSON=1|0` overrides either way.

Also owns the `livekit` logger burst-dedup filter: the Rust FFI core
(livekit 1.1.13) re-emits some INFO records many times within a few
milliseconds (observed 15× "attached packet_trailer handler ..."
bursts in prod). The filter passes the first occurrence and drops
identical repeats inside a short window.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone

# Railway's log explorer recognizes these level strings on JSON lines.
_RAILWAY_LEVELS = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "error",
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message.

    The exception traceback (when present) is appended to `message`
    rather than emitted as bare extra lines, so one log call is always
    exactly one entry in Railway.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{record.stack_info}"
        return json.dumps(
            {
                "time": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(timespec="milliseconds"),
                "level": _RAILWAY_LEVELS.get(record.levelname, "info"),
                "logger": record.name,
                "message": message,
            },
            ensure_ascii=False,
        )


class BurstDedupFilter(logging.Filter):
    """Drop a record identical (level + rendered message) to one seen
    within `window_seconds`. First occurrence always passes.

    Bounded: the seen-map is an LRU capped at `max_entries`, so the
    filter itself can never become a memory leak.
    """

    def __init__(self, *, window_seconds: float = 2.0, max_entries: int = 256) -> None:
        super().__init__()
        self._window = window_seconds
        self._max_entries = max_entries
        self._seen: OrderedDict[tuple[int, str], float] = OrderedDict()

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelno, record.getMessage())
        now = time.monotonic()
        last = self._seen.get(key)
        self._seen[key] = now
        self._seen.move_to_end(key)
        while len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)
        return last is None or (now - last) >= self._window


def _json_mode_enabled() -> bool:
    override = os.getenv("NEMARA_LOG_JSON")
    if override is not None:
        return override.strip().lower() in ("1", "true", "yes")
    # RAILWAY_ENVIRONMENT_NAME (and friends) are injected into every
    # Railway service container; local dev and CI never set them.
    return "RAILWAY_ENVIRONMENT_NAME" in os.environ or "RAILWAY_PROJECT_ID" in os.environ


def configure_logging() -> None:
    """Install the process-wide handler. Idempotent — safe under test
    re-imports and uvicorn worker respawns."""
    handler = logging.StreamHandler(sys.stdout)
    if _json_mode_enabled():
        handler.setFormatter(JsonLineFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("NEMARA_LOG_LEVEL", "INFO").upper())

    # Uvicorn installs its own stderr handlers before importing the app.
    # Strip them and let its records propagate through the root handler
    # so access/error lines share the stream and the JSON format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers[:] = []
        uv_logger.propagate = True

    lk_logger = logging.getLogger("livekit")
    if not any(isinstance(f, BurstDedupFilter) for f in lk_logger.filters):
        lk_logger.addFilter(BurstDedupFilter())
