"""Append-only JSON-lines event log: a durable, greppable record of what diskid did.

One JSON object per line, e.g.

    {"ts": "2026-11-14T17:30:02+00:00", "event": "file_moved", "source": "...", "target": "..."}

Unlike stdout/journald it is never rotated away, and `jq`/`grep` work on it
directly. Write failures never interrupt the work being logged.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("diskid.events")


class EventLog:
    def __init__(self, path: Path | None):
        """`path=None` makes a no-op log."""
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "event": event, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            log.warning("could not write event log %s: %s", self.path, e)
