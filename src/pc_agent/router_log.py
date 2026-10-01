from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RouterLogger:
    def __init__(self, path: Path, enabled: bool):
        self.path = path
        self.enabled = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_decision: dict[str, Any] | None = None

    def log(self, event: str, **data: Any) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **data}
        if event in {"route", "escalation"}:
            self.last_decision = record
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

