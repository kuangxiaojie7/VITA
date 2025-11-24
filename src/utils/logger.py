from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any


class Logger:
    """Minimal logger that prints to stdout and appends JSON lines to disk."""

    def __init__(self, log_dir: str | os.PathLike):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file = (self.log_dir / "train.log").open("a", encoding="utf-8")

    def log(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        payload = {"time": time.time(), **metrics}
        if step is not None:
            payload["step"] = step
        line = json.dumps(payload, ensure_ascii=False)
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
