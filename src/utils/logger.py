from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch.distributed as dist

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
        if not self._should_log():
            return
        line = json.dumps(payload, ensure_ascii=False)
        print(line)
        self.file.write(line + "\n")
        self.file.flush()

    @staticmethod
    def _should_log() -> bool:
        rank_env = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank_env is not None:
            try:
                if int(rank_env) != 0:
                    return False
            except ValueError:
                pass
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        try:
            return dist.get_rank() == 0
        except RuntimeError:
            return True

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
