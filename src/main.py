from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except Exception:  # pragma: no cover
    def record(fn):  # type: ignore[no-redef]
        return fn

from src.algorithms.onpolicy_baseline import run_onpolicy_smac
from src.utils import load_config, set_seed


@record
def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO/VITA on SMAC.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    algo_name = str(cfg.get("algorithm", "mappo")).lower()
    env_name = str(cfg.get("env", {}).get("name", "smac")).lower()
    if env_name != "smac":
        raise ValueError("Only SMAC environments are currently supported.")

    run_dir = Path(args.log_dir) / str(algo_name)
    run_onpolicy_smac(cfg, config_path=Path(args.config), run_dir=run_dir)


if __name__ == "__main__":
    main()
