from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.algorithms import MAPPOTrainer, VITATrainer
from src.envs import make_smac_env
from src.utils import Logger, load_config, set_seed


def build_trainer(name: str, env, cfg, logger: Logger, device: torch.device):
    name = name.lower()
    if name == "mappo":
        return MAPPOTrainer(env, cfg["model"], cfg["train"], logger, device)
    if name == "vita":
        return VITATrainer(env, cfg["model"], cfg["train"], logger, device)
    raise ValueError(f"Unsupported algorithm: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO/VITA on SMAC.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    algo_name = cfg.get("algorithm", "mappo")
    env_name = cfg.get("env", {}).get("name", "smac")
    if env_name != "smac":
        raise ValueError("Only SMAC environments are currently supported.")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("cuda", True) else "cpu")
    env = make_smac_env(cfg["env"])
    log_dir = Path(args.log_dir) / algo_name
    logger = Logger(log_dir)

    trainer = build_trainer(algo_name, env, cfg, logger, device)
    try:
        trainer.train()
    finally:
        env.close()
        logger.close()


if __name__ == "__main__":
    main()
