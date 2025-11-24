from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_log(path: Path) -> List[Dict[str, float]]:
    data: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not data:
        raise ValueError(f"No valid JSON lines found in {path}")
    return data


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    smoothed: List[float] = []
    acc = 0.0
    for idx, val in enumerate(values):
        acc += val
        if idx >= window:
            acc -= values[idx - window]
            smoothed.append(acc / window)
        elif idx == window - 1:
            smoothed.append(acc / window)
    if len(smoothed) < len(values):
        padding = [smoothed[-1]] * (len(values) - len(smoothed))
        smoothed.extend(padding)
    return smoothed


def plot_metrics(
    entries: List[Dict[str, float]],
    metrics: List[str],
    smooth: int,
    output: Path | None,
) -> None:
    steps = [entry.get("step", idx + 1) for idx, entry in enumerate(entries)]
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        values = [entry.get(metric) for entry in entries]
        if any(v is None for v in values):
            print(f"[WARN] Metric '{metric}' missing in some entries, skipping.")
            continue
        smoothed = moving_average(values, smooth)
        plt.plot(steps, smoothed, label=f"{metric} (window={smooth})")
    plt.xlabel("Training Update")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics from JSON log.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to train.log (JSON lines).")
    parser.add_argument(
        "--metrics",
        type=str,
        default="episode_reward,policy_loss,value_loss",
        help="Comma-separated metric names to visualize.",
    )
    parser.add_argument("--smooth", type=int, default=10, help="Moving-average window size.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the plot.")
    args = parser.parse_args()

    entries = load_log(args.log_file)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    plot_metrics(entries, metrics, max(1, args.smooth), args.output)


if __name__ == "__main__":
    main()
