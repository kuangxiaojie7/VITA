from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

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
    if window <= 1 or len(values) <= window:
        return values
    smoothed: List[float] = []
    acc = 0.0
    for idx, val in enumerate(values):
        acc += val
        if idx >= window:
            acc -= values[idx - window]
        if idx >= window - 1:
            smoothed.append(acc / window)
    # Pad the beginning with the first computed average to keep lengths matched
    if smoothed:
        padding = [smoothed[0]] * (len(values) - len(smoothed))
        return padding + smoothed
    return values


def _format_axes(metric: str, args) -> None:
    if args.winrate_style and "win_rate" in metric:
        plt.xlabel("T (mil)")
        plt.ylabel("Test Win Rate%")
        plt.ylim(0, 100)
        plt.xlim(left=0)
    else:
        plt.xlabel("Training Update")
        plt.ylabel(metric)


def plot_metric(
    metric: str,
    runs: Sequence[Dict[str, object]],
    smooth: int,
    output_dir: Path | None,
    args,
) -> None:
    plt.figure(figsize=(10, 5))
    if args.winrate_style:
        plt.gca().set_facecolor("#f9f9f9")
        plt.grid(True, alpha=0.4, color="#cccccc")
    for run in runs:
        entries: List[Dict[str, float]] = run["entries"]  # type: ignore
        filtered_steps: List[float] = []
        filtered_values: List[float] = []
        for idx, entry in enumerate(entries):
            if args.phase != "all":
                phase = entry.get("phase")
                if phase is None:
                    phase = "eval" if any(k.startswith("eval_") for k in entry.keys()) else "train"
                if phase != args.phase:
                    continue
            val = entry.get(metric)
            if val is None:
                continue
            step_val = entry.get("step", idx + 1)
            if args.winrate_style and "win_rate" in metric:
                step_val = step_val * args.timesteps_per_update / 1e6
                val = val * 100.0
            filtered_steps.append(step_val)
            filtered_values.append(val)
        if args.winrate_style and "win_rate" in metric and filtered_values and filtered_steps[0] > 0:
            # Ensure the curve anchors at the true origin for win-rate plots.
            filtered_steps.insert(0, 0.0)
            filtered_values.insert(0, 0.0)
        if not filtered_values:
            print(f"[WARN] '{metric}' not present for run {run['label']}, skipping.")
            continue
        # For win-rate plots, left-pad zeros to let smoothing start from origin smoothly.
        if args.winrate_style and "win_rate" in metric and smooth > 1:
            pad_len = smooth - 1
            padded_vals = [0.0] * pad_len + filtered_values
            padded_steps = [0.0] * pad_len + filtered_steps
            smoothed = moving_average(padded_vals, smooth)
            # Drop the padded prefix so lengths match the original data.
            smoothed = smoothed[pad_len:]
            x_vals = padded_steps[pad_len : pad_len + len(smoothed)]
            if x_vals:
                x_vals[0] = 0.0
                smoothed[0] = 0.0
        else:
            smoothed = moving_average(filtered_values, smooth)
            x_vals = filtered_steps[: len(smoothed)]
        line_color = run.get("color")
        line = plt.plot(
            x_vals,
            smoothed,
            linewidth=2,
            label=f"{run['label']} (window={smooth})",
            color=line_color,
        )[0]
        if args.winrate_style and not args.no_fill:
            plt.fill_between(x_vals, smoothed, alpha=args.fill_alpha, color=line.get_color())
    plt.title(args.title or metric)
    _format_axes(metric, args)
    if not args.winrate_style:
        plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{metric}.png"
        plt.savefig(out_path, dpi=200)
        print(f"Saved {out_path}")
        plt.close()
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics for multiple runs.")
    parser.add_argument(
        "--log-files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more train.log paths (JSON lines).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        help="Optional labels for each log file. Defaults to filename stem.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional matplotlib colors for each log file (e.g. '#1f77b4', 'tab:orange'). "
            "Must match the number of --log-files."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="episode_reward,policy_loss,value_loss,incre_win_rate,eval_win_rate",
        help="Comma-separated metric names to visualize.",
    )
    parser.add_argument("--smooth", type=int, default=10, help="Moving-average window size.")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override (defaults to metric name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If provided, saves each metric figure to this directory.",
    )
    parser.add_argument(
        "--winrate-style",
        action="store_true",
        help="Format plots similar to SMAC win-rate figures (T mil on x-axis, percentage on y-axis).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "train", "eval"],
        default="all",
        help="Filter log entries by phase (newer logs include `phase`; older logs are inferred).",
    )
    parser.add_argument(
        "--timesteps-per-update",
        type=float,
        default=1.0,
        help="Environment timesteps represented by one training update (used when --winrate-style).",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Disable the filled area under curves in --winrate-style plots.",
    )
    parser.add_argument(
        "--fill-alpha",
        type=float,
        default=0.15,
        help="Alpha used for the filled area (ignored when --no-fill).",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.log_files):
        raise ValueError("Number of labels must match number of log files.")
    if args.colors and len(args.colors) != len(args.log_files):
        raise ValueError("Number of colors must match number of log files.")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    runs = []
    for idx, log_path in enumerate(args.log_files):
        entries = load_log(log_path)
        label = args.labels[idx] if args.labels else log_path.stem
        color = args.colors[idx] if args.colors else None
        runs.append({"label": label, "entries": entries, "color": color})

    for metric in metrics:
        plot_metric(metric, runs, max(1, args.smooth), args.output_dir, args)


if __name__ == "__main__":
    main()
