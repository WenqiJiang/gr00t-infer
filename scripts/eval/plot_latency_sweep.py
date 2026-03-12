"""Plot success rate and average finish time vs latency from sweep results.

Produces a dual-axis plot (left: success rate for both original and extended
budgets, right: avg steps for succeeded episodes) for synchronous inference
(staleness=0).  Output defaults to plots/ directory.

Usage:
    uv run python scripts/eval/plot_latency_sweep.py data/robocasa/latency_sweep/
    uv run python scripts/eval/plot_latency_sweep.py data/robocasa/latency_sweep/ \
        --tasks CoffeeSetupMug_PandaOmron_Env --n-action-steps 8
    uv run python scripts/eval/plot_latency_sweep.py data/robocasa/latency_sweep/ \
        --output-dir plots/custom/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


def _load_results(results_dir: pathlib.Path) -> list[dict]:
    """Load all results_*.json files, extracting metadata from filenames."""
    result_files = sorted(results_dir.glob("results_*.json"))
    if not result_files:
        raise SystemExit(f"No results_*.json files found in {results_dir}")

    all_results = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        task_match = re.match(r"results_(.+?)_lat\d+", path.name)
        lat_match = re.search(r"lat(\d+)", path.name)
        nact_match = re.search(r"nact(\d+)", path.name)
        stale_match = re.search(r"stale(\d+)", path.name)

        data.setdefault("task", task_match.group(1) if task_match else "unknown")
        data.setdefault("latency", int(lat_match.group(1)) if lat_match else 0)
        data.setdefault("n_action_steps", int(nact_match.group(1)) if nact_match else 0)
        data.setdefault("staleness", int(stale_match.group(1)) if stale_match else 0)

        # Backwards compat: old results may use "success_rate" only.
        if "success_rate_extended_budget" not in data and "success_rate" in data:
            data["success_rate_extended_budget"] = data["success_rate"]
            data.setdefault("success_rate_original_budget", data["success_rate"])
        all_results.append(data)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Plot latency sweep results.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="data/robocasa/latency_sweep",
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Filter to these task names (default: all)"
    )
    parser.add_argument(
        "--latencies", nargs="+", type=int, default=None, help="Filter to these latency values"
    )
    parser.add_argument(
        "--n-action-steps", nargs="+", type=int, default=None, help="Filter to these nact values"
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots (default: plots/)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument(
        "--show", action="store_true", help="Show plots interactively instead of saving"
    )
    args = parser.parse_args()

    all_results = _load_results(pathlib.Path(args.results_dir))

    # Only synchronous (staleness=0).
    all_results = [r for r in all_results if r["staleness"] == 0]

    if args.tasks:
        all_results = [r for r in all_results if r["task"] in args.tasks]
    if args.latencies is not None:
        all_results = [r for r in all_results if r["latency"] in args.latencies]
    if args.n_action_steps is not None:
        all_results = [r for r in all_results if r["n_action_steps"] in args.n_action_steps]

    if not all_results:
        raise SystemExit("No results match the given filters.")

    tasks = sorted(set(r["task"] for r in all_results))
    latencies = sorted(set(r["latency"] for r in all_results))
    nact_vals = sorted(set(r["n_action_steps"] for r in all_results))

    # Build lookup: (task, latency, nact) -> result
    lookup = {(r["task"], r["latency"], r["n_action_steps"]): r for r in all_results}

    output_dir = pathlib.Path(args.output_dir)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _make_plot(
        x: np.ndarray,
        sr_ext: np.ndarray,
        sr_orig: np.ndarray,
        ft: np.ndarray,
        title: str,
        save_name: str,
    ):
        """Create and save/show a single dual-axis plot."""
        fig, ax1 = plt.subplots(figsize=(9, 5.5))

        color_sr_ext = "#2563eb"  # blue
        color_sr_orig = "#16a34a"  # green
        color_ft = "#dc2626"  # red

        # Left axis: success rates
        ax1.plot(
            x, sr_ext * 100, color=color_sr_ext, marker="o", markersize=7,
            linewidth=2, label="Success Rate (extended budget)",
        )
        ax1.plot(
            x, sr_orig * 100, color=color_sr_orig, marker="^", markersize=7,
            linewidth=2, label="Success Rate (original budget)",
        )
        ax1.set_xlabel("Latency (steps)", fontsize=12)
        ax1.set_ylabel("Success Rate (%)", fontsize=12)
        ax1.tick_params(axis="y")
        ax1.set_ylim(bottom=0)

        # Right axis: avg finish time (succeeded)
        ax2 = ax1.twinx()
        ax2.plot(
            x, ft, color=color_ft, marker="s", markersize=7,
            linewidth=2, label="Avg Steps (succeeded)",
        )
        ax2.set_ylabel("Avg Steps (succeeded episodes)", color=color_ft, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color_ft)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

        ax1.set_title(title, fontsize=13)
        ax1.set_xticks(x)
        ax1.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        if args.show:
            plt.show()
        else:
            out = output_dir / save_name
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)

    def _gather_metrics(
        task_list: list[str],
        lat_list: list[int],
        nact: int,
        require_all_tasks: bool = False,
    ):
        """Collect (avg across task_list) metrics for each latency.

        If *require_all_tasks* is True, a latency point is only included when
        every task in *task_list* has data for it.  This avoids biasing the
        average when some tasks are missing at certain latencies.
        """
        valid_lats, sr_ext_m, sr_orig_m, ft_m = [], [], [], []
        for lat in lat_list:
            sr_ext_vals, sr_orig_vals, ft_vals = [], [], []
            for t in task_list:
                r = lookup.get((t, lat, nact))
                if r is None:
                    if require_all_tasks:
                        break  # skip this latency entirely
                    continue
                v = r.get("success_rate_extended_budget")
                if v is not None:
                    sr_ext_vals.append(v)
                elif require_all_tasks:
                    break
                v = r.get("success_rate_original_budget")
                if v is not None:
                    sr_orig_vals.append(v)
                v = r.get("avg_steps_succeeded")
                if v is not None:
                    ft_vals.append(v)
            else:
                # Only reached if the for-loop didn't break.
                if sr_ext_vals or sr_orig_vals:
                    valid_lats.append(lat)
                    sr_ext_m.append(np.mean(sr_ext_vals) if sr_ext_vals else np.nan)
                    sr_orig_m.append(np.mean(sr_orig_vals) if sr_orig_vals else np.nan)
                    ft_m.append(np.mean(ft_vals) if ft_vals else np.nan)
                continue
            # break was hit — skip this latency when require_all_tasks
            if not require_all_tasks and (sr_ext_vals or sr_orig_vals):
                valid_lats.append(lat)
                sr_ext_m.append(np.mean(sr_ext_vals) if sr_ext_vals else np.nan)
                sr_orig_m.append(np.mean(sr_orig_vals) if sr_orig_vals else np.nan)
                ft_m.append(np.mean(ft_vals) if ft_vals else np.nan)
        return (
            np.array(valid_lats),
            np.array(sr_ext_m),
            np.array(sr_orig_m),
            np.array(ft_m),
        )

    for nact in nact_vals:
        # --- Per-task plots ---
        for task in tasks:
            x, sr_ext, sr_orig, ft = _gather_metrics([task], latencies, nact)
            if len(x) == 0:
                continue
            _make_plot(
                x, sr_ext, sr_orig, ft,
                title=f"{task}\nnact={nact}, staleness=0",
                save_name=f"latency_sweep_{task}_nact{nact}.png",
            )

        # --- Cross-task average (only when >1 task) ---
        if len(tasks) > 1:
            x, sr_ext, sr_orig, ft = _gather_metrics(
                tasks, latencies, nact, require_all_tasks=True,
            )
            if len(x) == 0:
                continue
            _make_plot(
                x, sr_ext, sr_orig, ft,
                title=f"Average across {len(tasks)} tasks\nnact={nact}, staleness=0",
                save_name=f"latency_sweep_avg_nact{nact}.png",
            )


if __name__ == "__main__":
    main()
