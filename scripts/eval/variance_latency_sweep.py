"""Show accuracy distribution across seeds for a latency sweep.

Loads results from multiple seed subdirectories, prints per-task and
cross-task summary tables (mean +/- std), and generates box/violin plots
with per-seed values annotated.

Usage:
    uv run python scripts/eval/variance_latency_sweep.py \
        data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/

    # Filter to specific tasks or latencies:
    uv run python scripts/eval/variance_latency_sweep.py \
        data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --tasks TurnOnStove_PandaOmron_Env --latencies 0 4 8 16

    # Use box plots instead of violin plots:
    uv run python scripts/eval/variance_latency_sweep.py \
        data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --plot-style box

    # Custom output directory:
    uv run python scripts/eval/variance_latency_sweep.py \
        data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --output-dir plots/variance/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_all_seeds(base_dir: pathlib.Path) -> list[dict]:
    """Load results from all seed*/ subdirectories under base_dir."""
    seed_dirs = sorted(base_dir.glob("seed*"))
    if not seed_dirs:
        raise SystemExit(f"No seed*/ subdirectories found in {base_dir}")

    all_results = []
    for seed_dir in seed_dirs:
        seed_match = re.search(r"seed(\d+)", seed_dir.name)
        seed_id = int(seed_match.group(1)) if seed_match else 0

        for path in sorted(seed_dir.glob("results_*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping corrupt file {path}: {e}")
                continue

            task_match = re.match(r"results_(.+?)_lat\d+", path.name)
            lat_match = re.search(r"lat(\d+)", path.name)
            nact_match = re.search(r"nact(\d+)", path.name)
            stale_match = re.search(r"stale(\d+)", path.name)

            data.setdefault("task", task_match.group(1) if task_match else "unknown")
            data.setdefault("latency", int(lat_match.group(1)) if lat_match else 0)
            data.setdefault("n_action_steps", int(nact_match.group(1)) if nact_match else 0)
            data.setdefault("staleness", int(stale_match.group(1)) if stale_match else 0)
            data["seed"] = seed_id

            # Backwards compat
            if "success_rate_extended_budget" not in data and "success_rate" in data:
                data["success_rate_extended_budget"] = data["success_rate"]
                data.setdefault("success_rate_original_budget", data["success_rate"])

            all_results.append(data)

    return all_results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _compute_seed_vals(
    all_results: list[dict],
    metric_key: str,
) -> dict[tuple, list[float]]:
    """Build lookup: (task, latency, nact) -> list of metric values across seeds."""
    seed_vals: dict[tuple, list[float]] = {}
    for r in all_results:
        key = (r["task"], r["latency"], r["n_action_steps"])
        val = r.get(metric_key)
        if val is not None:
            seed_vals.setdefault(key, []).append(val)
    return seed_vals


def _compute_cross_task_avg(
    all_results: list[dict],
    tasks: list[str],
    lat: int,
    nact: int,
    metric_key: str,
) -> list[float]:
    """Return per-seed cross-task average for a given (lat, nact)."""
    seeds_set = sorted(set(r["seed"] for r in all_results))
    per_seed_avg = []
    for seed in seeds_set:
        task_vals = []
        for task in tasks:
            matching = [
                r[metric_key]
                for r in all_results
                if r["task"] == task
                and r["latency"] == lat
                and r["n_action_steps"] == nact
                and r["seed"] == seed
                and r.get(metric_key) is not None
            ]
            if matching:
                task_vals.append(matching[0])
        if len(task_vals) == len(tasks):
            per_seed_avg.append(np.mean(task_vals))
    return per_seed_avg


def _print_single_table(
    all_results: list[dict],
    tasks: list[str],
    latencies: list[int],
    nact_vals: list[int],
    metric_key: str,
    metric_label: str,
    seed_vals: dict[tuple, list[float]],
    n_sd: int,
):
    """Print one table at a given SD multiplier."""
    ci_approx = {1: "~68%", 2: "~95%", 3: "~99.7%"}.get(n_sd, f"~{n_sd}SD")
    sd_label = f"{n_sd} SD" if n_sd > 1 else "1 SD"
    print(f"\n{metric_label} | mean +/- {sd_label} ({ci_approx} of runs):")

    for nact in nact_vals:
        print(f"  nact={nact}:")
        header = f"  {'Task':<43}"
        for lat in latencies:
            header += f" {'lat=' + str(lat):>14}"
        print(header)
        print("  " + "-" * (43 + 15 * len(latencies)))

        for task in tasks:
            row = f"  {task:<43}"
            for lat in latencies:
                vals = seed_vals.get((task, lat, nact), [])
                if vals:
                    mean = np.mean(vals) * 100
                    spread = (np.std(vals, ddof=1) * 100 * n_sd) if len(vals) > 1 else 0.0
                    row += f" {mean:5.1f}+/-{spread:4.1f}%"
                else:
                    row += f" {'--':>14}"
            print(row)

        # Cross-task average
        if len(tasks) > 1:
            row = f"  {'** AVG across tasks **':<43}"
            for lat in latencies:
                per_seed_avg = _compute_cross_task_avg(
                    all_results, tasks, lat, nact, metric_key
                )
                if per_seed_avg:
                    mean = np.mean(per_seed_avg) * 100
                    spread = (
                        np.std(per_seed_avg, ddof=1) * 100 * n_sd
                        if len(per_seed_avg) > 1
                        else 0.0
                    )
                    row += f" {mean:5.1f}+/-{spread:4.1f}%"
                else:
                    row += f" {'--':>14}"
            print(row)

        print("  " + "-" * (43 + 15 * len(latencies)))


def _print_tables(
    all_results: list[dict],
    tasks: list[str],
    latencies: list[int],
    nact_vals: list[int],
    metric_key: str,
    metric_label: str,
):
    """Print per-task and cross-task tables at 1 SD and 2 SD."""
    seed_vals = _compute_seed_vals(all_results, metric_key)

    _print_single_table(
        all_results, tasks, latencies, nact_vals,
        metric_key, metric_label, seed_vals, n_sd=1,
    )
    _print_single_table(
        all_results, tasks, latencies, nact_vals,
        metric_key, metric_label, seed_vals, n_sd=2,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _make_variance_plot(
    task_label: str,
    latencies: list[int],
    seed_data: dict[int, list[float]],
    nact: int,
    metric_label: str,
    save_path: pathlib.Path,
    plot_style: str = "violin",
    dpi: int = 150,
    n_episodes: int | None = None,
):
    """Create a box or violin plot for one task (or cross-task average).

    seed_data: {latency: [value_per_seed, ...]}  (values in [0, 1])
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(max(8, len(latencies) * 1.2), 6))

    positions = list(range(len(latencies)))
    data_lists = [np.array(seed_data.get(lat, [])) * 100 for lat in latencies]

    if plot_style == "violin":
        # Need at least 2 points for a violin; fall back to box for singletons
        parts = None
        valid_pos = [i for i, d in enumerate(data_lists) if len(d) >= 2]
        valid_data = [data_lists[i] for i in valid_pos]
        if valid_data:
            parts = ax.violinplot(
                valid_data, positions=valid_pos,
                showmeans=True, showmedians=True, showextrema=False,
            )
            # Style the violins
            if parts:
                for pc in parts.get("bodies", []):
                    pc.set_facecolor("#93c5fd")
                    pc.set_edgecolor("#2563eb")
                    pc.set_alpha(0.6)
                if "cmeans" in parts:
                    parts["cmeans"].set_color("#dc2626")
                    parts["cmeans"].set_linewidth(2)
                if "cmedians" in parts:
                    parts["cmedians"].set_color("#2563eb")
                    parts["cmedians"].set_linewidth(2)

        # Overlay individual seed points (jittered)
        for i, d in enumerate(data_lists):
            if len(d) > 0:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
                ax.scatter(
                    np.full(len(d), i) + jitter, d,
                    color="#2563eb", alpha=0.5, s=18, zorder=3,
                )
    else:  # box
        bp = ax.boxplot(
            data_lists, positions=positions, widths=0.5,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="#dc2626", markeredgecolor="#dc2626",
                           markersize=6),
            medianprops=dict(color="#2563eb", linewidth=2),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#93c5fd")
            patch.set_alpha(0.6)

        # Overlay individual seed points (jittered)
        for i, d in enumerate(data_lists):
            if len(d) > 0:
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
                ax.scatter(
                    np.full(len(d), i) + jitter, d,
                    color="#2563eb", alpha=0.5, s=18, zorder=3,
                )

    # Annotate mean, 1SD, and 2SD on top of each box/violin
    for i, d in enumerate(data_lists):
        if len(d) == 0:
            continue
        mean = np.mean(d)
        std = np.std(d, ddof=1) if len(d) > 1 else 0.0
        y_top = max(d) + 2.5
        ax.text(
            i, y_top,
            f"{mean:.1f}%\n1SD={std:.1f}\n2SD={2 * std:.1f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            color="#1e3a5f",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(lat) for lat in latencies], fontsize=10)
    ax.set_xlabel("Latency (steps)", fontsize=12)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=12)

    # Title with seed count and episodes per run
    n_seeds = len(seed_data.get(latencies[0], []))
    title = f"{task_label}\nnact={nact}, {n_seeds} seeds"
    if n_episodes is not None:
        title += f", {n_episodes} episodes/run"
    ax.set_title(title, fontsize=13)

    ax.set_ylim(bottom=0, top=min(105, ax.get_ylim()[1] + 8))
    ax.grid(axis="y", alpha=0.3)

    # Legend for the two horizontal bars and annotation
    legend_handles = [
        Line2D([0], [0], color="#dc2626", linewidth=2, label="Mean"),
        Line2D([0], [0], color="#2563eb", linewidth=2, label="Median"),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#2563eb",
            alpha=0.5, markersize=6, label="Individual seeds",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def _get_n_episodes(all_results: list[dict]) -> int | None:
    """Extract n_episodes from the first result that has it."""
    for r in all_results:
        if "n_episodes" in r:
            return r["n_episodes"]
    return None


def _generate_plots(
    all_results: list[dict],
    tasks: list[str],
    latencies: list[int],
    nact_vals: list[int],
    metric_key: str,
    metric_label: str,
    output_dir: pathlib.Path,
    plot_style: str,
    dpi: int,
):
    """Generate one plot per (task, nact) + cross-task average."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_episodes = _get_n_episodes(all_results)

    for nact in nact_vals:
        for task in tasks:
            # Collect per-latency seed values
            seed_data: dict[int, list[float]] = {}
            for lat in latencies:
                vals = sorted(
                    r[metric_key]
                    for r in all_results
                    if r["task"] == task
                    and r["latency"] == lat
                    and r["n_action_steps"] == nact
                    and r.get(metric_key) is not None
                )
                if vals:
                    seed_data[lat] = vals

            if not seed_data:
                continue

            save_name = f"variance_{task}_nact{nact}.png"
            _make_variance_plot(
                task, latencies, seed_data, nact, metric_label,
                output_dir / save_name, plot_style, dpi, n_episodes,
            )

        # Cross-task average
        if len(tasks) > 1:
            seeds_set = sorted(set(r["seed"] for r in all_results))
            seed_data: dict[int, list[float]] = {}
            for lat in latencies:
                per_seed_avg = []
                for seed in seeds_set:
                    task_vals = []
                    for task in tasks:
                        matching = [
                            r[metric_key]
                            for r in all_results
                            if r["task"] == task
                            and r["latency"] == lat
                            and r["n_action_steps"] == nact
                            and r["seed"] == seed
                            and r.get(metric_key) is not None
                        ]
                        if matching:
                            task_vals.append(matching[0])
                    if len(task_vals) == len(tasks):
                        per_seed_avg.append(np.mean(task_vals))
                if per_seed_avg:
                    seed_data[lat] = per_seed_avg

            if seed_data:
                _make_variance_plot(
                    f"Average across {len(tasks)} tasks",
                    latencies, seed_data, nact, metric_label,
                    output_dir / f"variance_avg_nact{nact}.png",
                    plot_style, dpi, n_episodes,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Show accuracy distribution across seeds for a latency sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing seed*/ subdirectories "
        "(e.g. data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Filter to these task names (default: all)",
    )
    parser.add_argument(
        "--latencies", nargs="+", type=int, default=None,
        help="Filter to these latency values (default: all)",
    )
    parser.add_argument(
        "--n-action-steps", nargs="+", type=int, default=None,
        help="Filter to these nact values (default: all)",
    )
    parser.add_argument(
        "--staleness", nargs="+", type=int, default=None,
        help="Filter to these staleness values (default: all)",
    )
    parser.add_argument(
        "--metric",
        choices=["extended", "original"],
        default="extended",
        help="Which success rate metric to use (default: extended)",
    )
    parser.add_argument(
        "--plot-style",
        choices=["violin", "box"],
        default="violin",
        help="Plot style: violin or box (default: violin)",
    )
    parser.add_argument(
        "--output-dir", default="plots/variance",
        help="Directory to save plots (default: plots/variance/)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument(
        "--no-plot", action="store_true", help="Only print tables, skip plot generation",
    )
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    all_results = _load_all_seeds(base_dir)

    # Filter to sync only by default
    all_results = [r for r in all_results if r["staleness"] == 0]

    # Apply filters
    if args.tasks:
        all_results = [r for r in all_results if r["task"] in args.tasks]
    if args.latencies is not None:
        all_results = [r for r in all_results if r["latency"] in args.latencies]
    if args.n_action_steps is not None:
        all_results = [r for r in all_results if r["n_action_steps"] in args.n_action_steps]
    if args.staleness is not None:
        all_results = [r for r in all_results if r["staleness"] in args.staleness]

    if not all_results:
        raise SystemExit("No results match the given filters.")

    tasks = sorted(set(r["task"] for r in all_results))
    latencies = sorted(set(r["latency"] for r in all_results))
    nact_vals = sorted(set(r["n_action_steps"] for r in all_results))
    seeds = sorted(set(r["seed"] for r in all_results))

    if args.metric == "original":
        metric_key = "success_rate_original_budget"
        metric_label = "Success Rate (original budget)"
    else:
        metric_key = "success_rate_extended_budget"
        metric_label = "Success Rate (extended budget)"

    n_episodes = _get_n_episodes(all_results)

    print("=" * 90)
    print(f"Variance Analysis — {len(seeds)} seeds, {len(tasks)} task(s), "
          f"{len(all_results)} experiments")
    print(f"  Seeds:       {seeds}")
    print(f"  Tasks:       {tasks}")
    print(f"  Latencies:   {latencies}")
    print(f"  nact:        {nact_vals}")
    print(f"  Metric:      {metric_label}")
    if n_episodes is not None:
        print(f"  Episodes/run:{n_episodes}")
    print("=" * 90)

    # Print summary tables
    _print_tables(all_results, tasks, latencies, nact_vals, metric_key, metric_label)

    # Generate plots
    if not args.no_plot:
        output_dir = pathlib.Path(args.output_dir)
        print(f"\nGenerating {args.plot_style} plots -> {output_dir}/\n")
        _generate_plots(
            all_results, tasks, latencies, nact_vals,
            metric_key, metric_label, output_dir, args.plot_style, args.dpi,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
