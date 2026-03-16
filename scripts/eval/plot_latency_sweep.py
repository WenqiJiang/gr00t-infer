"""Plot success rate and average finish time vs latency from sweep results.

Supports three plot types via --plot-type:

  per_task          — (default) Per-task dual-axis plot with success rate
                      (extended + original budget) and avg steps vs latency.
                      One plot per (task, nact), sync only (staleness=0).

  compare_nact      — Compare action chunk sizes. Each curve is a different
                      n_action_steps value. Left Y = success rate (extended
                      budget), right Y = avg steps (succeeded). Sync only.

  compare_staleness — Compare sync vs async inference. Each curve is a
                      different staleness value (0 = sync, >0 = async).
                      Left Y = success rate (extended budget), right Y =
                      avg steps (succeeded).

Usage:
    # Default: generate all plot types:
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/

    # Only per-task plots (sync, one plot per task x nact):
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --plot-type per_task

    # Compare action chunk sizes (sync only):
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --plot-type compare_nact

    # Compare sync vs async (staleness curves):
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --plot-type compare_staleness --n-action-steps 8

    # Filter to specific tasks:
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --plot-type compare_nact --tasks CoffeeSetupMug_PandaOmron_Env

    # Save to custom directory:
    uv run python scripts/eval/plot_latency_sweep.py data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \
        --output-dir plots/custom/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Distinct markers for multi-curve plots.
_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "h", "<", ">"]
# Qualitative color palette (tab10-ish but explicit hex for consistency).
_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _load_results(results_dir: pathlib.Path) -> list[dict]:
    """Load all results_*.json files, extracting metadata from filenames."""
    result_files = sorted(results_dir.glob("results_*.json"))
    if not result_files:
        raise SystemExit(f"No results_*.json files found in {results_dir}")

    all_results = []
    for path in result_files:
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

        # Backwards compat: old results may use "success_rate" only.
        if "success_rate_extended_budget" not in data and "success_rate" in data:
            data["success_rate_extended_budget"] = data["success_rate"]
            data.setdefault("success_rate_original_budget", data["success_rate"])
        all_results.append(data)
    return all_results


def _print_usage(args: argparse.Namespace) -> None:
    """Print a summary of the invocation and available plot types."""
    script = "scripts/eval/plot_latency_sweep.py"
    print("=" * 80)
    print(f"Plot Latency Sweep — plot-type: {args.plot_type}")
    print("=" * 80)
    print()
    print("Available plot types (--plot-type):")
    print(f"  {'all':<22} Generate all plot types below (default)")
    print(f"  {'per_task':<22} Per-task success rate + avg steps vs latency (sync only)")
    print(f"  {'compare_nact':<22} Compare action chunk sizes vs latency (sync only)")
    print(f"  {'compare_staleness':<22} Compare sync vs async (staleness curves) vs latency")
    print()
    rd = "data/latency_sweep/<env_prefix>/<model>/<trials>"
    print("Examples:")
    print(f"  uv run python {script} {rd}")
    print(f"  uv run python {script} {rd} --plot-type compare_nact")
    print(f"  uv run python {script} {rd} --plot-type compare_staleness --n-action-steps 8")
    print(f"  uv run python {script} {rd} --tasks CoffeeSetupMug_PandaOmron_Env")
    print(f"  uv run python {script} {rd} --output-dir plots/custom/")
    print()
    filters = []
    if args.tasks:
        filters.append(f"tasks={args.tasks}")
    if args.latencies is not None:
        filters.append(f"latencies={args.latencies}")
    if args.n_action_steps is not None:
        filters.append(f"n-action-steps={args.n_action_steps}")
    if args.staleness is not None:
        filters.append(f"staleness={args.staleness}")
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Filters     : {', '.join(filters) if filters else '(none)'}")
    print("-" * 80)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_or_show(fig, save_name: str, args: argparse.Namespace, output_dir: pathlib.Path):
    if args.show:
        plt.show()
    else:
        out = output_dir / save_name
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def _make_per_task_plot(
    x: np.ndarray,
    sr_ext: np.ndarray,
    sr_orig: np.ndarray,
    ft: np.ndarray,
    title: str,
    save_name: str,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
):
    """Dual-axis plot: success rate (extended + original) left, avg steps right."""
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color_sr_ext = "#2563eb"  # blue
    color_sr_orig = "#16a34a"  # green
    color_ft = "#dc2626"  # red

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

    ax2 = ax1.twinx()
    ax2.plot(
        x, ft, color=color_ft, marker="s", markersize=7,
        linewidth=2, label="Avg Steps (succeeded)",
    )
    ax2.set_ylabel("Avg Steps (succeeded episodes)", color=color_ft, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color_ft)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=10)

    ax1.set_title(title, fontsize=13)
    ax1.set_xticks(x)
    ax1.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_name, args, output_dir)


def _make_multi_curve_plot(
    x: np.ndarray,
    sr_curves: list[tuple],
    ft_curves: list[tuple],
    title: str,
    save_name: str,
    args: argparse.Namespace,
    output_dir: pathlib.Path,
):
    """Dual-axis plot with multiple labelled curves.

    *x* is used for x-tick positions. Each curve entry is one of:
      (label, y_array)                          — shared x, auto color/marker
      (label, y_array, color, marker)           — shared x, explicit color/marker
      (label, x_array, y_array, color, marker)  — per-curve x, explicit color/marker

    Success-rate curves use solid lines, avg-steps curves use dashed lines.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    n_series = max(len(sr_curves), len(ft_curves))

    def _unpack(entry, idx):
        """Return (label, x_vals, y_vals, color, marker) from a curve entry."""
        if len(entry) == 5:
            return entry[0], entry[1], entry[2], entry[3], entry[4]
        elif len(entry) == 4:
            return entry[0], x, entry[1], entry[2], entry[3]
        else:
            return (
                entry[0], x, entry[1],
                _COLORS[idx % len(_COLORS)], _MARKERS[idx % len(_MARKERS)],
            )

    # Left axis — success rate
    for i, entry in enumerate(sr_curves):
        label, cx, y, color, marker = _unpack(entry, i)
        ax1.plot(
            cx, y * 100,
            color=color, marker=marker,
            markersize=7, linewidth=2, label=label,
        )
    ax1.set_xlabel("Latency (steps)", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12)
    ax1.tick_params(axis="y")
    ax1.set_ylim(bottom=0)

    # Right axis — avg steps
    ax2 = ax1.twinx()
    for i, entry in enumerate(ft_curves):
        label, cx, y, color, marker = _unpack(entry, i)
        ax2.plot(
            cx, y,
            color=color, marker=marker,
            markersize=7, linewidth=2, linestyle="--", label=label,
        )
    ax2.set_ylabel("Avg Steps (succeeded episodes)", fontsize=12)
    ax2.tick_params(axis="y")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="best", fontsize=9, ncol=2 if n_series > 4 else 1,
    )

    ax1.set_title(title, fontsize=13)
    ax1.set_xticks(x)
    ax1.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_name, args, output_dir)


# ---------------------------------------------------------------------------
# Metric gathering
# ---------------------------------------------------------------------------


def _gather_metrics_single(
    lookup: dict,
    task_list: list[str],
    lat_list: list[int],
    nact: int,
    staleness: int = 0,
    require_all_tasks: bool = False,
):
    """Collect avg-across-tasks metrics for each latency at a fixed (nact, staleness).

    Returns (valid_latencies, sr_ext, sr_orig, ft) as numpy arrays.
    """
    valid_lats, sr_ext_m, sr_orig_m, ft_m = [], [], [], []
    for lat in lat_list:
        sr_ext_vals, sr_orig_vals, ft_vals = [], [], []
        skip = False
        for t in task_list:
            r = lookup.get((t, lat, nact, staleness))
            if r is None:
                if require_all_tasks:
                    skip = True
                    break
                continue
            v = r.get("success_rate_extended_budget")
            if v is not None:
                sr_ext_vals.append(v)
            elif require_all_tasks:
                skip = True
                break
            v = r.get("success_rate_original_budget")
            if v is not None:
                sr_orig_vals.append(v)
            v = r.get("avg_steps_succeeded")
            if v is not None:
                ft_vals.append(v)
        if skip:
            continue
        if sr_ext_vals or sr_orig_vals:
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


# ---------------------------------------------------------------------------
# Plot-type implementations
# ---------------------------------------------------------------------------


def _plot_per_task(all_results, args, output_dir):
    """Original behaviour: one dual-axis plot per (task, nact), sync only."""
    results = [r for r in all_results if r["staleness"] == 0]
    if not results:
        raise SystemExit("No synchronous (staleness=0) results match the given filters.")

    tasks = sorted(set(r["task"] for r in results))
    latencies = sorted(set(r["latency"] for r in results))
    nact_vals = sorted(set(r["n_action_steps"] for r in results))
    lookup = {
        (r["task"], r["latency"], r["n_action_steps"], r["staleness"]): r for r in results
    }

    for nact in nact_vals:
        for task in tasks:
            x, sr_ext, sr_orig, ft = _gather_metrics_single(
                lookup, [task], latencies, nact, staleness=0,
            )
            if len(x) == 0:
                continue
            _make_per_task_plot(
                x, sr_ext, sr_orig, ft,
                title=f"{task}\nnact={nact}, staleness=0",
                save_name=f"latency_sweep_{task}_nact{nact}.png",
                args=args, output_dir=output_dir,
            )

        if len(tasks) > 1:
            x, sr_ext, sr_orig, ft = _gather_metrics_single(
                lookup, tasks, latencies, nact, staleness=0, require_all_tasks=True,
            )
            if len(x) == 0:
                continue
            _make_per_task_plot(
                x, sr_ext, sr_orig, ft,
                title=f"Average across {len(tasks)} tasks\nnact={nact}, staleness=0",
                save_name=f"latency_sweep_avg_nact{nact}.png",
                args=args, output_dir=output_dir,
            )


def _plot_compare_nact(all_results, args, output_dir):
    """Compare action chunk sizes: one curve per nact, sync only."""
    results = [r for r in all_results if r["staleness"] == 0]
    if not results:
        raise SystemExit("No synchronous (staleness=0) results match the given filters.")

    tasks = sorted(set(r["task"] for r in results))
    latencies = sorted(set(r["latency"] for r in results))
    nact_vals = sorted(set(r["n_action_steps"] for r in results), reverse=True)
    lookup = {
        (r["task"], r["latency"], r["n_action_steps"], r["staleness"]): r for r in results
    }

    # Build a global color/marker mapping so nact=16 always gets the same
    # color regardless of which tasks have data for it.
    nact_color = {n: _COLORS[i % len(_COLORS)] for i, n in enumerate(nact_vals)}
    nact_marker = {n: _MARKERS[i % len(_MARKERS)] for i, n in enumerate(nact_vals)}

    if len(nact_vals) < 2:
        print(
            f"WARNING: only one n_action_steps value ({nact_vals}) found. "
            "compare_nact is most useful with multiple values."
        )

    def _make(task_list, label_prefix, save_prefix, require_all):
        # Use the union of latencies so each curve plots all points it has.
        all_lats = set()
        series_data = {}
        for nact in nact_vals:
            x, sr_ext, _, ft = _gather_metrics_single(
                lookup, task_list, latencies, nact, staleness=0,
                require_all_tasks=require_all,
            )
            if len(x) == 0:
                continue
            series_data[nact] = (x, sr_ext, ft)
            all_lats.update(x.tolist())

        if not all_lats or not series_data:
            return
        all_lats_arr = np.array(sorted(all_lats))

        sr_curves, ft_curves = [], []
        for nact in nact_vals:
            if nact not in series_data:
                continue
            x, sr_ext, ft = series_data[nact]
            sr_curves.append((f"nact={nact} SR(ext)", x, sr_ext, nact_color[nact],
                              nact_marker[nact]))
            ft_curves.append((f"nact={nact} AvgSteps", x, ft, nact_color[nact],
                              nact_marker[nact]))

        _make_multi_curve_plot(
            all_lats_arr, sr_curves, ft_curves,
            title=f"{label_prefix}\nCompare n_action_steps, staleness=0",
            save_name=f"{save_prefix}_compare_nact.png",
            args=args, output_dir=output_dir,
        )

    # Per-task
    for task in tasks:
        _make([task], task, f"latency_sweep_{task}", require_all=False)

    # Cross-task average
    if len(tasks) > 1:
        _make(
            tasks, f"Average across {len(tasks)} tasks",
            "latency_sweep_avg", require_all=True,
        )


def _plot_compare_staleness(all_results, args, output_dir):
    """Compare sync vs async: one curve per staleness value."""
    results = list(all_results)  # no staleness filter here
    if not results:
        raise SystemExit("No results match the given filters.")

    tasks = sorted(set(r["task"] for r in results))
    latencies = sorted(set(r["latency"] for r in results))
    nact_vals = sorted(set(r["n_action_steps"] for r in results))
    staleness_vals = sorted(set(r["staleness"] for r in results))
    lookup = {
        (r["task"], r["latency"], r["n_action_steps"], r["staleness"]): r for r in results
    }

    if len(staleness_vals) < 2:
        print(
            f"WARNING: only one staleness value ({staleness_vals}) found. "
            "compare_staleness is most useful with multiple values."
        )

    def _make(task_list, nact, label_prefix, save_prefix, require_all):
        # Collect per-staleness data; use the union of latencies so each
        # curve plots all points it has (not just the common subset).
        all_lats = set()
        series_data = {}
        for stale in staleness_vals:
            x, sr_ext, _, ft = _gather_metrics_single(
                lookup, task_list, latencies, nact, staleness=stale,
                require_all_tasks=require_all,
            )
            if len(x) == 0:
                continue
            series_data[stale] = (x, sr_ext, ft)
            all_lats.update(x.tolist())

        if not all_lats or not series_data:
            return
        all_lats_arr = np.array(sorted(all_lats))

        sr_curves, ft_curves = [], []
        for stale in staleness_vals:
            if stale not in series_data:
                continue
            x, sr_ext, ft = series_data[stale]
            tag = "sync" if stale == 0 else f"async(stale={stale})"
            i = staleness_vals.index(stale)
            color = _COLORS[i % len(_COLORS)]
            marker = _MARKERS[i % len(_MARKERS)]
            sr_curves.append((f"{tag} SR(ext)", x, sr_ext, color, marker))
            ft_curves.append((f"{tag} AvgSteps", x, ft, color, marker))

        _make_multi_curve_plot(
            all_lats_arr, sr_curves, ft_curves,
            title=f"{label_prefix}\nnact={nact}, sync vs async",
            save_name=f"{save_prefix}_nact{nact}_compare_staleness.png",
            args=args, output_dir=output_dir,
        )

    for nact in nact_vals:
        for task in tasks:
            _make([task], nact, task, f"latency_sweep_{task}", require_all=False)

        if len(tasks) > 1:
            _make(
                tasks, nact, f"Average across {len(tasks)} tasks",
                "latency_sweep_avg", require_all=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _plot_all(all_results, args, output_dir):
    """Run all plot types."""
    print("--- per_task ---")
    _plot_per_task(all_results, args, output_dir)
    print("\n--- compare_nact ---")
    _plot_compare_nact(all_results, args, output_dir)
    print("\n--- compare_staleness ---")
    _plot_compare_staleness(all_results, args, output_dir)


_PLOT_TYPES = {
    "all": _plot_all,
    "per_task": _plot_per_task,
    "compare_nact": _plot_compare_nact,
    "compare_staleness": _plot_compare_staleness,
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot latency sweep results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing results_*.json files"
        " (e.g. data/latency_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/)",
    )
    parser.add_argument(
        "--plot-type",
        choices=list(_PLOT_TYPES.keys()),
        default="all",
        help="Type of plot to produce (default: all)",
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
        "--staleness", nargs="+", type=int, default=None,
        help="Filter to these staleness values (only used by compare_staleness)",
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

    # Print usage / invocation summary.
    _print_usage(args)

    all_results = _load_results(pathlib.Path(args.results_dir))

    # Apply common filters.
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

    output_dir = pathlib.Path(args.output_dir)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(all_results)} result(s). Generating {args.plot_type} plots ...\n")

    _PLOT_TYPES[args.plot_type](all_results, args, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
