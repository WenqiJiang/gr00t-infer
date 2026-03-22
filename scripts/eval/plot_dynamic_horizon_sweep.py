"""Plot success rate from dynamic-horizon sweep results.

Supports three plot types via --plot-type:

  fix_phase1       — (default) Fix pre-grasp (lat1, nact1), plot success rate
                     vs latency_2 with curves for each nact_2.
                     Shows how post-grasp settings affect performance.

  fix_phase2       — Fix post-grasp (lat2, nact2), plot success rate
                     vs latency_1 with curves for each nact_1.
                     Shows how pre-grasp settings affect performance.

  heatmap          — For fixed (nact1, nact2), show a 2D heatmap of
                     latency_1 x latency_2 → success rate.

  phase2_curves    — For a specific task and fixed (lat1, nact1),
                     plot success rate vs latency_2 with one curve
                     per nact_2 value. Shows how post-grasp action
                     horizon interacts with post-grasp latency.

Usage:
    # Default: all fix_phase1 plots:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/

    # Fix phase 1 to lat1=10, nact1=8, compare phase 2 settings:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \\
        --plot-type fix_phase1 --latency-1 10 --nact-1 8

    # Fix phase 2 to lat2=0, nact2=4, compare phase 1 settings:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \\
        --plot-type fix_phase2 --latency-2 0 --nact-2 4

    # Heatmap for nact1=8, nact2=4:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \\
        --plot-type heatmap --nact-1 8 --nact-2 4

    # Filter to specific tasks:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \\
        --tasks CoffeeSetupMug_PandaOmron_Env

    # Phase 2 curves: fix task + phase1, one curve per nact2:
    uv run python scripts/eval/plot_dynamic_horizon_sweep.py \\
        data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/ \\
        --plot-type phase2_curves \\
        --tasks PnPCounterToMicrowave_PandaOmron_Env \\
        --latency-1 16 --nact-1 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
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

        # Parse filename: results_{task}_lat1-{L}_nact1-{N}_lat2-{L}_nact2-{N}[_stale...].json
        task_match = re.match(r"results_(.+?)_lat1-\d+", path.name)
        lat1_match = re.search(r"lat1-(\d+)", path.name)
        nact1_match = re.search(r"nact1-(\d+)", path.name)
        lat2_match = re.search(r"lat2-(\d+)", path.name)
        nact2_match = re.search(r"nact2-(\d+)", path.name)
        stale1_match = re.search(r"stale1-(\d+)", path.name)
        stale2_match = re.search(r"stale2-(\d+)", path.name)

        data.setdefault("task", task_match.group(1) if task_match else "unknown")
        data.setdefault("latency_1", int(lat1_match.group(1)) if lat1_match else 0)
        data.setdefault("n_action_steps_1", int(nact1_match.group(1)) if nact1_match else 0)
        data.setdefault("latency_2", int(lat2_match.group(1)) if lat2_match else 0)
        data.setdefault("n_action_steps_2", int(nact2_match.group(1)) if nact2_match else 0)
        data.setdefault("staleness_1", int(stale1_match.group(1)) if stale1_match else 0)
        data.setdefault("staleness_2", int(stale2_match.group(1)) if stale2_match else 0)

        all_results.append(data)
    return all_results


def _save_or_show(fig, save_name: str, args: argparse.Namespace, output_dir: pathlib.Path):
    if args.show:
        plt.show()
    else:
        out = output_dir / save_name
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fix_phase1: Fix (task, lat1, nact1), bars = (nact2, lat2) combos, y = accuracy
# ---------------------------------------------------------------------------


def _plot_fix_phase1(all_results, args, output_dir):
    """Bar chart: one plot per (task, nact1, lat1). Each bar is a (nact2, lat2) combo."""
    tasks = sorted(set(r["task"] for r in all_results))
    lat1_vals = sorted(set(r["latency_1"] for r in all_results))
    nact1_vals = sorted(set(r["n_action_steps_1"] for r in all_results))

    if args.latency_1 is not None:
        lat1_vals = [v for v in lat1_vals if v in args.latency_1]
    if args.nact_1 is not None:
        nact1_vals = [n for n in nact1_vals if n in args.nact_1]

    lookup = {
        (r["task"], r["latency_1"], r["n_action_steps_1"],
         r["latency_2"], r["n_action_steps_2"]): r
        for r in all_results
    }

    # Collect unique (nact2, lat2) pairs, sorted for consistent ordering.
    phase2_pairs = sorted(set(
        (r["n_action_steps_2"], r["latency_2"]) for r in all_results
    ))

    for task in tasks:
        for nact1 in nact1_vals:
            for lat1 in lat1_vals:
                _fix_phase1_bar(
                    lookup, task, lat1, nact1, phase2_pairs,
                    args=args, output_dir=output_dir,
                )


def _fix_phase1_bar(lookup, task, lat1, nact1, phase2_pairs, args, output_dir):
    """Single bar chart for one (task, lat1, nact1)."""
    labels = []
    values = []
    colors = []
    baseline_pair = (nact1, lat1)

    for nact2, lat2 in phase2_pairs:
        r = lookup.get((task, lat1, nact1, lat2, nact2))
        if r is None or "success_rate_extended_budget" not in r:
            continue
        is_baseline = (nact2, lat2) == baseline_pair
        label = f"nact2={nact2}\nlat2={lat2}"
        if is_baseline:
            label += "\n(baseline)"
        labels.append(label)
        values.append(r["success_rate_extended_budget"] * 100)
        colors.append(_COLORS[2] if is_baseline else _COLORS[0])

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5.5))
    bars = ax.bar(x, values, width, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bars with values.
    for i, v in enumerate(values):
        ax.text(x[i], v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Phase 2 Config (nact2, lat2)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(bottom=0, top=min(max(values) + 15, 105) if values else 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{task}\nphase1: nact={nact1}, lat={lat1}", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_name = f"fix_phase1_{task}_nact1-{nact1}_lat1-{lat1}.png"
    _save_or_show(fig, save_name, args, output_dir)


# ---------------------------------------------------------------------------
# fix_phase2: Fix (lat2, nact2), x=lat1, curves=nact1
# ---------------------------------------------------------------------------


def _plot_fix_phase2(all_results, args, output_dir):
    """Fix post-grasp settings, compare pre-grasp (lat1, nact1)."""
    tasks = sorted(set(r["task"] for r in all_results))
    lat1_vals = sorted(set(r["latency_1"] for r in all_results))
    lat2_vals = sorted(set(r["latency_2"] for r in all_results))
    nact1_vals = sorted(set(r["n_action_steps_1"] for r in all_results))
    nact2_vals = sorted(set(r["n_action_steps_2"] for r in all_results))

    # Filter to requested phase 2 settings if specified.
    if args.latency_2 is not None:
        lat2_vals = [l for l in lat2_vals if l in args.latency_2]
    if args.nact_2 is not None:
        nact2_vals = [n for n in nact2_vals if n in args.nact_2]

    lookup = {
        (r["task"], r["latency_1"], r["n_action_steps_1"],
         r["latency_2"], r["n_action_steps_2"]): r
        for r in all_results
    }

    nact1_color = {n: _COLORS[i % len(_COLORS)] for i, n in enumerate(nact1_vals)}
    nact1_marker = {n: _MARKERS[i % len(_MARKERS)] for i, n in enumerate(nact1_vals)}

    for lat2 in lat2_vals:
        for nact2 in nact2_vals:
            for task in tasks:
                _fix_phase2_plot(
                    lookup, [task], lat1_vals, nact1_vals,
                    lat2, nact2, nact1_color, nact1_marker,
                    title=f"{task}\nphase2: lat={lat2}, nact={nact2}",
                    save_name=f"fix_phase2_{task}_lat2-{lat2}_nact2-{nact2}.png",
                    args=args, output_dir=output_dir,
                )

            if len(tasks) > 1:
                _fix_phase2_plot(
                    lookup, tasks, lat1_vals, nact1_vals,
                    lat2, nact2, nact1_color, nact1_marker,
                    title=(
                        f"Average across {len(tasks)} tasks\n"
                        f"phase2: lat={lat2}, nact={nact2}"
                    ),
                    save_name=f"fix_phase2_avg_lat2-{lat2}_nact2-{nact2}.png",
                    args=args, output_dir=output_dir,
                )


def _fix_phase2_plot(
    lookup, task_list, lat1_vals, nact1_vals,
    lat2, nact2, nact1_color, nact1_marker,
    title, save_name, args, output_dir,
):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    has_data = False
    for nact1 in nact1_vals:
        x_pts, y_pts = [], []
        for lat1 in lat1_vals:
            sr_vals = []
            for task in task_list:
                r = lookup.get((task, lat1, nact1, lat2, nact2))
                if r is not None and "success_rate_extended_budget" in r:
                    sr_vals.append(r["success_rate_extended_budget"])
            if sr_vals:
                x_pts.append(lat1)
                y_pts.append(np.mean(sr_vals))
        if x_pts:
            has_data = True
            ax.plot(
                x_pts, [v * 100 for v in y_pts],
                color=nact1_color[nact1], marker=nact1_marker[nact1],
                markersize=7, linewidth=2, label=f"nact1={nact1}",
            )

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Latency Phase 1 (steps)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_xticks(lat1_vals)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_name, args, output_dir)


# ---------------------------------------------------------------------------
# heatmap: For fixed (nact1, nact2), 2D lat1 x lat2 -> success rate
# ---------------------------------------------------------------------------


def _plot_heatmap(all_results, args, output_dir):
    """2D heatmap: latency_1 x latency_2 -> success rate, for each (nact1, nact2)."""
    tasks = sorted(set(r["task"] for r in all_results))
    lat1_vals = sorted(set(r["latency_1"] for r in all_results))
    lat2_vals = sorted(set(r["latency_2"] for r in all_results))
    nact1_vals = sorted(set(r["n_action_steps_1"] for r in all_results))
    nact2_vals = sorted(set(r["n_action_steps_2"] for r in all_results))

    if args.nact_1 is not None:
        nact1_vals = [n for n in nact1_vals if n in args.nact_1]
    if args.nact_2 is not None:
        nact2_vals = [n for n in nact2_vals if n in args.nact_2]

    lookup = {
        (r["task"], r["latency_1"], r["n_action_steps_1"],
         r["latency_2"], r["n_action_steps_2"]): r
        for r in all_results
    }

    for nact1 in nact1_vals:
        for nact2 in nact2_vals:
            # Per-task heatmaps
            for task in tasks:
                _heatmap_single(
                    lookup, [task], lat1_vals, lat2_vals, nact1, nact2,
                    title=f"{task}\nnact1={nact1}, nact2={nact2}",
                    save_name=f"heatmap_{task}_nact1-{nact1}_nact2-{nact2}.png",
                    args=args, output_dir=output_dir,
                )

            # Cross-task average heatmap
            if len(tasks) > 1:
                _heatmap_single(
                    lookup, tasks, lat1_vals, lat2_vals, nact1, nact2,
                    title=(
                        f"Average across {len(tasks)} tasks\n"
                        f"nact1={nact1}, nact2={nact2}"
                    ),
                    save_name=f"heatmap_avg_nact1-{nact1}_nact2-{nact2}.png",
                    args=args, output_dir=output_dir,
                )


def _heatmap_single(
    lookup, task_list, lat1_vals, lat2_vals, nact1, nact2,
    title, save_name, args, output_dir,
):
    grid = np.full((len(lat1_vals), len(lat2_vals)), np.nan)
    for i, lat1 in enumerate(lat1_vals):
        for j, lat2 in enumerate(lat2_vals):
            sr_vals = []
            for task in task_list:
                r = lookup.get((task, lat1, nact1, lat2, nact2))
                if r is not None and "success_rate_extended_budget" in r:
                    sr_vals.append(r["success_rate_extended_budget"])
            if sr_vals:
                grid[i, j] = np.mean(sr_vals) * 100

    if np.all(np.isnan(grid)):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        grid, cmap="RdYlGn", aspect="auto",
        vmin=0, vmax=100,
        origin="lower",
    )

    ax.set_xticks(range(len(lat2_vals)))
    ax.set_xticklabels(lat2_vals)
    ax.set_yticks(range(len(lat1_vals)))
    ax.set_yticklabels(lat1_vals)
    ax.set_xlabel("Latency Phase 2 (steps)", fontsize=12)
    ax.set_ylabel("Latency Phase 1 (steps)", fontsize=12)

    # Annotate cells with values.
    for i in range(len(lat1_vals)):
        for j in range(len(lat2_vals)):
            val = grid[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 30 or val > 80 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Success Rate (%)")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, save_name, args, output_dir)


# ---------------------------------------------------------------------------
# phase2_curves: Fix (task, lat1, nact1), x=lat2, one curve per nact2
# ---------------------------------------------------------------------------


def _plot_phase2_curves(all_results, args, output_dir):
    """Fix task + phase1 settings, plot x=lat2 with curves for each nact2."""
    tasks = sorted(set(r["task"] for r in all_results))
    lat1_vals = sorted(set(r["latency_1"] for r in all_results))
    nact1_vals = sorted(set(r["n_action_steps_1"] for r in all_results))
    nact2_vals = sorted(set(r["n_action_steps_2"] for r in all_results))
    lat2_vals = sorted(set(r["latency_2"] for r in all_results))

    if args.latency_1 is not None:
        lat1_vals = [l for l in lat1_vals if l in args.latency_1]
    if args.nact_1 is not None:
        nact1_vals = [n for n in nact1_vals if n in args.nact_1]

    lookup = {
        (r["task"], r["latency_1"], r["n_action_steps_1"],
         r["latency_2"], r["n_action_steps_2"]): r
        for r in all_results
    }

    nact2_color = {n: _COLORS[i % len(_COLORS)] for i, n in enumerate(nact2_vals)}
    nact2_marker = {n: _MARKERS[i % len(_MARKERS)] for i, n in enumerate(nact2_vals)}

    for task in tasks:
        for lat1 in lat1_vals:
            for nact1 in nact1_vals:
                fig, ax = plt.subplots(figsize=(9, 5.5))
                has_data = False

                for nact2 in nact2_vals:
                    x_pts, y_pts = [], []
                    for lat2 in lat2_vals:
                        r = lookup.get((task, lat1, nact1, lat2, nact2))
                        if r is not None and "success_rate_extended_budget" in r:
                            x_pts.append(lat2)
                            y_pts.append(r["success_rate_extended_budget"])
                    if x_pts:
                        has_data = True
                        ax.plot(
                            x_pts, [v * 100 for v in y_pts],
                            color=nact2_color[nact2], marker=nact2_marker[nact2],
                            markersize=7, linewidth=2, label=f"nact2={nact2}",
                        )

                if not has_data:
                    plt.close(fig)
                    continue

                ax.set_xlabel("Latency Phase 2 (steps)", fontsize=12)
                ax.set_ylabel("Success Rate (%)", fontsize=12)
                ax.set_ylim(bottom=0)
                ax.set_xticks(lat2_vals)
                ax.legend(loc="best", fontsize=10)
                ax.set_title(
                    f"{task}\nphase1: lat={lat1}, nact={nact1}",
                    fontsize=13,
                )
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                save_name = (
                    f"phase2_curves_{task}_lat1-{lat1}_nact1-{nact1}.png"
                )
                _save_or_show(fig, save_name, args, output_dir)


# ---------------------------------------------------------------------------
# time_breakdown: Stacked bar — avg phase1 vs phase2 time for succeeded episodes
# ---------------------------------------------------------------------------


def _plot_time_breakdown(all_results, args, output_dir):
    """Stacked bar: average time in phase1 / phase2 for succeeded episodes only."""
    tasks = sorted(set(r["task"] for r in all_results))

    # Apply optional filters on phase configs.
    filtered = all_results
    if args.nact_1 is not None:
        filtered = [r for r in filtered if r["n_action_steps_1"] in args.nact_1]
    if args.nact_2 is not None:
        filtered = [r for r in filtered if r["n_action_steps_2"] in args.nact_2]
    if args.latency_1 is not None:
        filtered = [r for r in filtered if r["latency_1"] in args.latency_1]
    if args.latency_2 is not None:
        filtered = [r for r in filtered if r["latency_2"] in args.latency_2]

    for task in tasks:
        task_results = [r for r in filtered if r["task"] == task]
        if not task_results:
            continue

        # Sort configs for consistent ordering.
        task_results.sort(
            key=lambda r: (
                r["n_action_steps_1"], r["latency_1"],
                r["n_action_steps_2"], r["latency_2"],
            )
        )

        labels = []
        phase1_times = []
        phase2_times = []

        for r in task_results:
            successes = r.get("episode_successes", [])
            t_before = r.get("time_before_switch", [])
            t_after = r.get("time_after_switch", [])
            if not successes or not t_before or not t_after:
                continue

            # Average only over succeeded episodes.
            t1_succ = [t for t, s in zip(t_before, successes) if s]
            t2_succ = [t for t, s in zip(t_after, successes) if s]
            if not t1_succ:
                continue

            labels.append(
                f"l1={r['latency_1']} n1={r['n_action_steps_1']}\n"
                f"l2={r['latency_2']} n2={r['n_action_steps_2']}"
            )
            phase1_times.append(np.mean(t1_succ))
            phase2_times.append(np.mean(t2_succ))

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.6

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 6))
        bars1 = ax.bar(x, phase1_times, width, label="Phase 1", color=_COLORS[0])
        bars2 = ax.bar(
            x, phase2_times, width, bottom=phase1_times,
            label="Phase 2", color=_COLORS[1],
        )

        # Annotate bars with values.
        for i in range(len(labels)):
            t1, t2 = phase1_times[i], phase2_times[i]
            ax.text(
                x[i], t1 / 2, f"{t1:.1f}s", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
            )
            ax.text(
                x[i], t1 + t2 / 2, f"{t2:.1f}s", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
            )

        ax.set_xlabel("Config", fontsize=12)
        ax.set_ylabel("Avg Time (s) — succeeded episodes", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(loc="best", fontsize=10)
        ax.set_title(f"{task}\nTime Breakdown (succeeded episodes)", fontsize=13)
        fig.tight_layout()
        _save_or_show(
            fig, f"time_breakdown_{task}.png", args, output_dir,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _plot_all(all_results, args, output_dir):
    """Run all plot types into the same output directory."""
    for name, fn in _PLOT_TYPES.items():
        if name == "all":
            continue
        print(f"--- {name} ---")
        fn(all_results, args, output_dir)


_PLOT_TYPES = {
    "all": _plot_all,
    "fix_phase1": _plot_fix_phase1,
    "fix_phase2": _plot_fix_phase2,
    "heatmap": _plot_heatmap,
    "phase2_curves": _plot_phase2_curves,
    "time_breakdown": _plot_time_breakdown,
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot dynamic-horizon sweep results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing results_*.json files"
        " (e.g. data/dynamic_horizon_sweep/robocasa_panda_omron/GR00T-N1.6-3B/trials50/)",
    )
    parser.add_argument(
        "--plot-type",
        choices=list(_PLOT_TYPES.keys()),
        default="all",
        help="Type of plot to produce (default: all)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Filter to these task names",
    )
    parser.add_argument(
        "--latency-1", nargs="+", type=int, default=None,
        help="Filter to these phase 1 latency values (for fix_phase1)",
    )
    parser.add_argument(
        "--nact-1", nargs="+", type=int, default=None,
        help="Filter to these phase 1 n_action_steps values",
    )
    parser.add_argument(
        "--latency-2", nargs="+", type=int, default=None,
        help="Filter to these phase 2 latency values (for fix_phase2)",
    )
    parser.add_argument(
        "--nact-2", nargs="+", type=int, default=None,
        help="Filter to these phase 2 n_action_steps values",
    )
    parser.add_argument(
        "--output-dir", default="plots/dynamic_latency",
        help="Directory to save plots (default: plots/dynamic_latency/)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument(
        "--show", action="store_true", help="Show plots interactively instead of saving",
    )
    args = parser.parse_args()

    all_results = _load_results(pathlib.Path(args.results_dir))

    # Apply common filters.
    if args.tasks:
        all_results = [r for r in all_results if r["task"] in args.tasks]

    if not all_results:
        raise SystemExit("No results match the given filters.")

    output_dir = pathlib.Path(args.output_dir)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(set(r["task"] for r in all_results))
    configs = sorted(set(
        (r["latency_1"], r["n_action_steps_1"], r["latency_2"], r["n_action_steps_2"])
        for r in all_results
    ))
    print(f"Loaded {len(all_results)} result(s) across {len(tasks)} task(s), "
          f"{len(configs)} config(s). Generating {args.plot_type} plots ...\n")

    _PLOT_TYPES[args.plot_type](all_results, args, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
