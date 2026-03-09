"""Summarize results from a latency x n_action_steps (x staleness) sweep.

Usage:
    uv run python scripts/eval/summarize_latency_sweep.py data/robocasa/latency_sweep/
"""

from __future__ import annotations

import json
import pathlib
import re
import sys


def _fmt_val(value, fmt: str) -> str:
    """Format a value, returning '—' if None."""
    if value is None:
        return f"{'—':>10}"
    return f"{value:{fmt}}"


def _print_table(
    title: str,
    lookup: dict,
    tasks: list[str],
    latencies: list[int],
    nact_vals: list[int],
    staleness_vals: list[int],
    has_staleness: bool,
    metric_key: str,
    fmt: str = ">9.1%",
    avg_across_tasks: bool = False,
):
    """Print a 2D (latency x nact) or 3D (latency x staleness per nact) table."""
    if avg_across_tasks:
        # Averaged across tasks
        if has_staleness:
            for nact in nact_vals:
                print(f"\nn_action_steps={nact} — {title}:")
                header = f"{'Latency':>8}"
                for stale in staleness_vals:
                    header += f" {'s=' + str(stale):>10}"
                print(header)
                print("-" * 90)
                for lat in latencies:
                    row = f"{lat:>8}"
                    for stale in staleness_vals:
                        vals = [
                            lookup[(t, lat, nact, stale)][metric_key]
                            for t in tasks
                            if (t, lat, nact, stale) in lookup
                            and lookup[(t, lat, nact, stale)].get(metric_key) is not None
                        ]
                        if vals:
                            row += f" {sum(vals) / len(vals):{fmt}}"
                        else:
                            row += f" {'—':>10}"
                    print(row)
                print("-" * 90)
        else:
            print(f"\n{title}:")
            header = f"{'Latency':>8}"
            for nact in nact_vals:
                header += f" {'nact=' + str(nact):>10}"
            print(header)
            print("-" * 90)
            for lat in latencies:
                row = f"{lat:>8}"
                for nact in nact_vals:
                    vals = [
                        lookup[(t, lat, nact, 0)][metric_key]
                        for t in tasks
                        if (t, lat, nact, 0) in lookup
                        and lookup[(t, lat, nact, 0)].get(metric_key) is not None
                    ]
                    if vals:
                        row += f" {sum(vals) / len(vals):{fmt}}"
                    else:
                        row += f" {'—':>10}"
                print(row)
            print("-" * 90)
        return

    # Per-task tables
    for task in tasks:
        if has_staleness:
            for nact in nact_vals:
                print(f"\n{task} | n_action_steps={nact} — {title}:")
                header = f"{'Latency':>8}"
                for stale in staleness_vals:
                    header += f" {'s=' + str(stale):>10}"
                print(header)
                print("-" * 90)
                for lat in latencies:
                    row = f"{lat:>8}"
                    for stale in staleness_vals:
                        r = lookup.get((task, lat, nact, stale))
                        if r and r.get(metric_key) is not None:
                            row += f" {r[metric_key]:{fmt}}"
                        else:
                            row += f" {'—':>10}"
                    print(row)
                print("-" * 90)
        else:
            print(f"\n{task} — {title}:")
            header = f"{'Latency':>8}"
            for nact in nact_vals:
                header += f" {'nact=' + str(nact):>10}"
            print(header)
            print("-" * 90)
            for lat in latencies:
                row = f"{lat:>8}"
                for nact in nact_vals:
                    r = lookup.get((task, lat, nact, 0))
                    if r and r.get(metric_key) is not None:
                        row += f" {r[metric_key]:{fmt}}"
                    else:
                        row += f" {'—':>10}"
                print(row)
            print("-" * 90)


def main():
    results_dir = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else pathlib.Path("data/robocasa/latency_sweep")
    )

    result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print(f"No results_*.json files found in {results_dir}")
        sys.exit(1)

    all_results = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        # Extract fields from filename as fallback.
        task_match = re.match(r"results_(.+?)_lat\d+", path.name)
        lat_match = re.search(r"lat(\d+)", path.name)
        nact_match = re.search(r"nact(\d+)", path.name)
        stale_match = re.search(r"stale(\d+)", path.name)

        data.setdefault("task", task_match.group(1) if task_match else "unknown")
        data.setdefault("latency", int(lat_match.group(1)) if lat_match else 0)
        data.setdefault("n_action_steps", int(nact_match.group(1)) if nact_match else 0)
        if stale_match:
            data.setdefault("staleness", int(stale_match.group(1)))
        else:
            data.setdefault("staleness", 0)
        all_results.append(data)

    tasks = sorted(set(r["task"] for r in all_results))
    latencies = sorted(set(r["latency"] for r in all_results))
    nact_vals = sorted(set(r["n_action_steps"] for r in all_results))
    staleness_vals = sorted(set(r["staleness"] for r in all_results))
    has_staleness = len(staleness_vals) > 1 or staleness_vals != [0]

    # Build lookup: (task, latency, n_action_steps, staleness) -> result
    lookup = {
        (r["task"], r["latency"], r["n_action_steps"], r["staleness"]): r
        for r in all_results
    }

    print("=" * 90)
    print(f"Latency Sweep Summary — {len(tasks)} task(s), {len(all_results)} experiments")
    print("=" * 90)

    # Metrics to display: (title, key, format)
    metrics = [
        ("success rate", "success_rate", ">9.1%"),
        ("avg steps (all episodes)", "avg_steps_all", ">9.1f"),
        ("avg steps (succeeded)", "avg_steps_succeeded", ">9.1f"),
    ]

    for title, key, fmt in metrics:
        _print_table(
            title, lookup, tasks, latencies, nact_vals,
            staleness_vals, has_staleness, key, fmt,
        )

    # --- Cross-task average ---
    if len(tasks) > 1:
        print(f"\n{'AVERAGE across tasks'}")
        for title, key, fmt in metrics:
            _print_table(
                f"avg {title}", lookup, tasks, latencies, nact_vals,
                staleness_vals, has_staleness, key, fmt,
                avg_across_tasks=True,
            )

    print("=" * 90)


if __name__ == "__main__":
    main()
