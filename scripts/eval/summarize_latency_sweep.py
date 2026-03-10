"""Summarize results from a latency x n_action_steps (x staleness) sweep.

Usage:
    uv run python scripts/eval/summarize_latency_sweep.py data/robocasa/latency_sweep/
    uv run python scripts/eval/summarize_latency_sweep.py data/robocasa/latency_sweep/ \
        --staleness 0 --tasks CoffeeSetupMug_PandaOmron_Env
    uv run python scripts/eval/summarize_latency_sweep.py data/robocasa/latency_sweep/ \
        --latencies 0 5 10 --n-action-steps 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


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
    parser = argparse.ArgumentParser(description="Summarize latency sweep results.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="data/robocasa/latency_sweep",
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Filter to these task names (default: all)",
    )
    parser.add_argument(
        "--latencies", nargs="+", type=int, default=None,
        help="Filter to these latency values (default: all)",
    )
    parser.add_argument(
        "--n-action-steps", nargs="+", type=int, default=None,
        help="Filter to these n_action_steps values (default: all)",
    )
    parser.add_argument(
        "--staleness", nargs="+", type=int, default=None,
        help="Filter to these staleness values (default: all). Use --staleness 0 for sync only.",
    )
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print(f"No results_*.json files found in {results_dir}")
        raise SystemExit(1)

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

    # Apply filters.
    if args.tasks:
        all_results = [r for r in all_results if r["task"] in args.tasks]
    if args.latencies is not None:
        all_results = [r for r in all_results if r["latency"] in args.latencies]
    if args.n_action_steps is not None:
        all_results = [r for r in all_results if r["n_action_steps"] in args.n_action_steps]
    if args.staleness is not None:
        all_results = [r for r in all_results if r["staleness"] in args.staleness]

    if not all_results:
        print("No results match the given filters.")
        raise SystemExit(1)

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

    # Build filter description for header.
    filters = []
    if args.tasks:
        filters.append(f"tasks={args.tasks}")
    if args.latencies is not None:
        filters.append(f"latencies={args.latencies}")
    if args.n_action_steps is not None:
        filters.append(f"nact={args.n_action_steps}")
    if args.staleness is not None:
        filters.append(f"staleness={args.staleness}")
    filter_str = f" | filters: {', '.join(filters)}" if filters else ""

    print("=" * 90)
    print(
        f"Latency Sweep Summary — {len(tasks)} task(s), "
        f"{len(all_results)} experiments{filter_str}"
    )
    print("=" * 90)

    # Metrics to display: (title, key, format)
    metrics = [
        ("success rate (extended budget)", "success_rate_extended_budget", ">9.1%"),
        ("success rate (original budget)", "success_rate_original_budget", ">9.1%"),
        ("avg steps (all episodes)", "avg_steps_all", ">9.1f"),
        ("avg steps (succeeded)", "avg_steps_succeeded", ">9.1f"),
    ]

    # Backwards compat: old results may use "success_rate" instead of the new keys,
    # and avg_steps may be in macro steps (not raw) if total_action_steps_per_macro
    # is missing.
    for r in all_results:
        if "success_rate_extended_budget" not in r and "success_rate" in r:
            r["success_rate_extended_budget"] = r["success_rate"]
            r["success_rate_original_budget"] = r.get(
                "success_rate_original_budget", r["success_rate"]
            )
        # Old results stored macro-step counts; convert to raw if needed.
        if "episode_lengths" in r and "episode_lengths_raw" not in r:
            lat = r.get("latency", 0)
            stale = r.get("staleness", 0)
            nact = r.get("n_action_steps", 8)
            total = (lat - stale) + nact
            if total > nact and r.get("avg_steps_all") is not None:
                r["avg_steps_all"] = r["avg_steps_all"] * total
            if total > nact and r.get("avg_steps_succeeded") is not None:
                r["avg_steps_succeeded"] = r["avg_steps_succeeded"] * total

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
