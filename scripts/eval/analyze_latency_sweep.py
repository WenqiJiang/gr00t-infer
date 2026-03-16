"""Statistical analysis of latency sweep results.

Analyses (sync case, staleness=0):
  1. Spearman rank correlation of success rate vs latency per (task, nact).
  2. Wilson score confidence intervals for each success rate data point.

Usage:
    uv run python scripts/eval/analyze_latency_sweep.py \
        data/robocasa/latency_sweep/trials50/

    # Filter to specific nact / tasks:
    uv run python scripts/eval/analyze_latency_sweep.py \
        data/robocasa/latency_sweep/trials50/ \
        --n-action-steps 8 --tasks CoffeeSetupMug_PandaOmron_Env

    # Save CI plots:
    uv run python scripts/eval/analyze_latency_sweep.py \
        data/robocasa/latency_sweep/trials50/ --plot --output-dir plots/analysis/
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading (reuses pattern from plot/summarize scripts)
# ---------------------------------------------------------------------------


def _load_results(results_dir: pathlib.Path) -> list[dict]:
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

        # Backwards compat
        if "success_rate_extended_budget" not in data and "success_rate" in data:
            data["success_rate_extended_budget"] = data["success_rate"]
            data.setdefault("success_rate_original_budget", data["success_rate"])
        all_results.append(data)
    return all_results


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------


def wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (center, lower, upper) where center is the Wilson point estimate
    and [lower, upper] is the (1-alpha) CI.  z=1.96 gives a 95 % CI.
    """
    if n_total == 0:
        return (0.0, 0.0, 0.0)
    p_hat = n_success / n_total
    z2 = z * z
    denom = 1 + z2 / n_total
    center = (p_hat + z2 / (2 * n_total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n_total)) / n_total) / denom
    return (center, max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze(all_results: list[dict], args: argparse.Namespace):
    # Filter to sync only
    results = [r for r in all_results if r["staleness"] == 0]
    if not results:
        raise SystemExit("No synchronous (staleness=0) results found.")

    tasks = sorted(set(r["task"] for r in results))
    nact_vals = sorted(set(r["n_action_steps"] for r in results))
    lookup = {(r["task"], r["latency"], r["n_action_steps"]): r for r in results}

    # For both extended and original budget metrics
    metric_keys = [
        ("success_rate_extended_budget", "SR (extended budget)"),
        ("success_rate_original_budget", "SR (original budget)"),
    ]

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("LEGEND")
    print("=" * 90)
    print()
    print("Spearman's rho (ρ): rank correlation in [-1, +1].")
    print("  Measures whether success rate monotonically increases or decreases")
    print("  with latency (no linearity assumption, robust to outliers).")
    print("    ρ ≈ +1 : success rate consistently rises with latency")
    print("    ρ ≈ -1 : success rate consistently falls with latency")
    print("    ρ ≈  0 : no consistent directional trend")
    print()
    print("p-value: probability of observing |ρ| this large under the null")
    print("  hypothesis of NO monotonic relationship.")
    print("    p < 0.05 : statistically significant (unlikely due to chance)")
    print("    p < 0.10 : marginal / suggestive")
    print("    p ≥ 0.10 : not significant (trend may be noise)")
    print()
    print("Wilson 95% CI: confidence interval for a binomial proportion.")
    print("  With n=50 trials, typical CI width is ~15-25% for mid-range rates.")
    print("  Overlapping CIs between two conditions means their difference is")
    print("  NOT statistically significant.")
    print()

    # -----------------------------------------------------------------------
    # Helper: compute Spearman rho/pval for a (task, nact, metric) combo
    # -----------------------------------------------------------------------
    def _spearman_for(task, nact, metric_key):
        """Return (rho, pval, n_points) or None if insufficient data."""
        latencies = sorted(
            lat for (t, lat, na) in lookup if t == task and na == nact
        )
        if len(latencies) < 3:
            return None
        sr_vals = []
        for lat in latencies:
            r = lookup.get((task, lat, nact))
            if r is None or r.get(metric_key) is None:
                sr_vals.append(np.nan)
            else:
                sr_vals.append(r[metric_key])
        lat_arr = np.array(latencies, dtype=float)
        sr_arr = np.array(sr_vals)
        mask = ~np.isnan(sr_arr)
        if mask.sum() < 3:
            return None
        rho, pval = stats.spearmanr(lat_arr[mask], sr_arr[mask])
        return (rho, pval, int(mask.sum()))

    def _spearman_avg(tasks, nact, metric_key):
        """Return (rho, pval, n_points) for cross-task average, or None."""
        all_lats = sorted(set(lat for (t, lat, na) in lookup if na == nact))
        if len(all_lats) < 3:
            return None
        avg_sr, valid_lats = [], []
        for lat in all_lats:
            vals = [
                lookup[(t, lat, nact)][metric_key]
                for t in tasks
                if (t, lat, nact) in lookup
                and lookup[(t, lat, nact)].get(metric_key) is not None
            ]
            if vals:
                avg_sr.append(np.mean(vals))
                valid_lats.append(lat)
        if len(valid_lats) < 3:
            return None
        rho, pval = stats.spearmanr(valid_lats, avg_sr)
        return (rho, pval, len(valid_lats))

    def _interp(rho, pval):
        if np.isnan(rho) or np.isnan(pval):
            return "  n/a"
        if pval < 0.05:
            return "* SIG increase" if rho > 0 else "* SIG decrease"
        elif pval < 0.10:
            return "~ marginal"
        else:
            return "  no sig trend"

    def _interp_short(rho, pval):
        """Short symbol for the cross-nact consistency table."""
        if np.isnan(rho) or np.isnan(pval):
            return "  —  "
        if pval < 0.05:
            return " ↑↑↑ " if rho > 0 else " ↓↓↓ "
        elif pval < 0.10:
            return " ↑↑  " if rho > 0 else " ↓↓  "
        else:
            if rho > 0.3:
                return " ↑   "
            elif rho < -0.3:
                return " ↓   "
            return " ·   "

    # -----------------------------------------------------------------------
    # 1. Spearman correlation table
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("SPEARMAN RANK CORRELATION: success_rate vs latency (sync only)")
    print("  rho > 0 => higher latency tends to IMPROVE success rate")
    print("  rho < 0 => higher latency tends to HURT success rate")
    print("  p-value tests H0: no monotonic relationship")
    print("=" * 90)

    for metric_key, metric_label in metric_keys:
        print(f"\n--- {metric_label} ---")
        print(f"{'Task':<45} {'nact':>5} {'N':>4} {'rho':>7} {'p-val':>8}  Interpretation")
        print("-" * 90)

        for nact in nact_vals:
            for task in tasks:
                result = _spearman_for(task, nact, metric_key)
                if result is None:
                    continue
                rho, pval, n = result
                print(
                    f"{task:<45} {nact:>5} {n:>4} "
                    f"{rho:>+7.3f} {pval:>8.4f}  {_interp(rho, pval)}"
                )

            if len(tasks) > 1:
                result = _spearman_avg(tasks, nact, metric_key)
                if result is not None:
                    rho, pval, n = result
                    print(
                        f"{'** AVG across tasks **':<45} {nact:>5} "
                        f"{n:>4} {rho:>+7.3f} {pval:>8.4f}  {_interp(rho, pval)}"
                    )
        print("-" * 90)

    # -----------------------------------------------------------------------
    # 1b. Cross-nact consistency table (extended budget only)
    # -----------------------------------------------------------------------
    metric_key_ext = "success_rate_extended_budget"
    if len(nact_vals) > 1:
        print("\n" + "=" * 90)
        print("CROSS-NACT CONSISTENCY: SR (extended budget) trend across n_action_steps")
        print("  ↓↓↓ = significant decrease (p<.05)   ↑↑↑ = significant increase (p<.05)")
        print("  ↓↓  = marginal decrease (p<.10)      ↑↑  = marginal increase (p<.10)")
        print("  ↓   = weak negative (|ρ|>.3)         ↑   = weak positive (|ρ|>.3)")
        print("  ·   = no trend (|ρ|<.3)              —   = insufficient data")
        print("  ✓ = consistent trend: all ρ have the same sign across ALL nact,")
        print("      AND at least one is significant (p<.05)")
        print("=" * 90)

        header = f"{'Task':<45}"
        for nact in nact_vals:
            header += f" {'nact=' + str(nact):>8}"
        header += "  Consistent?"
        print(header)
        print("-" * (50 + 9 * len(nact_vals) + 14))

        def _consistency_tag(entries):
            """Check if all rho have same sign and at least one is significant.

            entries: list of (rho, pval) tuples.
            Returns '✓' if consistent, '' otherwise.
            """
            if len(entries) < 2:
                return ""
            all_pos = all(rho > 0 for rho, _ in entries)
            all_neg = all(rho < 0 for rho, _ in entries)
            any_sig = any(pval < 0.05 for _, pval in entries)
            if (all_pos or all_neg) and any_sig:
                return "✓"
            return ""

        for task in tasks:
            row = f"{task:<45}"
            entries = []  # collect (rho, pval) for consistency check
            for nact in nact_vals:
                result = _spearman_for(task, nact, metric_key_ext)
                if result is None:
                    row += f" {'—':>8}"
                else:
                    rho, pval, _ = result
                    row += f" {_interp_short(rho, pval):>8}"
                    if not np.isnan(rho) and not np.isnan(pval):
                        entries.append((rho, pval))

            row += f"  {_consistency_tag(entries)}"
            print(row)

        # AVG across tasks row
        if len(tasks) > 1:
            row = f"{'** AVG across tasks **':<45}"
            entries = []
            for nact in nact_vals:
                result = _spearman_avg(tasks, nact, metric_key_ext)
                if result is None:
                    row += f" {'—':>8}"
                else:
                    rho, pval, _ = result
                    row += f" {_interp_short(rho, pval):>8}"
                    if not np.isnan(rho) and not np.isnan(pval):
                        entries.append((rho, pval))
            row += f"  {_consistency_tag(entries)}"
            print(row)

        print("-" * (50 + 9 * len(nact_vals) + 14))

    # -----------------------------------------------------------------------
    # 2. Wilson confidence intervals table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("WILSON 95% CONFIDENCE INTERVALS: success_rate per (task, latency, nact)")
    print("  Overlapping CIs => difference is NOT statistically significant")
    print("=" * 90)

    for metric_key, metric_label in metric_keys:
        # Determine the successes field to use
        if metric_key == "success_rate_original_budget":
            successes_key = "episode_successes_original_budget"
        else:
            successes_key = None  # will compute from rate * n

        print(f"\n--- {metric_label} ---")
        for nact in nact_vals:
            for task in tasks:
                latencies = sorted(
                    lat for (t, lat, na) in lookup if t == task and na == nact
                )
                if not latencies:
                    continue

                print(f"\n  {task} | nact={nact}:")
                print(
                    f"  {'Latency':>8}  {'n':>4}  {'k':>4}  "
                    f"{'p_hat':>7}  {'Wilson':>7}  {'95% CI':>16}  {'CI width':>8}"
                )
                print("  " + "-" * 70)

                for lat in latencies:
                    r = lookup.get((task, lat, nact))
                    if r is None:
                        continue

                    n_total = r.get("n_episodes", 50)

                    # Try to get exact success count from episode list
                    if successes_key and successes_key in r:
                        k = sum(r[successes_key])
                    elif r.get(metric_key) is not None:
                        # Reconstruct from rate (may lose a fraction)
                        k = round(r[metric_key] * n_total)
                    else:
                        continue

                    p_hat = k / n_total
                    center, lo, hi = wilson_ci(k, n_total)

                    print(
                        f"  {lat:>8}  {n_total:>4}  {k:>4}  "
                        f"{p_hat:>7.1%}  {center:>7.1%}  "
                        f"[{lo:>6.1%}, {hi:>6.1%}]  {hi - lo:>7.1%}"
                    )
                print("  " + "-" * 70)

    # -----------------------------------------------------------------------
    # 3. Optional: CI plots
    # -----------------------------------------------------------------------
    if args.plot:
        _plot_with_ci(results, lookup, tasks, nact_vals, args)


# ---------------------------------------------------------------------------
# Plotting with CI error bars
# ---------------------------------------------------------------------------


def _plot_with_ci(
    results: list[dict],
    lookup: dict,
    tasks: list[str],
    nact_vals: list[int],
    args: argparse.Namespace,
):
    import matplotlib.pyplot as plt

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_key = "success_rate_extended_budget"

    for nact in nact_vals:
        for task in tasks:
            latencies = sorted(
                lat for (t, lat, na) in lookup if t == task and na == nact
            )
            if len(latencies) < 2:
                continue

            lats, p_hats, los, his = [], [], [], []
            for lat in latencies:
                r = lookup.get((task, lat, nact))
                if r is None or r.get(metric_key) is None:
                    continue
                n_total = r.get("n_episodes", 50)
                k = round(r[metric_key] * n_total)
                _, lo, hi = wilson_ci(k, n_total)
                lats.append(lat)
                p_hats.append(r[metric_key] * 100)
                los.append(r[metric_key] * 100 - lo * 100)
                his.append(hi * 100 - r[metric_key] * 100)

            lats = np.array(lats)
            p_hats = np.array(p_hats)

            # Spearman annotation
            if len(lats) >= 3:
                rho, pval = stats.spearmanr(lats, p_hats)
                trend_text = f"Spearman ρ={rho:+.3f}, p={pval:.3f}"
            else:
                trend_text = ""

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.errorbar(
                lats, p_hats, yerr=[los, his],
                fmt="o-", capsize=5, capthick=1.5, linewidth=2, markersize=7,
                color="#2563eb", ecolor="#93c5fd", label="SR (ext budget) ± 95% Wilson CI",
            )
            ax.set_xlabel("Latency (steps)", fontsize=12)
            ax.set_ylabel("Success Rate (%)", fontsize=12)
            ax.set_title(f"{task}\nnact={nact}, staleness=0", fontsize=13)
            ax.set_xticks(lats)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=10, loc="best")
            if trend_text:
                ax.annotate(
                    trend_text, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=10, color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                )
            fig.tight_layout()
            out = output_dir / f"ci_{task}_nact{nact}.png"
            fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {out}")
            plt.close(fig)

        # Cross-task average plot
        if len(tasks) > 1:
            all_lats = sorted(set(
                lat for (t, lat, na) in lookup if na == nact
            ))
            lats, p_hats, los, his = [], [], [], []
            for lat in all_lats:
                # Pool all episodes across tasks at this latency
                total_k, total_n = 0, 0
                for t in tasks:
                    r = lookup.get((t, lat, nact))
                    if r is None or r.get(metric_key) is None:
                        continue
                    n = r.get("n_episodes", 50)
                    total_k += round(r[metric_key] * n)
                    total_n += n
                if total_n == 0:
                    continue
                p = total_k / total_n
                _, lo, hi = wilson_ci(total_k, total_n)
                lats.append(lat)
                p_hats.append(p * 100)
                los.append(p * 100 - lo * 100)
                his.append(hi * 100 - p * 100)

            if len(lats) >= 2:
                lats = np.array(lats)
                p_hats = np.array(p_hats)

                if len(lats) >= 3:
                    rho, pval = stats.spearmanr(lats, p_hats)
                    trend_text = f"Spearman ρ={rho:+.3f}, p={pval:.3f}"
                else:
                    trend_text = ""

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.errorbar(
                    lats, p_hats, yerr=[los, his],
                    fmt="o-", capsize=5, capthick=1.5, linewidth=2, markersize=7,
                    color="#2563eb", ecolor="#93c5fd",
                    label="SR (ext budget) ± 95% Wilson CI (pooled)",
                )
                ax.set_xlabel("Latency (steps)", fontsize=12)
                ax.set_ylabel("Success Rate (%)", fontsize=12)
                ax.set_title(
                    f"Average across {len(tasks)} tasks\nnact={nact}, staleness=0",
                    fontsize=13,
                )
                ax.set_xticks(lats)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.3)
                ax.legend(fontsize=10, loc="best")
                if trend_text:
                    ax.annotate(
                        trend_text, xy=(0.02, 0.02), xycoords="axes fraction",
                        fontsize=10, color="gray",
                        bbox=dict(
                            boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                        ),
                    )
                fig.tight_layout()
                out = output_dir / f"ci_avg_nact{nact}.png"
                fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
                print(f"Saved: {out}")
                plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of latency sweep results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing results_*.json files",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="Filter to these task names",
    )
    parser.add_argument(
        "--latencies", nargs="+", type=int, default=None, help="Filter to these latency values",
    )
    parser.add_argument(
        "--n-action-steps", nargs="+", type=int, default=None, help="Filter to these nact values",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate CI plots",
    )
    parser.add_argument(
        "--output-dir", default="plots/analysis", help="Directory to save plots (default: plots/analysis/)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    args = parser.parse_args()

    all_results = _load_results(pathlib.Path(args.results_dir))

    # Apply filters
    if args.tasks:
        all_results = [r for r in all_results if r["task"] in args.tasks]
    if args.latencies is not None:
        all_results = [r for r in all_results if r["latency"] in args.latencies]
    if args.n_action_steps is not None:
        all_results = [r for r in all_results if r["n_action_steps"] in args.n_action_steps]

    if not all_results:
        raise SystemExit("No results match the given filters.")

    print(f"Loaded {len(all_results)} result(s) from {args.results_dir}\n")
    _analyze(all_results, args)


if __name__ == "__main__":
    main()
