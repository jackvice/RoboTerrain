#!/usr/bin/env python3
"""
Statistical significance analysis for social navigation experiments.

Fixes over previous version:
  1. Primary metric is now goals_per_encounter (higher = better),
     matching the paper's "Goals per <0.5m encounter" column.
  2. Unit of analysis is the independent RUN (one 30-min CSV), not
     autocorrelated goal-windows carved from within a run.
  3. Exact permutation test (n_a + n_b choose n_a enumerations)
     gives honest p-values given the true sample size.
  4. Cluster bootstrap provides smoothed confidence intervals.

Usage:
    python statistical_analysis.py /path/to/metrics_data/island --output island_results.csv
"""

import argparse
import glob
import os
import sys
from itertools import combinations
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Types ───────────────────────────────────────────────────────────────────

class RunSummary(NamedTuple):
    """Summary statistics for a single independent run."""
    csv_path: str
    total_goals: int
    total_encounters: int          # discrete <0.5 m events (hysteresis)
    goals_per_encounter: float     # primary metric (higher = better)
    total_duration_s: float
    secs_per_goal: float           # secondary metric (lower = faster)


class TestResult(NamedTuple):
    environment: str
    metric: str
    condition_a: str
    condition_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float               # mean_a - mean_b
    ci_low: float                  # bootstrap 95% CI lower
    ci_high: float                 # bootstrap 95% CI upper
    p_perm: float                  # exact permutation p-value
    cohen_d: float
    significant: str               # "*" / "**" / "***" / ""


# ── Constants ───────────────────────────────────────────────────────────────

ENVIRONMENTS: List[str] = ["construct", "inspect", "island"]

CONDITION_DIRS: Dict[str, str] = {
    "Active Vision":    "Active_Vision",
    "With Attention":   "attention",
    "No Attention":     "no_attention",
    "No Social Nav":    "no_soc_nav",
    "Nav2 (forward)":   "Nav2_lidar_forward",
    "Nav2 (w/reverse)": "Nav2_lidar_reverse",
}

PLANNED_COMPARISONS: List[Tuple[str, str]] = [
    ("Active Vision",   "With Attention"),
    ("Active Vision",   "No Attention"),
    ("Active Vision",   "No Social Nav"),
    ("Active Vision",   "Nav2 (forward)"),
    ("Active Vision",   "Nav2 (w/reverse)"),
    ("With Attention",  "No Attention"),
    ("With Attention",  "No Social Nav"),
    ("With Attention",  "Nav2 (forward)"),
    ("With Attention",  "Nav2 (w/reverse)"),
]

METRICS: List[str] = ["goals_per_encounter", "secs_per_goal"]

# Directionality: condition_a is the hypothesised-better method.
# "greater" = one-sided test that condition_a > condition_b (min p = 0.05 with n=3).
# "two-sided" = no assumed direction (min p = 0.10 with n=3).
METRIC_ALTERNATIVE: Dict[str, str] = {
    "goals_per_encounter": "greater",    # higher = better, one-sided
    "secs_per_goal":       "two-sided",  # direction unclear (social methods trade speed for safety)
}

ENTER_THRESHOLD: float = 0.5      # metres — start of encounter
EXIT_THRESHOLD: float = 0.55      # metres — end of encounter (hysteresis)
ENCOUNTER_FLOOR: float = 0.5      # continuity correction for 0-encounter runs
BOOTSTRAP_ITERATIONS: int = 10_000
RNG_SEED: int = 42


# ── Data loading ────────────────────────────────────────────────────────────

def scan_csv_files(directory: str) -> List[str]:
    """Return sorted list of CSV files in a directory."""
    pattern: str = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def load_single_csv(path: str) -> pd.DataFrame:
    """Load one CSV, return DataFrame."""
    return pd.read_csv(path)


def extract_dmin(df: pd.DataFrame) -> pd.Series:
    """Extract or compute dmin from available columns."""
    if "dmin" in df.columns:
        return df["dmin"].astype(float)
    dist_cols: List[str] = [
        c for c in df.columns
        if c.startswith("d") and c not in ("dmin", "step", "time_s")
    ]
    if not dist_cols:
        raise ValueError(f"No distance columns found. Columns: {list(df.columns)}")
    return df[dist_cols].astype(float).min(axis=1)


def extract_goals(df: pd.DataFrame) -> pd.Series:
    """Extract cumulative goals column."""
    for col in ("goals", "goals_count"):
        if col in df.columns:
            return df[col].astype(int)
    raise ValueError(f"No goals column found. Columns: {list(df.columns)}")


def extract_time_s(df: pd.DataFrame) -> pd.Series:
    """Extract time column in seconds."""
    if "time_s" in df.columns:
        return df["time_s"].astype(float)
    raise ValueError(f"No time_s column found. Columns: {list(df.columns)}")


# ── Per-run metric computation ──────────────────────────────────────────────

def count_encounter_events(
    dmin_series: pd.Series,
    enter_threshold: float = ENTER_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
) -> int:
    """
    Count discrete encounter events using hysteresis.
    An encounter starts when dmin drops below enter_threshold,
    and ends when dmin rises above exit_threshold.
    """
    in_encounter: bool = False
    count: int = 0
    for d in dmin_series:
        if pd.isna(d):
            continue
        if not in_encounter and d < enter_threshold:
            in_encounter = True
            count += 1
        elif in_encounter and d > exit_threshold:
            in_encounter = False
    return count


def summarise_run(csv_path: str) -> RunSummary:
    """Compute summary metrics for one independent 30-min run."""
    df: pd.DataFrame = load_single_csv(csv_path)
    dmin: pd.Series = extract_dmin(df)
    goals_series: pd.Series = extract_goals(df)
    time_s: pd.Series = extract_time_s(df)

    total_goals: int = int(goals_series.iloc[-1] - goals_series.iloc[0])
    total_encounters: int = count_encounter_events(dmin)
    duration_s: float = float(time_s.iloc[-1] - time_s.iloc[0])

    # Primary metric: goals per encounter.
    # Continuity correction (+ ENCOUNTER_FLOOR) avoids division by zero
    # when a run has zero encounters — a genuinely good outcome.
    goals_per_enc: float = float(total_goals) / (float(total_encounters) + ENCOUNTER_FLOOR)

    secs_per_goal: float = duration_s / total_goals if total_goals > 0 else float("nan")

    return RunSummary(
        csv_path=csv_path,
        total_goals=total_goals,
        total_encounters=total_encounters,
        goals_per_encounter=goals_per_enc,
        total_duration_s=duration_s,
        secs_per_goal=secs_per_goal,
    )


def load_condition_runs(
    base_path: str,
    condition_dir: str,
) -> Optional[List[RunSummary]]:
    """Load all runs for one condition, return list of RunSummary or None."""
    csv_dir: str = os.path.join(base_path, condition_dir)
    if not os.path.isdir(csv_dir):
        return None
    csv_paths: List[str] = scan_csv_files(csv_dir)
    if not csv_paths:
        return None
    return [summarise_run(p) for p in csv_paths]


# ── Statistical tests ───────────────────────────────────────────────────────

def exact_permutation_p(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "greater",
) -> float:
    """
    Exact permutation test.

    Enumerates all C(n_a + n_b, n_a) ways to split the pooled values.

    alternative:
      "greater"  — one-sided: H1 is mean(a) > mean(b).
                    With n=3 vs 3, minimum p = 1/20 = 0.05.
      "two-sided" — H1 is mean(a) != mean(b).
                    With n=3 vs 3, minimum p = 2/20 = 0.10.

    Default is "greater" because our planned comparisons place the
    hypothesised-better condition first (e.g., Active Vision vs Nav2
    for goals_per_encounter where higher is better).
    """
    observed_diff: float = float(np.mean(a) - np.mean(b))
    pooled: np.ndarray = np.concatenate([a, b])
    n_a: int = len(a)
    n_total: int = len(pooled)

    count_extreme: int = 0
    count_total: int = 0

    for indices in combinations(range(n_total), n_a):
        idx_set: FrozenSet[int] = frozenset(indices)
        group_a: np.ndarray = pooled[list(idx_set)]
        group_b: np.ndarray = pooled[[i for i in range(n_total) if i not in idx_set]]
        perm_diff: float = float(np.mean(group_a) - np.mean(group_b))
        count_total += 1

        if alternative == "greater":
            if perm_diff >= observed_diff - 1e-12:
                count_extreme += 1
        else:  # two-sided
            if abs(perm_diff) >= abs(observed_diff) - 1e-12:
                count_extreme += 1

    return float(count_extreme) / float(count_total)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> Tuple[float, float]:
    """
    Cluster bootstrap 95% CI for difference in means (a - b).

    Resamples RUNS with replacement (the independent unit),
    computes difference in means for each bootstrap replicate.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    diffs: np.ndarray = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot_a: np.ndarray = rng.choice(a, size=len(a), replace=True)
        boot_b: np.ndarray = rng.choice(b, size=len(b), replace=True)
        diffs[i] = float(np.mean(boot_a) - np.mean(boot_b))
    lo: float = float(np.percentile(diffs, 100 * alpha / 2))
    hi: float = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled SD)."""
    n_a: int = len(a)
    n_b: int = len(b)
    var_a: float = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b: float = float(np.var(b, ddof=1)) if n_b > 1 else 0.0
    pooled_std: float = float(np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    ))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction, returns corrected p in original order."""
    n: int = len(p_values)
    if n == 0:
        return []
    indexed: List[Tuple[int, float]] = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected: List[float] = [0.0] * n
    cumulative_max: float = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted: float = p * (n - rank)
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)
    return corrected


def significance_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Analysis orchestration ──────────────────────────────────────────────────

def run_environment_analysis(
    base_path: str,
    environment: str,
    comparisons: List[Tuple[str, str]],
) -> List[TestResult]:
    """Run all planned comparisons for one environment."""
    # Determine which conditions we need
    needed: set = set()
    for a, b in comparisons:
        needed.add(a)
        needed.add(b)

    # Load run-level data
    condition_runs: Dict[str, Optional[List[RunSummary]]] = {}
    for cond_name in needed:
        cond_dir: str = CONDITION_DIRS[cond_name]
        runs: Optional[List[RunSummary]] = load_condition_runs(base_path, cond_dir)
        condition_runs[cond_name] = runs
        if runs is not None:
            print(f"  {environment}/{cond_dir}: {len(runs)} runs loaded")
            for r in runs:
                print(f"    {os.path.basename(r.csv_path)}: "
                      f"{r.total_goals} goals, {r.total_encounters} encounters, "
                      f"g/e={r.goals_per_encounter:.2f}, s/g={r.secs_per_goal:.1f}s")
        else:
            print(f"  {environment}/{cond_dir}: NOT FOUND — skipping")

    # Run tests per metric
    results_by_metric: Dict[str, List[TestResult]] = {m: [] for m in METRICS}

    for metric_idx, metric_name in enumerate(METRICS):
        raw_p_values: List[float] = []
        partial_results: List[Optional[TestResult]] = []

        for cond_a, cond_b in comparisons:
            runs_a: Optional[List[RunSummary]] = condition_runs.get(cond_a)
            runs_b: Optional[List[RunSummary]] = condition_runs.get(cond_b)

            if runs_a is None or runs_b is None:
                partial_results.append(None)
                continue

            # Extract the metric array from RunSummary
            arr_a: np.ndarray = np.array(
                [getattr(r, metric_name) for r in runs_a], dtype=np.float64
            )
            arr_b: np.ndarray = np.array(
                [getattr(r, metric_name) for r in runs_b], dtype=np.float64
            )

            # Drop NaN
            arr_a = arr_a[~np.isnan(arr_a)]
            arr_b = arr_b[~np.isnan(arr_b)]

            if len(arr_a) < 2 or len(arr_b) < 2:
                partial_results.append(None)
                continue

            p_perm: float = exact_permutation_p(
                arr_a, arr_b, alternative=METRIC_ALTERNATIVE[metric_name]
            )
            ci_lo, ci_hi = bootstrap_ci(arr_a, arr_b)
            d: float = cohen_d(arr_a, arr_b)
            mean_diff: float = float(np.mean(arr_a) - np.mean(arr_b))

            raw_p_values.append(p_perm)
            partial_results.append(TestResult(
                environment=environment,
                metric=metric_name,
                condition_a=cond_a,
                condition_b=cond_b,
                n_a=len(arr_a),
                n_b=len(arr_b),
                mean_a=float(np.mean(arr_a)),
                mean_b=float(np.mean(arr_b)),
                mean_diff=mean_diff,
                ci_low=ci_lo,
                ci_high=ci_hi,
                p_perm=p_perm,
                cohen_d=d,
                significant="",  # placeholder
            ))

        # Holm-Bonferroni correction across comparisons within this metric
        corrected: List[float] = holm_bonferroni(raw_p_values)

        corr_idx: int = 0
        for result in partial_results:
            if result is not None:
                p_corr: float = corrected[corr_idx]
                corr_idx += 1
                results_by_metric[metric_name].append(result._replace(
                    p_perm=p_corr,
                    significant=significance_marker(p_corr),
                ))

    all_results: List[TestResult] = []
    for metric_name in METRICS:
        all_results.extend(results_by_metric[metric_name])

    return all_results


def run_full_analysis(
    base_path: str,
    comparisons: List[Tuple[str, str]],
) -> pd.DataFrame:
    """Run analysis for a single environment directory."""
    env_name: str = os.path.basename(os.path.normpath(base_path))
    print(f"\nProcessing environment: {env_name}")

    all_results: List[TestResult] = run_environment_analysis(
        base_path, env_name, comparisons,
    )

    columns: List[str] = [
        "environment", "metric", "condition_a", "condition_b",
        "n_a", "n_b", "mean_a", "mean_b", "mean_diff",
        "ci_low", "ci_high", "p_perm", "cohen_d", "significant",
    ]
    return pd.DataFrame([r._asdict() for r in all_results], columns=columns)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Statistical analysis of social navigation experiments"
    )
    parser.add_argument("data_path", type=str,
                        help="Path to one environment's metrics_data directory "
                             "(e.g., metrics_data/island)")
    parser.add_argument("--output", type=str, default="statistical_results.csv",
                        help="Output CSV path (default: statistical_results.csv)")
    args: argparse.Namespace = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"Error: {args.data_path} is not a directory")
        sys.exit(1)

    results_df: pd.DataFrame = run_full_analysis(args.data_path, PLANNED_COMPARISONS)

    results_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nResults saved to {args.output}")
    print(f"Total tests: {len(results_df)}")

    # Print summary
    print(f"\n{'='*80}")
    print("NOTE: With n=3 runs per condition and C(6,3)=20 permutations:")
    print("  goals_per_encounter (one-sided): minimum p = 1/20 = 0.05")
    print("  secs_per_goal       (two-sided): minimum p = 2/20 = 0.10")
    print("  Large effect sizes (Cohen's d) may be more informative than p-values.")
    print(f"{'='*80}")

    sig: pd.DataFrame = results_df[results_df["significant"] != ""]
    if not sig.empty:
        print(f"\nSignificant results ({len(sig)}):")
        for _, row in sig.iterrows():
            print(f"  {row['environment']} | {row['metric']} | "
                  f"{row['condition_a']} vs {row['condition_b']} | "
                  f"p_corr={row['p_perm']:.4f}{row['significant']} | "
                  f"d={row['cohen_d']:.2f}")
    else:
        print("\nNo statistically significant results after correction.")

    # Always print effect sizes for primary metric
    primary: pd.DataFrame = results_df[results_df["metric"] == "goals_per_encounter"]
    if not primary.empty:
        print(f"\nGoals per encounter (effect sizes):")
        for _, row in primary.iterrows():
            print(f"  {row['condition_a']} vs {row['condition_b']}: "
                  f"d={row['cohen_d']:.2f}, "
                  f"diff={row['mean_diff']:.2f} "
                  f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
