#!/usr/bin/env python3
"""
Statistical significance analysis for social navigation experiments.
Segments 90-minute runs into 5-minute blocks, computes per-block metrics,
runs Welch t-tests on planned comparisons, applies Holm-Bonferroni correction.

Usage:
    python statistical_analysis.py /path/to/metrics_data --output results.csv
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ── Types ───────────────────────────────────────────────────────────────────

class BlockMetrics(NamedTuple):
    p05: float               # fraction of seconds with dmin < 0.5
    goal_rate: float         # goals per minute in this block
    encounters_per_goal: float  # encounters(<0.5m) per goal achieved


class TestResult(NamedTuple):
    environment: str
    metric: str
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    p_corrected: float
    cohen_d: float
    significant: str   # "*" if p_corrected < 0.05, "**" if < 0.01, "" otherwise


# ── Constants ───────────────────────────────────────────────────────────────

BLOCK_MINUTES: int = 5

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
    ("Active Vision",  "With Attention"),
    ("With Attention",  "No Attention"),
    ("Active Vision",   "No Attention"),
    ("With Attention",  "No Social Nav"),
    ("With Attention",  "Nav2 (w/reverse)"),
]

METRICS: List[str] = ["p05", "goal_rate", "encounters_per_goal"]


# ── Data loading ────────────────────────────────────────────────────────────

def scan_csv_files(directory: str) -> List[str]:
    """Return sorted list of CSV files in a directory."""
    pattern: str = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def load_and_concatenate(csv_paths: List[str]) -> pd.DataFrame:
    """Load multiple CSVs, concatenate, reset row index to sequential seconds."""
    frames: List[pd.DataFrame] = [pd.read_csv(p) for p in csv_paths]
    df: pd.DataFrame = pd.concat(frames, ignore_index=True)
    return df


def extract_dmin(df: pd.DataFrame) -> pd.Series:
    """Extract or compute dmin from available columns."""
    if "dmin" in df.columns:
        return df["dmin"].astype(float)

    # Fallback: compute from individual actor distance columns
    dist_cols: List[str] = [c for c in df.columns if c.startswith("d") and c not in
                            ("dmin", "step", "time_s")]
    if not dist_cols:
        raise ValueError(f"No distance columns found. Columns: {list(df.columns)}")
    return df[dist_cols].astype(float).min(axis=1)


def extract_goals(df: pd.DataFrame) -> pd.Series:
    """Extract cumulative goals column."""
    for col in ["goals", "goals_count"]:
        if col in df.columns:
            return df[col].astype(int)
    raise ValueError(f"No goals column found. Columns: {list(df.columns)}")


# ── Block segmentation and metrics ──────────────────────────────────────────

def compute_block_metrics_array(
    dmin: pd.Series,
    goals: pd.Series,
    block_minutes: int
) -> np.ndarray:
    """
    Segment time series into fixed blocks, compute metrics per block.

    Returns: np.ndarray of shape (n_blocks, 3) with columns [p05, goal_rate, encounters_per_goal].
    Data is at ~1 Hz, so index ≈ seconds.
    """
    block_size: int = block_minutes * 60  # seconds per block
    n_rows: int = len(dmin)
    n_blocks: int = n_rows // block_size

    if n_blocks == 0:
        raise ValueError(f"Not enough data for even one {block_minutes}-min block "
                         f"({n_rows} rows, need {block_size})")

    results: List[List[float]] = []

    for i in range(n_blocks):
        start: int = i * block_size
        end: int = (i + 1) * block_size

        block_dmin: pd.Series = dmin.iloc[start:end].dropna()
        block_goals: pd.Series = goals.iloc[start:end]

        n_valid: int = len(block_dmin)
        p05: float = float((block_dmin < 0.5).sum() / n_valid) if n_valid > 0 else 0.0

        goals_in_block: float = float(block_goals.iloc[-1] - block_goals.iloc[0])
        goal_rate: float = goals_in_block / block_minutes

        # Encounters per goal (lower is better; NaN if no goals in block)
        count_05: float = float((block_dmin < 0.5).sum())
        encounters_per_goal: float = count_05 / goals_in_block if goals_in_block > 0 else float('nan')

        results.append([p05, goal_rate, encounters_per_goal])

    return np.array(results)


def load_condition_metrics(
    base_path: str,
    condition_dir: str,
    block_minutes: int
) -> Optional[np.ndarray]:
    """
    Load CSVs for one condition, return block metrics array.
    Returns None if directory doesn't exist or has no CSVs.
    """
    csv_dir: str = os.path.join(base_path, condition_dir)

    if not os.path.isdir(csv_dir):
        return None

    csv_paths: List[str] = scan_csv_files(csv_dir)
    if not csv_paths:
        return None

    df: pd.DataFrame = load_and_concatenate(csv_paths)
    dmin: pd.Series = extract_dmin(df)
    goals: pd.Series = extract_goals(df)

    return compute_block_metrics_array(dmin, goals, block_minutes)


# ── Statistical tests ───────────────────────────────────────────────────────

def welch_t_test(
    a: np.ndarray,
    b: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """
    Welch's two-sample t-test (unequal variance).

    Returns: (t_stat, p_value, mean_diff, ci_low, ci_high)
    """
    t_stat: float
    p_value: float
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    mean_diff: float = float(np.mean(a) - np.mean(b))

    # 95% CI for the difference in means
    se: float = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))

    # Welch-Satterthwaite degrees of freedom
    va: float = float(np.var(a, ddof=1))
    vb: float = float(np.var(b, ddof=1))
    na: int = len(a)
    nb: int = len(b)
    df_num: float = (va / na + vb / nb) ** 2
    df_den: float = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df: float = df_num / df_den if df_den > 0 else min(na, nb) - 1

    t_crit: float = float(stats.t.ppf(0.975, df))
    ci_low: float = mean_diff - t_crit * se
    ci_high: float = mean_diff + t_crit * se

    return (float(t_stat), float(p_value), mean_diff, ci_low, ci_high)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    na: int = len(a)
    nb: int = len(b)
    pooled_std: float = float(np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    ))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down correction to a list of p-values.
    Returns corrected p-values in the original order.
    """
    n: int = len(p_values)
    if n == 0:
        return []

    # Sort by p-value, keeping track of original indices
    indexed: List[Tuple[int, float]] = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])

    corrected: List[float] = [0.0] * n
    cumulative_max: float = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted: float = p * (n - rank)
        # Enforce monotonicity: corrected p can't decrease as we step through
        cumulative_max = max(cumulative_max, adjusted)
        corrected[orig_idx] = min(cumulative_max, 1.0)

    return corrected


def significance_marker(p: float) -> str:
    """Return significance marker for corrected p-value."""
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
    block_minutes: int
) -> List[TestResult]:
    """
    Run all planned comparisons for one environment.
    Returns list of TestResult (one per comparison × metric).
    """
    # Load all conditions needed
    needed_conditions: set = set()
    for a, b in comparisons:
        needed_conditions.add(a)
        needed_conditions.add(b)

    condition_data: Dict[str, Optional[np.ndarray]] = {}
    for cond_name in needed_conditions:
        cond_dir: str = CONDITION_DIRS[cond_name]
        data: Optional[np.ndarray] = load_condition_metrics(
            base_path, cond_dir, block_minutes
        )
        condition_data[cond_name] = data
        if data is not None:
            print(f"  {environment}/{cond_dir}: {data.shape[0]} blocks loaded")
        else:
            print(f"  {environment}/{cond_dir}: NOT FOUND — skipping")

    # Run tests per metric, collect p-values for correction
    results_by_metric: Dict[str, List[TestResult]] = {m: [] for m in METRICS}

    for metric_idx, metric_name in enumerate(METRICS):
        raw_p_values: List[float] = []
        partial_results: List[Optional[TestResult]] = []

        for cond_a, cond_b in comparisons:
            data_a: Optional[np.ndarray] = condition_data.get(cond_a)
            data_b: Optional[np.ndarray] = condition_data.get(cond_b)

            if data_a is None or data_b is None:
                partial_results.append(None)
                continue

            arr_a: np.ndarray = data_a[:, metric_idx]
            arr_b: np.ndarray = data_b[:, metric_idx]

            # Drop NaN values (e.g. encounters_per_goal when block has 0 goals)
            arr_a = arr_a[~np.isnan(arr_a)]
            arr_b = arr_b[~np.isnan(arr_b)]

            if len(arr_a) < 2 or len(arr_b) < 2:
                partial_results.append(None)
                continue

            t_stat, p_val, mean_diff, ci_lo, ci_hi = welch_t_test(arr_a, arr_b)
            d: float = cohen_d(arr_a, arr_b)

            raw_p_values.append(p_val)
            partial_results.append(TestResult(
                environment=environment,
                metric=metric_name,
                condition_a=cond_a,
                condition_b=cond_b,
                mean_a=float(np.mean(arr_a)),
                mean_b=float(np.mean(arr_b)),
                mean_diff=mean_diff,
                ci_low=ci_lo,
                ci_high=ci_hi,
                t_stat=t_stat,
                p_value=p_val,
                p_corrected=0.0,   # placeholder
                cohen_d=d,
                significant=""     # placeholder
            ))

        # Apply Holm-Bonferroni correction across comparisons within this metric
        corrected: List[float] = holm_bonferroni(raw_p_values)

        # Merge corrected p-values back
        corr_idx: int = 0
        for result in partial_results:
            if result is not None:
                p_corr: float = corrected[corr_idx]
                corr_idx += 1
                results_by_metric[metric_name].append(result._replace(
                    p_corrected=p_corr,
                    significant=significance_marker(p_corr)
                ))

    # Flatten
    all_results: List[TestResult] = []
    for metric_name in METRICS:
        all_results.extend(results_by_metric[metric_name])

    return all_results


def run_full_analysis(
    base_path: str,
    comparisons: List[Tuple[str, str]],
    block_minutes: int
) -> pd.DataFrame:
    """Run analysis for a single environment directory, return results DataFrame."""
    env_name: str = os.path.basename(os.path.normpath(base_path))
    print(f"\nProcessing environment: {env_name}")

    all_results: List[TestResult] = run_environment_analysis(
        base_path, env_name, comparisons, block_minutes
    )

    columns: List[str] = [
        "environment", "metric", "condition_a", "condition_b",
        "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
        "t_stat", "p_value", "p_corrected", "cohen_d", "significant"
    ]
    return pd.DataFrame([r._asdict() for r in all_results], columns=columns)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Statistical analysis of social navigation experiments"
    )
    parser.add_argument("data_path", type=str,
                        help="Path to metrics_data directory")
    parser.add_argument("--output", type=str, default="statistical_results.csv",
                        help="Output CSV path (default: statistical_results.csv)")
    parser.add_argument("--block_minutes", type=int, default=BLOCK_MINUTES,
                        help="Block size in minutes (default: 5)")
    args: argparse.Namespace = parser.parse_args()

    if not os.path.isdir(args.data_path):
        print(f"Error: {args.data_path} is not a directory")
        sys.exit(1)

    results_df: pd.DataFrame = run_full_analysis(
        args.data_path, PLANNED_COMPARISONS, args.block_minutes
    )

    results_df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nResults saved to {args.output}")
    print(f"Total tests: {len(results_df)}")

    # Print summary of significant results
    sig: pd.DataFrame = results_df[results_df["significant"] != ""]
    if not sig.empty:
        print(f"\nSignificant results ({len(sig)}):")
        for _, row in sig.iterrows():
            print(f"  {row['environment']} | {row['metric']} | "
                  f"{row['condition_a']} vs {row['condition_b']} | "
                  f"p_corr={row['p_corrected']:.4f}{row['significant']} | "
                  f"d={row['cohen_d']:.2f}")
    else:
        print("\nNo statistically significant results after correction.")


if __name__ == "__main__":
    main()
