#!/usr/bin/env python3
"""
Run-level (30-min CSV) stats for RoboTerrain social-nav metrics.

What it does (honest inference):
- Treats each CSV as ONE independent replicate ("run").
- Computes per-run metrics:
    1) risk_secs_per_goal  = (# seconds with dmin < 0.5) / (goals achieved)
    2) secs_per_goal       = (run duration in seconds) / (goals achieved)
    3) goals_per_encounter = (goals achieved) / (# <0.5m encounter EVENTS)   [optional output]
- Compares conditions using an EXACT permutation test at the RUN level
  (no pseudoreplication from 5-min blocks or goal-windows).
- Optionally applies Holm correction across multiple planned comparisons.

Assumptions:
- 1 CSV row = 1 second of sim time (as you stated).
- goals_count is cumulative (nondecreasing); goals achieved per run is its net increase.

Folder layout expected:
  <root>/<env>/<condition_dir>/*.csv
where condition_dir is one of:
  attention, no_attention, no_soc_nav, Active_Vision, Nav2_lidar_forward, Nav2_lidar_reverse

Usage:
  python run_level_stats.py island --root metrics_data --output island_run_stats.csv
  python run_level_stats.py --all --root metrics_data --output all_envs_run_stats.csv

Planned comparisons:
- Default: only Active_Vision vs no_soc_nav (primary)
- Add --compare_all to test Active_Vision against all other conditions (Holm-corrected)

"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Condition = str
Env = str
MetricName = Literal["risk_secs_per_goal", "secs_per_goal", "goals_per_encounter"]
Alternative = Literal["two-sided", "greater", "less"]


# --- Config (edit if your folder names differ) --------------------------------
CONDITION_DIRS: Dict[Condition, str] = {
    "With Attention": "attention",
    "No Attention": "no_attention",
    "No Social Nav": "no_soc_nav",
    "Active Vision": "Active_Vision",
    "Nav2 (forward)": "Nav2_lidar_forward",
    "Nav2 (w/reverse)": "Nav2_lidar_reverse",
}

# Which direction is "better" for each metric:
# - risk_secs_per_goal: lower is better -> expect A < B, so use alternative="less" when A is "better"
# - secs_per_goal:      lower is better -> expect A < B
# - goals_per_encounter: higher is better -> expect A > B
METRIC_DIRECTION: Dict[MetricName, Literal["lower", "higher"]] = {
    "risk_secs_per_goal": "lower",
    "secs_per_goal": "lower",
    "goals_per_encounter": "higher",
}


# --- Data structures -----------------------------------------------------------
@dataclass(frozen=True)
class RunMetrics:
    env: Env
    condition: Condition
    run_file: str
    goals: int
    seconds: int
    risk_secs: int
    encounter_events: int
    risk_secs_per_goal: float
    secs_per_goal: float
    goals_per_encounter: float


@dataclass(frozen=True)
class TestResult:
    env: Env
    metric: MetricName
    cond_a: Condition
    cond_b: Condition
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    diff_a_minus_b: float
    p_perm_raw: float
    p_perm_holm: Optional[float]
    alternative: Alternative


# --- Core computations ---------------------------------------------------------
def _safe_int(x: float) -> int:
    return int(np.round(float(x)))


def extract_dmin(df: pd.DataFrame) -> pd.Series:
    if "dmin" in df.columns:
        return df["dmin"].astype(float)
    # Fallbacks: if you ever need them, expand here.
    # For your schema (diag_actor_dist, linear_actor_dist, triangle_actor_dist),
    # you can uncomment the following lines to compute dmin from *_dist columns.
    dist_cols = [c for c in df.columns if c.endswith("_dist") or c.endswith("actor_dist")]
    if dist_cols:
        return df[dist_cols].astype(float).min(axis=1)
    raise ValueError("No 'dmin' column found and no usable *_dist columns for fallback.")


def goals_achieved(df: pd.DataFrame) -> int:
    if "goals" not in df.columns:
        raise ValueError("CSV missing required column: goals")
    g = df["goals"].astype(float).to_numpy()
    # Net increase handles cumulative counter starting at nonzero.
    return max(0, _safe_int(g[-1] - g[0]))


def count_encounter_events(dmin: pd.Series, threshold: float = 0.5) -> int:
    """
    Counts discrete encounter EVENTS as the number of transitions into the threshold region:
      event when dmin[t] < thr and dmin[t-1] >= thr
    """
    x = dmin.to_numpy(dtype=float)
    below = x < threshold
    if below.size == 0:
        return 0
    # transitions: False->True
    return int(np.sum((~below[:-1]) & (below[1:])))


def compute_run_metrics(env: Env, condition: Condition, csv_path: Path) -> RunMetrics:
    df = pd.read_csv(csv_path)
    dmin = extract_dmin(df)

    secs = int(len(df))  # 1 row = 1 second (your assumption)
    goals = goals_achieved(df)
    risk_secs = int(np.sum(dmin.to_numpy(dtype=float) < 0.5))
    events = count_encounter_events(dmin, threshold=0.5)

    # If goals=0, the run isn't usable for per-goal metrics.
    if goals <= 0:
        raise ValueError(f"{csv_path.name}: goals achieved is 0 (cannot compute per-goal metrics)")

    risk_per_goal = float(risk_secs) / float(goals)
    secs_per_goal = float(secs) / float(goals)

    # goals_per_encounter: use continuity correction so events=0 doesn't go infinite.
    # This mirrors your earlier choice; adjust if you prefer inf handling.
    goals_per_enc = float(goals) / (float(events) + 0.5)

    return RunMetrics(
        env=env,
        condition=condition,
        run_file=csv_path.name,
        goals=goals,
        seconds=secs,
        risk_secs=risk_secs,
        encounter_events=events,
        risk_secs_per_goal=risk_per_goal,
        secs_per_goal=secs_per_goal,
        goals_per_encounter=goals_per_enc,
    )


def load_env_runs(root: Path, env: Env) -> List[RunMetrics]:
    out: List[RunMetrics] = []
    env_dir = root / env
    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    for cond_label, cond_dirname in CONDITION_DIRS.items():
        cond_dir = env_dir / cond_dirname
        if not cond_dir.exists():
            continue
        for csv_path in sorted(cond_dir.glob("*.csv")):
            out.append(compute_run_metrics(env, cond_label, csv_path))
    return out


def exact_permutation_p(
    a: Sequence[float],
    b: Sequence[float],
    alternative: Alternative,
) -> float:
    """
    Exact permutation p-value for difference in means: mean(a) - mean(b),
    permuting labels over pooled samples.

    For n_a=3, n_b=3 -> 20 permutations.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a = a.size
    n_b = b.size
    pooled = np.concatenate([a, b])
    n = pooled.size

    obs = float(np.mean(a) - np.mean(b))

    idx = range(n)
    diffs: List[float] = []
    for choose_a in itertools.combinations(idx, n_a):
        mask = np.zeros(n, dtype=bool)
        mask[list(choose_a)] = True
        aa = pooled[mask]
        bb = pooled[~mask]
        diffs.append(float(np.mean(aa) - np.mean(bb)))

    if alternative == "two-sided":
        num = sum(1 for d in diffs if abs(d) >= abs(obs))
        return num / float(len(diffs))
    if alternative == "greater":
        num = sum(1 for d in diffs if d >= obs)
        return num / float(len(diffs))
    if alternative == "less":
        num = sum(1 for d in diffs if d <= obs)
        return num / float(len(diffs))
    raise ValueError(f"Unknown alternative: {alternative}")


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """
    Holm step-down adjusted p-values.
    Returns adjusted p-values in original order.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for k, i in enumerate(order):
        adj = (m - k) * pvals[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    return adjusted


def metric_values(runs: Sequence[RunMetrics], metric: MetricName) -> List[float]:
    if metric == "risk_secs_per_goal":
        return [r.risk_secs_per_goal for r in runs]
    if metric == "secs_per_goal":
        return [r.secs_per_goal for r in runs]
    if metric == "goals_per_encounter":
        return [r.goals_per_encounter for r in runs]
    raise ValueError(f"Unknown metric: {metric}")


def pick_alternative(metric: MetricName) -> Alternative:
    # For your use: we pick "less" when metric is "lower is better", else "greater".
    return "less" if METRIC_DIRECTION[metric] == "lower" else "greater"


# --- Analysis runner -----------------------------------------------------------
def analyze_env(
    root: Path,
    env: Env,
    compare_all: bool,
    metrics: Sequence[MetricName],
) -> Tuple[List[RunMetrics], List[TestResult]]:
    runs = load_env_runs(root, env)
    by_cond: Dict[Condition, List[RunMetrics]] = {}
    for r in runs:
        by_cond.setdefault(r.condition, []).append(r)

    # Planned comparisons
    if compare_all:
        comparisons: List[Tuple[Condition, Condition]] = [
            ("Active Vision", c) for c in by_cond.keys() if c != "Active Vision"
        ]
    else:
        comparisons = [("Active Vision", "No Social Nav")]

    results: List[TestResult] = []

    for metric in metrics:
        alt = pick_alternative(metric)
        p_raws: List[float] = []
        tmp: List[TestResult] = []

        for a_name, b_name in comparisons:
            if a_name not in by_cond or b_name not in by_cond:
                continue
            a_runs = by_cond[a_name]
            b_runs = by_cond[b_name]
            a_vals = metric_values(a_runs, metric)
            b_vals = metric_values(b_runs, metric)

            p = exact_permutation_p(a_vals, b_vals, alternative=alt)
            p_raws.append(p)

            tmp.append(
                TestResult(
                    env=env,
                    metric=metric,
                    cond_a=a_name,
                    cond_b=b_name,
                    n_a=len(a_vals),
                    n_b=len(b_vals),
                    mean_a=float(np.mean(a_vals)),
                    mean_b=float(np.mean(b_vals)),
                    diff_a_minus_b=float(np.mean(a_vals) - np.mean(b_vals)),
                    p_perm_raw=p,
                    p_perm_holm=None,
                    alternative=alt,
                )
            )

        # Holm correction within metric (only if compare_all implies multiple tests)
        if compare_all and tmp:
            p_adj = holm_bonferroni(p_raws)
            tmp = [
                TestResult(**{**tr.__dict__, "p_perm_holm": p_adj[i]})  # type: ignore[arg-type]
                for i, tr in enumerate(tmp)
            ]

        results.extend(tmp)

    return runs, results


def write_runs_csv(path: Path, runs: Sequence[RunMetrics]) -> None:
    cols = [
        "env",
        "condition",
        "run_file",
        "goals",
        "seconds",
        "risk_secs",
        "encounter_events",
        "risk_secs_per_goal",
        "secs_per_goal",
        "goals_per_encounter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            w.writerow(
                [
                    r.env,
                    r.condition,
                    r.run_file,
                    r.goals,
                    r.seconds,
                    r.risk_secs,
                    r.encounter_events,
                    f"{r.risk_secs_per_goal:.6f}",
                    f"{r.secs_per_goal:.6f}",
                    f"{r.goals_per_encounter:.6f}",
                ]
            )


def write_tests_csv(path: Path, tests: Sequence[TestResult]) -> None:
    cols = [
        "env",
        "metric",
        "cond_a",
        "cond_b",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "diff_a_minus_b",
        "alternative",
        "p_perm_raw",
        "p_perm_holm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in tests:
            w.writerow(
                [
                    t.env,
                    t.metric,
                    t.cond_a,
                    t.cond_b,
                    t.n_a,
                    t.n_b,
                    f"{t.mean_a:.6f}",
                    f"{t.mean_b:.6f}",
                    f"{t.diff_a_minus_b:.6f}",
                    t.alternative,
                    f"{t.p_perm_raw:.6f}",
                    "" if t.p_perm_holm is None else f"{t.p_perm_holm:.6f}",
                ]
            )


def print_summary(runs: Sequence[RunMetrics], tests: Sequence[TestResult], compare_all: bool) -> None:
    by_cond: Dict[Condition, List[RunMetrics]] = {}
    for r in runs:
        by_cond.setdefault(r.condition, []).append(r)

    print("Runs loaded:")
    for cond, rs in sorted(by_cond.items(), key=lambda kv: kv[0]):
        print(f"  {cond}: {len(rs)} runs")
        for r in rs:
            print(
                f"    {r.run_file}: goals={r.goals}, risk_secs={r.risk_secs}, "
                f"risk/g={r.risk_secs_per_goal:.3f}, s/g={r.secs_per_goal:.3f}, g/e={r.goals_per_encounter:.3f}"
            )

    print("\nExact permutation tests (run-level):")
    if not tests:
        print("  (no tests ran — missing conditions?)")
        return

    for t in tests:
        p_show = t.p_perm_holm if (compare_all and t.p_perm_holm is not None) else t.p_perm_raw
        p_kind = "p_holm" if (compare_all and t.p_perm_holm is not None) else "p_raw"
        print(
            f"  [{t.env}] {t.metric}: {t.cond_a} vs {t.cond_b} "
            f"(n={t.n_a} vs {t.n_b}, alt={t.alternative}) "
            f"diff={t.diff_a_minus_b:.3f}  {p_kind}={p_show:.3f}"
        )

    # Useful reminder about granularity
    # If n=3 vs 3, there are 20 permutations, so min nonzero p is 0.05 for the standard exact test.
    for t in tests[:1]:
        if t.n_a + t.n_b == 6 and t.n_a == 3 and t.n_b == 3:
            print("\nNOTE: with 3 vs 3 runs, there are C(6,3)=20 permutations -> p values in steps of 0.05.")


# --- CLI ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("env", nargs="?", help="Environment name (e.g., island, inspection, construction)")
    p.add_argument("--all", action="store_true", help="Analyze all environment directories under --root")
    p.add_argument("--root", type=str, default="metrics_data", help="Root metrics directory (default: metrics_data)")
    p.add_argument("--output", type=str, default="run_level_stats_tests.csv", help="Output CSV for test results")
    p.add_argument("--runs_output", type=str, default="run_level_stats_runs.csv", help="Output CSV for per-run metrics")
    p.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare Active Vision against all other available conditions (Holm-corrected per metric). "
        "Default compares only Active Vision vs No Social Nav.",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="risk_secs_per_goal,secs_per_goal",
        help="Comma-separated metrics: risk_secs_per_goal,secs_per_goal,goals_per_encounter",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    allowed = {"risk_secs_per_goal", "secs_per_goal", "goals_per_encounter"}
    if any(m not in allowed for m in metrics):
        bad = [m for m in metrics if m not in allowed]
        raise SystemExit(f"Unknown metric(s): {bad}. Allowed: {sorted(allowed)}")
    metrics_typed = tuple(metrics)  # type: ignore[assignment]

    envs: List[str]
    if args.all:
        envs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    else:
        if not args.env:
            raise SystemExit("Provide an env (e.g., island) or use --all")
        envs = [args.env]

    all_runs: List[RunMetrics] = []
    all_tests: List[TestResult] = []

    for env in envs:
        runs, tests = analyze_env(root, env, compare_all=args.compare_all, metrics=metrics_typed)  # type: ignore[arg-type]
        all_runs.extend(runs)
        all_tests.extend(tests)
        print(f"\n=== {env} ===")
        print_summary(runs, tests, compare_all=args.compare_all)

    write_runs_csv(Path(args.runs_output), all_runs)
    write_tests_csv(Path(args.output), all_tests)
    print(f"\nWrote per-run metrics to: {args.runs_output}")
    print(f"Wrote test results   to: {args.output}")


if __name__ == "__main__":
    main()
