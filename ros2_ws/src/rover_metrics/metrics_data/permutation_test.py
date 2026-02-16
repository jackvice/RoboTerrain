#!/usr/bin/env python3
"""
Permutation test over 5-minute blocks for a single environment + condition pair.

Example:
  python permutation_test_blocks.py island \
    --base_dir /path/to/metrics_data \
    --metric p05 \
    --cond_a "Active Vision" \
    --cond_b "With Attention" \
    --n_perm 50000 \
    --seed 0

Directory layout expected (same as statistical_analysis.py):
  <base_dir>/<environment>/<condition_dir>/*.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Config: match your statistical_analysis.py ──────────────────────────────

CONDITION_DIRS: Dict[str, str] = {
    "Active_Vision":  "Active_Vision",
    "attention":      "attention",
    "no_attention":    "no_attention",
    "no_soc_nav":     "no_soc_nav",
    "Nav2_lidar_forward":   "Nav2_lidar_forward",
    "Nav2_lidar_reverse":   "Nav2_lidar_reverse",
}

METRIC_INDEX: Dict[str, int] = {
    "p05": 0,
    "goal_rate": 1,
    "encounters_per_goal": 2,
}


# ── IO ──────────────────────────────────────────────────────────────────────

def _scan_csvs(dir_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(dir_path, "*.csv")))


def _load_concat(csv_paths: List[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)


def _extract_dmin(df: pd.DataFrame) -> pd.Series:
    if "dmin" in df.columns:
        return df["dmin"].astype(float)
    dist_cols: List[str] = [c for c in df.columns if c.startswith("d") and c not in ("dmin", "step", "time_s")]
    if not dist_cols:
        raise ValueError(f"No dmin or distance columns found: {list(df.columns)}")
    return df[dist_cols].astype(float).min(axis=1)


def _extract_goals(df: pd.DataFrame) -> pd.Series:
    for col in ("goals", "goals_count"):
        if col in df.columns:
            return df[col].astype(int)
    raise ValueError(f"No goals column found: {list(df.columns)}")


# ── Blocks + metrics (matches your row-count block logic) ───────────────────

def _block_metrics(
    dmin: pd.Series,
    goals: pd.Series,
    block_minutes: int,
) -> np.ndarray:
    """
    Returns array shape (n_blocks, 3) for [p05, goal_rate, encounters_per_goal].
    Uses row-count blocks (block_size = block_minutes * 60), matching your stats script.
    """
    block_size: int = block_minutes * 60
    n_blocks: int = len(dmin) // block_size
    if n_blocks <= 0:
        raise ValueError(f"Not enough rows for {block_minutes} min blocks.")

    out: List[Tuple[float, float, float]] = []
    for i in range(n_blocks):
        s: int = i * block_size
        e: int = (i + 1) * block_size

        bd: pd.Series = dmin.iloc[s:e].dropna()
        bg: pd.Series = goals.iloc[s:e]

        n_valid: int = int(len(bd))
        p05: float = float((bd < 0.5).sum() / n_valid) if n_valid > 0 else 0.0

        goals_in_block: float = float(bg.iloc[-1] - bg.iloc[0])
        goal_rate: float = goals_in_block / float(block_minutes)

        count_05: float = float((bd < 0.5).sum())
        epg: float = (count_05 / goals_in_block) if goals_in_block > 0 else float("nan")

        out.append((p05, goal_rate, epg))

    return np.asarray(out, dtype=float)


def load_condition_blocks(
    base_dir: str,
    environment: str,
    condition_name: str,
    block_minutes: int,
) -> np.ndarray:
    cond_dir: str = CONDITION_DIRS[condition_name]
    csv_dir: str = os.path.join(base_dir, environment, cond_dir)

    csvs: List[str] = _scan_csvs(csv_dir)
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {csv_dir}")

    df: pd.DataFrame = _load_concat(csvs)
    dmin: pd.Series = _extract_dmin(df)
    goals: pd.Series = _extract_goals(df)
    return _block_metrics(dmin, goals, block_minutes)


# ── Permutation test ────────────────────────────────────────────────────────

def permutation_test_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int,
    seed: int,
    alternative: str,
) -> Tuple[float, float]:
    """
    Permutation test for difference in means (A - B).
    Returns (observed_diff, p_value).
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be: two-sided | greater | less")

    rng: np.random.Generator = np.random.default_rng(seed)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least 2 non-NaN blocks per group.")

    obs: float = float(np.mean(a) - np.mean(b))

    x: np.ndarray = np.concatenate([a, b])
    n_a: int = len(a)
    n: int = len(x)

    # Permute labels; compute diffs
    diffs: np.ndarray = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm: np.ndarray = rng.permutation(n)
        xa: np.ndarray = x[perm[:n_a]]
        xb: np.ndarray = x[perm[n_a:]]
        diffs[i] = float(np.mean(xa) - np.mean(xb))

    if alternative == "two-sided":
        p: float = float((np.abs(diffs) >= abs(obs)).mean())
    elif alternative == "greater":
        p = float((diffs >= obs).mean())
    else:  # "less"
        p = float((diffs <= obs).mean())

    # Add a tiny “+1 / (n_perm+1)” style continuity correction (optional but common)
    p = float((p * n_perm + 1.0) / (n_perm + 1.0))
    return obs, p


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("environment", type=str, help="e.g., island | construct | inspect")
    ap.add_argument("--base_dir", type=str, required=True, help="Path to metrics_data")
    ap.add_argument("--metric", type=str, required=True, choices=sorted(METRIC_INDEX.keys()))
    ap.add_argument("--cond_a", type=str, required=True, choices=sorted(CONDITION_DIRS.keys()))
    ap.add_argument("--cond_b", type=str, required=True, choices=sorted(CONDITION_DIRS.keys()))
    ap.add_argument("--block_minutes", type=int, default=5)
    ap.add_argument("--n_perm", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alternative", type=str, default="two-sided",
                    choices=["two-sided", "greater", "less"])
    args = ap.parse_args()

    blocks_a: np.ndarray = load_condition_blocks(args.base_dir, args.environment, args.cond_a, args.block_minutes)
    blocks_b: np.ndarray = load_condition_blocks(args.base_dir, args.environment, args.cond_b, args.block_minutes)

    idx: int = METRIC_INDEX[args.metric]
    a: np.ndarray = blocks_a[:, idx]
    b: np.ndarray = blocks_b[:, idx]

    obs, p = permutation_test_mean_diff(a, b, args.n_perm, args.seed, args.alternative)

    print(f"{args.environment} | {args.metric} | {args.cond_a} vs {args.cond_b}")
    print(f"blocks: nA={np.sum(~np.isnan(a))}, nB={np.sum(~np.isnan(b))}, block_minutes={args.block_minutes}")
    print(f"observed Δ(mean A−B) = {obs:.6f}")
    print(f"permutation p ({args.alternative}, n_perm={args.n_perm}) = {p:.6g}")


if __name__ == "__main__":
    main()
