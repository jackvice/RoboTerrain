#!/usr/bin/env python3
"""
Summarize encounter ratios per CSV file.

Writes: filename,total_goals,goal_dot5_ratio,goal_dot5_dot8_ratio,goal_all_three_ratio

Ratios (as you specified):
  goal_dot5_ratio        = round(total_goals / c_lt_05, 2)
  goal_dot5_dot8_ratio   = round(total_goals / (c_lt_05 + c_05_08), 2)
  goal_all_three_ratio   = round(total_goals / (c_lt_05 + c_05_08 + c_08_12), 2)

Usage:
  python summarize_encounters.py <env> <input_dir> <output_csv>

Example:
  python summarize_encounters.py construct ./csv_runs ./summary.csv
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import csv
import glob
import os

import pandas as pd


EncounterPoint = Tuple[float, float]  # (time, min_distance)


# (filename, total_goals, c_08_12, c_05_08, c_lt_05)
FileCounts = Tuple[str, int, int, int, int]


def summarize_one_file_counts(path: str, actor_cols: Sequence[str]) -> FileCounts:
    df: pd.DataFrame = pd.read_csv(path)

    total_goals: int = get_total_goals(df)
    encounter_points = detect_all_actor_encounters(df, actor_cols)
    c_08_12, c_05_08, c_lt_05 = classify_encounters(encounter_points)

    return (os.path.basename(path), total_goals, c_08_12, c_05_08, c_lt_05)


def compute_ratios_from_counts(
    total_goals: int,
    c_08_12: int,
    c_05_08: int,
    c_lt_05: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Your exact formulas:
    #   goal_dot5_ratio      = goals / c_lt_05
    #   goal_dot5_dot8_ratio = goals / (c_lt_05 + c_05_08)
    #   goal_all_three_ratio = goals / (c_lt_05 + c_05_08 + c_08_12)
    goal_dot5_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05)
    goal_dot5_dot8_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05 + c_05_08)
    goal_all_three_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05 + c_05_08 + c_08_12)
    return (goal_dot5_ratio, goal_dot5_dot8_ratio, goal_all_three_ratio)


def print_grand_totals(file_counts: Sequence[FileCounts]) -> None:
    total_files: int = len(file_counts)
    goals_sum: int = sum(x[1] for x in file_counts)
    c_08_12_sum: int = sum(x[2] for x in file_counts)
    c_05_08_sum: int = sum(x[3] for x in file_counts)
    c_lt_05_sum: int = sum(x[4] for x in file_counts)

    r05, r058, rall = compute_ratios_from_counts(goals_sum, c_08_12_sum, c_05_08_sum, c_lt_05_sum)

    def fmt(x: Optional[float]) -> str:
        return "" if x is None else f"{x:.2f}"

    print("=== Grand totals across all CSVs ===")
    print(f"files: {total_files}")
    print(f"total_goals: {goals_sum}")
    print(f"encounters_c_lt_05: {c_lt_05_sum}")
    print(f"encounters_c_05_08: {c_05_08_sum}")
    print(f"encounters_c_08_12: {c_08_12_sum}")
    print(f"encounters_total: {c_lt_05_sum + c_05_08_sum + c_08_12_sum}")
    print(f"goal_dot5_ratio: {fmt(r05)}")
    print(f"goal_dot5_dot8_ratio: {fmt(r058)}")
    print(f"goal_all_three_ratio: {fmt(rall)}")



def scan_csv_files(directory: str) -> List[str]:
    pattern: str = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def get_total_goals(df: pd.DataFrame) -> int:
    # Supports both old and new goal column names.
    for col in ("goals", "goals_count"):
        if col in df.columns and not df.empty:
            try:
                return int(pd.to_numeric(df[col], errors="coerce").max())
            except Exception:
                return 0
    return 0


def detect_encounters_for_single_actor(times: pd.Series, distances: pd.Series) -> List[EncounterPoint]:
    """
    Same logic as your multi_plot scripts:
      - "Encounter" when dist < 1.2
      - Record the minimum distance within each contiguous encounter segment
    """
    encounters: List[EncounterPoint] = []
    if distances.empty:
        return encounters

    in_encounter: bool = False
    current_min_dist: float = float("inf")
    current_min_time: float = 0.0

    for t, d in zip(times, distances):
        if pd.isna(d):
            continue

        within: bool = float(d) < 1.2

        if within:
            if not in_encounter:
                in_encounter = True
                current_min_dist = float(d)
                current_min_time = float(t)
            else:
                if float(d) < current_min_dist:
                    current_min_dist = float(d)
                    current_min_time = float(t)
        else:
            if in_encounter:
                encounters.append((current_min_time, current_min_dist))
                in_encounter = False
                current_min_dist = float("inf")
                current_min_time = 0.0

    if in_encounter:
        encounters.append((current_min_time, current_min_dist))

    return encounters


def detect_all_actor_encounters(df: pd.DataFrame, actor_cols: Sequence[str]) -> List[EncounterPoint]:
    if df.empty:
        return []

    times: pd.Series = pd.Series(range(len(df)), dtype=float) / 60.0  # minutes, same convention as your scripts
    all_encounters: List[EncounterPoint] = []

    for col in actor_cols:
        if col in df.columns:
            all_encounters.extend(detect_encounters_for_single_actor(times, df[col]))

    return all_encounters


def classify_encounters(encounter_points: Sequence[EncounterPoint]) -> Tuple[int, int, int]:
    """
    Returns (c_08_12, c_05_08, c_lt_05) based on the min distance of each encounter.
    """
    c_08_12: int = 0
    c_05_08: int = 0
    c_lt_05: int = 0

    for _, d in encounter_points:
        if d < 0.5:
            c_lt_05 += 1
        elif d < 0.8:
            c_05_08 += 1
        else:
            c_08_12 += 1

    return (c_08_12, c_05_08, c_lt_05)


def safe_ratio(numer: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return round(numer / denom, 2)


def actor_columns_for_env(env: str) -> List[str]:
    """
    Handles "old vs new" actor distance column naming by trying a superset per environment.
    - construct: supports both the newer lower/upper naming and the older triangle naming :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    - island: older triangle naming (but we include the construct-new names too just in case)
    - inspect: linear/triangle/diag naming :contentReference[oaicite:5]{index=5}
    """
    env_l: str = env.lower()

    construct_cols: List[str] = [
        # "new-ish" names
        "d1_lower", "d2_upper", "lower_actor_dist", "upper_actor_dist",
        # "old-ish" names
        "d1_triangle", "d2_triangle2", "d3_triangle3",
    ]

    island_cols: List[str] = [
        # typical island file names
        "d1_triangle", "d2_triangle2", "d3_triangle3",
        # tolerate newer naming too
        "d1_lower", "d2_upper", "lower_actor_dist", "upper_actor_dist",
    ]

    inspect_cols: List[str] = [
        # current inspect/nav2 lidar logs
        "linear_actor_dist", "triangle_actor_dist", "diag_actor_dist",
        # if you log dmin, it's the simplest + most reliable
        "dmin",

        # older naming (keep for backward compat)
        "d1_linear", "d2_triangle", "d3_diag",
        "d1_triangle", "d2_triangle2", "d3_triangle3",
    ]

    mapping: Dict[str, List[str]] = {
        "construct": construct_cols,
        "island": island_cols,
        "inspect": inspect_cols,
    }

    if env_l not in mapping:
        raise ValueError(f"env must be one of: {', '.join(mapping.keys())}")

    return mapping[env_l]


def summarize_one_file(path: str, actor_cols: Sequence[str]) -> Tuple[str, int, Optional[float], Optional[float], Optional[float]]:
    df: pd.DataFrame = pd.read_csv(path)

    total_goals: int = get_total_goals(df)
    encounter_points: List[EncounterPoint] = detect_all_actor_encounters(df, actor_cols)
    c_08_12, c_05_08, c_lt_05 = classify_encounters(encounter_points)

    # Your exact formulas (goals divided by encounter counts) :contentReference[oaicite:6]{index=6}
    goal_dot5_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05)
    goal_dot5_dot8_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05 + c_05_08)
    goal_all_three_ratio: Optional[float] = safe_ratio(total_goals, c_lt_05 + c_05_08 + c_08_12)

    return (os.path.basename(path), total_goals, goal_dot5_ratio, goal_dot5_dot8_ratio, goal_all_three_ratio)


def write_summary_csv(
    rows: Iterable[Tuple[str, int, Optional[float], Optional[float], Optional[float]]],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename",
            "total_goals",
            "goal_dot5_ratio",
            "goal_dot5_dot8_ratio",
            "goal_all_three_ratio",
        ])
        for filename, total_goals, r05, r058, rall in rows:
            w.writerow([
                filename,
                total_goals,
                "" if r05 is None else f"{r05:.2f}",
                "" if r058 is None else f"{r058:.2f}",
                "" if rall is None else f"{rall:.2f}",
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize encounter ratios per CSV file.")
    p.add_argument("env", choices=["inspect", "construct", "island"], help="Environment name.")
    p.add_argument("input_dir", help="Directory containing CSV files.")
    p.add_argument("output_csv", help="Output CSV path to write.")
    return p.parse_args()





def main() -> None:
    args = parse_args()

    csv_paths: List[str] = scan_csv_files(args.input_dir)
    if not csv_paths:
        raise SystemExit(f"No .csv files found in: {args.input_dir}")

    actor_cols: List[str] = actor_columns_for_env(args.env)

    # --- Per-file counts
    file_counts: List[FileCounts] = [
        summarize_one_file_counts(p, actor_cols) for p in csv_paths
    ]

    # --- Per-file output rows (filename, total_goals, r05, r058, rall)
    rows_for_csv: List[Tuple[str, int, Optional[float], Optional[float], Optional[float]]] = [
        (fn, goals, *compute_ratios_from_counts(goals, c0812, c0508, clt05))
        for (fn, goals, c0812, c0508, clt05) in file_counts
    ]

    write_summary_csv(rows_for_csv, args.output_csv)

    # --- Grand totals to terminal (pooled counts across all CSVs)
    print_grand_totals(file_counts)

    print(f"Wrote {args.output_csv} from {len(csv_paths)} input files.")



if __name__ == "__main__":
    main()
