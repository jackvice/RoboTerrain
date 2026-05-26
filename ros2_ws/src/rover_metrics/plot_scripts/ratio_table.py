#!/usr/bin/env python3
"""Per-CSV table of goals / <0.5 m encounters.

Reuses encounter-detection logic from inspect_multi_plot_new.py so the
ratio number prints exactly the same way as the existing plot caption.
Run on a directory of CSVs to bisect which configuration produced which
proxemic-safety ratio.

Usage:
    python3 ratio_table.py /path/to/Nav2CAN
    python3 ratio_table.py /path/to/single.csv
"""

import os
import sys
import glob
from typing import List, Tuple

import pandas as pd

from inspect_multi_plot_new import (
    detect_all_actor_encounters,
    get_total_goals,
)


def list_csvs(path: str) -> List[str]:
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.csv")))
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]
    return []


def per_file_metrics(path: str) -> Tuple[int, int, int, int, float, float, float]:
    df = pd.read_csv(path)
    if "time_s" in df.columns:
        t0 = float(df["time_s"].iloc[0])
        times = (df["time_s"].astype(float) - t0) / 60.0
        duration_min = float(times.iloc[-1])
    else:
        times = pd.Series(range(len(df))) / 60.0
        duration_min = len(df) / 60.0

    encounters = detect_all_actor_encounters(df, times)
    n_total = len(encounters)
    n_05_08 = sum(1 for _, d in encounters if 0.5 <= d < 0.8)
    n_lt_05 = sum(1 for _, d in encounters if d < 0.5)
    n_08_12 = sum(1 for _, d in encounters if 0.8 <= d < 1.2)
    goals = get_total_goals(df)
    ratio_05 = (goals / n_lt_05) if n_lt_05 else float("inf")
    return goals, n_total, n_08_12, n_05_08, n_lt_05, ratio_05, duration_min


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: ratio_table.py <csv_file | csv_directory>")
        sys.exit(1)

    csvs = list_csvs(sys.argv[1])
    if not csvs:
        print(f"No CSV files found for: {sys.argv[1]}")
        sys.exit(1)

    headers = ("file", "min", "goals", "enc<1.2", "0.8-1.2", "0.5-0.8", "<0.5", "goals/<0.5")
    print("{:<40} {:>5} {:>6} {:>8} {:>8} {:>8} {:>6} {:>10}".format(*headers))
    print("-" * 100)

    total_goals = 0
    total_lt_05 = 0
    for path in csvs:
        name = os.path.basename(path)
        goals, n_total, n_08_12, n_05_08, n_lt_05, ratio_05, duration = per_file_metrics(path)
        total_goals += goals
        total_lt_05 += n_lt_05
        ratio_str = f"{ratio_05:.2f}" if n_lt_05 else "inf"
        print(
            f"{name:<40} {duration:>5.1f} {goals:>6d} "
            f"{n_total:>8d} {n_08_12:>8d} {n_05_08:>8d} {n_lt_05:>6d} {ratio_str:>10}"
        )

    if len(csvs) > 1:
        agg_ratio = (total_goals / total_lt_05) if total_lt_05 else float("inf")
        agg_ratio_str = f"{agg_ratio:.2f}" if total_lt_05 else "inf"
        print("-" * 100)
        print(
            f"{'AGGREGATE':<40} {'':>5} {total_goals:>6d} "
            f"{'':>8} {'':>8} {'':>8} {total_lt_05:>6d} {agg_ratio_str:>10}"
        )


if __name__ == "__main__":
    main()
