# plot_simple.py
# Usage: python plot_simple.py my_metrics.csv
from __future__ import annotations
import argparse
import sys
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

# -- Load CSV ---------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    """Read CSV and fail fast if empty."""
    df = pd.read_csv(path)
    if df.empty:
        print("CSV is empty.", file=sys.stderr)
        sys.exit(1)
    return df

# -- Time axis --------------------------------------------------------------
def time_axis(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Return relative time series in seconds and label."""
    if "time_s" in df.columns:
        t0 = float(df["time_s"].iloc[0])
        return df["time_s"] - t0, "time (s)"
    if "step" in df.columns:
        s0 = float(df["step"].iloc[0])
        return df["step"] - s0, "time (s)"
    return pd.Series(range(len(df)), name="idx"), "time (s)"

# -- Main plotting ----------------------------------------------------------
def plot_simple(csv_path: str) -> None:
    """Plot d1_linear and d2_triangle vs time; show only."""
    df = load_csv(csv_path)
    for col in ("d1_linear", "d2_triangle"):
        if col not in df.columns:
            print(f"Missing column: {col}. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # --- Encounter counting (print steps per contiguous run below threshold) ---
    THRESH = 0.8  # meters; use 0.08 if you truly meant 8 cm
    for name in ("d1_linear", "d2_triangle"):
        inside = (df[name] < THRESH)
        cnt = 0
        for flag in inside:
            if flag:
                cnt += 1
            elif cnt > 0:
                print(f"Encounter steps are {cnt} [{name}]")
                cnt = 0
        if cnt > 0:
            print(f"Encounter steps are {cnt} [{name}]")

            
    t, xlabel = time_axis(df)

    plt.figure()
    plt.plot(t, df["d1_linear"], linestyle="--", label="d1_linear")
    plt.plot(t, df["d2_triangle"], linestyle=":", label="d2_triangle")
    plt.xlabel(xlabel)
    plt.ylabel("distance (m)")
    plt.ylim(0, 5)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot d1 and d2 distances vs time.")
    ap.add_argument("csv_path", help="Path to metrics CSV")
    args = ap.parse_args()
    plot_simple(args.csv_path)

if __name__ == "__main__":
    main()
