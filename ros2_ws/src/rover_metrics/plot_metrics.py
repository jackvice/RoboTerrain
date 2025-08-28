# metrics_plot.py
# Usage: python metrics_plot.py my_csv_file.csv
from __future__ import annotations
import argparse
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os, time

# --- pure helpers ------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit("CSV is empty.")
    return df

def make_time_axis(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "time_s" in df.columns:
        t0 = float(df["time_s"].iloc[0])
        return df["time_s"] - t0, "time (s)"
    elif "step" in df.columns:
        s0 = float(df["step"].iloc[0])
        return df["step"] - s0, "time (s)"  # 1 Hz logging => steps ~ seconds
    else:
        return pd.Series(range(len(df)), name="idx"), "time (s)"

def pick_metrics(df: pd.DataFrame):
    # Only plot these on primary axis (if present). Never plot rx/ry.
    primary = [c for c in ["speed_mps", "d1_linear", "d2_triangle", "dmin"] if c in df.columns]
    secondary = "goals" if "goals" in df.columns else None
    return primary, secondary


    
def pick_metrics_old(df: pd.DataFrame) -> Tuple[List[str], str | None]:
    # Plot these on primary axis if present
    primary_pref = ["speed_mps", "d1_linear", "d2_triangle", "dmin"]
    # Secondary axis metric
    secondary = "goals" if "goals" in df.columns else None

    # Collect numeric columns
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Remove time-ish columns from plotting set
    for drop in ("time_s", "step"):
        if drop in numeric:
            numeric.remove(drop)
    # Build primary list in preferred order + any other leftover numeric cols
    primary = [c for c in primary_pref if c in numeric]
    leftovers = [c for c in numeric if c not in primary and c != secondary]
    primary += leftovers
    return primary, secondary

def savefig_name(csv_path: str) -> str:
    stamp = time.strftime("%m_%d_%H-%M")
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return f"{base}_plot_{stamp}.png"

# --- main plotting -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot metrics vs time from CSV.")
    ap.add_argument("csv_path", help="Path to metrics CSV")
    args = ap.parse_args()

    df = load_csv(args.csv_path)
    t, xlabel = make_time_axis(df)
    primary, secondary = pick_metrics(df)

    if not primary and not secondary:
        raise SystemExit("No plottable metrics found.")

    fig, ax = plt.subplots()
    lines = []
    labels = []

    # Primary axis metrics
    for col in primary:
        ln, = ax.plot(t, df[col], label=col)  # default styles/colors
        lines.append(ln); labels.append(col)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("metrics")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Secondary axis for goals (if present)
    if secondary is not None:
        ax2 = ax.twinx()
        ln2, = ax2.plot(t, df[secondary], label=secondary, linewidth=1.2)
        ax2.set_ylabel(secondary)
        lines.append(ln2); labels.append(secondary)

    ax.legend(lines, labels, loc="best")
    fig.tight_layout()

    out_path = savefig_name(args.csv_path)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
