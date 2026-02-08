#!/usr/bin/env python3
# ecdf_dmin.py
# Usage:
#   python ecdf_dmin.py --cond "Attention" att1.csv att2.csv ... \
#                       --cond "No Attention" noatt1.csv noatt2.csv ... \
#                       --max 1.2 --out ecdf_dmin.png --show

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_group(label, files, max_val=None):
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if "dmin" not in df.columns:
            raise ValueError(f"{f} has no 'dmin' column.")
        v = df["dmin"].astype(float).to_numpy()
        v = v[~np.isnan(v)]
        v = v[v < 1.2]
        if max_val is not None:
            v = v[v <= max_val]
        vals.append(v)
    if not vals:
        raise ValueError(f"No CSVs provided for condition '{label}'.")
    v = np.concatenate(vals)
    if v.size == 0:
        raise ValueError(f"Condition '{label}' has no valid dmin values after filtering.")
    return label, v

def ecdf(y):
    # Return sorted data and ECDF values in [0,1]
    x = np.sort(y)
    n = x.size
    p = np.arange(1, n + 1) / n
    return x, p

def main():
    ap = argparse.ArgumentParser(description="ECDF of dmin across conditions.")
    ap.add_argument("--cond", nargs="+", action="append", required=True,
                    metavar=("LABEL", "CSV"),
                    help="Condition label followed by one or more CSV paths.")
    ap.add_argument("--max", type=float, default=None,
                    help="Optional upper filter for dmin (e.g., 1.2 to focus on social zone).")
    ap.add_argument("--out", type=str, default="ecdf_dmin.png",
                    help="Output image filename (default: ecdf_dmin.png)")
    ap.add_argument("--show", action="store_true",
                    help="Display the plot window after saving.")
    args = ap.parse_args()

    labels, data = [], []
    for group in args.cond:
        label, *files = group
        lab, arr = load_group(label, files, max_val=args.max)
        labels.append(lab)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for lab, arr in zip(labels, data):
        x, p = ecdf(arr)
        ax.step(x, p, where="post", label=f"{lab} (n={arr.size})")

    ax.set_xlabel("Minimum distance to human, dmin (m)")
    ax.set_ylabel("ECDF: P(dmin ≤ x)")
    ttl = "ECDF of encounter proximity"
    if args.max is not None:
        ttl += f" (≤ {args.max} m)"
    ax.set_title(ttl)

    # Social safety reference lines
    for y in (0.5, 0.8, 1.2):
        ax.axvline(y, linestyle="--", linewidth=1)

    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    out = Path(args.out)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"Saved {out.resolve()}")

    # Quick stats
    for lab, arr in zip(labels, data):
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        print(f"[{lab}] n={arr.size}  mean={arr.mean():.3f}  "
              f"median={q[1]:.3f}  IQR=({q[0]:.3f}, {q[2]:.3f})")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
