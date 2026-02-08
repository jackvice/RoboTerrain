#!/usr/bin/env python3
# violin_dmin.py
# Usage:
#   python violin_dmin.py --cond "Attention" att1.csv att2.csv ... \
#                         --cond "No Attention" noatt1.csv noatt2.csv ...
# (You can pass just one CSV per condition too.)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_group(label, files):
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if "dmin" not in df.columns:
            raise ValueError(f"{f} has no 'dmin' column.")
        vals.append(df["dmin"].astype(float).to_numpy())
    if not vals:
        raise ValueError(f"No CSVs provided for condition '{label}'.")
    v = np.concatenate(vals)
    v = v[~np.isnan(v)]
    v = v[v < 1.2]  # keep only encounters under 1.2 m

    return label, v

def main():
    parser = argparse.ArgumentParser(description="Violin plot of dmin across conditions.")
    # Repeatable --cond blocks: label then 1+ files
    parser.add_argument("--cond", nargs="+", action="append",
                        metavar=("LABEL", "CSV"), required=True,
                        help="Condition label followed by one or more CSV paths.")
    parser.add_argument("--out", type=str, default="violin_dmin.png",
                        help="Output image filename (default: violin_dmin.png)")
    args = parser.parse_args()

    labels, data = [], []
    for group in args.cond:
        label, *files = group
        if not files:
            raise ValueError(f"Condition '{label}' needs at least one CSV.")
        lab, arr = load_group(label, files)
        if arr.size == 0:
            raise ValueError(f"Condition '{label}' has no valid dmin values.")
        labels.append(lab)
        data.append(arr)

    # Create violin plot (single figure with both conditions)
    fig, ax = plt.subplots(figsize=(6, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)

    # color violins (indexing matches your labels order)
    for i, b in enumerate(parts['bodies']):
        b.set_edgecolor('black')
        b.set_alpha(0.85)
        b.set_facecolor('#8c8c8c')  # default for all

    if len(parts['bodies']) > 1:
        parts['bodies'][1].set_facecolor('#1f77b4')  # different color for the second violin

    
    # X tick labels at positions 1..N
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_ylabel("Minimum distance to human, dmin (m)")
    ax.set_title("Social navigation proximity: dmin distribution by condition")

    # Social safety reference lines
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.axhline(0.8, linestyle="--", linewidth=1)
    ax.axhline(1.2, linestyle="--", linewidth=1)
    ax.text(0.02, 0.5, "<0.5 m critical", transform=ax.get_yaxis_transform(), va="center")
    ax.text(0.02, 0.8, "0.5–0.8 m close", transform=ax.get_yaxis_transform(), va="center")
    ax.text(0.02, 1.2, "0.8–1.2 m personal", transform=ax.get_yaxis_transform(), va="center")

    # Save
    out = Path(args.out)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.show() 
    print(f"Saved {out.resolve()}")

    # Optional: print quick stats
    for lab, arr in zip(labels, data):
        q = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"[{lab}] n={arr.size}  mean={arr.mean():.3f}  median={q[2]:.3f}  "
              f"p10={q[0]:.3f}  p25={q[1]:.3f}  p75={q[3]:.3f}  p90={q[4]:.3f}")

if __name__ == "__main__":
    main()
