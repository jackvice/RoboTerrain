#!/usr/bin/env python3
# box_beehive_dmin.py
# Usage:
#   python box_beehive_dmin.py --cond "Attention" att1.csv att2.csv ... \
#                              --cond "No Attention" noatt1.csv ... \
#                              --out box_beehive.png --show

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_group(label, files, max_val=1.2):
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if "dmin" not in df.columns:
            raise ValueError(f"{f} has no 'dmin' column.")
        v = df["dmin"].astype(float).to_numpy()
        v = v[~np.isnan(v)]
        if max_val is not None:
            v = v[v <= max_val]  # keep only encounters within social zone
        vals.append(v)
    if not vals:
        raise ValueError(f"No CSVs for condition '{label}'.")
    v = np.concatenate(vals)
    if v.size == 0:
        raise ValueError(f"Condition '{label}' has no valid dmin values after filtering.")
    return label, v

def main():
    ap = argparse.ArgumentParser(description="Box + beeswarm (beehive) of dmin by condition (≤ 1.2 m).")
    ap.add_argument("--cond", nargs="+", action="append", required=True,
                    metavar=("LABEL", "CSV"),
                    help="Condition label followed by one or more CSV paths.")
    ap.add_argument("--out", type=str, default="box_beehive_dmin.png",
                    help="Output image filename.")
    ap.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    args = ap.parse_args()

    labels, data = [], []
    for group in args.cond:
        label, *files = group
        lab, arr = load_group(label, files, max_val=1.2)
        labels.append(lab)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(7, 5))

    # --- Boxplot ---
    box = ax.boxplot(
        data,
        positions=np.arange(1, len(labels) + 1),
        widths=0.5,
        showmeans=True,
        patch_artist=True,
        manage_ticks=False,
    )

    # Color boxes (2nd a different color)
    facecolors = ["#8c8c8c"] * len(labels)
    if len(labels) > 1:
        facecolors[1] = "#1f77b4"
    for patch, fc in zip(box["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)
    for elem in ("whiskers", "caps", "medians", "means"):
        for line in box[elem]:
            line.set_color("black")
            line.set_linewidth(1.2)

    # --- Beeswarm / beehive (jittered points) ---
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        # jitter width relative to box width
        x = np.full_like(arr, i, dtype=float) + rng.normal(0, 0.06, size=arr.size)
        ax.scatter(x, arr, s=16, alpha=0.7, edgecolors="none", color=facecolors[i-1])

    # Reference lines for social zones
    for y in (0.5, 0.8, 1.2):
        ax.axhline(y, linestyle="--", linewidth=1, color="k")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.set_ylim(0.0, 1.25)  # focus on ≤1.2 m (with a little headroom)
    ax.set_ylabel("Minimum distance to human, dmin (m)")
    ax.set_title("Encounter proximity (≤ 1.2 m): box + beehive by condition")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=fc, edgecolor="black", label=lab) for fc, lab in zip(facecolors, labels)],
              frameon=False, loc="upper right")

    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=300)
    print(f"Saved {out.resolve()}")

    # Quick stats
    for lab, arr in zip(labels, data):
        q = np.quantile(arr, [0.25, 0.5, 0.75])
        print(f"[{lab}] n={arr.size}  mean={arr.mean():.3f}  median={q[1]:.3f}  IQR=({q[0]:.3f}, {q[2]:.3f})")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
