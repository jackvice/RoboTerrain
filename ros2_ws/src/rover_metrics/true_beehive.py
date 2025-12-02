#!/usr/bin/env python3
# box_beehive_dmin.py
# Usage:
#   python box_beehive_dmin.py --cond "Attention" path/to/dir_or.csv ... \
#                              --cond "No Attention" path/to/dir_or.csv ... \
#                              --out box_beehive.png --show

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



# ↑ Set a larger global font size for everything on the plot
plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.titlesize": 18,     # axes title
    "axes.labelsize": 16,     # axes labels
    "xtick.labelsize": 14,    # tick labels
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def expand_csvs(paths):
    """Expand a list of file/dir paths into a flat list of CSV files."""
    csvs = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            csvs.extend(sorted(str(x) for x in pth.glob("*.csv")))
        else:
            csvs.append(str(pth))
    return csvs

def load_group(label, files_or_dirs, max_val=1.2):
    """Load and concatenate dmin from CSVs (≤ max_val meters)."""
    files = expand_csvs(files_or_dirs)
    if not files:
        raise ValueError(f"No CSVs found for condition '{label}'.")
    vals = []
    for f in files:
        df = pd.read_csv(f)
        if "dmin" not in df.columns:
            raise ValueError(f"{f} has no 'dmin' column.")
        v = df["dmin"].astype(float).to_numpy()
        v = v[~np.isnan(v)]
        if max_val is not None:
            v = v[v <= max_val]  # keep only encounters within social zone
        if v.size:
            vals.append(v)
    if not vals:
        raise ValueError(f"Condition '{label}' has no valid dmin values after filtering.")
    v = np.concatenate(vals)
    return label, v



def main():
    import seaborn as sns
    
    ap = argparse.ArgumentParser(description="Box + beeswarm (beehive) of dmin by condition (≤ 1.2 m).")
    ap.add_argument("--cond", nargs="+", action="append", required=True,
                    metavar=("LABEL", "PATH"),
                    help="Condition label followed by one or more CSV files or directories containing CSVs.")
    ap.add_argument("--out", type=str, default="box_beehive_dmin2.png",
                    help="Output image filename.")
    ap.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    args = ap.parse_args()

    labels, data = [], []
    for group in args.cond:
        label, *paths = group
        lab, arr = load_group(label, paths, max_val=1.2)
        labels.append(lab)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color boxes (2nd a different color)
    facecolors = ["#8c8c8c"] * len(labels)
    if len(labels) > 1:
        facecolors[1] = "#1f77b4"

    # --- Beeswarm / beehive using seaborn ---
    # Prepare data in long format
    plot_data = pd.DataFrame({
        'condition': np.concatenate([np.full(len(arr), labels[i]) for i, arr in enumerate(data)]),
        'dmin': np.concatenate(data)
    })
    
    # Create swarmplot (seaborn uses 0-indexed positions)
    sns.swarmplot(
        data=plot_data,
        x='condition',
        y='dmin',
        order=labels,
        hue='condition',
        hue_order=labels,
        palette=facecolors,
        size=5.5,
        alpha=0.6,
        legend=False,
        ax=ax
    )

    # --- Boxplot (now using 0-indexed positions to match seaborn) ---
    box = ax.boxplot(
        data,
        positions=np.arange(len(labels)),  # Changed from np.arange(1, len(labels) + 1)
        widths=0.5,
        showmeans=True,
        patch_artist=True,
        manage_ticks=False,
    )

    for patch, fc in zip(box["boxes"], facecolors):
        patch.set_facecolor(fc)
        patch.set_edgecolor("black")
        patch.set_alpha(0.4)
    for elem in ("whiskers", "caps", "medians", "means"):
        for line in box[elem]:
            line.set_color("black")
            line.set_linewidth(1.4)

    # Reference lines for social zones
    for y in (0.5, 0.8, 1.2):
        ax.axhline(y, linestyle="--", linewidth=1, color="k")

    ax.set_xticks(range(len(labels)))  # Changed to 0-indexed
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)  # Adjusted for 0-indexing
    ax.set_ylim(0.0, 1.25)
    ax.set_ylabel("Minimum distance to human, dmin (m)")
    ax.set_title("Encounter proximity (≤ 1.2 m): box + beehive by condition")

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=fc, edgecolor="black", label=lab) for fc, lab in zip(facecolors, labels)],
              frameon=False, loc="lower right")

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


        
def main_old():
    ap = argparse.ArgumentParser(description="Box + beeswarm (beehive) of dmin by condition (≤ 1.2 m).")
    ap.add_argument("--cond", nargs="+", action="append", required=True,
                    metavar=("LABEL", "PATH"),
                    help="Condition label followed by one or more CSV files or directories containing CSVs.")
    ap.add_argument("--out", type=str, default="box_beehive_dmin2.png",
                    help="Output image filename.")
    ap.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    args = ap.parse_args()

    labels, data = [], []
    for group in args.cond:
        label, *paths = group
        lab, arr = load_group(label, paths, max_val=1.2)
        labels.append(lab)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(8, 6))

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
            line.set_linewidth(1.4)

    # --- Beeswarm / beehive (jittered points) ---
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        x = np.full_like(arr, i, dtype=float) + rng.normal(0, 0.06, size=arr.size)
        ax.scatter(x, arr, s=22, alpha=0.7, edgecolors="none", color=facecolors[i-1])

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
              frameon=False, loc="lower right")

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
