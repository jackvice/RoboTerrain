#!/usr/bin/env python3
"""
Convert statistical results CSV to compact LaTeX table for publication.
Usage: python csv_to_latex.py island_results.csv --output island_table.tex
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd


METRIC_LABELS: Dict[str, str] = {
    "p05": r"$p_{0.5}$",
    "goal_rate": "Goal rate",
    "encounters_per_goal": r"Encounters ($<$0.5\,m) per goal",
}

METRIC_ORDER: List[str] = ["p05", "goal_rate", "encounters_per_goal"]


def format_p_value(p: float) -> str:
    """Format p-value for compact display."""
    if p < 0.001:
        return "$<$0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def build_latex_table(df: pd.DataFrame) -> str:
    """Build compact LaTeX table: Comparison | delta | d | p | sig."""
    env_name: str = df["environment"].iloc[0].capitalize()

    lines: List[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrl}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Comparison} & "
        r"\textbf{$\Delta$} & \textbf{$d$} & "
        r"\textbf{$p_{\mathrm{corr}}$} & \\"
    )
    lines.append(r"\midrule")

    grouped: Dict[str, pd.DataFrame] = {m: g for m, g in df.groupby("metric", sort=False)}

    for metric_idx, metric_key in enumerate(METRIC_ORDER):
        if metric_key not in grouped:
            continue

        metric_df: pd.DataFrame = grouped[metric_key]
        metric_label: str = METRIC_LABELS[metric_key]

        # Metric as a spanning header row
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{metric_label}}}}} \\")

        for _, row in metric_df.iterrows():
            comparison: str = f"{row['condition_a']} vs {row['condition_b']}"
            delta: str = f"{row['mean_diff']:.4f}"
            d_str: str = f"{row['cohen_d']:.2f}"
            p_str: str = format_p_value(row["p_corrected"])
            sig: str = row["significant"] if pd.notna(row["significant"]) else ""

            lines.append(
                f"\\quad {comparison} & {delta} & {d_str} & {p_str} & {sig} \\\\"
            )

        if metric_idx < len(METRIC_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Statistical comparisons for " + env_name + r" environment. "
        r"$\Delta$: difference in mean per-block rates (A$-$B). "
        r"$d$: Cohen's $d$. "
        r"$p_{\mathrm{corr}}$: Holm--Bonferroni corrected. "
        r"$p_{0.5}$: fraction of time $d_{\min} < 0.5$\,m; "
        r"goal rate in goals/min; "
        r"encounters per goal: count of $d_{\min} < 0.5$\,m seconds per goal achieved. "
        r"*\,$p < .05$, **\,$p < .01$, ***\,$p < .001$.}"
    )
    lines.append(r"\label{tab:stats_" + env_name.lower() + r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert statistical results CSV to LaTeX table"
    )
    parser.add_argument("csv_path", type=str, help="Path to results CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .tex path (default: same name as input with .tex)")
    args: argparse.Namespace = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"Error: {args.csv_path} not found")
        sys.exit(1)

    output_path: str = args.output or args.csv_path.rsplit(".", 1)[0] + ".tex"

    df: pd.DataFrame = pd.read_csv(args.csv_path)
    latex: str = build_latex_table(df)

    with open(output_path, "w") as f:
        f.write(latex)

    print(f"LaTeX table written to {output_path}")
    print(f"\nPreview:\n{latex}")


if __name__ == "__main__":
    main()
