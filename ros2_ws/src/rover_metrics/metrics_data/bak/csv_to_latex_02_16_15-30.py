#!/usr/bin/env python3
"""
Convert statistical results CSV to compact LaTeX table for publication.
Updated for run-level permutation test output.

Usage: python csv_to_latex.py island_results.csv --output island_table.tex
"""

import argparse
import os
import sys
from typing import Dict, List

import pandas as pd


METRIC_LABELS: Dict[str, str] = {
    "goals_per_encounter": r"Goals per encounter ($d_{\min} < 0.5$\,m)",
    "secs_per_goal": "Seconds per goal",
}

METRIC_ORDER: List[str] = ["goals_per_encounter", "secs_per_goal"]


def format_p_value(p: float) -> str:
    """Format p-value for compact display."""
    if p < 0.001:
        return "$<$0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def build_latex_table(df: pd.DataFrame) -> str:
    """Build compact LaTeX table: Comparison | diff | 95% CI | d | p | sig."""
    env_name: str = df["environment"].iloc[0].capitalize()

    lines: List[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Comparison} & "
        r"\textbf{$\Delta$} & \textbf{95\% CI} & "
        r"\textbf{$d$} & "
        r"\textbf{$p_{\mathrm{corr}}$} & \\"
    )
    lines.append(r"\midrule")

    grouped: Dict[str, pd.DataFrame] = {
        m: g for m, g in df.groupby("metric", sort=False)
    }

    for metric_idx, metric_key in enumerate(METRIC_ORDER):
        if metric_key not in grouped:
            continue

        metric_df: pd.DataFrame = grouped[metric_key]
        metric_label: str = METRIC_LABELS[metric_key]

        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{metric_label}}}}} \\")

        for _, row in metric_df.iterrows():
            comparison: str = f"{row['condition_a']} vs {row['condition_b']}"
            delta: str = f"{row['mean_diff']:.2f}"
            ci_str: str = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
            d_str: str = f"{row['cohen_d']:.2f}"
            p_str: str = format_p_value(row["p_perm"])
            sig: str = row["significant"] if pd.notna(row["significant"]) else ""

            lines.append(
                f"\\quad {comparison} & {delta} & {ci_str} & "
                f"{d_str} & {p_str} & {sig} \\\\"
            )

        if metric_idx < len(METRIC_ORDER) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Statistical comparisons for " + env_name + r" environment. "
        r"Each condition comprises $n = 3$ independent 30-minute runs. "
        r"$\Delta$: difference in means (A$-$B). "
        r"95\% CI: bootstrap confidence interval (10\,000 resamples). "
        r"$d$: Cohen's $d$. "
        r"$p_{\mathrm{corr}}$: exact permutation $p$-value with "
        r"Holm--Bonferroni correction. "
        r"Goals per encounter: navigation goals achieved per close-proximity "
        r"event ($d_{\min} < 0.5$\,m; higher = safer). "
        r"Seconds per goal: mean time to complete one goal (lower = faster). "
        r"With $n = 3$ per condition, the minimum achievable $p = 0.05$. "
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
