#!/usr/bin/env python3
"""
Multi-CSV timeline plotter for robot encounter metrics.
Usage: python multi_plot.py /path/to/csvs
"""

from typing import List
import sys
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def scan_csv_files(directory: str) -> List[str]:
    """Scan directory for CSV files, return sorted by filename."""
    pattern: str = os.path.join(directory, "*.csv")
    return sorted(glob.glob(pattern))


def load_and_concatenate_csvs(file_paths: List[str]) -> pd.DataFrame:
    """Load multiple CSV files and concatenate preserving timestamps."""
    dfs: List[pd.DataFrame] = []
    for path in file_paths:
        df: pd.DataFrame = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def calculate_min_distances(df: pd.DataFrame) -> pd.Series:
    """Extract minimum distances from available columns."""
    # Check if we have dmin column (pre-calculated minimum)
    if 'dmin' in df.columns:
        return df['dmin']
    
    # Fallback: calculate from individual distance columns
    distance_cols: List[str] = []
    for col in ['d1_linear', 'd2_triangle', 'd3_diag']:
        if col in df.columns:
            distance_cols.append(col)
    
    if distance_cols:
        distances = df[distance_cols].values
        return pd.Series(np.nanmin(distances, axis=1))
    
    # No distance data found
    return pd.Series(np.full(len(df), np.nan))


def get_total_goals(df: pd.DataFrame) -> int:
    """Get total number of goals reached."""
    if df.empty or 'goals' not in df.columns:
        return 0
    return int(df['goals'].max())


def count_encounters_below_threshold(min_dists: pd.Series, threshold: float) -> int:
    """Count distinct encounters below threshold."""
    if min_dists.empty:
        return 0
    
    encounter_count: int = 0
    in_encounter: bool = False
    
    for dist in min_dists:
        if pd.isna(dist):
            continue
            
        currently_within: bool = dist < threshold
        
        if currently_within and not in_encounter:
            encounter_count += 1
            in_encounter = True
        elif not currently_within:
            in_encounter = False
    
    return encounter_count


def plot_timeline(times: pd.Series, min_dists: pd.Series, total_goals: int) -> None:
    """Plot timeline with encounter points below 1.2m threshold."""
    plt.figure(figsize=(12, 6))
    
    # Filter to only plot encounters below 1.2m threshold
    encounter_mask = min_dists < 1.2
    encounter_times = times[encounter_mask]
    encounter_dists = min_dists[encounter_mask]
    
    # Count encounters below thresholds
    encounters_05: int = count_encounters_below_threshold(min_dists, 0.5)
    encounters_08: int = count_encounters_below_threshold(min_dists, 0.8)
    
    # Plot encounter points with lines
    #plt.plot(encounter_times, encounter_dists, 'bo-', markersize=3, linewidth=1, label='Encounters')
    plt.plot(encounter_times, encounter_dists, 'bo', markersize=3, label='Encounters')

    
    # Add horizontal threshold lines with encounter counts in legend
    plt.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='1.2m threshold')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label=f'0.8m threshold ({encounters_08} encounters)') 
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label=f'0.5m threshold ({encounters_05} encounters)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (m)')
    plt.title(f'Robot-Actor Encounter Timeline (Total Goals: {total_goals})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python multi_plot.py /path/to/csvs")
        sys.exit(1)
    
    directory: str = sys.argv[1]
    file_paths: List[str] = scan_csv_files(directory)
    
    if not file_paths:
        print(f"No CSV files found in {directory}")
        sys.exit(1)
    
    print(f"Found {len(file_paths)} CSV files")
    
    df: pd.DataFrame = load_and_concatenate_csvs(file_paths)
    min_dists: pd.Series = calculate_min_distances(df)
    total_goals: int = get_total_goals(df)
    
    times: pd.Series = df['time_s'] if 'time_s' in df.columns else pd.Series(range(len(df)))
    
    plot_timeline(times, min_dists, total_goals)


if __name__ == "__main__":
    main()
