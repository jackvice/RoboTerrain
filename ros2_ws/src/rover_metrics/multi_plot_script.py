#!/usr/bin/env python3
"""
Multi-CSV timeline plotter for robot encounter metrics.
Usage: python multi_plot.py /path/to/csvs_directory
   or: python multi_plot.py /path/to/single_file.csv
"""

from typing import List, Tuple
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


def classify_encounters_by_min_distance(min_dists: pd.Series) -> Tuple[int, int, int]:
    """Classify encounters by their minimum distance reached. Returns (count_08_12, count_05_08, count_below_05)."""
    if min_dists.empty:
        return (0, 0, 0)
    
    encounters_08_12: int = 0
    encounters_05_08: int = 0
    encounters_below_05: int = 0
    
    in_encounter: bool = False
    current_encounter_min: float = float('inf')
    
    for dist in min_dists:
        if pd.isna(dist):
            continue
            
        currently_within: bool = dist < 1.2
        
        if currently_within:
            if not in_encounter:
                # Starting new encounter
                in_encounter = True
                current_encounter_min = dist
            else:
                # Continuing encounter, update minimum
                current_encounter_min = min(current_encounter_min, dist)
        else:
            if in_encounter:
                # Ending encounter, classify it
                if current_encounter_min < 0.5:
                    encounters_below_05 += 1
                elif current_encounter_min < 0.8:
                    encounters_05_08 += 1
                else:  # current_encounter_min < 1.2
                    encounters_08_12 += 1
                
                in_encounter = False
                current_encounter_min = float('inf')
    
    # Handle case where data ends during an encounter
    if in_encounter:
        if current_encounter_min < 0.5:
            encounters_below_05 += 1
        elif current_encounter_min < 0.8:
            encounters_05_08 += 1
        else:
            encounters_08_12 += 1
    
    return (encounters_08_12, encounters_05_08, encounters_below_05)


def plot_timeline(times: pd.Series, min_dists: pd.Series, total_goals: int) -> None:
    """Plot timeline with encounter points in separate distance ranges."""
    plt.figure(figsize=(12, 6))
    
    # Filter to only plot encounters below 1.2m threshold
    encounter_mask = min_dists < 1.2
    encounter_times = times[encounter_mask]
    encounter_dists = min_dists[encounter_mask]
    
    # Classify encounters by their minimum distance reached
    encounters_08_12, encounters_05_08, encounters_below_05 = classify_encounters_by_min_distance(min_dists)
    
    # Create masks for each distance range
    range_08_12 = (encounter_dists >= 0.8) & (encounter_dists < 1.2)
    range_05_08 = (encounter_dists >= 0.5) & (encounter_dists < 0.8)
    range_below_05 = encounter_dists < 0.5
    
    # Plot each range with different colors (no lines, just dots)
    plt.plot(encounter_times[range_08_12], encounter_dists[range_08_12], 'yo', markersize=5, label=f'0.8m-1.2m ({encounters_08_12} encounters)')
    plt.plot(encounter_times[range_05_08], encounter_dists[range_05_08], 'o', color='orange', markersize=5, label=f'0.5m-0.8m ({encounters_05_08} encounters)')
    plt.plot(encounter_times[range_below_05], encounter_dists[range_below_05], 'ro', markersize=5, label=f'<0.5m ({encounters_below_05} encounters)')
    
    # Add horizontal threshold lines
    plt.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='1.2m threshold')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='0.8m threshold') 
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='0.5m threshold')
    
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