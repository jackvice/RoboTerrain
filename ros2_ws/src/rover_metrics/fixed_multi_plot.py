#!/usr/bin/env python3
"""
Multi-CSV timeline plotter for robot encounter metrics.
Usage: python multi_plot.py /path/to/csvs
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


def get_total_goals_from_files(file_paths: List[str]) -> int:
    """Get total goals by summing max from each file."""
    total: int = 0
    for path in file_paths:
        df: pd.DataFrame = pd.read_csv(path)
        if not df.empty and 'goals' in df.columns:
            total += int(df['goals'].max())
    return total


def detect_encounters_for_single_actor(times: pd.Series, distances: pd.Series) -> List[Tuple[float, float]]:
    """Detect encounter periods for one actor, return (time, min_distance) for each encounter."""
    encounters: List[Tuple[float, float]] = []
    
    if distances.empty:
        return encounters
    
    in_encounter: bool = False
    current_encounter_min_dist: float = float('inf')
    current_encounter_min_time: float = 0.0
    
    for i, (time, dist) in enumerate(zip(times, distances)):
        # Skip NaN values
        if pd.isna(dist):
            continue
            
        currently_within: bool = dist < 1.2
        
        if currently_within:
            if not in_encounter:
                # Starting new encounter
                in_encounter = True
                current_encounter_min_dist = dist
                current_encounter_min_time = time
            else:
                # Continuing encounter, update minimum if this is closer
                if dist < current_encounter_min_dist:
                    current_encounter_min_dist = dist
                    current_encounter_min_time = time
        else:
            if in_encounter:
                # Ending encounter, record the minimum point
                encounters.append((current_encounter_min_time, current_encounter_min_dist))
                in_encounter = False
                current_encounter_min_dist = float('inf')
                current_encounter_min_time = 0.0
    
    # Handle case where data ends during an encounter
    if in_encounter:
        encounters.append((current_encounter_min_time, current_encounter_min_dist))
    
    return encounters


def detect_all_actor_encounters(df: pd.DataFrame, times: pd.Series) -> List[Tuple[float, float]]:
    """Detect encounters for all actors, return combined list of (time, min_distance) points."""
    all_encounters: List[Tuple[float, float]] = []
    
    # Process each actor separately
    for col in ['d1_linear', 'd2_triangle', 'd3_diag']:
        if col in df.columns:
            actor_encounters = detect_encounters_for_single_actor(times, df[col])
            all_encounters.extend(actor_encounters)
    
    return all_encounters


def get_total_goals(df: pd.DataFrame) -> int:
    """Get total number of goals reached."""
    if df.empty or 'goals' not in df.columns:
        return 0
    return int(df['goals'].max())


def classify_encounters_by_min_distance(encounter_points: List[Tuple[float, float]]) -> Tuple[int, int, int]:
    """Classify encounter points by their minimum distance. Returns (count_08_12, count_05_08, count_below_05)."""
    if not encounter_points:
        return (0, 0, 0)
    
    encounters_08_12: int = 0
    encounters_05_08: int = 0
    encounters_below_05: int = 0
    
    for time, min_dist in encounter_points:
        if min_dist < 0.5:
            encounters_below_05 += 1
        elif min_dist < 0.8:
            encounters_05_08 += 1
        else:  # min_dist < 1.2 (since these are encounter points)
            encounters_08_12 += 1
    
    return (encounters_08_12, encounters_05_08, encounters_below_05)


def plot_timeline(encounter_points: List[Tuple[float, float]], total_goals: int) -> None:
    """Plot timeline with encounter points in separate distance ranges."""
    plt.figure(figsize=(12, 6))
    
    if not encounter_points:
        plt.text(0.5, 0.5, 'No encounters detected', ha='center', va='center', transform=plt.gca().transAxes)
        plt.xlabel('Time (min)')
        plt.ylabel('Distance (m)')
        plt.title(f'Robot-Actor Encounter Timeline without Attention Mechanism (Success Rate: {total_goals})')
        plt.show()
        return
    
    # Extract times and distances from encounter points
    encounter_times: List[float] = [point[0] for point in encounter_points]
    encounter_dists: List[float] = [point[1] for point in encounter_points]
    
    # Convert to numpy arrays for easier masking
    encounter_times_array = np.array(encounter_times)
    encounter_dists_array = np.array(encounter_dists)
    
    # Classify encounters by their minimum distance reached
    encounters_08_12, encounters_05_08, encounters_below_05 = classify_encounters_by_min_distance(encounter_points)
    
    # Create masks for each distance range
    range_08_12 = (encounter_dists_array >= 0.8) & (encounter_dists_array < 1.2)
    range_05_08 = (encounter_dists_array >= 0.5) & (encounter_dists_array < 0.8)
    range_below_05 = encounter_dists_array < 0.5
    
    # Plot each range with different colors (no lines, just dots)
    plt.plot(encounter_times_array[range_08_12], encounter_dists_array[range_08_12], 'go', markersize=5, label=f'0.8m-1.2m ({encounters_08_12} encounters)')
    plt.plot(encounter_times_array[range_05_08], encounter_dists_array[range_05_08], 'bo', markersize=5, label=f'0.5m-0.8m ({encounters_05_08} encounters)')
    plt.plot(encounter_times_array[range_below_05], encounter_dists_array[range_below_05], 'ro', markersize=5, label=f'<0.5m ({encounters_below_05} encounters)')
    
    # Add horizontal threshold lines
    plt.axhline(y=1.2, color='red', linestyle='--', alpha=0.7, label='1.2m threshold')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='0.8m threshold') 
    plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='0.5m threshold')
    
    plt.xlabel('Time (min)')
    plt.ylabel('Distance (m)')
    plt.title(f'Robot-Actor Encounter Timeline without Attention Mechanism (Success Rate: {total_goals})')
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
    times: pd.Series = pd.Series(range(len(df))) / 60.0
    encounter_points: List[Tuple[float, float]] = detect_all_actor_encounters(df, times)
    total_goals: int = get_total_goals_from_files(file_paths)
    
    plot_timeline(encounter_points, total_goals)


if __name__ == "__main__":
    main()