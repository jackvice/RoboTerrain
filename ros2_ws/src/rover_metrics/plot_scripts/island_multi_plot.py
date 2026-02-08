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


def resolve_csv_inputs(path: str) -> List[str]:
    """Return a list of CSV files from a directory or a single CSV file."""
    if os.path.isfile(path):
        return [path] if path.endswith(".csv") else []
    if os.path.isdir(path):
        return scan_csv_files(path)
    return []


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
    for col in ['d1_triangle', 'd2_triangle2', 'd3_triangle3']:
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


from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def plot_timeline(encounter_points: List[Tuple[float, float]], total_goals: int) -> None:
    """
    Plot encounter minima and place a single-row legend under the x-axis.
    Order: dot series first, then dotted threshold guides.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # -- Empty case
    if not encounter_points:
        ax.text(0.5, 0.5, 'No encounters detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Distance (m)')
        ax.set_title(f'Robot–Actor Encounter Timeline (Goals: {total_goals})')
        plt.show()
        return

    # -- Unpack
    times = np.array([t for t, _ in encounter_points], dtype=float)
    dists = np.array([d for _, d in encounter_points], dtype=float)

    # -- Mutually exclusive bins
    m_08_12 = (dists >= 0.8) & (dists < 1.2)
    m_05_08 = (dists >= 0.5) & (dists < 0.8)
    m_lt_05 = dists < 0.5

    # -- Counts (for labels)
    c_08_12 = int(m_08_12.sum())
    c_05_08 = int(m_05_08.sum())
    c_lt_05 = int(m_lt_05.sum())

    # -- Scatter (legend handles)
    h_08_12, = ax.plot(times[m_08_12], dists[m_08_12], 'yo', markersize=5, label=f'0.8–1.2 m ({c_08_12})')
    h_05_08, = ax.plot(times[m_05_08], dists[m_05_08], 'o', color='orange', markersize=5, label=f'0.5–0.8 m ({c_05_08})')
    h_lt_05, = ax.plot(times[m_lt_05], dists[m_lt_05], 'ro', markersize=5, label=f'<0.5 m ({c_lt_05})')

    # -- Dotted threshold guides (include in legend)
    th_12 = ax.axhline(1.2, color='red', linestyle='--', alpha=0.6, label='Threshold 1.2 m')
    th_08 = ax.axhline(0.8, color='orange', linestyle='--', alpha=0.6, label='Threshold 0.8 m')
    th_05 = ax.axhline(0.5, color='green', linestyle='--', alpha=0.6, label='Threshold 0.5 m')

    # -- Labels, grid, title
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Distance (m)')
    ax.set_title(f'Robot–Actor Encounter Timeline with Attention Mechanism (Success Rate: {total_goals})')
    ax.grid(True, alpha=0.3, linestyle='--')

    # -- Legend: dots first, then dashed thresholds, single row under x-axis
    handles = [h_08_12, h_05_08, h_lt_05, th_12, th_08, th_05]
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),     # adjust closer/farther as needed
        bbox_transform=ax.transAxes,     # relative to axes box
        ncol=len(labels),
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    # -- Reserve bottom space so legend isn't clipped
    plt.subplots_adjust(bottom=0.30)

    plt.show()


def main() -> None:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python multi_plot.py /path/to/csvs")
        sys.exit(1)
    
    #directory: str = sys.argv[1]
    #file_paths: List[str] = scan_csv_files(directory)

    input_path: str = sys.argv[1]
    file_paths: List[str] = resolve_csv_inputs(input_path)

    
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
