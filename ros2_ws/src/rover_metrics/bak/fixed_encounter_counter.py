#!/usr/bin/env python3
"""
Encounter counter for robot-actor proximity analysis.
Usage: python encounter_counter.py <csv_file>
"""

from typing import List
import pandas as pd
import sys


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file using pandas."""
    return pd.read_csv(filepath)


def filter_usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where at least one of d1_linear, d2_triangle, or d3_diag is not NaN."""
    return df[~(df['d1_linear'].isna() & df['d2_triangle'].isna() & df['d3_diag'].isna())]


def count_encounters_for_single_actor(distances: pd.Series, threshold: float) -> int:
    """Count distinct encounters for one actor at given threshold."""
    if distances.empty:
        return 0
    
    encounter_count: int = 0
    in_encounter: bool = False
    
    for dist in distances:
        # Skip NaN values
        if pd.isna(dist):
            continue
            
        currently_within: bool = dist < threshold
        
        if currently_within and not in_encounter:
            encounter_count += 1
            in_encounter = True
        elif not currently_within:
            in_encounter = False
    
    return encounter_count


def count_total_encounters(df: pd.DataFrame, threshold: float) -> int:
    """Count total encounters across all actors at given threshold."""
    total_encounters: int = 0
    
    # Count encounters for each actor separately
    for col in ['d1_linear', 'd2_triangle', 'd3_diag']:
        if col in df.columns:
            actor_encounters = count_encounters_for_single_actor(df[col], threshold)
            total_encounters += actor_encounters
    
    return total_encounters


def get_total_goals(df: pd.DataFrame) -> int:
    """Get total number of goals reached during the period."""
    if df.empty or 'goals' not in df.columns:
        return 0
    
    # Goals column is cumulative, so take the maximum value
    return int(df['goals'].max())


def main() -> None:
    """Main function to process CSV and count encounters."""
    if len(sys.argv) != 2:
        print("Usage: python encounter_counter.py <csv_file>")
        sys.exit(1)
    
    filepath: str = sys.argv[1]
    df: pd.DataFrame = load_csv(filepath)
    filtered_df: pd.DataFrame = filter_usable_rows(df)
    
    encounters_1_2: int = count_total_encounters(filtered_df, 1.2)
    encounters_0_8: int = count_total_encounters(filtered_df, 0.8)
    encounters_0_5: int = count_total_encounters(filtered_df, 0.5)
    total_goals: int = get_total_goals(df)
    
    print(f"Encounters < 1.2m: {encounters_1_2}")
    print(f"Encounters < 0.8m: {encounters_0_8}")
    print(f"Encounters < 0.5m: {encounters_0_5}")
    print(f"SR (Success Rate): {total_goals}")


if __name__ == "__main__":
    main()