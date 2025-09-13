#
from typing import List
import pandas as pd
import sys

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file using pandas."""
    return pd.read_csv(filepath)

def filter_usable_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where at least one of d1_linear or d2_triangle is not NaN."""
    return df[~(df['d1_linear'].isna() & df['d2_triangle'].isna())]

def is_within_threshold(d1: float, d2: float, threshold: float) -> bool:
    """Check if either distance is within threshold, handling NaN values."""
    valid_distances: List[float] = []
    
    if not pd.isna(d1):
        valid_distances.append(d1)
    if not pd.isna(d2):
        valid_distances.append(d2)
    
    if not valid_distances:
        return False
    
    return min(valid_distances) < threshold

def count_encounters(df: pd.DataFrame, threshold: float) -> int:
    """Count distinct encounters where robot gets within threshold distance."""
    if df.empty:
        return 0
    
    encounter_count: int = 0
    in_encounter: bool = False
    
    for _, row in df.iterrows():
        d1: float = row['d1_linear']
        d2: float = row['d2_triangle']
        
        currently_within: bool = is_within_threshold(d1, d2, threshold)
        
        if currently_within and not in_encounter:
            encounter_count += 1
            in_encounter = True
        elif not currently_within:
            in_encounter = False
    
    return encounter_count

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
    
    encounters_1_2: int = count_encounters(filtered_df, 1.2)
    encounters_0_8: int = count_encounters(filtered_df, 0.8)
    encounters_0_5: int = count_encounters(filtered_df, 0.5)
    total_goals: int = get_total_goals(df)
    
    print(f"Encounters < 1.2m: {encounters_1_2}")
    print(f"Encounters < 0.8m: {encounters_0_8}")
    print(f"Encounters < 0.5m: {encounters_0_5}")
    print(f"SR (Success Rate): {total_goals}")

if __name__ == "__main__":
    main()
