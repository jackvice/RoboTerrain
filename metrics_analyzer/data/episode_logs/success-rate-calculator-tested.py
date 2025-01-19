#!/usr/bin/env python3

import argparse
import pandas as pd
from datetime import datetime

def calculate_success_rate(log_file, start_time_str, window_minutes):
    # Read lines and parse manually to ensure correct types
    timestamps = []
    events = []
    episodes = []
    positions = []
    
    with open(log_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                timestamps.append(float(parts[0]))
                events.append(parts[1])
                episodes.append(int(parts[2]))
                positions.append(','.join(parts[3:]))
    
    # Create DataFrame with correct types
    df = pd.DataFrame({
        'timestamp': timestamps,
        'event': events,
        'episode': episodes,
        'position': positions
    })
    
    # Convert input time to timestamp
    dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    epoch_offset = 1737225559.0245466 - datetime(2024, 1, 18, 14, 45, 0).timestamp()
    start_time = dt.timestamp() + epoch_offset
    end_time = start_time + (window_minutes * 60)
    
    print(f"\nAnalyzing from: {start_time_str} for {window_minutes} minutes")
    print(f"Using time window: {start_time} to {end_time}")
    
    # Filter for time window
    window_data = df[
        (df['timestamp'] >= start_time) & 
        (df['timestamp'] <= end_time)
    ].copy()
    
    window_data = window_data.sort_values('timestamp')
    print(f"Found {len(window_data)} entries in time window")
    
    # Process episodes
    valid_episodes = set()
    completed_episodes = set()
    episode_start_times = {}
    current_episode = None
    
    for idx, row in window_data.iterrows():
        if row['event'] == 'episode_start':
            # If we had a previous episode, check if it was valid
            if current_episode is not None:
                episode_duration = row['timestamp'] - episode_start_times[current_episode]
                if episode_duration > 5.0:
                    valid_episodes.add(current_episode)
            
            current_episode = row['episode']
            episode_start_times[current_episode] = row['timestamp']
            
        elif row['event'] == 'goal_reached':
            if current_episode == row['episode']:
                episode_duration = row['timestamp'] - episode_start_times[current_episode]
                if episode_duration > 5.0:
                    valid_episodes.add(current_episode)
                    completed_episodes.add(current_episode)
    
    # Check the last episode if it exists
    if current_episode is not None and window_data['timestamp'].max() - episode_start_times[current_episode] > 3.0:
        valid_episodes.add(current_episode)
    
    # Calculate success rate
    total_valid = len(valid_episodes)
    total_completed = len(completed_episodes)
    success_rate = (total_completed / total_valid * 100) if total_valid > 0 else 0
    
    print(f"\nSuccess Rate Analysis:")
    print(f"Time Window: {start_time_str} to {datetime.fromtimestamp(start_time + window_minutes*60 - epoch_offset)}")
    print(f"Valid Episodes (>3s duration): {total_valid}")
    print(f"Successful Episodes: {total_completed}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    if total_valid > 0:
        print("\nValid Episodes:", sorted(list(valid_episodes)))
        print("Completed Episodes:", sorted(list(completed_episodes)))

def main():
    parser = argparse.ArgumentParser(description='Calculate robot navigation success rate')
    parser.add_argument('log_file', help='Path to episode log CSV file')
    parser.add_argument('start_time', help='Start time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('window_minutes', type=int, help='Analysis window in minutes')
    
    args = parser.parse_args()
    calculate_success_rate(args.log_file, args.start_time, args.window_minutes)

if __name__ == '__main__':
    main()
