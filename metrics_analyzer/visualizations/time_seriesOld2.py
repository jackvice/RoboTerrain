#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import argparse
from data_loader import MetricsDataLoader
from metrics_statistics import MetricsStatistics

class TimeSeriesVisualizer:
    def __init__(self):
        print("Initializing TimeSeriesVisualizer")
        sns.set_style("whitegrid")
        sns.set_context("paper")
        
        self.style_settings = {
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.title_fontsize': 10
        }
        plt.rcParams.update(self.style_settings)

    def plot_metric_time_series(self,
                              df: pd.DataFrame,
                              metric: str,
                              title: Optional[str] = None,
                              ci: bool = True) -> plt.Figure:
        print(f"Plotting time series for metric: {metric}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Create a copy of the dataframe to modify
        plot_df = df.copy()
        
        # Convert Unix timestamp to minutes from start
        start_time = plot_df['Timestamp'].min()
        plot_df['Minutes'] = (plot_df['Timestamp'] - start_time) / 60
        
        # Define metric mapping
        metric_mapping = {
            'TC': 'Total Collisions',
            'CS': 'Current Collision Status',
            'SM': 'Smoothness Metric',
            'OC': 'Obstacle Clearance',
            'DT': 'Distance Traveled',
            'CV': 'Current Velocity',
            'IM': 'IMU Acceleration Magnitude',
            'RT': 'Is Rough Terrain'
        }
        
        # Get the full column name
        column_name = metric_mapping.get(metric, metric)
        print(f"Looking for column: {column_name}")
        
        if column_name not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(metric_mapping.keys())}")
        
        print(f"Value range for {column_name}: {plot_df[column_name].min()} to {plot_df[column_name].max()}")
        print(f"Time range in minutes: 0 to {plot_df['Minutes'].max():.2f}")
        
        fig, ax = plt.subplots()
        
        # Plot the main line
        sns.lineplot(
            data=plot_df,
            x='Minutes',
            y=column_name,
            errorbar=('ci', 95) if ci else None,
            ax=ax
        )
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(column_name)
        
        # Format x-axis with appropriate intervals
        max_minutes = plot_df['Minutes'].max()
        if max_minutes <= 5:
            ax.set_xticks(np.arange(0, max_minutes + 0.5, 0.5))
        elif max_minutes <= 10:
            ax.set_xticks(np.arange(0, max_minutes + 1, 1))
        else:
            ax.set_xticks(np.arange(0, max_minutes + 2, 2))
        
        if title:
            ax.set_title(title)
            
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        print("Plot created successfully")
        return fig

    def save_plots(self,
                  figs: Dict[str, plt.Figure],
                  output_path: Path,
                  formats: List[str] = ['png', 'pdf']) -> None:
        print(f"Saving {len(figs)} plots to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figs.items():
            for fmt in formats:
                save_path = output_path / f'{name}.{fmt}'
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot to {save_path}")
            plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze and visualize metrics data.')
    parser.add_argument('input_file', type=Path, help='Path to input CSV file')
    parser.add_argument('--metrics', nargs='+', required=True,
                      help='List of metrics to analyze')
    parser.add_argument('--plot-types', nargs='+', default=['time_series'],
                      choices=['time_series', 'statistics'],
                      help='Types of plots to generate')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                      help='Output directory for plots and statistics')
    return parser.parse_args()

def main():
    print("Starting main function")
    args = parse_args()
    print(f"Arguments: {args}")
    
    visualizer = TimeSeriesVisualizer()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the CSV file directly
    try:
        print(f"Loading data from {args.input_file}")
        df = pd.read_csv(args.input_file)
        print(f"Loaded data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
        
    # Display available metrics
    available_metrics = [col for col in df.columns if col != 'Timestamp' and col != 'Notes']
    print(f"Available metrics: {available_metrics}")
    
    # Generate plots
    figs = {}
    if 'time_series' in args.plot_types:
        print("Generating time series plots")
        for metric in args.metrics:
            print(f"Processing metric: {metric}")
            try:
                fig = visualizer.plot_metric_time_series(
                    df,
                    metric,
                    title=f'{metric} Time Series'
                )
                figs[f'{metric}_time_series'] = fig
                print(f"Successfully created plot for {metric}")
            except ValueError as e:
                print(f"Error plotting metric '{metric}': {e}")
                continue
    
    # Save plots
    if figs:
        visualizer.save_plots(figs, args.output_dir)
    else:
        print("No plots were generated. Please check your metric names and data.")

if __name__ == '__main__':
    main()

    
