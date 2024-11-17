#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class TimeSeriesVisualizer:
    """Visualization class for time series metrics data."""
    
    def __init__(self):
        """Initialize the visualizer with publication-ready style settings."""
        # Set up the style
        plt.style.use('seaborn-paper')
        sns.set_context("paper")
        
        # Custom style settings
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
        """
        Create a publication-ready time series plot for a single metric.
        """
        # Create a copy of the dataframe to modify
        plot_df = df.copy()

        # Convert Unix timestamp to minutes from start
        start_time = plot_df['Timestamp'].min()
        plot_df['Minutes'] = (plot_df['Timestamp'] - start_time) / 60
        
        # Define metric mapping from short names to actual column names
        metric_mapping = {
            'TC': 'Total Collisions',
            'CS': 'Current Collision Status',
            'SM': 'Smoothness Metric',
            'CS': 'Current Smoothness',
            'OC': 'Obstacle Clearance',
            'DT': 'Distance Traveled',
            'CV': 'Current Velocity',
            'IM': 'IMU Acceleration Magnitude',
            'RT': 'Is Rough Terrain'
        }
        
        # Get the full column name
        column_name = metric_mapping.get(metric, metric)
        if column_name not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {list(metric_mapping.keys())}")
        
        fig, ax = plt.subplots()
        
        # Plot the main line
        sns.lineplot(
            data=plot_df,
            x='Minutes',
            y=column_name,
            errorbar=('ci', 95) if ci else None,
            ax=ax
        )
    
        # Customize the plot
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
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
        # Tight layout
        plt.tight_layout()
    
        return fig

        
    def plot_multi_trial_comparison(self,
                                  trial_data: Dict[str, pd.DataFrame],
                                  metric: str,
                                  title: Optional[str] = None) -> plt.Figure:
        """
        Create a comparison plot of multiple trials for a single metric.
        
        Args:
            trial_data (Dict[str, pd.DataFrame]): Dictionary of trial DataFrames
            metric (str): Metric to plot
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        fig, ax = plt.subplots()
        
        # Plot each trial
        for trial_name, df in trial_data.items():
            sns.lineplot(
                data=df,
                x='Timestamp',
                y=metric,
                label=trial_name,
                ax=ax
            )
        
        # Customize the plot
        ax.set_xlabel('Time' + (' (%)' if all(df['Timestamp'].max() <= 100 for df in trial_data.values()) else ' (s)'))
        ax.set_ylabel(metric)
        
        if title:
            ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust legend
        ax.legend(title='Trials', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def plot_metric_with_statistics(self,
                                  df: pd.DataFrame,
                                  metric: str,
                                  rolling_stats: pd.DataFrame,
                                  title: Optional[str] = None) -> plt.Figure:
        """
        Create a time series plot with rolling statistics.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            metric (str): Metric to plot
            rolling_stats (pd.DataFrame): Rolling statistics DataFrame
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        fig, ax = plt.subplots()
        
        # Plot raw data
        ax.plot(df['Timestamp'], df[metric], 'b-', alpha=0.3, label='Raw Data')
        
        # Plot rolling mean
        ax.plot(df['Timestamp'], rolling_stats['mean'], 'r-', label='Rolling Mean')
        
        # Plot confidence interval
        ax.fill_between(
            df['Timestamp'],
            rolling_stats['lower_ci'],
            rolling_stats['upper_ci'],
            color='r',
            alpha=0.2,
            label='95% CI'
        )
        
        # Customize the plot
        ax.set_xlabel('Time' + (' (%)' if df['Timestamp'].max() <= 100 else ' (s)'))
        ax.set_ylabel(metric)
        
        if title:
            ax.set_title(title)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def save_plots(self,
                  figs: Dict[str, plt.Figure],
                  output_path: Path,
                  formats: List[str] = ['png', 'pdf']) -> None:
        """
        Save generated figures in multiple formats.
        
        Args:
            figs (Dict[str, plt.Figure]): Dictionary of figures to save
            output_path (Path): Output directory
            formats (List[str]): List of formats to save in
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figs.items():
            for fmt in formats:
                fig.savefig(
                    output_path / f'{name}.{fmt}',
                    dpi=300,
                    bbox_inches='tight'
                )


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize components
    loader = MetricsDataLoader()
    stats = MetricsStatistics()
    visualizer = TimeSeriesVisualizer()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the CSV file directly
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
        
    # Display available metrics if requested metric not found
    available_metrics = [col for col in df.columns if col != 'Timestamp' and col != 'Notes']
    print(f"Available metrics: {available_metrics}")
    
    # Generate plots
    figs = {}
    if 'time_series' in args.plot_types:
        for metric in args.metrics:
            try:
                fig = visualizer.plot_metric_time_series(
                    df,
                    metric,
                    title=f'{metric} Time Series'
                )
                figs[f'{metric}_time_series'] = fig
            except ValueError as e:
                print(f"Error plotting metric '{metric}': {e}")
                continue
    
    # Save plots
    if figs:
        visualizer.save_plots(figs, args.output_dir)
        print(f"Plots saved to {args.output_dir}")
    else:
        print("No plots were generated. Please check your metric names and data.")
