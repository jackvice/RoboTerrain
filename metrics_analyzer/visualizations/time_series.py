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
        sns.set_style("whitegrid")  # Use seaborn's set_style instead of plt.style.use
        sns.set_context("paper")    # Set the plotting context
        
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
        """Create a publication-ready time series plot for a single metric."""
        # Convert time to minutes
        plot_df = df.copy()
        start_time = plot_df['Timestamp'].min()
        plot_df['Minutes'] = (plot_df['Timestamp'] - start_time) / 60
        
        fig, ax = plt.subplots()
    
        # Plot the main line
        sns.lineplot(
            data=plot_df,
            x='Minutes',
            y=metric,  # Use the metric name passed in (should already be mapped)
            errorbar=('ci', 95) if ci else None,
            ax=ax
        )
    
        # Customize the plot
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel(metric)
    
        if title:
            ax.set_title(title)
        
        ax.grid(True, linestyle='--', alpha=0.7)
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
            # Convert timestamp to proper minutes
            df = df.copy()
            start_time = df['Timestamp'].iloc[0]
            # Convert elapsed seconds to minutes
            df['Minutes'] = (df['Timestamp'] - start_time) / 60.0
            
            sns.lineplot(
                data=df,
                x='Minutes',
                y=metric,
                label=trial_name,
                ax=ax
            )
        
        # Add collision threshold line if this is Obstacle Clearance
        if 'Obstacle Clearance' in metric:
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
            ax.text(ax.get_xlim()[1], 0.4, 'Collision Threshold', 
                   color='red', va='bottom', ha='right')
        
        # Customize the plot
        ax.set_xlabel('Time (minutes)')
        if 'Obstacle Clearance' in metric:
            ax.set_ylabel('Obstacle Clearance (meters)')
        else:
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

    
    def plot_multi_trial_comparisonOld(self,
                                  trial_data: Dict[str, pd.DataFrame],
                                  metric: str,
                                  title: Optional[str] = None) -> plt.Figure:
        # Debug prints at start
        for trial_name, df in trial_data.items():
            print(f"\nDebug - {trial_name}:")
            print(f"First timestamp: {df['Timestamp'].iloc[0]}")
            print(f"Last timestamp: {df['Timestamp'].iloc[-1]}")
            print(f"Timestamp type: {type(df['Timestamp'].iloc[0])}")
            print(f"DataFrame info:")
            print(df.info())
        
    

        fig, ax = plt.subplots()
        
        # Plot each trial
        for trial_name, df in trial_data.items():
            # Convert timestamp to proper minutes
            df = df.copy()  # Create a copy to avoid modifying original
            start_time = float(df['Timestamp'].iloc[0])  # Ensure float conversion
            # Convert to float and ensure proper arithmetic
            df['Minutes'] = df['Timestamp'].astype(float).sub(start_time).div(60.0)
            
            print(f"Debug - Time range: {df['Minutes'].min():.2f} to {df['Minutes'].max():.2f} minutes")  # Debug print
            
            sns.lineplot(
                data=df,
                x='Minutes',
                y=metric,
                label=trial_name,
                ax=ax
            )
        
        # Add collision threshold line if this is Obstacle Clearance
        if 'Obstacle Clearance' in metric:
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
            ax.text(ax.get_xlim()[1], 0.4, 'Collision Threshold', 
                   color='red', va='bottom', ha='right')
        
        # Customize the plot
        ax.set_xlabel('Time (minutes)')
        if 'Obstacle Clearance' in metric:
            ax.set_ylabel('Obstacle Clearance (meters)')
        else:
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
    
    def plot_multi_trial_comparisonOld(self,
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
            # Convert timestamp to minutes
            start_time = df['Timestamp'].min()
            df = df.copy()  # Create a copy to avoid modifying original
            df['Minutes'] = (df['Timestamp'] - start_time) / 60
            
            sns.lineplot(
                data=df,
                x='Minutes',
                y=metric,
                label=trial_name,
                ax=ax
            )
        
        # Add collision threshold line if this is Obstacle Clearance
        if 'Obstacle Clearance' in metric:
            ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
            ax.text(ax.get_xlim()[1], 0.4, 'Collision Threshold', 
                   color='red', va='bottom', ha='right')
        
        # Customize the plot
        ax.set_xlabel('Time (minutes)')
        # Modify y-label based on metric
        if 'Obstacle Clearance' in metric:
            ax.set_ylabel('Obstacle Clearance (meters)')
        else:
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
    
    def plot_multi_trial_comparisonOld(self,
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
        ax.set_xlabel('Time' + (' (%)'
                                if all(df['Timestamp'].max() <= 100
                                              for df in trial_data.values())
                                else ' (s)'))
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

if __name__ == '__main__':
    # Example usage
    visualizer = TimeSeriesVisualizer()
    # Add example usage code here
