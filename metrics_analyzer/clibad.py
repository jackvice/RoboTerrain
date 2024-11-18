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
    """Visualization class for time series metrics data."""
    
    def __init__(self):
        """Initialize the visualizer with publication-ready style settings."""
        # Set up the style
        sns.set_style("whitegrid")
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
        fig, ax = plt.subplots()
        
        # Plot the main line
        sns.lineplot(
            data=df,
            x='Timestamp',
            y=metric,
            ci=95 if ci else None,
            ax=ax
        )
        
        # Customize the plot
        ax.set_xlabel('Time' + (' (%)' if df['Timestamp'].max() <= 100 else ' (s)'))
        ax.set_ylabel(metric)
        
        if title:
            ax.set_title(title)
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return fig

    def save_plots(self,
                  figs: Dict[str, plt.Figure],
                  output_path: Path,
                  formats: List[str] = ['png', 'pdf']) -> None:
        """Save generated figures in multiple formats."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figs.items():
            for fmt in formats:
                fig.savefig(
                    output_path / f'{name}.{fmt}',
                    dpi=300,
                    bbox_inches='tight'
                )
                plt.close(fig)  # Close the figure to free memory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and visualize metrics data.')
    parser.add_argument('input_file', type=Path, help='Path to input CSV file')
    parser.add_argument('--metrics', nargs='+', required=True,
                      help='List of metrics to analyze')
    parser.add_argument('--plot-types', nargs='+', default=['time_series'],
                      choices=['time_series', 'statistics'],
                      help='Types of plots to generate')
    parser.add_argument('--output-dir', type=Path, default=Path('output'),
                      help='Output directory for plots and statistics')
    parser.add_argument('--time-normalize', choices=['percentage', 'fixed_interval', 'none'],
                      default='percentage', help='Time normalization method')
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize components
    loader = MetricsDataLoader()
    stats = MetricsStatistics()
    visualizer = TimeSeriesVisualizer()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    data_dict = loader.process_data(
        args.input_file,
        args.metrics,
        normalize_method=args.time_normalize
    )
    
    # Generate plots
    figs = {}
    if 'time_series' in args.plot_types:
        for metric in args.metrics:
            fig = visualizer.plot_metric_time_series(
                data_dict['normalized'],
                metric,
                title=f'{metric} Time Series'
            )
            figs[f'{metric}_time_series'] = fig
    
    # Save plots
    if figs:
        visualizer.save_plots(figs, args.output_dir)
        print(f"Plots saved to {args.output_dir}")

if __name__ == '__main__':
    main()
