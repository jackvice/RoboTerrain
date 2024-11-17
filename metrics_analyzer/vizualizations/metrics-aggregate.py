#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class AggregateVisualizer:
    """Visualization class for aggregate statistics."""
    
    def __init__(self):
        """Initialize the visualizer with publication-ready style settings."""
        plt.style.use('seaborn-paper')
        sns.set_context("paper")
        
        self.style_settings = {
            'figure.figsize': (10, 6),
            'figure.dpi': 300,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        }
        plt.rcParams.update(self.style_settings)
        
        self.logger = logging.getLogger(__name__)

    def plot_metric_distribution(self,
                               trial_data: Dict[str, pd.DataFrame],
                               metric: str,
                               title: Optional[str] = None) -> plt.Figure:
        """
        Create violin plots showing metric distribution across trials.
        
        Args:
            trial_data (Dict[str, pd.DataFrame]): Dictionary of trial DataFrames
            metric (str): Metric to plot
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        # Prepare data for plotting
        plot_data = []
        for trial_name, df in trial_data.items():
            if metric in df.columns:
                temp_df = pd.DataFrame({
                    'Trial': trial_name,
                    'Value': df[metric]
                })
                plot_data.append(temp_df)
        
        if not plot_data:
            self.logger.warning(f"No data available for metric: {metric}")
            return None
            
        plot_df = pd.concat(plot_data, ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Create violin plot
        sns.violinplot(
            data=plot_df,
            x='Trial',
            y='Value',
            ax=ax,
            inner='box'  # Show box plot inside violin
        )
        
        ax.set_ylabel(metric)
        if title:
            ax.set_title(title)
            
        # Rotate x-axis labels if needed
        if len(trial_data) > 4:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        return fig

    def plot_summary_statistics(self,
                              trial_stats: Dict[str, Dict],
                              metric: str,
                              title: Optional[str] = None) -> plt.Figure:
        """
        Create bar plot of summary statistics with error bars.
        
        Args:
            trial_stats (Dict[str, Dict]): Dictionary of trial statistics
            metric (str): Metric to plot
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        # Extract statistics
        means = []
        errors = []
        trial_names = []
        
        for trial, stats in trial_stats.items():
            if metric in stats:
                means.append(stats[metric]['stats']['mean'])
                ci = stats[metric]['confidence_interval']
                errors.append((ci[1] - ci[0]) / 2)  # Convert CI to error bar
                trial_names.append(trial)
        
        # Create figure
        fig, ax = plt.subplots()
        
        # Create bar plot
        x = np.arange(len(trial_names))
        bars = ax.bar(x, means, yerr=errors, capsize=5)
        
        # Customize plot
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(trial_names)
        
        if title:
            ax.set_title(title)
            
        # Rotate x-axis labels if needed
        if len(trial_names) > 4:
            plt.xticks(rotation=45, ha='right')
            
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
            
        plt.tight_layout()
        return fig

    def plot_metric_boxplots(self,
                            trial_data: Dict[str, pd.DataFrame],
                            metrics: List[str],
                            title: Optional[str] = None) -> plt.Figure:
        """
        Create multi-metric box plots for comparison across trials.
        
        Args:
            trial_data (Dict[str, pd.DataFrame]): Dictionary of trial DataFrames
            metrics (List[str]): List of metrics to plot
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        # Prepare data for plotting
        plot_data = []
        for trial_name, df in trial_data.items():
            for metric in metrics:
                if metric in df.columns:
                    temp_df = pd.DataFrame({
                        'Trial': trial_name,
                        'Metric': metric,
                        'Value': df[metric]
                    })
                    plot_data.append(temp_df)
        
        if not plot_data:
            self.logger.warning("No data available for plotting")
            return None
            
        plot_df = pd.concat(plot_data, ignore_index=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create box plot
        sns.boxplot(
            data=plot_df,
            x='Trial',
            y='Value',
            hue='Metric',
            ax=ax
        )
        
        if title:
            ax.set_title(title)
            
        # Rotate x-axis labels if needed
        if len(trial_data) > 4:
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()
        return fig

    def plot_performance_radar(self,
                             trial_stats: Dict[str, Dict],
                             metrics: List[str],
                             title: Optional[str] = None) -> plt.Figure:
        """
        Create radar plot of normalized metrics across trials.
        
        Args:
            trial_stats (Dict[str, Dict]): Dictionary of trial statistics
            metrics (List[str]): List of metrics to plot
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot for each trial
        for trial, stats in trial_stats.items():
            # Extract and normalize values
            values = []
            for metric in metrics:
                if metric in