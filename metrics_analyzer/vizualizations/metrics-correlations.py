#!/usr/bin/env python3

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class CorrelationVisualizer:
    """Visualization class for correlation analysis."""
    
    def __init__(self):
        """Initialize the visualizer with publication-ready style settings."""
        plt.style.use('seaborn-paper')
        sns.set_context("paper")
        
        self.style_settings = {
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'axes.titlesize': 11,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        }
        plt.rcParams.update(self.style_settings)
        
        # Define color scheme
        self.cmap = sns.diverging_palette(230, 20, as_cmap=True)

    def plot_correlation_matrix(self,
                              corr_matrix: pd.DataFrame,
                              p_values: pd.DataFrame,
                              title: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation heatmap with significance markers.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            p_values (pd.DataFrame): Matrix of p-values
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=self.cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            annot=True,
            fmt='.2f',
            cbar_kws={"shrink": .5},
            ax=ax
        )
        
        # Add significance markers
        significance_mask = (p_values < 0.05) & ~mask
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if significance_mask.iloc[i, j]:
                    ax.text(j + 0.5, i + 0.85, '*',
                           ha='center', va='center',
                           color='black', fontsize=12)
        
        if title:
            ax.set_title(title)
            
        plt.tight_layout()
        return fig

    def plot_metric_pair_correlation(self,
                                   df: pd.DataFrame,
                                   metric1: str,
                                   metric2: str,
                                   title: Optional[str] = None) -> plt.Figure:
        """
        Create a scatter plot with regression line for a pair of metrics.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            metric1 (str): First metric
            metric2 (str): Second metric
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        fig, ax = plt.subplots()
        
        # Create scatter plot with regression line
        sns.regplot(
            data=df,
            x=metric1,
            y=metric2,
            scatter_kws={'alpha':0.5},
            line_kws={'color': 'red'},
            ax=ax
        )
        
        # Calculate correlation coefficient and p-value
        corr_coef = df[metric1].corr(df[metric2])
        _, p_value = stats.pearsonr(df[metric1], df[metric2])
        
        # Add correlation information
        ax.text(0.05, 0.95,
                f'r = {corr_coef:.2f}\np = {p_value:.3f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title:
            ax.set_title(title)
            
        plt.tight_layout()
        return fig

    def plot_correlation_clustermap(self,
                                  corr_matrix: pd.DataFrame,
                                  title: Optional[str] = None) -> plt.Figure:
        """
        Create a hierarchically-clustered correlation matrix.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            title (Optional[str]): Plot title
            
        Returns:
            plt.Figure: Generated figure
        """
        # Create clustermap
        g = sns.clustermap(
            corr_matrix,
            cmap=self.cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt='.2f',
            square=True,
            figsize=(12, 12),
            dendrogram_ratio=(.1, .1),
            cbar_pos=(0.02, 0.8, 0.03, 0.2)
        )
        
        if title:
            g.fig.suptitle(title, y=1.02)
            
        return g.fig

    def plot_cross_trial_correlations(self,
                                    trial_data: Dict[str, pd.DataFrame],
                                    metrics: List[str],
                                    title: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create correlation plots for multiple trials.
        
        Args:
            trial_data (Dict[str, pd.DataFrame]): Dictionary of trial DataFrames
            metrics (List[str]): List of metrics to analyze
            title (Optional[str]): Base title for plots
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of generated figures
        """
        figures = {}
        
        # Create correlation matrices for each trial
        for trial_name, df in trial_data.items():
            corr_matrix = df[metrics].corr()
            p_values = pd.DataFrame(
                np.zeros_like(corr_matrix),
                index=corr_matrix.index,
                columns=corr_matrix.columns
            )
            
            # Calculate p-values
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    if i != j:
                        _, p = stats.pearsonr(df[metric1], df[metric2])
                        p_values.iloc[i, j] = p
            
            # Create heatmap
            fig = self.plot_correlation_matrix(
                corr_matrix,
                p_values,
                title=f"{title} - {trial_name}" if title else trial_name
            )
            figures[f"correlation_matrix_{trial_name}"] = fig
            
            # Create clustermap
            fig_cluster = self.plot_correlation_clustermap(
                corr_matrix,
                title=f"Clustered Correlations - {trial_name}"
            )
            figures[f"correlation_cluster_{trial_name}"] = fig_cluster
        
        return figures

    def save_correlation_plots(self,
                             figs: Dict[str, plt.Figure],
                             output_path: Path,
                             formats: List[str] = ['png', 'pdf']) -> None:
        """
        Save correlation plots in multiple formats.
        
        Args:
            figs (Dict[str, plt.Figure]): Dictionary of figures to save
            output_path (Path): Output directory
            formats (List[str]): List of formats to save in
        """
        correlations_path = output_path / 'correlations'
        correlations_path.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figs.items():
            for fmt in formats:
                fig.savefig(
                    correlations_path / f'{name}.{fmt}',
                    dpi=300,
                    bbox_inches='tight'
                )

if __name__ == '__main__':
    # Example usage
    visualizer = CorrelationVisualizer()
    # Add example usage code here
