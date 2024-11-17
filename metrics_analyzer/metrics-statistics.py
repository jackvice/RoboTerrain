#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

class MetricsStatistics:
    """Statistical analysis for navigation metrics."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analysis module.
        
        Args:
            confidence_level (float): Confidence level for intervals (0-1)
        """
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def calculate_basic_stats(self, 
                            data: pd.Series) -> Dict[str, float]:
        """
        Calculate basic statistical measures.
        
        Args:
            data (pd.Series): Input data series
            
        Returns:
            Dict[str, float]: Dictionary of statistical measures
        """
        return {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'count': len(data)
        }

    def calculate_confidence_interval(self, 
                                   data: pd.Series) -> Tuple[float, float]:
        """
        Calculate confidence interval using t-distribution.
        
        Args:
            data (pd.Series): Input data series
            
        Returns:
            Tuple[float, float]: Lower and upper confidence bounds
        """
        alpha = 1 - self.confidence_level
        n = len(data)
        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(self.confidence_level, n-1, mean, se)
        return ci

    def calculate_rolling_stats(self,
                              data: pd.Series,
                              window: int = 10) -> pd.DataFrame:
        """
        Calculate rolling statistics.
        
        Args:
            data (pd.Series): Input data series
            window (int): Rolling window size
            
        Returns:
            pd.DataFrame: DataFrame with rolling statistics
        """
        rolling = data.rolling(window=window)
        return pd.DataFrame({
            'mean': rolling.mean(),
            'std': rolling.std(),
            'lower_ci': rolling.apply(
                lambda x: self.calculate_confidence_interval(x)[0]
                if len(x.dropna()) > 1 else np.nan
            ),
            'upper_ci': rolling.apply(
                lambda x: self.calculate_confidence_interval(x)[1]
                if len(x.dropna()) > 1 else np.nan
            )
        })

    def calculate_correlation_matrix(self,
                                   df: pd.DataFrame,
                                   metrics: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate correlation matrix and p-values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            metrics (List[str]): Metrics to analyze
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Correlation matrix and p-values
        """
        # Calculate correlation matrix
        corr_matrix = df[metrics].corr()
        
        # Calculate p-values
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                              index=corr_matrix.index,
                              columns=corr_matrix.columns)
        
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                if i != j:
                    stat, p = stats.pearsonr(
                        df[metrics[i]].dropna(),
                        df[metrics[j]].dropna()
                    )
                    p_values.iloc[i,j] = p
        
        return corr_matrix, p_values

    def analyze_trial(self,
                     df: pd.DataFrame,
                     metrics: List[str]) -> Dict[str, Dict]:
        """
        Perform complete statistical analysis for a single trial.
        
        Args:
            df (pd.DataFrame): Trial data
            metrics (List[str]): Metrics to analyze
            
        Returns:
            Dict[str, Dict]: Complete statistical analysis
        """
        results = {}
        
        for metric in metrics:
            if metric not in df.columns:
                self.logger.warning(f"Metric {metric} not found in data")
                continue
                
            data = df[metric].dropna()
            
            # Basic statistics
            basic_stats = self.calculate_basic_stats(data)
            
            # Confidence intervals
            ci_lower, ci_upper = self.calculate_confidence_interval(data)
            
            # Rolling statistics
            rolling_stats = self.calculate_rolling_stats(data)
            
            results[metric] = {
                'basic_stats': basic_stats,
                'confidence_interval': {
                    'lower': ci_lower,
                    'upper': ci_upper
                },
                'rolling_stats': rolling_stats
            }
        
        # Add correlation analysis
        corr_matrix, p_values = self.calculate_correlation_matrix(df, metrics)
        results['correlations'] = {
            'correlation_matrix': corr_matrix,
            'p_values': p_values
        }
        
        return results

    def compare_trials(self,
                      trial_data: Dict[str, pd.DataFrame],
                      metrics: List[str]) -> Dict[str, Dict]:
        """
        Compare statistics across multiple trials.
        
        Args:
            trial_data (Dict[str, pd.DataFrame]): Dictionary of trial DataFrames
            metrics (List[str]): Metrics to compare
            
        Returns:
            Dict[str, Dict]: Cross-trial statistical comparison
        """
        comparison = {}
        
        for metric in metrics:
            metric_stats = {}
            
            # Collect metric data from all trials
            trial_values = {
                trial: data[metric].dropna()
                for trial, data in trial_data.items()
                if metric in data.columns
            }
            
            # Calculate statistics for each trial
            for trial, values in trial_values.items():
                metric_stats[trial] = {
                    'stats': self.calculate_basic_stats(values),
                    'confidence_interval': self.calculate_confidence_interval(values)
                }
            
            # Add aggregate statistics across all trials
            all_values = pd.concat([values for values in trial_values.values()])
            metric_stats['aggregate'] = {
                'stats': self.calculate_basic_stats(all_values),
                'confidence_interval': self.calculate_confidence_interval(all_values)
            }
            
            comparison[metric] = metric_stats
        
        return comparison

    def save_statistics(self,
                       stats_dict: Dict,
                       output_path: Path,
                       prefix: str = '') -> None:
        """
        Save statistical results to CSV files.
        
        Args:
            stats_dict (Dict): Statistical results
            output_path (Path): Output directory
            prefix (str): Prefix for output files
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save basic statistics
        for metric, data in stats_dict.items():
            if metric == 'correlations':
                # Save correlation matrix
                data['correlation_matrix'].to_csv(
                    output_path / f'{prefix}correlation_matrix.csv'
                )
                # Save p-values
                data['p_values'].to_csv(
                    output_path / f'{prefix}correlation_pvalues.csv'
                )
            else:
                # Save metric statistics
                pd.DataFrame(data['basic_stats'], index=[0]).to_csv(
                    output_path / f'{prefix}{metric}_stats.csv'
                )
                # Save rolling statistics
                data['rolling_stats'].to_csv(
                    output_path / f'{prefix}{metric}_rolling_stats.csv'
                )

if __name__ == '__main__':
    # Example usage
    stats = MetricsStatistics()
    # Add example usage code here
