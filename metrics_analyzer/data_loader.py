#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

class MetricsDataLoader:
    """Load and preprocess metrics data from CSV files."""
    
    def __init__(self, iqr_multiplier: float = 1.5):
        """
        Initialize the data loader.
        
        Args:
            iqr_multiplier (float): Multiplier for IQR in outlier detection
        """
        self.iqr_multiplier = iqr_multiplier
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path (Path): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            self._validate_dataframe(df)
            return df
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the DataFrame has required columns.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {'Timestamp'} # Add other required columns as needed
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def remove_outliers(self, 
                       df: pd.DataFrame, 
                       columns: List[str],
                       save_outliers: bool = False,
                       output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Remove outliers using the IQR method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to check for outliers
            save_outliers (bool): Whether to save removed outliers
            output_path (Optional[Path]): Path to save outliers data
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: 
                - Clean DataFrame
                - DataFrame containing outliers (if save_outliers=True)
        """
        outliers_df = pd.DataFrame()
        clean_df = df.copy()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (self.iqr_multiplier * IQR)
            upper_bound = Q3 + (self.iqr_multiplier * IQR)
            
            # Identify outliers
            mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            if save_outliers:
                outliers = df[mask].copy()
                outliers['metric'] = column
                outliers['reason'] = np.where(
                    df[column] < lower_bound,
                    'below_lower_bound',
                    'above_upper_bound'
                )
                outliers_df = pd.concat([outliers_df, outliers])
            
            # Remove outliers
            clean_df = clean_df[~mask]
            
            self.logger.info(
                f"Removed {mask.sum()} outliers from {column}"
            )
        
        if save_outliers and not outliers_df.empty and output_path:
            outliers_df.to_csv(output_path / 'outliers.csv', index=False)
            
        return clean_df, outliers_df if save_outliers else None

    def normalize_time(self, 
                      df: pd.DataFrame,
                      method: str = 'percentage',
                      fixed_interval: float = 1.0) -> pd.DataFrame:
        """
        Normalize time data using specified method.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            method (str): Normalization method ('percentage', 'fixed_interval', 'none')
            fixed_interval (float): Interval for fixed_interval method
            
        Returns:
            pd.DataFrame: DataFrame with normalized time
        """
        if method == 'none':
            return df
            
        normalized_df = df.copy()
        
        # Ensure timestamp is numeric
        if pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            normalized_df['Timestamp'] = (
                df['Timestamp'] - df['Timestamp'].min()
            ).dt.total_seconds()
        
        if method == 'percentage':
            # Convert to percentage of total time
            max_time = normalized_df['Timestamp'].max()
            normalized_df['Timestamp'] = (
                normalized_df['Timestamp'] / max_time * 100
            )
            
        elif method == 'fixed_interval':
            # Resample to fixed intervals
            normalized_df.set_index('Timestamp', inplace=True)
            normalized_df = normalized_df.resample(
                f'{fixed_interval}S'
            ).mean().interpolate()
            normalized_df.reset_index(inplace=True)
            
        return normalized_df

    def process_data(self,
                    file_path: Path,
                    metrics: List[str],
                    normalize_method: str = 'percentage',
                    fixed_interval: float = 1.0,
                    save_outliers: bool = False,
                    output_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """
        Complete data processing pipeline.
        
        Args:
            file_path (Path): Path to input CSV
            metrics (List[str]): Metrics to process
            normalize_method (str): Time normalization method
            fixed_interval (float): Interval for fixed_interval normalization
            save_outliers (bool): Whether to save outliers
            output_path (Optional[Path]): Path to save outliers
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing processed DataFrames
        """

        # Add at the start of the method
        metric_mapping = {
            'TC': 'Total Collisions',
            'SR': 'Success Rate',
            'MTT': 'Mean Time to Traverse',
            'TR': 'Traverse Rate',
            'TSR': 'Total Smoothness of Route',
            'OC': 'Obstacle Clearance',
            'VOR': 'Velocity Over Rough Terrain'
        }
    
        # Map the metric codes to actual column names
        mapped_metrics = [metric_mapping.get(m, m) for m in metrics]
        
        # Load data
        df = self.load_csv(file_path)
        
        # Remove outliers
        clean_df, outliers = self.remove_outliers(
            df, 
            metrics,
            save_outliers,
            output_path
        )
        
        # Normalize time
        normalized_df = self.normalize_time(
            clean_df,
            normalize_method,
            fixed_interval
        )
        
        return {
            'raw': df,
            'clean': clean_df,
            'normalized': normalized_df,
            'outliers': outliers
        }

if __name__ == '__main__':
    # Example usage
    loader = MetricsDataLoader()
    # Add example usage code here
