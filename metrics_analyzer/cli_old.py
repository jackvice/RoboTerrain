#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

class MetricsAnalyzerCLI:
    """Command line interface for the metrics analyzer."""
    
    # Define valid metrics and their descriptions
    VALID_METRICS = {
        'SR': 'Success Rate (%)',
        'TC': 'Total Collisions (#)',
        'MTT': 'Mean Time to Traverse (s)',
        'TR': 'Traverse Rate (%)',
        'TSR': 'Total Smoothness of Route (m)',
        'OC': 'Obstacle Clearance (%)',
        'VOR': 'Velocity Over Rough Terrain (m/s)'
    }
    
    # Define valid plot types
    VALID_PLOT_TYPES = [
        'time_series',
        'correlation',
        'aggregate',
        'comparison'
    ]
    
    # Define valid time normalization methods
    VALID_TIME_NORMS = [
        'percentage',
        'fixed_interval',
        'none'
    ]

    def __init__(self):
        """Initialize the CLI parser."""
        self.parser = argparse.ArgumentParser(
            description='Analyze and visualize navigation metrics from CSV files.',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self._setup_arguments()

    def _setup_arguments(self):
        """Set up command line arguments."""
        # Required arguments
        self.parser.add_argument(
            'csv_files',
            type=Path,
            nargs='+',
            help='One or more CSV files containing metrics data'
        )
        
        # Optional arguments
        self.parser.add_argument(
            '-m', '--metrics',
            choices=list(self.VALID_METRICS.keys()),
            nargs='+',
            default=list(self.VALID_METRICS.keys()),
            help='Metrics to analyze (default: all)'
        )
        
        self.parser.add_argument(
            '-p', '--plot-types',
            choices=self.VALID_PLOT_TYPES,
            nargs='+',
            default=['time_series'],
            help='Types of plots to generate (default: time_series)'
        )
        
        self.parser.add_argument(
            '-o', '--output-dir',
            type=Path,
            default=Path('output'),
            help='Output directory for plots and analysis (default: ./output)'
        )
        
        self.parser.add_argument(
            '-n', '--normalize',
            choices=self.VALID_TIME_NORMS,
            default='percentage',
            help='Time normalization method (default: percentage)'
        )
        
        self.parser.add_argument(
            '--iqr-multiplier',
            type=float,
            default=1.5,
            help='IQR multiplier for outlier detection (default: 1.5)'
        )
        
        self.parser.add_argument(
            '--save-outliers',
            action='store_true',
            help='Save removed outliers to a separate file'
        )
        
        self.parser.add_argument(
            '--confidence-level',
            type=float,
            default=0.95,
            help='Confidence level for statistical analysis (default: 0.95)'
        )
        
        self.parser.add_argument(
            '--fixed-interval',
            type=float,
            default=1.0,
            help='Interval (seconds) for fixed-interval normalization (default: 1.0)'
        )

    def validate_files(self, files: List[Path]) -> None:
        """Validate that all input files exist and are CSV files."""
        for file in files:
            if not file.exists():
                sys.exit(f"Error: File {file} does not exist")
            if file.suffix.lower() != '.csv':
                sys.exit(f"Error: File {file} is not a CSV file")

    def validate_output_dir(self, output_dir: Path) -> None:
        """Validate and create output directory if it doesn't exist."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            sys.exit(f"Error creating output directory: {e}")

    def parse_args(self) -> Dict[str, Any]:
        """Parse and validate command line arguments."""
        args = self.parser.parse_args()
        
        # Validate input files
        self.validate_files(args.csv_files)
        
        # Validate output directory
        self.validate_output_dir(args.output_dir)
        
        # Validate confidence level
        if not 0 < args.confidence_level < 1:
            sys.exit("Error: Confidence level must be between 0 and 1")
        
        # Validate IQR multiplier
        if args.iqr_multiplier <= 0:
            sys.exit("Error: IQR multiplier must be positive")
        
        # Validate fixed interval
        if args.fixed_interval <= 0:
            sys.exit("Error: Fixed interval must be positive")
        
        return vars(args)

def main():
    """Main entry point for the CLI."""
    cli = MetricsAnalyzerCLI()
    args = cli.parse_args()
    return args

if __name__ == '__main__':
    main()
