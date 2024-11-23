import os
import csv
import time
import torch as th
from pathlib import Path

import torch.nn as nn
from torch import Tensor

class LoggingWrapper(nn.Module):
    def __init__(self, custom_extractor, log_dir="./logs"):
        super().__init__()
        self.extractor = custom_extractor
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.input_file = self.log_dir / "input_stats.csv"
        self.output_file = self.log_dir / "output_stats.csv"
        
        # Initialize files if they don't exist
        self.setup_logging()
        
        # Get the last sequence number or start from 0
        self.sequence = self.get_last_sequence()
        
        # Directly set features_dim from the extractor
        self._features_dim = custom_extractor.features_dim

    @property
    def features_dim(self):
        return self._features_dim

    def forward(self, observations) -> Tensor:
        """Process observations and log statistics"""
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        
        # Log input statistics
        self.log_input_stats(observations, timestamp)
        
        # Process through extractor
        output = self.extractor(observations)
        
        # Log output statistics
        self.log_output_stats(output, timestamp)
        
        # Increment sequence number
        self.sequence += 1
        
        return output

        
    def get_last_sequence(self):
        """Get the last sequence number from both files and return the max + 1"""
        last_seq = 0
        
        for file in [self.input_file, self.output_file]:
            if file.exists() and file.stat().st_size > 0:
                with open(file, 'r') as f:
                    # Skip header
                    next(f)
                    # Read all lines and get last sequence
                    lines = f.readlines()
                    if lines:
                        try:
                            last_seq = max(last_seq, int(lines[-1].split(',')[0]))
                        except (ValueError, IndexError):
                            pass
                            
        return last_seq + 1
    
    def setup_logging(self):
        """Create CSV files with headers if they don't exist"""
        input_header = [
            "sequence", "timestamp",
            "lidar_min", "lidar_max", "lidar_mean", "lidar_std",
            "pose_min", "pose_max", "pose_mean", "pose_std",
            "imu_min", "imu_max", "imu_mean", "imu_std",
            "target_min", "target_max", "target_mean", "target_std"
        ]
        
        output_header = [
            "sequence", "timestamp",
            "concat_min", "concat_max", "concat_mean", "concat_std"
        ]
        
        # Create input stats file if it doesn't exist
        if not self.input_file.exists():
            with open(self.input_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(input_header)
                
        # Create output stats file if it doesn't exist
        if not self.output_file.exists():
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(output_header)
    
    def calculate_stats(self, tensor):
        """Calculate statistics for a tensor, handling NaN values"""
        try:
            if th.isnan(tensor).any():
                return 'NaN', 'NaN', 'NaN', 'NaN'
            
            return (
                float(tensor.min().item()),
                float(tensor.max().item()),
                float(tensor.mean().item()),
                float(tensor.std().item())
            )
        except:
            return 'NaN', 'NaN', 'NaN', 'NaN'
    
    def log_input_stats(self, observations, timestamp):
        """Log statistics for each input observation"""
        stats = []
        for key in ['lidar', 'pose', 'imu', 'target']:
            tensor = observations[key]
            stats.extend(self.calculate_stats(tensor))
            
        row = [self.sequence, timestamp] + stats
        
        with open(self.input_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_output_stats(self, output_tensor, timestamp):
        """Log statistics for the concatenated output"""
        stats = self.calculate_stats(output_tensor)
        row = [self.sequence, timestamp] + list(stats)
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    
    def __call__(self, observations):
        """Allow the wrapper to be called like the original extractor"""
        return self.forward(observations)
