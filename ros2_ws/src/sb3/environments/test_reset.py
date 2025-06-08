#!/usr/bin/env python3
"""Simple test to detect ign service calls for resets."""

import subprocess
import time
from datetime import datetime

def monitor_ign_processes():
    """Monitor for ign service processes."""
    print("Monitoring for 'ign service' processes...")
    print("Run your RL training and watch for reset detection.")
    print("Press Ctrl+C to stop.")
    
    reset_count = 0
    
    try:
        while True:
            try:
                # Look for any ign service processes
                result = subprocess.run(
                    ['pgrep', '-f', 'ign.*service'],
                    capture_output=True,
                    text=True,
                    timeout=0.5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    reset_count += 1
                    print(f"[{current_time}] RESET DETECTED #{reset_count}")
                    
                    # Small delay to avoid double-counting same reset
                    time.sleep(0.5)
                    
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\nStopped. Total resets detected: {reset_count}")

if __name__ == "__main__":
    monitor_ign_processes()
