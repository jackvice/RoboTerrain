#!/usr/bin/env python3
"""
Simple metrics collector for 20-minute robot evaluation.
Records velocity, actor distances, and goal count to CSV.
"""

from typing import List, Tuple, Optional
import math
import time
import csv
import rclpy
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import String


def calculate_distance(pos1: Tuple[float, float, float], 
                      pos2: Tuple[float, float, float]) -> float:
    """Calculate 2D distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_velocity(pos1: Tuple[float, float, float], 
                      pos2: Tuple[float, float, float], 
                      time_diff: float) -> float:
    """Calculate velocity from position change over time."""
    if time_diff <= 0:
        return 0.0
    distance = calculate_distance(pos1, pos2)
    return distance / time_diff


def write_csv_header(filepath: str) -> None:
    """Write CSV header row."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'robot_x', 'robot_y', 'robot_z', 
            'velocity_mps', 'actor1_dist', 'actor2_dist', 'goals_total'
        ])


def write_csv_row(filepath: str, timestamp: float, robot_pos: Tuple[float, float, float],
                 velocity: float, actor_distances: List[float], goals_count: int) -> None:
    """Write one data row to CSV."""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{timestamp:.2f}", f"{robot_pos[0]:.3f}", f"{robot_pos[1]:.3f}", f"{robot_pos[2]:.3f}",
            f"{velocity:.3f}", f"{actor_distances[0]:.3f}", f"{actor_distances[1]:.3f}", goals_count
        ])


def main():
    """Collect metrics for 20 minutes and write to CSV."""
    
    world_name = "inspect"
    csv_filename = f"robot_metrics_{world_name}_{int(time.time())}.csv"
    
    # Data storage
    robot_position: Optional[Tuple[float, float, float]] = None
    actor_positions: List[Tuple[float, float, float]] = [None, None]
    goals_count: int = 0
    
    # Velocity calculation
    last_robot_position: Optional[Tuple[float, float, float]] = None
    last_timestamp: float = 0.0
    
    # Initialize ROS
    rclpy.init()
    node = rclpy.create_node('metrics_collector')
    
    def robot_pose_callback(msg: PoseArray) -> None:
        nonlocal robot_position
        if msg.poses:
            pose = msg.poses[0]
            robot_position = (pose.position.x, pose.position.y, pose.position.z)
    
    def actor_pose_callback(msg: PoseStamped) -> None:
        nonlocal actor_positions
        # Assuming single actor publisher - extend for multiple actors if needed
        actor_positions[0] = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        # For now, duplicate for second actor (modify when you have second actor publisher)
        actor_positions[1] = (msg.pose.position.x + 1.0, msg.pose.position.y + 1.0, msg.pose.position.z)
    
    def goal_event_callback(msg: String) -> None:
        nonlocal goals_count
        if "goal_reached" in msg.data.lower():
            goals_count += 1
    
    # Subscribers
    robot_sub = node.create_subscription(PoseArray, '/rover/pose_array', robot_pose_callback, 10)
    actor_sub = node.create_subscription(PoseStamped, '/actor/pose', actor_pose_callback, 10)
    event_sub = node.create_subscription(String, '/robot/events', goal_event_callback, 10)
    
    # Create CSV file
    write_csv_header(csv_filename)
    print(f"Starting 20-minute metrics collection...")
    print(f"Writing to: {csv_filename}")
    
    start_time = time.time()
    next_log_time = start_time + 1.0
    
    try:
        while time.time() - start_time < 1200:  # 20 minutes = 1200 seconds
            rclpy.spin_once(node, timeout_sec=0.01)
            
            current_time = time.time()
            
            # Log metrics every second
            if current_time >= next_log_time:
                if robot_position is not None and all(pos is not None for pos in actor_positions):
                    
                    # Calculate velocity
                    velocity = 0.0
                    if last_robot_position is not None and last_timestamp > 0:
                        time_diff = current_time - last_timestamp
                        velocity = calculate_velocity(last_robot_position, robot_position, time_diff)
                    
                    # Calculate distances to actors
                    actor_distances = [
                        calculate_distance(robot_position, actor_positions[0]) if actor_positions[0] else 0.0,
                        calculate_distance(robot_position, actor_positions[1]) if actor_positions[1] else 0.0
                    ]
                    
                    # Write to CSV
                    write_csv_row(csv_filename, current_time, robot_position, 
                                 velocity, actor_distances, goals_count)
                    
                    # Update for next velocity calculation
                    last_robot_position = robot_position
                    last_timestamp = current_time
                    
                    # Progress update
                    elapsed = current_time - start_time
                    remaining = 1200 - elapsed
                    if int(elapsed) % 60 == 0:  # Print every minute
                        print(f"Progress: {elapsed/60:.1f}/20.0 minutes, Goals: {goals_count}")
                
                next_log_time += 1.0
                
    except KeyboardInterrupt:
        print("\nCollection stopped by user")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\nMetrics collection complete!")
        print(f"Duration: {elapsed/60:.1f} minutes")
        print(f"Goals reached: {goals_count}")
        print(f"Data saved to: {csv_filename}")
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
