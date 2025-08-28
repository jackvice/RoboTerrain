#!/usr/bin/env python3
"""
Working actor position publisher that calculates real trajectory positions.
"""

from typing import List, Tuple, Optional
import math
import xml.etree.ElementTree as ET
import subprocess
import re
import rclpy
from geometry_msgs.msg import PoseStamped


def load_trajectory_waypoints(trajectory_file: str) -> List[Tuple[float, float, float, float]]:
    """Load waypoints from trajectory file and return list of (x, y, z, time)."""
    try:
        with open(trajectory_file, 'r') as f:
            content = f.read()
        
        root = ET.fromstring(content)
        waypoints = root.findall('.//waypoint')
        
        positions: List[Tuple[float, float, float]] = []
        for wp in waypoints:
            pose_element = wp.find('pose')
            if pose_element is not None:
                coords = [float(x) for x in pose_element.text.strip().split()]
                x, y, z = coords[0], coords[1], coords[2] + 1.0
                positions.append((x, y, z))
        
        waypoints_with_time: List[Tuple[float, float, float, float]] = []
        velocity = 1.0
        cumulative_time = 0.0
        
        waypoints_with_time.append((positions[0][0], positions[0][1], positions[0][2], 0.0))
        
        for i in range(1, len(positions)):
            distance = math.sqrt(
                (positions[i][0] - positions[i-1][0])**2 + 
                (positions[i][1] - positions[i-1][1])**2
            )
            time_needed = distance / velocity
            cumulative_time += time_needed
            waypoints_with_time.append((positions[i][0], positions[i][1], positions[i][2], cumulative_time))
        
        # Add return to start
        distance_to_start = math.sqrt(
            (positions[0][0] - positions[-1][0])**2 + 
            (positions[0][1] - positions[-1][1])**2
        )
        return_time = distance_to_start / velocity
        cumulative_time += return_time
        waypoints_with_time.append((positions[0][0], positions[0][1], positions[0][2], cumulative_time))
        
        return waypoints_with_time
        
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return []


def get_simulation_time(world_name: str) -> Optional[float]:
    """Get current simulation time from Gazebo."""
    try:
        result = subprocess.run([
            'ign', 'topic', '-e', '-t', f'/world/{world_name}/stats', '-n', '1'
        ], capture_output=True, text=True, timeout=0.5)
        
        if result.returncode == 0:
            sim_time_match = re.search(r'sim_time\s*\{\s*sec:\s*(\d+)\s*nsec:\s*(\d+)', result.stdout)
            if sim_time_match:
                sec = int(sim_time_match.group(1))
                nsec = int(sim_time_match.group(2))
                return sec + nsec / 1e9
        return None
    except:
        return None


def interpolate_actor_position(waypoints: List[Tuple[float, float, float, float]], 
                              current_time: float) -> Optional[Tuple[float, float, float]]:
    """Calculate actor position at current_time by interpolating between waypoints."""
    if not waypoints:
        return None
    
    total_duration = waypoints[-1][3]
    if total_duration > 0:
        current_time = current_time % total_duration
    
    for i in range(len(waypoints) - 1):
        start_wp = waypoints[i]
        end_wp = waypoints[i + 1]
        
        if start_wp[3] <= current_time <= end_wp[3]:
            segment_duration = end_wp[3] - start_wp[3]
            
            if segment_duration == 0:
                return (start_wp[0], start_wp[1], start_wp[2])
            
            t = (current_time - start_wp[3]) / segment_duration
            x = start_wp[0] + t * (end_wp[0] - start_wp[0])
            y = start_wp[1] + t * (end_wp[1] - start_wp[1])
            z = start_wp[2] + t * (end_wp[2] - start_wp[2])
            
            return (x, y, z)
    
    last_wp = waypoints[-1]
    return (last_wp[0], last_wp[1], last_wp[2])


def main():
    """Publish calculated actor position to ROS topic."""
    
    trajectory_file = "./trajectories/inspect_corner_triangle.sdf"
    world_name = "inspect"
    
    # Load trajectory
    print("Loading trajectory...")
    waypoints = load_trajectory_waypoints(trajectory_file)
    if not waypoints:
        print("Failed to load trajectory, exiting")
        return
    
    print(f"Loaded {len(waypoints)} waypoints")
    
    # Initialize ROS
    rclpy.init()
    node = rclpy.create_node('actor_position_publisher')
    pub = node.create_publisher(PoseStamped, '/actor/pose', 10)
    
    print("Publishing calculated actor position to /actor/pose")
    print("Press Ctrl+C to stop")
    
    try:
        count = 0
        while rclpy.ok():
            # Get current simulation time
            sim_time = get_simulation_time(world_name)
            
            if sim_time is not None:
                # Calculate actual actor position
                actor_pos = interpolate_actor_position(waypoints, sim_time)
                
                if actor_pos is not None:
                    msg = PoseStamped()
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.header.frame_id = 'world'
                    msg.pose.position.x = actor_pos[0]
                    msg.pose.position.y = actor_pos[1]
                    msg.pose.position.z = actor_pos[2]
                    msg.pose.orientation.w = 1.0
                    
                    pub.publish(msg)
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"Published {count} messages - Actor at ({actor_pos[0]:.2f}, {actor_pos[1]:.2f}, {actor_pos[2]:.2f})")
            
            rclpy.spin_once(node, timeout_sec=0.01)
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    
