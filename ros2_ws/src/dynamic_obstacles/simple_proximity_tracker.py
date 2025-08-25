#!/usr/bin/env python3
"""
Single functional script to track robot-actor proximity.
No dependencies on other scripts - gets robot pose directly from Ignition.
"""

from typing import List, Tuple, Optional
import math
import xml.etree.ElementTree as ET
import subprocess
import time
import re


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
                x, y, z = coords[0], coords[1], coords[2] + 1.0  # Add 1 to z like spawn.py
                positions.append((x, y, z))
        
        # Calculate timing based on velocity (like spawn.py)
        waypoints_with_time: List[Tuple[float, float, float, float]] = []
        velocity = 1.0  # m/s
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



def interpolate_actor_position(waypoints: List[Tuple[float, float, float, float]], 
                              current_time: float) -> Optional[Tuple[float, float, float]]:
    """Calculate actor position at current_time by interpolating between waypoints."""
    if not waypoints:
        return None
    
    # Handle looping trajectory
    total_duration = waypoints[-1][3]
    if total_duration > 0:
        current_time = current_time % total_duration
    
    # Find segment we're in
    for i in range(len(waypoints) - 1):
        start_wp = waypoints[i]
        end_wp = waypoints[i + 1]
        
        if start_wp[3] <= current_time <= end_wp[3]:
            segment_duration = end_wp[3] - start_wp[3]
            
            if segment_duration == 0:
                return (start_wp[0], start_wp[1], start_wp[2])
            
            # Linear interpolation
            t = (current_time - start_wp[3]) / segment_duration
            x = start_wp[0] + t * (end_wp[0] - start_wp[0])
            y = start_wp[1] + t * (end_wp[1] - start_wp[1])
            z = start_wp[2] + t * (end_wp[2] - start_wp[2])
            
            return (x, y, z)
    
    # Use last waypoint if time is beyond trajectory
    last_wp = waypoints[-1]
    return (last_wp[0], last_wp[1], last_wp[2])


def calculate_distance(pos1: Tuple[float, float, float], 
                      pos2: Tuple[float, float, float]) -> float:
    """Calculate 2D distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)



def get_simulation_time(world_name: str) -> Optional[float]:
    """Get current simulation time from Gazebo - optimized for speed."""
    try:
        result = subprocess.run([
            'ign', 'topic', '-e', '-t', f'/world/{world_name}/stats', '-n', '1'
        ], capture_output=True, text=True, timeout=0.5)  # Reduced timeout
        
        if result.returncode == 0:
            sim_time_match = re.search(r'sim_time\s*\{\s*sec:\s*(\d+)\s*nsec:\s*(\d+)', result.stdout)
            if sim_time_match:
                sec = int(sim_time_match.group(1))
                nsec = int(sim_time_match.group(2))
                return sec + nsec / 1e9
        return None
    except:
        return None


def get_robot_position(world_name: str, robot_name: str) -> Optional[Tuple[float, float, float]]:
    """Get robot position directly from Ignition pose topic - optimized for speed."""
    try:
        result = subprocess.run([
            'ign', 'topic', '-e', '-t', f'/world/{world_name}/pose/info', '-n', '1'
        ], capture_output=True, text=True, timeout=0.5)  # Reduced timeout
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            robot_found = False
            x, y, z = None, None, None
            
            for i, line in enumerate(lines):
                if f'name: "{robot_name}"' in line:
                    robot_found = True
                elif robot_found and 'position {' in line:
                    for j in range(i+1, min(i+8, len(lines))):  # Reduced search range
                        coord_line = lines[j]
                        if 'x:' in coord_line:
                            x = float(coord_line.split(':')[1].strip())
                        elif 'y:' in coord_line:
                            y = float(coord_line.split(':')[1].strip())
                        elif 'z:' in coord_line:
                            z = float(coord_line.split(':')[1].strip())
                        elif '}' in coord_line:
                            break
                    
                    if x is not None and y is not None and z is not None:
                        return (x, y, z)
                    break
        return None
    except:
        return None



def get_actor_heading(waypoints: List[Tuple[float, float, float, float]], 
                     current_time: float) -> Optional[float]:
    """Calculate actor's current heading based on trajectory direction."""
    if not waypoints:
        return None
    
    total_duration = waypoints[-1][3]
    if total_duration > 0:
        current_time = current_time % total_duration
    
    for i in range(len(waypoints) - 1):
        if waypoints[i][3] <= current_time <= waypoints[i + 1][3]:
            start_wp = waypoints[i]
            end_wp = waypoints[i + 1]
            
            dx = end_wp[0] - start_wp[0]
            dy = end_wp[1] - start_wp[1]
            return math.atan2(dy, dx)
    
    return None


def distance_to_line_segment(point: Tuple[float, float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
    """Calculate perpendicular distance from point to line segment."""
    px, py = point[0], point[1]
    x1, y1 = line_start
    x2, y2 = line_end
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def is_robot_in_forward_path(robot_pos: Tuple[float, float, float],
                           actor_pos: Tuple[float, float, float],
                           actor_heading: float) -> Tuple[bool, float]:
    """Check if robot is within 0.3m of 2m line in front of actor."""
    line_start = (actor_pos[0], actor_pos[1])
    line_end = (
        actor_pos[0] + 2.0 * math.cos(actor_heading),
        actor_pos[1] + 2.0 * math.sin(actor_heading)
    )
    
    distance_to_path = distance_to_line_segment(robot_pos, line_start, line_end)
    
    return (distance_to_path <= 0.3, distance_to_path)  # 0.3 meter threshold



def main():
    """Main function to track robot-actor proximity - maximum speed."""
    
    trajectory_file = "./trajectories/inspect_corner_triangle.sdf"
    world_name = "inspect"
    robot_name = "leo_rover"
    
    print(f"Loading trajectory from {trajectory_file}...")
    waypoints = load_trajectory_waypoints(trajectory_file)
    if not waypoints:
        print("Failed to load trajectory, exiting")
        return
    
    print(f"Loaded {len(waypoints)} waypoints")
    print("High-speed tracking active...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            sim_time = get_simulation_time(world_name)
            robot_pos = get_robot_position(world_name, robot_name)
            
            if sim_time is not None and robot_pos is not None:
                actor_pos = interpolate_actor_position(waypoints, sim_time)
                actor_heading = get_actor_heading(waypoints, sim_time)
                
                if actor_pos is not None and actor_heading is not None:
                    distance = calculate_distance(robot_pos, actor_pos)
                    in_path, path_distance = is_robot_in_forward_path(robot_pos, actor_pos, actor_heading)
                    
                    if in_path:
                        print("Robot in the way.")
                    
                    if distance < 5.0:
                        print(f"Distance between actor and robot is: {distance:.2f} meters")
                        
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
