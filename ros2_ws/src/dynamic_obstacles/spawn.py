#!/usr/bin/env python3
from typing import List, Optional
import math
import subprocess
import xml.etree.ElementTree as ET


def check_ign_available() -> bool:
    """Verify that the `ign` command is available."""
    try:
        subprocess.run(["ign", "gazebo", "--version"], check=True, capture_output=True)
        return True
    except FileNotFoundError:
        print("Error: 'ign' command not found. Is Gazebo Fortress installed?")
        return False


def load_trajectory_file(filepath: str) -> Optional[str]:
    """Load trajectory file from disk."""
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Trajectory file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading trajectory file: {e}")
        return None


def extract_pose_coordinates(pose_str: str) -> List[float]:
    """Convert a pose string to [x, y, z] with z offset."""
    coords = [float(x) for x in pose_str.split()]
    coords[2] += 1  # Add 1 to z-coordinate
    return coords[:3]


def calculate_yaw(current_pos: List[float], next_pos: List[float]) -> float:
    """Compute heading (yaw) from current_pos to next_pos."""
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    return math.atan2(dy, dx)


def calculate_distance(pose1: List[float], pose2: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(pose1, pose2)))


def process_trajectory(trajectory_content: str, desired_velocity: float = 1.0,
                       sample_interval: int = 1) -> Optional[str]:
    """
    Process trajectory XML:
    1. Extract and downsample waypoints
    2. Calculate timing based on desired velocity
    3. Calculate yaw for each waypoint pointing to next waypoint
    4. Return formatted trajectory XML
    """
    try:
        # Parse XML and extract waypoints
        root = ET.fromstring(trajectory_content)
        all_waypoints = root.findall('.//waypoint')
        
        # Downsample waypoints
        sampled_waypoints = all_waypoints[::sample_interval]
        if not sampled_waypoints:
            print("No waypoints found after downsampling!")
            return None
        
        # Extract positions
        positions: List[List[float]] = []
        for wp in sampled_waypoints:
            pose_str = wp.find('pose').text.strip()
            position = extract_pose_coordinates(pose_str)
            positions.append(position)
        
        # Calculate cumulative times based on distances
        times: List[float] = [0.0]
        cumulative_time = 0.0
        
        for i in range(1, len(positions)):
            distance = calculate_distance(positions[i-1], positions[i])
            time_needed = distance / desired_velocity
            cumulative_time += time_needed
            times.append(cumulative_time)
        
        # Add straight-line return to start (like original code)
        first_pos = positions[0]
        last_pos = positions[-1]
        return_distance = calculate_distance(last_pos, first_pos)
        return_time = return_distance / desired_velocity
        cumulative_time += return_time
        
        positions.append(first_pos)
        times.append(cumulative_time)
        
        # Build trajectory XML
        new_trajectory = '<trajectory id="0" type="walk">\n'
        
        for i in range(len(positions)):
            # Calculate yaw pointing to next waypoint
            if i < len(positions) - 1:
                # For all waypoints except the last, point to next waypoint
                yaw = calculate_yaw(positions[i], positions[i + 1])
            else:
                # For the last waypoint (back at start), maintain the yaw from the return journey
                # This prevents mid-journey rotation
                yaw = calculate_yaw(positions[i - 1], positions[i])
            
            # Format pose with calculated yaw
            pos = positions[i]
            pose_with_yaw = f"{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw}"
            
            new_trajectory += f"  <waypoint>\n"
            new_trajectory += f"    <time>{times[i]:.2f}</time>\n"
            new_trajectory += f"    <pose>{pose_with_yaw}</pose>\n"
            new_trajectory += f"  </waypoint>\n"
            
            print(f"Waypoint {i:03d}: yaw={yaw:+.3f} rad ({math.degrees(yaw):+.1f}°)")
        
        new_trajectory += "</trajectory>"
        print(f"Generated trajectory with {len(positions)} waypoints (including return to start).")
        return new_trajectory
        
    except Exception as e:
        print(f"Error creating trajectory: {e}")
        return None

def old_process_trajectory(trajectory_content: str, desired_velocity: float = 1.0, sample_interval: int = 1) -> Optional[str]:
    """
    Process trajectory XML:
    1. Extract and downsample waypoints
    2. Calculate timing based on desired velocity
    3. Calculate yaw for each waypoint pointing to next waypoint
    4. Return formatted trajectory XML
    """
    try:
        # Parse XML and extract waypoints
        root = ET.fromstring(trajectory_content)
        all_waypoints = root.findall('.//waypoint')
        
        # Downsample waypoints
        sampled_waypoints = all_waypoints[::sample_interval]
        if not sampled_waypoints:
            print("No waypoints found after downsampling!")
            return None
        
        # Extract positions
        positions: List[List[float]] = []
        for wp in sampled_waypoints:
            pose_str = wp.find('pose').text.strip()
            position = extract_pose_coordinates(pose_str)
            positions.append(position)
        
        # Calculate cumulative times based on distances
        times: List[float] = [0.0]
        cumulative_time = 0.0
        
        for i in range(1, len(positions)):
            distance = calculate_distance(positions[i-1], positions[i])
            time_needed = distance / desired_velocity
            cumulative_time += time_needed
            times.append(cumulative_time)
        
        # Add straight-line return to start (like original code)
        first_pos = positions[0]
        last_pos = positions[-1]
        return_distance = calculate_distance(last_pos, first_pos)
        return_time = return_distance / desired_velocity
        cumulative_time += return_time
        
        positions.append(first_pos)
        times.append(cumulative_time)
        
        # Build trajectory XML
        new_trajectory = '<trajectory id="0" type="walk">\n'
        
        for i in range(len(positions)):
            # Calculate yaw pointing to next waypoint
            if i == len(positions) - 1:
                # Last waypoint (back at start) should point to first trajectory waypoint (index 1)
                next_index = 1
            else:
                # All other waypoints point to the next waypoint in sequence
                next_index = i + 1
            
            yaw = calculate_yaw(positions[i], positions[next_index])
            
            # Format pose with calculated yaw
            pos = positions[i]
            pose_with_yaw = f"{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw}"
            
            new_trajectory += f"  <waypoint>\n"
            new_trajectory += f"    <time>{times[i]:.2f}</time>\n"
            new_trajectory += f"    <pose>{pose_with_yaw}</pose>\n"
            new_trajectory += f"  </waypoint>\n"
            
            print(f"Waypoint {i:03d}: yaw={yaw:+.3f} rad ({math.degrees(yaw):+.1f}°)")
        
        new_trajectory += "</trajectory>"
        print(f"Generated trajectory with {len(positions)} waypoints (including return to start).")
        return new_trajectory
        
    except Exception as e:
        print(f"Error creating trajectory: {e}")
        return None


def spawn_actor(trajectory_sdf: str, name: str = "walking_actor", world: str = "inspect") -> bool:
    """Create actor SDF and spawn it in Gazebo."""
    # Create the complete actor SDF
    actor_sdf = f'''<?xml version="1.0" ?>
    <sdf version="1.9">
    <actor name="{name}">
    <pose>0 0 0 0 0 0</pose>
    
    <skin>
    <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
    <scale>1.0</scale>
    </skin>

    <animation name="walk">
    <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
    <interpolate_x>true</interpolate_x>
    <loop>true</loop>
    </animation>

    <script>
    <loop>true</loop>
    <delay_start>0.0</delay_start>
    <auto_start>true</auto_start>
    {trajectory_sdf}
    </script>
    </actor>
    </sdf>'''

    # Write to temp file
    temp_sdf_path = '/tmp/actor_with_trajectory.sdf'
    try:
        with open(temp_sdf_path, 'w') as f:
            f.write(actor_sdf)
        print("Wrote actor SDF to", temp_sdf_path)
    except Exception as e:
        print(f"Error writing SDF: {e}")
        return False

    # Spawn the actor using ign service
    try:
        command = [
            'ign', 'service',
            '-s', f'/world/{world}/create',
            '--reqtype', 'ignition.msgs.EntityFactory',
            '--reptype', 'ignition.msgs.Boolean',
            '--timeout', '1000',
            '--req', f'sdf_filename: "{temp_sdf_path}"'
        ]

        print("\nExecuting command:", ' '.join(command))
        result = subprocess.run(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)

        print("Command output:")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("Return code:", result.returncode)

        return (result.returncode == 0)
    except Exception as e:
        print(f"Error spawning actor: {e}")
        return False


def main() -> None:
    """Main function to process trajectory and spawn actor."""
    # Check if ign command is available
    if not check_ign_available():
        return
    
    # Load trajectory file
    #trajectory_file = 'trajectory_short.sdf'
    trajectory_file = 'triangle_trajectory.sdf'
    
    raw_trajectory = load_trajectory_file(trajectory_file)
    if raw_trajectory is None:
        return

    # Process trajectory
    final_trajectory_sdf = process_trajectory(
        raw_trajectory,
        desired_velocity=1.0,
        sample_interval=1
    )
    if final_trajectory_sdf is None:
        return

    # Spawn actor
    success = spawn_actor(
        final_trajectory_sdf, 
        name="walking_actor",
        #world="inspect"
        world="default"
    )
    
    if success:
        print("\nActor successfully spawned with trajectory.")
    else:
        print("\nFailed to spawn actor.")


if __name__ == '__main__':
    main()
