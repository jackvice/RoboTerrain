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
import argparse

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

        TURN_DURATION = 0.1     # seconds for the snap-turn

        for i in range(len(positions)):
            pos = positions[i]
            t   = times[i]

            # -------- 1) heading to KEEP while walking the current leg --------
            if i == 0:
                # first point: only a next segment exists
                yaw_hold = calculate_yaw(positions[0], positions[1])
            else:
                # use heading of the segment we just walked (prev → current)
                yaw_hold = calculate_yaw(positions[i - 1], positions[i])

            # keep this heading until we arrive at the corner
            pose_hold = f"{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw_hold}"
            new_trajectory += (
                f"  <waypoint>\n"
                f"    <time>{t:.2f}</time>\n"
                f"    <pose>{pose_hold}</pose>\n"
                f"  </waypoint>\n"
            )
            print(
                f"Corner {i:02d} arrive  t={t:.2f}s  hold_yaw={math.degrees(yaw_hold):+.1f}°"
            )

            # -------- 2) tiny turn waypoint (skip for final point that loops) ------
            if i < len(positions) - 1:
                yaw_next = calculate_yaw(positions[i], positions[i + 1])

                pose_turn = f"{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw_next}"
                t_turn    = t + TURN_DURATION
                new_trajectory += (
                    f"  <waypoint>\n"
                    f"    <time>{t_turn:.2f}</time>\n"
                    f"    <pose>{pose_turn}</pose>\n"
                    f"  </waypoint>\n"
                )
                print(
                    f"           turn → yaw_next={math.degrees(yaw_next):+.1f}° "
                    f"(Δt={TURN_DURATION}s)"
                )


        
        new_trajectory += "</trajectory>"
        print(f"Generated trajectory with {len(positions)} waypoints (including return to start).")
        return new_trajectory
        
    except Exception as e:
        print(f"Error creating trajectory: {e}")
        return None



def sdf_header():
    actor_sdf= f'''<?xml version="1.0" ?>
    <sdf version="1.9">
    <actor name="{name}_actor">
    <pose>0 0 0 0 0 0</pose>
    
    <skin>
    <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
    <scale>1.0</scale>
    </skin>

    <animation name="{animate_name}_walk">
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
    return actor_sdf

    
def spawn_actor(trajectory_sdf: str, name: str = "first", animate_name: str = 'one',
                world: str = "inspect") -> bool:
    """Create actor SDF and spawn it in Gazebo."""
    # Create the complete actor SDF

    actor_sdf = sdf_header()

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



def main(trajectory_file: str = "flat_triangle_traject_rev_2nd.sdf",
         actor_name: str = "walking_actor",
         animate_name: str = "walk",
         world_name: str = "moon") -> None:
    """
    Spawn actor (processed SDF) then publish pose computed from RAW trajectory file.
    No changes to existing helper functions.
    """
    import time

    # --- Spawn (as in spawn.py) ---
    if not check_ign_available():
        return

    raw_trajectory = load_trajectory_file(trajectory_file)
    if raw_trajectory is None:
        return

    final_trajectory_sdf = process_trajectory(raw_trajectory,
                                              desired_velocity=1.0,
                                              sample_interval=1)
    if final_trajectory_sdf is None:
        return

    # Seed globals for sdf_header() which uses free vars: name, animate_name, trajectory_sdf
    globals().update({"name": actor_name,
                      "animate_name": animate_name,
                      "trajectory_sdf": final_trajectory_sdf})  # needed by sdf_header()

    success = spawn_actor(final_trajectory_sdf,
                          name=actor_name,
                          animate_name=animate_name,
                          world=world_name)
    if not success:
        print("\nFailed to spawn actor.")
        return
    print("\nActor successfully spawned with trajectory.")

    # Give Gazebo a moment to register the new entity before publishing
    time.sleep(1.0)

    # --- Publish (from actor_publisher.py) against RAW file ---
    waypoints = load_trajectory_waypoints(trajectory_file)
    if not waypoints:
        print("Failed to load trajectory for publishing, exiting")
        return

    rclpy.init()
    node = rclpy.create_node('actor_position_publisher')
    pub = node.create_publisher(PoseStamped, '/'+ actor_name +'_actor/pose', 10)
    print("Publishing calculated actor position to /actor/pose")
    print("Press Ctrl+C to stop")


    try:
        count = 0
        sim_time = 0.0
    
        while rclpy.ok():
            # Update simulation time every 5 iterations (gives ~25 Hz publishing)
            if count % 5 == 0:
                new_sim_time = get_simulation_time(world_name)
                if new_sim_time is not None:
                    sim_time = new_sim_time
        
            if sim_time > 0:
                pos = interpolate_actor_position(waypoints, sim_time)
                if pos is not None:
                    msg = PoseStamped()
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.header.frame_id = 'world'
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
                    msg.pose.orientation.w = 1.0
                    pub.publish(msg)
                    count += 1
                    if count % 200 == 0:
                        print(f"Published {count} messages - Actor at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
            rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    """
    try:
        count = 0
        while rclpy.ok():
            sim_time = get_simulation_time(world_name)
            if sim_time is not None:
                pos = interpolate_actor_position(waypoints, sim_time)
                if pos is not None:
                    msg = PoseStamped()
                    msg.header.stamp = node.get_clock().now().to_msg()
                    msg.header.frame_id = 'world'
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pos
                    msg.pose.orientation.w = 1.0
                    pub.publish(msg)
                    count += 1
                    if count % 100 == 0:
                        print(f"Published {count} messages - Actor at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            rclpy.spin_once(node, timeout_sec=0.01)
    """



        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Spawn a walking actor with custom trajectory.")
    parser.add_argument('--trajectory_file', type=str, default='flat_triangle_traject_rev.sdf',
                        help='Path to trajectory .sdf file (default: flat_triangle_traject_rev_2nd.sdf)')
    parser.add_argument('--actor_name', type=str, default='first',
                        help='Unique name for the Gazebo actor (default: first)')
    parser.add_argument('--animate_name', type=str, default='actor1',
                        help='Unique animation name to avoid conflicts (default: actor1)')
    parser.add_argument('--world_name', type=str, default='moon',
                        help='Unique world name (default: moon)')

    args = parser.parse_args()
    main(args.trajectory_file, args.actor_name, args.animate_name, args.world_name)

