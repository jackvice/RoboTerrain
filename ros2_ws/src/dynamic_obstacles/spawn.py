#!/usr/bin/env python3
from typing import Tuple, List, Optional, Callable
import math

import subprocess
import xml.etree.ElementTree as ET
import math

class ActorSpawner:
    def __init__(self):
        self.check_ign_available()

    def check_ign_available(self):
        """Verify that the `ign` command is available."""
        try:
            subprocess.run(["ign", "gazebo", "--version"], check=True)
        except FileNotFoundError:
            print("Error: 'ign' command not found. Is Gazebo Fortress installed?")
            exit(1)

    def extract_pose_coordinates(self, pose_str):
        """Convert a <pose> string (x y z roll pitch yaw) to just [x, y, z]."""
        coords = [float(x) for x in pose_str.split()]
        coords[2] += 1
        return coords[:3]

    def calculate_distance(self, pose1, pose2):
        """Euclidean distance between two [x, y, z] points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pose1, pose2)))


    def optimize_trajectory_yaw(
            self,
            waypoints: list[tuple[list[float], float]] ) -> list[tuple[list[float], float, float]]:
        """
        Return (pos, time, yaw) where yaw follows the direction of travel
        and is unwrapped so adjacent values never differ by > π.
        This prevents Gazebo from interpolating a 180° spin.
        """
        positions = [p for p, _ in waypoints]
        times     = [t for _, t in waypoints]

        # --- 1. raw yaw for every segment i → i+1 --------------------------
        raw_yaws = []
        for i in range(len(positions) - 1):
            dx = positions[i+1][0] - positions[i][0]
            dy = positions[i+1][1] - positions[i][1]
            raw_yaws.append(math.atan2(dy, dx))
        raw_yaws.append(raw_yaws[0])          # closing edge

        # --- 2. unwrap so successive values differ by ≤ π -----------------
        unwrapped = [raw_yaws[0]]
        for r in raw_yaws[1:]:
            y = r
            prev = unwrapped[-1]
            # shift y by ±2π until it is within π of prev
            while y - prev >  math.pi:
                y -= 2 * math.pi
            while y - prev < -math.pi:
                y += 2 * math.pi
            unwrapped.append(y)

        # --- 3. (optional) normalise back into [-π, π] for prettiness -----
        yaws = [((y + math.pi) % (2 * math.pi)) - math.pi for y in unwrapped]

        return [(positions[i], times[i], yaws[i]) for i in range(len(positions))]


    
    def sample_waypoints(self, trajectory_content: str,
                         desired_velocity: float = 1.0,
                         sample_interval: int = 10):
        """
        Parse the original trajectory file, downsample waypoints,
        and compute strictly increasing times based on distance & velocity.
        Uses pre-processing to ensure optimal yaw values.
        """
        try:
            root = ET.fromstring(trajectory_content)
            all_waypoints = root.findall('.//waypoint')
            
            # Downsample (e.g. take every Nth waypoint)
            sampled_waypoints = all_waypoints[::sample_interval]
            if not sampled_waypoints:
                print("No waypoints found after downsampling!")
                return None
                
            # Extract positions and calculate timing
            positions = []
            times = []
            cumulative_time = 0.0
            prev_pos = None
            
            # Process each waypoint
            for wp in sampled_waypoints:
                original_pose_str = wp.find('pose').text.strip()
                current_pos = self.extract_pose_coordinates(original_pose_str)
                positions.append(current_pos)
                
                if prev_pos is not None:
                    distance = self.calculate_distance(prev_pos, current_pos)
                    time_needed = distance / desired_velocity
                    cumulative_time += time_needed
                
                times.append(cumulative_time)
                prev_pos = current_pos
            
            # Add the loop-closing waypoint
            first_pos = positions[0]
            last_pos = positions[-1]
            distance = self.calculate_distance(last_pos, first_pos)
            time_needed = distance / desired_velocity
            cumulative_time += time_needed
            positions.append(first_pos)
            times.append(cumulative_time)
            
            # Calculate optimized yaw values
            waypoints_with_times = list(zip(positions, times))
            optimized_waypoints = self.optimize_trajectory_yaw(waypoints_with_times)
            
            # Build the trajectory XML
            new_trajectory = '<trajectory id="0" type="walk">\n'
            
            # For diagnostic purposes, track the previous yaw
            prev_yaw = None
            
            for i, (pos, time, yaw) in enumerate(optimized_waypoints):
                pose_with_yaw = f"{pos[0]} {pos[1]} {pos[2]} 0 0 {yaw}"
                new_trajectory += f"  <waypoint>\n"
                new_trajectory += f"    <time>{time:.2f}</time>\n"
                new_trajectory += f"    <pose>{pose_with_yaw}</pose>\n"
                new_trajectory += f"  </waypoint>\n"
                
                # Print diagnostic info
                print(f"{i:03d}  yaw={yaw:+.3f}  Δ={yaw-prev_yaw if prev_yaw is not None else 0:+.3f}")
                prev_yaw = yaw
            
            new_trajectory += "</trajectory>"
            print(f"Generated trajectory with {len(optimized_waypoints)} waypoints (including loop closure).")
            return new_trajectory
            
        except Exception as e:
            print(f"Error creating trajectory: {e}")
            return None

        
    def load_trajectory_file(self, filepath: str):
        """Load original trajectory file from disk."""
        try:
            with open(filepath, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: Trajectory file not found: {filepath}")
            return None
        except Exception as e:
            print(f"Error reading trajectory file: {e}")
            return None


    def spawn_actor(self, trajectory_sdf: str, name: str = "walking_actor"):
        """
        Create a complete SDF for the actor, write it to a temp file,
        and use ign service call to spawn it in the default world.
        """
        #initial_pose = self.get_first_waypoint(trajectory_sdf)
        #print(f"Setting initial pose to: {initial_pose}")
        initial_pose = "0 0 0 0 0 0"
        print("Setting initial pose to neutral (0 0 0 0 0 0)")

        # final_trajectory_sdf already contains just <trajectory>…</trajectory>
        actor_sdf = f'''<?xml version="1.0" ?>
        <sdf version="1.9">
        <actor name="{name}">
        <pose>{initial_pose}</pose>
        
        <skin>
        <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
        <scale>1.0</scale>
        </skin>

        <animation name="walk">
        <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
        <interpolate_x>true</interpolate_x>
        <loop>true</loop>          <!-- keep skeleton cycling -->
        </animation>

        <script>
        <loop>true</loop>          <!-- makes the whole path repeat -->
        <delay_start>0.0</delay_start>
        <auto_start>true</auto_start>
        {trajectory_sdf}     <!-- your way‑points go here -->
        </script>
        </actor>
        </sdf>'''

        
        temp_sdf_path = '/tmp/actor_with_trajectory.sdf'
        try:
            # Write the SDF to a temporary file
            with open(temp_sdf_path, 'w') as f:
                f.write(actor_sdf)
            print("Wrote actor SDF to", temp_sdf_path)

            # Use ign-transport to spawn the actor in /world/default
            command = [
                'ign', 'service',
                '-s', '/world/inspect/create',
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

            print("\nCommand output:")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            print("Return code:", result.returncode)

            return (result.returncode == 0)

        except Exception as e:
            print(f"Error spawning actor: {e}")
            return False



        
def main():
    spawner = ActorSpawner()

    # Path to your input trajectory (the large, dense one)
    #trajectory_file = 'trajectory_1.sdf'
    trajectory_file = 'trajectory_short.sdf'
    #trajectory_file = 'trajectory_shortest.sdf'
    raw_trajectory = spawner.load_trajectory_file(trajectory_file)
    if raw_trajectory is None:
        return

    # Create a trajectory with consistent speed, no looping, and downsampling
    # Adjust desired_velocity if you want faster/slower
    # Adjust sample_interval if you want more/fewer points
    final_trajectory_sdf = spawner.sample_waypoints(raw_trajectory,
                                                    desired_velocity=1.0,
                                                    sample_interval=1)
    if final_trajectory_sdf is None:
        return

    # Now spawn the actor with this new trajectory
    success = spawner.spawn_actor(final_trajectory_sdf, name="walking_actor")
    if success:
        print("\nActor successfully spawned with trajectory.")
    else:
        print("\nFailed to spawn actor.")


if __name__ == '__main__':
    main()
