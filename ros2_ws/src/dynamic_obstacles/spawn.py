#!/usr/bin/env python3

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
        return coords[:3]

    def calculate_distance(self, pose1, pose2):
        """Euclidean distance between two [x, y, z] points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pose1, pose2)))

    def calculate_yaw(self, current_pos, next_pos):
        """Compute heading (yaw) from current_pos to next_pos."""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        return math.atan2(dy, dx)

    def sample_waypoints(self, trajectory_content: str,
                         desired_velocity: float = 1.0,
                         sample_interval: int = 10):
        """
        Parse the original trajectory file, downsample waypoints,
        and compute strictly increasing times based on distance & velocity.
        """
        try:
            root = ET.fromstring(trajectory_content)
            all_waypoints = root.findall('.//waypoint')

            # Downsample (e.g. take every Nth waypoint)
            sampled_waypoints = all_waypoints[::sample_interval]
            if not sampled_waypoints:
                print("No waypoints found after downsampling!")
                return None

            # Start building our new <trajectory> element:
            new_trajectory = '<trajectory id="0" type="walk">\n'
            # Force no looping so the actor won't snap back:
            new_trajectory += '  <loop>false</loop>\n'

            cumulative_time = 0.0
            prev_pose = None

            for i, wp in enumerate(sampled_waypoints):
                # Get X, Y, Z from the <pose>
                original_pose_str = wp.find('pose').text.strip()
                current_pose = self.extract_pose_coordinates(original_pose_str)

                # Calculate yaw to the next waypoint
                yaw = 0.0
                if i < len(sampled_waypoints) - 1:
                    next_pose_str = sampled_waypoints[i + 1].find('pose').text
                    next_pose = self.extract_pose_coordinates(next_pose_str)
                    yaw = self.calculate_yaw(current_pose, next_pose)

                # Accumulate time based on distance / velocity
                if prev_pose is not None:
                    distance = self.calculate_distance(prev_pose, current_pose)
                    time_needed = distance / desired_velocity
                    cumulative_time += time_needed

                # Build <pose> with the computed yaw
                pose_with_yaw = f"{current_pose[0]} {current_pose[1]} {current_pose[2]} 0 0 {yaw}"

                # Add this waypoint to the new trajectory
                new_trajectory += f"  <waypoint>\n"
                new_trajectory += f"    <time>{cumulative_time:.2f}</time>\n"
                new_trajectory += f"    <pose>{pose_with_yaw}</pose>\n"
                new_trajectory += f"  </waypoint>\n"

                prev_pose = current_pose

            new_trajectory += "</trajectory>"
            print(f"Generated trajectory with {len(sampled_waypoints)} waypoints.")
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

    def get_first_waypoint(self, trajectory_content: str):
        """Extract the pose from the first <waypoint> in the final generated trajectory."""
        try:
            root = ET.fromstring(trajectory_content)
            first_waypoint = root.find('.//waypoint')
            if first_waypoint is not None:
                pose = first_waypoint.find('pose').text
                return pose.strip()
        except Exception as e:
            print(f"Error getting first waypoint: {e}")
        return "0 0 1.0 0 0 0"

    def spawn_actor(self, trajectory_sdf: str, name: str = "walking_actor"):
        """
        Create a complete SDF for the actor, write it to a temp file,
        and use ign service call to spawn it in the default world.
        """
        initial_pose = self.get_first_waypoint(trajectory_sdf)
        print(f"Setting initial pose to: {initial_pose}")

        # Insert <loop>false</loop> under the <animation> block as well, just to be safe
        actor_sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <actor name="{name}">
    <pose>{initial_pose}</pose>
    <skin>
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
    </skin>
    <animation name="walking">
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae</filename>
      <scale>1.0</scale>
      <interpolate_x>true</interpolate_x>
      <loop>false</loop>
    </animation>
    {trajectory_sdf}
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
    trajectory_file = 'trajectory_short.sdf'
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
