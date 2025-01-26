#!/usr/bin/env python3

import subprocess
import xml.etree.ElementTree as ET
import os
import math

class ActorSpawner:
    def __init__(self):
        self.check_ign_available()

    def check_ign_available(self):
        try:
            subprocess.run(['ign', '--version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("Error: 'ign' command not found. Is Gazebo Fortress installed?")
            sys.exit(1)

    def extract_pose_coordinates(self, pose_str):
        """Convert pose string to coordinates"""
        coords = [float(x) for x in pose_str.split()]
        return coords[:3]  # Return just x, y, z

    def calculate_distance(self, pose1, pose2):
        """Calculate distance between two poses"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pose1, pose2)))

    def calculate_yaw(self, current_pos, next_pos):
        """Calculate yaw angle between current and next position"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        return math.atan2(dy, dx)

    def sample_waypoints(self, trajectory_content: str, desired_velocity: float = 1.0):
        try:
            root = ET.fromstring(trajectory_content)
            waypoints = root.findall('.//waypoint')
            
            # Sample every 10th point
            sample_interval = 2
            sampled_waypoints = waypoints[::sample_interval]
            
            new_trajectory = "<trajectory id=\"0\" type=\"walk\">\n"
            prev_pose = None
            
            for i, wp in enumerate(sampled_waypoints):
                pose_str = wp.find('pose').text
                current_pose = self.extract_pose_coordinates(pose_str)
                
                # Use original time from trajectory
                time = float(wp.find('time').text)
                
                yaw = 0
                if i < len(sampled_waypoints) - 1:
                    next_pose_str = sampled_waypoints[i+1].find('pose').text
                    next_pose = self.extract_pose_coordinates(next_pose_str)
                    yaw = self.calculate_yaw(current_pose, next_pose)
                elif prev_pose is not None:
                    yaw = self.calculate_yaw(prev_pose, current_pose)
                
                pose_with_yaw = f"{current_pose[0]} {current_pose[1]} {current_pose[2]} 0 0 {yaw}"
                new_trajectory += f"  <waypoint>\n    <time>{time:.2f}</time>\n    <pose>{pose_with_yaw}</pose>\n  </waypoint>\n"
                prev_pose = current_pose
            
            print(f"Generated trajectory with {len(sampled_waypoints)} points")
            new_trajectory += "</trajectory>"
            return new_trajectory
                
        except Exception as e:
            print(f"Error creating trajectory: {e}")
            return None
    

    def load_trajectory_file(self, filepath: str):
        """Load trajectory from file"""
        try:
            with open(filepath, 'r') as f:
                trajectory_content = f.read().strip()
            return trajectory_content
        except FileNotFoundError:
            print(f'Error: Trajectory file not found: {filepath}')
            return None
        except Exception as e:
            print(f'Error reading trajectory file: {e}')
            return None

    def get_first_waypoint(self, trajectory_content: str):
        """Extract position from first waypoint"""
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
        """Spawn an actor with the given trajectory"""
        # Get the initial pose from the first waypoint
        initial_pose = self.get_first_waypoint(trajectory_sdf)
        print(f"Setting initial pose to: {initial_pose}")

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
    </animation>
    {trajectory_sdf}
  </actor>
</sdf>'''

        # Write to temp file
        temp_sdf_path = '/tmp/actor_with_trajectory.sdf'
        try:
            with open(temp_sdf_path, 'w') as f:
                f.write(actor_sdf)
            print("SDF file written to", temp_sdf_path)

            command = [
                'ign', 'service', 
                '-s', '/world/default/create',
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

            return result.returncode == 0

        except Exception as e:
            print(f'Error spawning actor: {e}')
            return False

def main():
    spawner = ActorSpawner()

    # Load trajectory from file
    trajectory_file = 'trajectory.sdf'
    trajectory_sdf = spawner.load_trajectory_file(trajectory_file)
    
    if trajectory_sdf is None:
        return

    # Create trajectory with proper sequential timing
    sampled_trajectory = spawner.sample_waypoints(trajectory_sdf, desired_velocity=1.0)
    
    if sampled_trajectory is None:
        return

    # Spawn actor with the trajectory
    success = spawner.spawn_actor(sampled_trajectory)
    
    if success:
        print('\nActor successfully spawned with trajectory')
    else:
        print('\nFailed to spawn actor')

if __name__ == '__main__':
    main()
