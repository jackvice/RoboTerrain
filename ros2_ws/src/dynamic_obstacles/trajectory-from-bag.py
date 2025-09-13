#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import math
import sqlite3
from geometry_msgs.msg import PoseArray

class TrajectoryGeneratorNode(Node):
    def __init__(self):
        super().__init__('trajectory_generator')
        
    def read_bag_file(self, bag_path):
        """Read poses from bag file"""
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        
        poses = []
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == '/rover/pose_array':
                msg_type = get_message(type_map[topic])
                pose_array_msg = deserialize_message(data, msg_type)
                pose = pose_array_msg.poses[0]  # Assuming single pose in array
                poses.append((
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ))
        return poses

    def calculate_3d_waypoints(self, poses_3d, velocity, speed_multiplier=2.0):
        """Calculate waypoints with timing information"""
        waypoints_3d = []
        time = 0
        
        # Apply speed multiplier by dividing the time intervals
        adjusted_velocity = velocity * speed_multiplier
        
        for i, (x, y, z) in enumerate(poses_3d):
            if i > 0:
                # Calculate distance from previous waypoint
                prev_x, prev_y, prev_z = poses_3d[i-1]
                distance = math.sqrt(
                    (x - prev_x) ** 2 + 
                    (y - prev_y) ** 2 + 
                    (z - prev_z) ** 2
                )
                # Calculate time needed at desired velocity
                time += distance / adjusted_velocity
            waypoints_3d.append((x, y, z, time))
        
        return waypoints_3d

    def generate_sdf_trajectory_block(self, waypoints_3d):
        """Generate SDF trajectory XML block"""
        trajectory_block = "<trajectory id=\"0\" type=\"walk\">\n"
        for wp in waypoints_3d:
            x, y, z, time = wp
            yaw = 0  # Default yaw, could be calculated from consecutive points if needed
            trajectory_block += f"  <waypoint>\n    <time>{time:.2f}</time>\n    <pose>{x} {y} {z} 0 0 {yaw}</pose>\n  </waypoint>\n"
        trajectory_block += "</trajectory>"
        return trajectory_block

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()

    # Path to your recorded bag file
    #bag_path = 'rover_trajectory'  # Update this to your bag file path
    #bag_path = 'inspect_linear'  # Update this to your bag file path
    bag_path = 'rosbag2_2025_09_12-19_30_13'  # Update this to your bag file path
    
    # Read poses from bag
    poses_3d = node.read_bag_file(bag_path)
    
    # Base velocity (robot's velocity)
    base_velocity = 0.5  # m/s
    
    # Make actor move 2x faster than the robot
    speed_multiplier = 2.0
    
    # Calculate waypoints with timing
    waypoints_3d = node.calculate_3d_waypoints(
        poses_3d, 
        velocity=base_velocity,
        speed_multiplier=speed_multiplier
    )
    
    # Generate and print SDF trajectory block
    sdf_trajectory_block = node.generate_sdf_trajectory_block(waypoints_3d)
    print(sdf_trajectory_block)
    
    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
