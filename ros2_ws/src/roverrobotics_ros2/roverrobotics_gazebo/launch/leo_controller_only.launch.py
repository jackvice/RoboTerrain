"""
Minimal Nav2 launch: Controller Server only for local obstacle avoidance.
Use this for benchmarking depth-based reactive navigation.

Usage:
  ros2 launch roverrobotics_gazebo leo_controller_only.launch.py
  ros2 launch roverrobotics_gazebo leo_controller_only.launch.py params_file:=/path/to/params.yaml
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    
    # Default path - use source directory
    default_params = os.path.join(
        os.path.expanduser('~'),
        'src/RoboTerrain/ros2_ws/src/roverrobotics_ros2/roverrobotics_gazebo/config',
        'leo_controller_only_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),
        
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Full path to the params file'
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}]
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}]
        ),
    ])
