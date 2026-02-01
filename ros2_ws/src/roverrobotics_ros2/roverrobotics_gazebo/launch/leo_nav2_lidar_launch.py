"""
Nav2 launch for 2D LiDAR-based obstacle avoidance with 2D Nav Goals.
No map/AMCL - uses odom frame as global frame for mapless navigation.

Configured for Leo Rover with gpu_lidar sensor on /scan topic.

Includes:
  - controller_server (DWB local planner)
  - planner_server (NavFn global planner)
  - behavior_server (recovery behaviors)
  - bt_navigator (behavior tree executor)
  - smoother_server (path smoothing)
  - velocity_smoother (velocity smoothing)
  - waypoint_follower (multi-goal navigation)

Usage:
  ros2 launch roverrobotics_gazebo leo_nav2_lidar_launch.py
  ros2 launch roverrobotics_gazebo leo_nav2_lidar_launch.py headless:=true
"""

from __future__ import annotations

import os
from typing import Final, Sequence

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Type aliases
NodeConfig = tuple[str, str, str]  # (package, executable, name)


# Nav2 node configurations
NAV2_NODES: Final[Sequence[NodeConfig]] = (
    ('nav2_controller', 'controller_server', 'controller_server'),
    ('nav2_planner', 'planner_server', 'planner_server'),
    ('nav2_behaviors', 'behavior_server', 'behavior_server'),
    ('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
    ('nav2_smoother', 'smoother_server', 'smoother_server'),
    ('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother'),
    ('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower'),
    ('nav2_lifecycle_manager', 'lifecycle_manager', 'lifecycle_manager_navigation'),
)


def create_nav2_node(
    package: str,
    executable: str,
    name: str,
    params_file: LaunchConfiguration,
    use_sim_time: LaunchConfiguration,
) -> Node:
    """
    Create a Nav2 node with standard configuration.
    
    Args:
        package: ROS2 package name
        executable: Executable name within package
        name: Node name
        params_file: Path to parameters YAML file
        use_sim_time: Whether to use simulation time
        
    Returns:
        Configured Node action
    """
    return Node(
        package=package,
        executable=executable,
        name=name,
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
    )


def build_nav2_nodes(
    params_file: LaunchConfiguration,
    use_sim_time: LaunchConfiguration,
) -> list[Node]:
    """
    Build list of all Nav2 nodes for navigation stack.
    
    Args:
        params_file: Path to parameters YAML file
        use_sim_time: Whether to use simulation time
        
    Returns:
        List of configured Node actions
    """
    return [
        create_nav2_node(pkg, exe, name, params_file, use_sim_time)
        for pkg, exe, name in NAV2_NODES
    ]


def get_default_params_path() -> str:
    """
    Get default path to Nav2 parameters file.
    
    Returns:
        Absolute path to leo_nav2_lidar_params.yaml
    """
    return os.path.join(
        os.path.expanduser('~'),
        'src/RoboTerrain/ros2_ws/src/roverrobotics_ros2/roverrobotics_gazebo/config',
        'leo_nav2_lidar_params.yaml',
    )


def generate_launch_description() -> LaunchDescription:
    """
    Generate launch description for Nav2 LiDAR navigation.
    
    Returns:
        LaunchDescription with all Nav2 nodes configured
    """
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    launch_args: list = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=get_default_params_path(),
            description='Full path to the Nav2 params file',
        ),
    ]

    nav2_nodes = build_nav2_nodes(params_file, use_sim_time)

    return LaunchDescription(launch_args + nav2_nodes)
