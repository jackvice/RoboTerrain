#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, TextSubstitution, PythonExpression
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration, ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter


def generate_launch_description():

    bringup_dir = get_package_share_directory('context_aware_navigation')
    launch_dir = os.path.join(bringup_dir, 'launch')

    # Create the launch configuration variables
    slam = LaunchConfiguration('slam')
    namespace = LaunchConfiguration('namespace')
    use_namespace = LaunchConfiguration('use_namespace')
    map_yaml_file = LaunchConfiguration('map')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    autostart = LaunchConfiguration('autostart')
    use_composition = LaunchConfiguration('use_composition')
    use_respawn = LaunchConfiguration('use_respawn')

    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Top-level namespace')

    declare_use_namespace_cmd = DeclareLaunchArgument(
        'use_namespace',
        default_value='False',
        description='Whether to apply a namespace to the navigation stack')

    declare_slam_cmd = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Whether run a SLAM')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(
            bringup_dir, 'world', 'maps', 'comparison.yaml'),
        description='Full path to map file to load')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')

    declare_params_file_cmd = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            bringup_dir, 'params', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='True',
        description='Automatically startup the nav2 stack')

    declare_use_composition_cmd = DeclareLaunchArgument(
        'use_composition', default_value='False',
        description='Whether to use composed bringup')

    declare_use_respawn_cmd = DeclareLaunchArgument(
        'use_respawn', default_value='False',
        description='Whether to respawn if a node crashes. Applied when composition is disabled.')

    urdf_file_name = 'world/urdf/pr2.urdf'
    urdf = os.path.join(bringup_dir,
                        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    bringup_cmd = IncludeLaunchDescription(

        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'bringup_launch.py')),
        launch_arguments={'namespace': namespace,
                          'use_namespace': use_namespace,
                          'slam': slam,
                          'map': map_yaml_file,
                          'use_sim_time': use_sim_time,
                          'params_file': params_file,
                          'autostart': autostart,
                          'use_composition': use_composition,
                          'use_respawn': use_respawn}.items())

    return LaunchDescription([
        SetParameter(name='output', value="screen"),
        declare_map_yaml_cmd,
        declare_namespace_cmd,
        declare_use_namespace_cmd,
        declare_slam_cmd,
        declare_use_sim_time_cmd,
        declare_params_file_cmd,
        declare_autostart_cmd,
        declare_use_composition_cmd,
        declare_use_respawn_cmd,
        bringup_cmd,
        SetParameter(
            name='use_sim_time', value=True),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[urdf]),
        Node(
            name='joint_state_publisher',
            package='joint_state_publisher',
            executable='joint_state_publisher',
            parameters=[
                {'source_list': ["/stage_joint_states"], 'use_sim_time': use_sim_time}],
        ),
        Node(
            name='stage_joints',
            package='context_aware_navigation',
            executable='stage_joints.py',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            name='odom_tf_publisher',
            package='context_aware_navigation',
            executable='odom_tf_publisher.py',
            parameters=[{'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[
                '-d', [os.path.join(bringup_dir, 'config', 'comparison.rviz')]]
        ),
        Node(
            package="context_aware_navigation",
            executable="social_map_generator.py",
            parameters=[{"use_sim_time": use_sim_time}],
            arguments=["253"]
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "-.055", "0", "0.285",
                       "0", "0", "camera_link", "camera2_link"]
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", ".055", "0", "-0.285",
                       "0", "0", "camera_link", "camera1_link"]
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["15.0", "8.0", "1.05", "0",
                       "0", "0", "map", "camera_link"]
        ),
        # Node(
        #         package="tf2_ros",
        #         executable="static_transform_publisher",
        #         arguments=["30.0", "2.0", "0", "0",
        #                    "0", "0", "odom", "base_footprint"]
        #     ),
        # Node(
        #         package="tf2_ros",
        #         executable="static_transform_publisher",
        #         arguments=["0", "0", "0", "0",
        #                    "0", "0", "map", "odom"]
        #     ),

    ])



