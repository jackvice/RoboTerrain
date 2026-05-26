"""Full Nav2CAN demo on a TurtleBot3 in Gazebo.

This brings up:
    * Gazebo with the TurtleBot3 waffle in the nav2_bringup stock world
    * Nav2 configured with this project's SocialLayer + InteractionLayer
      (params/nav2_params_tb3.yaml)
    * The Nav2CAN context module:
        - social_map_generator (People -> /social_map image)
        - interaction_detection (YOLOv7 -> /interaction_bb)
    * Optional sources of /people:
        - fake_people_publisher (default — hard-coded poses)
        - multi_person_tracker (YOLO-pose on Gazebo's RealSense)
    * Optional Gazebo human actor for the tracker to see
    * RViz with the standard nav2 layout

Examples:

    # Default — fake people, no detection, no Gazebo human
    ros2 launch context_aware_navigation nav2can_tb3_launch.py

    # Move the fake people around
    ros2 launch context_aware_navigation nav2can_tb3_launch.py \\
        people:="[1.0, 0.5, 3.14, 1.0, -0.5, 0.0]"

    # Visual-detection pipeline: real tracker + actor in Gazebo
    ros2 launch context_aware_navigation nav2can_tb3_launch.py \\
        use_fake_people:=False use_tracker:=True spawn_person:=True
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Hard-coded RealSense topic names used by multi_person_tracker.  We always
# remap *both* "cameras" (because the tracker defaults to n_cameras=2 even
# when we override the ROS param, the second Camera instance still subscribes
# during __init__ — the remap stops it from spamming "topic not found").
TB3_RGB_TOPIC = '/intel_realsense_r200_depth/image_raw'
TB3_DEPTH_TOPIC = '/intel_realsense_r200_depth/depth/image_raw'

# Frame TB3's RealSense actually publishes in.
TB3_CAMERA_FRAME = 'camera_rgb_frame'


def _truthy(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _nav2can_nodes(context):
    """Resolve launch args to plain Python values before starting nodes."""
    people_text = LaunchConfiguration('people').perform(context)
    social_max_cost = LaunchConfiguration('social_max_cost').perform(context)
    use_fake_people = _truthy(
        LaunchConfiguration('use_fake_people').perform(context))
    use_tracker = _truthy(
        LaunchConfiguration('use_tracker').perform(context))
    spawn_person = _truthy(
        LaunchConfiguration('spawn_person').perform(context))
    tracker_frame = LaunchConfiguration('tracker_frame').perform(context)
    person_x = LaunchConfiguration('person_x').perform(context)
    person_y = LaunchConfiguration('person_y').perform(context)
    person_yaw = LaunchConfiguration('person_yaw').perform(context)

    pkg_share = get_package_share_directory('context_aware_navigation')

    actions = []

    actions.append(Node(
        package='context_aware_navigation',
        executable='social_map_generator.py',
        name='social_map_generator',
        output='screen',
        arguments=[social_max_cost],
        parameters=[{'use_sim_time': True}]))

    actions.append(Node(
        package='interaction_detection',
        executable='interaction_detection',
        name='interaction_detection',
        output='screen',
        parameters=[{'use_sim_time': True}]))

    if use_fake_people and use_tracker:
        print('[nav2can_tb3_launch] WARNING: use_fake_people and use_tracker '
              'are both True — two publishers will fight on /people.  '
              'Disabling fake_people_publisher.')
        use_fake_people = False

    if use_fake_people:
        actions.append(Node(
            package='context_aware_navigation',
            executable='fake_people_publisher.py',
            name='fake_people_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'frame_id': 'map',
                'rate_hz': 10.0,
                'people': people_text,
            }]))

    if use_tracker:
        # The tracker expects ``camera1_color_frame``/``camera2_color_frame``
        # in TF; alias them to the frame the TB3 camera actually publishes in.
        # n_cameras=1 → tracker uses "camera_color_frame" (singular)
        # n_cameras>1 → tracker uses "camera1_color_frame", "camera2_color_frame"
        # Alias all three to the TB3 camera frame so the tracker's TF lookup
        # succeeds in either configuration.
        for ns in ('camera', 'camera1', 'camera2'):
            actions.append(Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name=f'{ns}_frame_alias',
                arguments=[
                    '0', '0', '0', '0', '0', '0',
                    tracker_frame, f'{ns}_color_frame',
                ],
                parameters=[{'use_sim_time': True}],
                output='log'))

        actions.append(Node(
            package='multi_person_tracker',
            executable='multi_person_tracker',
            name='multi_person_tracker',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                # TB3 only has one camera; keep n_cameras=1 so we get a
                # ``/camera/...`` namespace and a single set of remaps.
                'n_cameras': 1,
                'target_frame': 'map',
                'dt': 0.1,
                'keeptime': 5.0,
            }],
            remappings=[
                ('/camera/color/image_raw', TB3_RGB_TOPIC),
                ('/camera/aligned_depth_to_color/image_raw', TB3_DEPTH_TOPIC),
            ]))

    if spawn_person:
        # The default world (turtlebot3_world_with_actor.model) already
        # includes a Gazebo <actor> at (1.5, 0).  Only fall back to
        # spawn_entity + the primitive humanoid SDF when an alternate world
        # is used.
        sdf_path = os.path.join(pkg_share, 'world', 'person_standing.sdf')
        actions.append(ExecuteProcess(
            cmd=[
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', 'nav2can_person1',
                '-file', sdf_path,
                '-x', person_x, '-y', person_y, '-z', '0.0',
                '-Y', person_yaw,
            ],
            output='screen'))

    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory('context_aware_navigation')

    params_file = LaunchConfiguration('params_file')
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    world = LaunchConfiguration('world')

    declare_params = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'params', 'nav2_params_tb3.yaml'),
        description='Nav2 parameters file (defaults to the TB3 + Nav2CAN preset)')
    declare_headless = DeclareLaunchArgument(
        'headless', default_value='False',
        description='Set True to skip gzclient (server-only)')
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='True',
        description='Start RViz alongside the sim')
    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            pkg_share, 'world', 'turtlebot3_world_with_actor.model'),
        description='Gazebo .world file.  Default loads the standard TB3 '
                    'world plus a Gazebo <actor> at (1.5, 0) — a real walking '
                    'human mesh, not a primitive humanoid.  Set to '
                    '/opt/ros/humble/share/nav2_bringup/worlds/waffle.model '
                    'for the stock empty world.')
    declare_people = DeclareLaunchArgument(
        'people',
        # Wider spacing than the paper demo so the TB3 can detour in sim (≥2 m
        # between people reduces overlapping 253-cost zones).
        default_value='[2.0, 1.2, 3.14, 2.0, -1.2, 0.0]',
        description='Flat list of fake people poses (x,y,theta repeated). '
                    'Pass "[]" to publish no people.')
    declare_social_max_cost = DeclareLaunchArgument(
        'social_max_cost',
        default_value='240',
        description='Peak social-map cost (253 is impassable in Nav2). '
                    'Use 253 for paper-faithful behaviour.')
    declare_use_fake_people = DeclareLaunchArgument(
        'use_fake_people', default_value='True',
        description='Publish hard-coded /people poses.  Set False when testing '
                    'real visual detection with use_tracker:=True.')
    declare_use_tracker = DeclareLaunchArgument(
        'use_tracker', default_value='False',
        description='Start multi_person_tracker (YOLO-pose) on Gazebo\'s '
                    'RealSense topics so /people comes from real detections.')
    declare_spawn_person = DeclareLaunchArgument(
        'spawn_person', default_value='False',
        description='Spawn a Gazebo actor for the tracker to see.')
    declare_tracker_frame = DeclareLaunchArgument(
        'tracker_frame', default_value=TB3_CAMERA_FRAME,
        description='TF frame the tracker should treat as the camera origin '
                    '(aliased to camera1/2_color_frame).')
    declare_person_x = DeclareLaunchArgument(
        'person_x', default_value='1.5',
        description='Gazebo spawn X of the actor (map frame).')
    declare_person_y = DeclareLaunchArgument(
        'person_y', default_value='0.0',
        description='Gazebo spawn Y of the actor (map frame).')
    declare_person_yaw = DeclareLaunchArgument(
        'person_yaw', default_value='3.14',
        description='Gazebo spawn yaw of the actor (radians; facing -x by '
                    'default so they look toward the TB3 spawn at x=-2).')

    tb3_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'tb3_simulation_launch.py')),
        launch_arguments={
            'params_file': params_file,
            'headless': headless,
            'use_rviz': use_rviz,
            'world': world,
        }.items())

    return LaunchDescription([
        declare_params,
        declare_headless,
        declare_use_rviz,
        declare_world,
        declare_people,
        declare_social_max_cost,
        declare_use_fake_people,
        declare_use_tracker,
        declare_spawn_person,
        declare_tracker_frame,
        declare_person_x,
        declare_person_y,
        declare_person_yaw,
        tb3_sim,
        OpaqueFunction(function=_nav2can_nodes),
    ])
