"""End-to-end Nav2CAN demo on the Leo Rover in a DUnE world.

Composes:
  * leo_nav2can_sim.launch.py   - Gazebo Fortress + Leo (RGBD + LiDAR + IMU)
  * pose_topic/ign_ros2_Nav2_topics.py - publishes /odom_ground_truth + TF
  * Nav2 stack pointed at nav2_params_leo.yaml (mapless, SocialLayer + InteractionLayer)
  * multi_person_tracker (YOLOv8-pose) on the RGBD topics
  * social_map_generator + interaction_detection (Nav2CAN context)
  * (optional) fake_people_publisher for ground-truth smoke tests

Examples:

    # YOLOv8-pose-driven /people, inspect world
    ros2 launch context_aware_navigation leo_nav2can_launch.py

    # Headless training/eval run
    ros2 launch context_aware_navigation leo_nav2can_launch.py headless:=true

    # Fake-people smoke test (skip the tracker)
    ros2 launch context_aware_navigation leo_nav2can_launch.py \\
        use_tracker:=false use_fake_people:=true \\
        people:='[2.0, 1.0, 0.0, -1.5, -1.5, 1.57]'

    # Tighten / loosen social cost (default 240; 253 = impassable)
    ros2 launch context_aware_navigation leo_nav2can_launch.py social_max_cost:=220
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Nav2 lifecycle nodes parameterised by nav2_params_leo.yaml.
NAV2_NODES = (
    ('nav2_controller', 'controller_server', 'controller_server'),
    ('nav2_planner', 'planner_server', 'planner_server'),
    ('nav2_behaviors', 'behavior_server', 'behavior_server'),
    ('nav2_bt_navigator', 'bt_navigator', 'bt_navigator'),
    ('nav2_smoother', 'smoother_server', 'smoother_server'),
    ('nav2_velocity_smoother', 'velocity_smoother', 'velocity_smoother'),
    ('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower'),
    ('nav2_lifecycle_manager', 'lifecycle_manager', 'lifecycle_manager_navigation'),
)


def _nav2_node(pkg: str, exe: str, name: str, params_file) -> Node:
    return Node(package=pkg, executable=exe, name=name, output='screen',
                parameters=[params_file, {'use_sim_time': True}])


def _truthy(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _nav2can_nodes(context):
    """Resolve runtime LaunchConfigurations to nodes (single OpaqueFunction)."""
    use_tracker = _truthy(LaunchConfiguration('use_tracker').perform(context))
    use_fake = _truthy(LaunchConfiguration('use_fake_people').perform(context))
    use_interaction = _truthy(
        LaunchConfiguration('use_interaction').perform(context))
    people = LaunchConfiguration('people').perform(context)
    target_frame = LaunchConfiguration('target_frame').perform(context)
    social_max_cost = LaunchConfiguration('social_max_cost').perform(context)

    if use_fake and use_tracker:
        print('[leo_nav2can_launch] use_fake_people and use_tracker both true; '
              'disabling fake_people_publisher to avoid double-publishing /people.')
        use_fake = False

    nodes = [
        # social_map_generator.py reads social_max_cost as a positional CLI
        # arg (`int(sys.argv[1])`), so it MUST be passed via `arguments=`,
        # not `parameters=`.  Without it the node crashes on startup with
        # IndexError and /social_map is never published.
        Node(package='context_aware_navigation',
             executable='social_map_generator.py',
             name='social_map_generator',
             output='screen',
             arguments=[social_max_cost],
             parameters=[{'use_sim_time': True}]),
    ]

    # interaction_detection (YOLOv7-context) is the second half of Nav2CAN's
    # perception stack and feeds the InteractionLayer in global_costmap.  It
    # only matters when scenes contain *group* interactions (pairs talking,
    # queues, etc.).  For independent-pedestrian benchmarks (the default in
    # this repo's dynamic_obstacles scenarios) it produces no useful signal,
    # adds a heavy YOLOv7 model, and is a frequent crash source — so it's
    # opt-in.  Pass `use_interaction:=true` to enable for paper-faithful runs.
    if use_interaction:
        nodes.append(Node(
            package='interaction_detection',
            executable='interaction_detection',
            name='interaction_detection',
            output='screen',
            parameters=[{'use_sim_time': True}]))

    if use_fake:
        nodes.append(Node(
            package='context_aware_navigation',
            executable='fake_people_publisher.py',
            name='fake_people_publisher',
            output='screen',
            parameters=[{'use_sim_time': True, 'frame_id': target_frame,
                         'rate_hz': 10.0, 'people': people}]))

    if use_tracker:
        nodes.append(Node(
            package='multi_person_tracker',
            executable='multi_person_tracker',
            name='multi_person_tracker',
            output='screen',
            parameters=[{'use_sim_time': True, 'n_cameras': 1,
                         'target_frame': target_frame, 'dt': 0.1, 'keeptime': 5.0}],
            remappings=[
                ('/camera/color/image_raw', '/depth_camera/image'),
                ('/camera/aligned_depth_to_color/image_raw', '/depth_camera/depth_image'),
            ],
            additional_env={'POSE_BACKEND': 'ultralytics',
                            'POSE_MODEL': 'yolov8n-pose.pt',
                            # Default 0.3 fires on Gazebo scenery (railings,
                            # posts) at camera elevation, producing phantom
                            # tracklets at z~6m that the keeptime-based pruner
                            # can't clear because the false positive keeps
                            # refreshing the tracklet timestamp.  0.5 keeps
                            # YOLOv8-pose recall on actor meshes while
                            # rejecting most static-scenery hallucinations.
                            'POSE_THRESHOLD': '0.5'}))

    return nodes


def generate_launch_description() -> LaunchDescription:
    pkg_can = get_package_share_directory('context_aware_navigation')
    pkg_rover = get_package_share_directory('roverrobotics_gazebo')
    ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_rover))))

    declarations = (
        DeclareLaunchArgument('world', default_value='inspection_boxes_v4.world',
                              description='World file under roverrobotics_gazebo/worlds'),
        DeclareLaunchArgument('world_name', default_value='inspect',
                              description='Gazebo world name (used by /world/<name>/* and the pose converter)'),
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_tracker', default_value='true',
                              description='Run multi_person_tracker on the RGBD stream'),
        DeclareLaunchArgument('use_fake_people', default_value='false',
                              description='Publish hard-coded /people poses (mutually exclusive with use_tracker)'),
        DeclareLaunchArgument('use_interaction', default_value='false',
                              description='Run interaction_detection (YOLOv7 group context) and '
                                          'feed InteractionLayer.  Off by default — only useful '
                                          'when scenes contain multi-person social interactions.'),
        DeclareLaunchArgument('people', default_value='[2.0, 1.0, 0.0, -1.5, -1.5, 1.57]',
                              description='Flat [x,y,theta,...] list for fake_people_publisher'),
        DeclareLaunchArgument('target_frame', default_value='odom',
                              description='TF frame /people poses are expressed in'),
        DeclareLaunchArgument('social_max_cost', default_value='220',
                              description='Peak SocialLayer cost.  220 is below INSCRIBED '
                                          '(253) so a pedestrian on the robot will never by '
                                          'itself trip "Starting point in lethal space", but '
                                          'high enough that Smac2D (cost_travel_multiplier=3.5) '
                                          'will detour ~1 m around a pedestrian rather than '
                                          'brush past — keeps min_dist > 0.5 m for the '
                                          'comparison-vs-AVSN proxemic metric.  Lower to 180 '
                                          'if blobs seal off corridors; raise to 253 only for '
                                          'paper-faithful Nav2CAN behaviour.'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(pkg_can, 'params', 'nav2_params_leo.yaml'),
                              description='Nav2 params (Leo + Nav2CAN plugins)'),
    )

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_rover, 'launch', 'leo_nav2can_sim.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world'),
                          'world_name': LaunchConfiguration('world_name'),
                          'headless': LaunchConfiguration('headless')}.items())

    pose_converter = ExecuteProcess(
        cmd=['python3', os.path.join(ws_root, 'src', 'pose_topic', 'ign_ros2_Nav2_topics.py'),
             LaunchConfiguration('world_name'), 'leo_rover'],
        output='screen')

    nav2 = [_nav2_node(pkg, exe, name, LaunchConfiguration('params_file'))
            for pkg, exe, name in NAV2_NODES]

    return LaunchDescription([*declarations, sim, pose_converter, *nav2,
                              OpaqueFunction(function=_nav2can_nodes)])
