"""Spawn the Nav2CAN-flavoured Leo Rover (RGBD + 360 LiDAR + IMU) and bridge
its sensors into ROS 2.  Composed by ``leo_nav2can_launch.py``; can also be
launched standalone for a sensors-only smoke test.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Bridges: '@' = bidirectional, '[' = gz->ros, ']' = ros->gz.
#
# `/tf` is intentionally NOT bridged: gz-sim's DiffDrive plugin publishes
# `odom -> base_footprint` to /tf, and pose_topic/ign_ros2_Nav2_topics.py
# publishes its own ground-truth `odom -> base_footprint`.  Two publishers
# on the same transform race and produce extrapolation errors that break
# costmap and tracker TF lookups.  Let pose_topic own that edge.
STATIC_BRIDGES = (
    '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
    '/odometry/wheels@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
    '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
    '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
    '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
    '/depth_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
    '/depth_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
    '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
    '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
)

# Per-world actors (Gazebo entity names → /<actor>/pose topics).
ACTORS_BY_WORLD = {
    'inspect': ('linear_actor', 'diag_actor', 'triangle_actor'),
    'moon':    ('triangle_actor', 'triangle2_actor', 'triangle3_actor'),
    'island':  ('triangle_actor', 'triangle2_actor', 'triangle3_actor'),
    'default': ('upper_actor', 'lower_actor'),
}


def _world_bridges(world_name: str) -> tuple[str, ...]:
    actors = ACTORS_BY_WORLD.get(world_name, ())
    return (
        f'/world/{world_name}/dynamic_pose/info@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
        f'/world/{world_name}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        *(f'/{a}/pose@geometry_msgs/msg/Pose[gz.msgs.Pose' for a in actors),
    )


def _resource_path(pkg_source: str) -> str:
    return ':'.join((
        os.path.join(os.path.expanduser('~'),
                     'worlds/gazebo_models_worlds_collection/models/cpr_office_construction'),
        pkg_source,
        os.path.dirname(pkg_source),
    ))


def _static_tf(name: str, parent: str, child: str,
               x: float = 0.0, y: float = 0.0, z: float = 0.0,
               roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> Node:
    """Spawn a static_transform_publisher.  All numerics in metres / radians."""
    return Node(
        package='tf2_ros', executable='static_transform_publisher', name=name,
        arguments=[
            '--x', str(x), '--y', str(y), '--z', str(z),
            '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
            '--frame-id', parent, '--child-frame-id', child,
        ],
        parameters=[{'use_sim_time': True}], output='log',
    )


def _bridge_node(context):
    world_name = LaunchConfiguration('world_name').perform(context)
    return [Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[*STATIC_BRIDGES, *_world_bridges(world_name)],
        output='screen',
    )]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('roverrobotics_gazebo')
    pkg_source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_share)))),
        'src/roverrobotics_ros2/roverrobotics_gazebo')
    leo_sdf = os.path.join(get_package_share_directory('leo_description'),
                           'sdf', 'leo_nav2can.sdf')

    world = LaunchConfiguration('world')
    headless = LaunchConfiguration('headless')
    world_path = PathJoinSubstitution([
        FindPackageShare('roverrobotics_gazebo'), 'worlds', world])

    declarations = (
        DeclareLaunchArgument('world', default_value='inspection_boxes_v4.world',
                              description='World file under roverrobotics_gazebo/worlds'),
        DeclareLaunchArgument('world_name', default_value='inspect',
                              description=f'Gazebo world name; selects per-world bridges. '
                                          f'Known: {", ".join(ACTORS_BY_WORLD)}'),
        DeclareLaunchArgument('headless', default_value='false'),
    )

    env = (
        SetEnvironmentVariable('PATH', os.environ['PATH']),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', _resource_path(pkg_source)),
        SetEnvironmentVariable('IGN_GAZEBO_MODEL_PATH', pkg_source),
    )

    sim = (
        ExecuteProcess(cmd=['ign', 'gazebo', world_path], output='screen',
                       condition=UnlessCondition(headless)),
        ExecuteProcess(cmd=['ign', 'gazebo', '-s', '-r', world_path], output='screen',
                       condition=IfCondition(headless)),
    )

    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-file', leo_sdf, '-name', 'leo_rover',
                   '-allow_renaming', 'true', '-x', '0', '-y', '0', '-z', '1.0'],
        output='screen',
    )

    # The Leo SDF declares the link/joint/sensor frames as <frame>s, but
    # gz-sim only auto-broadcasts `odom -> base_footprint` (DiffDrive) -- the
    # rest of the kinematic chain never reaches ROS TF without either a
    # robot_state_publisher or explicit static_transform_publishers.  We pick
    # the latter.  Numbers come straight from leo_nav2can.sdf:
    #   base_joint:         <pose>0 0 0.19783 0 -0 0</pose>          (base_footprint -> base_link)
    #   lidar_mount:        <pose>0.174 0 0.6 0 -0 3.1415</pose>     (base_link -> lidar_link)
    #   camera_joint:       <pose>0.0971 0 -0.0427 0 0.2094 0</pose> (base_link -> camera_frame)
    #   camera_optical_jt:  <pose>0 0 0 -1.5708 -0 -1.5708</pose>    (camera_frame -> camera_optical_frame)
    static_tfs = (
        _static_tf('tf_base_link', 'base_footprint', 'base_link',
                   z=0.19783),
        _static_tf('tf_lidar_link', 'base_link', 'lidar_link',
                   x=0.174, z=0.6, yaw=3.1415),
        _static_tf('tf_camera_frame', 'base_link', 'camera_frame',
                   x=0.0971, z=-0.0427, pitch=0.2094),
        _static_tf('tf_camera_optical_frame', 'camera_frame', 'camera_optical_frame',
                   roll=-1.5708, yaw=-1.5708),
        # multi_person_tracker (n_cameras=1) expects 'camera_color_frame' /
        # 'camera_depth_frame'; our rgbd_camera publishes everything in
        # camera_optical_frame.  Alias both.
        _static_tf('tf_camera_color_alias', 'camera_optical_frame', 'camera_color_frame'),
        _static_tf('tf_camera_depth_alias', 'camera_optical_frame', 'camera_depth_frame'),
        # Some Nav2 obstacle-layer configs reference 'base_scan'; alias to lidar_link.
        _static_tf('tf_base_scan_alias', 'lidar_link', 'base_scan'),
        # Nav2CAN's SocialLayer / InteractionLayer / detectContextNode hard-code
        # 'map' as the fixed frame.  We run mapless (global_frame: odom), so
        # publish an identity odom -> map alias.  Ground-truth odom == map here.
        _static_tf('tf_map_alias', 'odom', 'map'),
    )

    return LaunchDescription([
        *declarations, *env, *sim, spawn,
        OpaqueFunction(function=_bridge_node),
        *static_tfs,
    ])
