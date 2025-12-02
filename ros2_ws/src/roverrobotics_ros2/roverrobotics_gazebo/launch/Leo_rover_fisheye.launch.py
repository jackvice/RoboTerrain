import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    world = LaunchConfiguration('world')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    declare_world_cmd = DeclareLaunchArgument(
        'world',

        #default_value='inspection_boxes_x10.world',
        default_value='inspection_boxes_x10_v2.world', # social nave testing world for MDPI publication
        #default_value='inspection_boxes_v3.world', # lots of boxes for Active vision
        #default_value='office_cpr_construction.world',
        #default_value='island.sdf',

        #default_value='inspection_simple.world',
        #default_value='rubicon.sdf',
        #default_value='simplecave3.sdf',
        #default_value='maze_simple.sdf',
        #default_value='maze_pillars.sdf',
        #default_value='inspection.world',
        #default_value='simple_40m2.world',
        #default_value='maze_clean.sdf',
        #default_value='maze_empty.sdf',
        description='World file to use in Gazebo')
    
    # Construct the world path using substitutions
    world_path = PathJoinSubstitution([
        FindPackageShare('roverrobotics_gazebo'),
        'worlds',
        world
    ])

    # Get both source and install directories
    pkg_share = get_package_share_directory('roverrobotics_gazebo')
    pkg_source = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkg_share)))),  # go up to workspace
        'src/roverrobotics_ros2/roverrobotics_gazebo'  # go down to package
    )
    
    # Create combined resource path
    resource_path = ':'.join([
        os.path.join(os.path.expanduser('~'), 'worlds/gazebo_models_worlds_collection/models/cpr_office_construction'),
        pkg_source,                   # Source directory
        os.path.dirname(pkg_source)   # Parent of source directory
    ])

    print(f"Package share directory: {pkg_share}")
    print(f"Package source directory: {pkg_source}")
    print(f"Resource path: {resource_path}")
    
    # Set up environment variables
    resource_env = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=resource_path
    )

    model_env = SetEnvironmentVariable(
        name='IGN_GAZEBO_MODEL_PATH',
        value=f"{pkg_source}"
    )

    
    # Set up all environment variables
    path_env = SetEnvironmentVariable(
        name='PATH',
        value=os.environ['PATH']
    )
    
    # Launch Gazebo
    gz_sim = ExecuteProcess(
        cmd=['ign', 'gazebo', world_path],
        output='screen'
    )
    
    # Spawn Rover Robot
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', os.path.join(
                get_package_share_directory('leo_description'),
                'sdf', 'leo_sim_fisheye.sdf'),
            '-name', 'leo_rover',
            '-allow_renaming', 'true',
            '-x', '0',
            '-y', '0',
            '-z', '1.0',
        ],
        output='screen'
    )


    # Bridge between ROS 2 and Ignition Gazebo
    gz_ros2_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Fix bi-directional topics (use '@' instead of mixed symbols)
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',  # This one-way is correct
            '/odometry/wheels@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',     # This one-way is correct
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',  # This one-way is correct
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            # Fix this direction (it was reversed)
            '/world/default/dynamic_pose/info@geometry_msgs/msg/PoseArray[ignition.msgs.Pose_V',
            # Remove the spawn service bridge as it's causing issues
            #'/world/default/create@ros_gz_interfaces/srv/SpawnEntity@ignition.msgs.EntityFactory',
        ],
        output='screen'
    )
    
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Add all actions to the launch description
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    
    # Add environment variables
    ld.add_action(path_env)
    ld.add_action(resource_env)
    ld.add_action(model_env)
    
    # Add nodes and processes
    ld.add_action(gz_sim)
    ld.add_action(gz_spawn_entity)
    ld.add_action(gz_ros2_bridge)
    
    return ld

