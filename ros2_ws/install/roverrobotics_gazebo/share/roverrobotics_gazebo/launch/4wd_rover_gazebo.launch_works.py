import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue



def generate_launch_description():
    # Create the launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    #urdf = os.path.join(get_package_share_directory(
    #    'roverrobotics_description'), 'urdf', 'rover_4wd.urdf')
    world = LaunchConfiguration('world')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        #default_value='maze.sdf',
        #default_value='rubicon.sdf',
        default_value='simplecave3.sdf',
        #default_value='fortress.sdf',
        #default_value='island.sdf',
        #default_value='maze_and_person.sdf',
        description='World file to use in Gazebo')
    
    gz_world_arg = PathJoinSubstitution([
        get_package_share_directory('roverrobotics_gazebo'), 'worlds', world])

    # Include the gz sim launch file  
    gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_sim_share, "launch", "gz_sim.launch.py")),
        launch_arguments={
            "gz_args" : gz_world_arg,

        }.items()
    )
    
    # Spawn Rover Robot
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", os.path.join(get_package_share_directory('roverrobotics_description'), 'urdf', 'camera_rover_4wd.sdf'), #jmv
            "-name", "rover_zero4wd",
            "-allow_renaming", "true",
            "-x", "0",  # Desired X spawn
            "-y", "0",  # Desired Y spawn
            "-z", "0.1", # Desired Z spawn
        ]
    )
    
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/odometry/wheels@nav_msgs/msg/Odometry@ignition.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',  # Added camera image
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'  # Added camera info

        ],
    )

    # Robot state publisher
    params = {'use_sim_time': use_sim_time} # jmv

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)

    # Launch Gazebo
    ld.add_action(gz_sim)
    ld.add_action(gz_spawn_entity)
    ld.add_action(gz_ros2_bridge)


    return ld
