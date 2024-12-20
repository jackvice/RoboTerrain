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
        #default_value='simplecave3.sdf,'
        #default_value='maze_and_person.sdf',
        #default_value='upb12.sdf',
        #default_value='office_cpr_construction.world',
        #default_value='maze_simple.sdf',
        #default_value='inspection.world',
        #default_value='island.sdf',
        default_value='inspection_simple.world',
        #default_value='simple_40m2.world',
        
        description='World file to use in Gazebo')
    
    # Construct the world path using substitutions
    world_path = PathJoinSubstitution([
        FindPackageShare('roverrobotics_gazebo'),
        'worlds',
        world
    ])

    # Ensure the 'ign' executable is in the PATH
    ignition_env = SetEnvironmentVariable(
        name='PATH',
        value=os.environ['PATH']
    )

    env = {
        'IGN_GAZEBO_RESOURCE_PATH': os.path.join(
            os.path.expanduser('~'),
            'worlds/gazebo_models_worlds_collection/models/cpr_office_construction'
        )
    }


    
    # Launch Gazebo in server-only mode (no GUI)
    gz_sim = ExecuteProcess(
        # -s for server/headless, -r for start running(play), -v verbose
        #cmd=['ign', 'gazebo', '-s', '-r', '-v 4', world_path], 
        #cmd=['ign', 'gazebo', world_path, 'additional_env=env'],
        cmd=['ign', 'gazebo', world_path],
        output='screen'
    )
    
    # Spawn Rover Robot
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-file', os.path.join(
                get_package_share_directory('roverrobotics_description'),
                'urdf', 'camera_rover_4wd.sdf'),
            '-name', 'rover_zero4wd',
            '-allow_renaming', 'true',
            '-x', '0',
            '-y', '0',
            '-z', '1.0',#'0.1',
        ],
        output='screen'
    )
    
    # Bridge between ROS 2 and Ignition Gazebo
    gz_ros2_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/odometry/wheels@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            #'/world/maze/reset@std_srvs/srv/Empty',
            #'/world/maze/reset@std_srvs/srv/Empty@ignition.msgs.Empty@ignition.msgs.Empty',
            #'/world/maze/control@ros_gz_interfaces/srv/ControlWorld@ignition.msgs.WorldControl@ignition.msgs.Boolean',
            #'/world/maze/dynamic_pose/info@ignition.msgs.Pose_V@ros2_geometry_msgs/PoseArray'
            '/world/default/dynamic_pose/info@ignition.msgs.Pose_V@geometry_msgs/msg/PoseArray'
        ],
        output='screen'
    )
    
    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Declare the launch options
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    
    # Set environment variables
    ld.add_action(ignition_env)
    
    # Launch Gazebo in server-only mode
    ld.add_action(gz_sim)
    ld.add_action(gz_spawn_entity)
    ld.add_action(gz_ros2_bridge)
    
    return ld
