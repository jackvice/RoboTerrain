# Copyright 2023 Fictionlab sp. z o.o.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def spawn_robot(context: LaunchContext, namespace: LaunchConfiguration):
    pkg_project_description = get_package_share_directory("leo_description")
    robot_ns = context.perform_substitution(namespace)

    robot_desc = xacro.process(
        os.path.join(
            pkg_project_description,
            "urdf",
            "leo_sim.urdf.xacro",
        ),
        mappings={"robot_ns": robot_ns},
    )

    if robot_ns == "":
        robot_gazebo_name = "leo_rover"
        node_name_prefix = ""
    else:
        robot_gazebo_name = "leo_rover_" + robot_ns
        node_name_prefix = robot_ns + "_"

    # Launch robot state publisher node
    robot_state_publisher = Node(
        namespace=robot_ns,
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"use_sim_time": True},
            {"robot_description": robot_desc},
        ],
    )
    # Spawn a robot inside a simulation
    leo_rover = Node(
        namespace=robot_ns,
        package="ros_gz_sim",
        executable="create",
        name="ros_gz_sim_create",
        output="both",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            robot_gazebo_name,
            "-z",
            "1.65",
        ],
    )

    # Bridge ROS topics and Gazebo messages for establishing communication
    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name=node_name_prefix + "parameter_bridge",
        arguments=[
            robot_ns + "/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
            robot_ns + "/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
            robot_ns + "/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V",
            robot_ns + "/imu/data_raw@sensor_msgs/msg/Imu[ignition.msgs.IMU",
            robot_ns
            + "/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo",
            robot_ns + "/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model",
        ],
        parameters=[
            {
                "qos_overrides./tf_static.publisher.durability": "transient_local",
            }
        ],
        output="screen",
    )

    # Camera image bridge
    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        name=node_name_prefix + "image_bridge",
        arguments=[robot_ns + "/camera/image_raw"],
        output="screen",
    )
    return [
        robot_state_publisher,
        leo_rover,
        topic_bridge,
        image_bridge,
    ]


def generate_launch_description():
    name_argument = DeclareLaunchArgument(
        "robot_ns",
        default_value="",
        description="Robot namespace",
    )

    namespace = LaunchConfiguration("robot_ns")

    return LaunchDescription(
        [name_argument, OpaqueFunction(function=spawn_robot, args=[namespace])]
    )
