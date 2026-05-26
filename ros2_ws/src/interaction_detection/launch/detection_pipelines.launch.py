import os
from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch_ros.actions import Node 
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


realsense_dir = get_package_share_directory('realsense2_camera')

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='interaction_detection',
            executable='interaction_detection',
            name='interaction_detection'),
        Node(
            package='multi_person_tracker',
            executable='multi_person_tracker',
            name='multi_person_tracker'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
            os.path.join(realsense_dir,'launch', 'rs_ours_launch.py')),)])