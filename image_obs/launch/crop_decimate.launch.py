from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

    container = ComposableNodeContainer(
        name='image_proc_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen'
    )

    crop_decimate_loader = LoadComposableNodes(
        target_container='image_proc_container',
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::CropDecimateNode',
                name='crop_decimate',
                remappings=[
                    # Remap default "in" topics to your camera feed
                    ('in/image_raw', '/leo1/camera/image_raw'),
                    ('in/camera_info', '/leo1/camera/camera_info'),

                    # Remap output to a nicer name
                    ('out/image_raw', '/leo1/camera/image_raw_decimate'),
                    ('out/camera_info', '/leo1/camera/camera_info_decimate'),
                ],
                parameters=[
                    # Adjust these so that (original_width / decimation_x)
                    # and (original_height / decimation_y) is close to 96x96
                    {'decimation_x': 6},
                    {'decimation_y': 5},
                ],
            )
        ]
    )

    return LaunchDescription([
        container,
        crop_decimate_loader
    ])
