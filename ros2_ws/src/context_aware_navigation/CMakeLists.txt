cmake_minimum_required(VERSION 3.8)
project(context_aware_navigation)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav2_common REQUIRED)
find_package(navigation2 REQUIRED)
find_package(multi_person_tracker_interfaces)
find_package(sensor_msgs)
find_package(geometry_msgs)
find_package(nav2_costmap_2d REQUIRED)
find_package(pluginlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

add_compile_options(-g)
set(dep_pkgs
    rclcpp
    nav2_costmap_2d
    OpenCV
    cv_bridge
    multi_person_tracker_interfaces
    geometry_msgs
    pluginlib
    tf2
    tf2_geometry_msgs
    tf2_ros
    tf2_sensor_msgs)



add_library(
  ${PROJECT_NAME} SHARED
            src/social_layer.cpp
            src/interaction_layer.cpp)

include_directories(include)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)

# Install python modules
ament_python_install_package(${PROJECT_NAME})

# Install python executables
install(PROGRAMS
  scripts/social_map_generator.py
  scripts/stage_joints.py
  scripts/odom_tf_publisher.py
  scripts/people_publisher.py
  scripts/fake_people_publisher.py
  scripts/measureTraveledDistance.py
  scripts/distToPeople.py
  scripts/mapRepublisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  world
  params
  config
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(nav2_costmap_2d layers.xml)
ament_target_dependencies(${PROJECT_NAME} ${dep_pkgs})

ament_package()
