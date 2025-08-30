cmake_minimum_required(VERSION 3.10)
project(actor_pose_publisher)

find_package(ignition-cmake2 REQUIRED)
ign_configure_project()

# Find required packages
ign_find_package(ignition-gazebo6 REQUIRED)
ign_find_package(ignition-transport11 REQUIRED)
ign_find_package(ignition-msgs8 REQUIRED)
ign_find_package(ignition-plugin1 REQUIRED COMPONENTS register)

add_library(ActorPosePublisher SHARED src/actor_pos_publisher.cc)

target_link_libraries(ActorPosePublisher
  ignition-gazebo6::ignition-gazebo6
  ignition-transport11::ignition-transport11
  ignition-msgs8::ignition-msgs8
  ignition-plugin1::ignition-plugin1  # Add this back
  ignition-plugin1::register          # Keep this too
)
