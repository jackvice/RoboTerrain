cmake_minimum_required(VERSION 3.5)
project(leo_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    models
    urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/gz_sim_resource_path.sh")
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/gz_sim_resource_path.dsv")

ament_package()

install(
  DIRECTORY sdf
  DESTINATION share/${PROJECT_NAME}
)
