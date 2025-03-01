cmake_minimum_required(VERSION 3.8)
project(leo_gz_plugins)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

add_library(leo_gz_differential_plugin SHARED
src/differential_system.cpp)

if(DEFINED ENV{GZ_VERSION} AND "$ENV{GZ_VERSION}" STREQUAL "garden")
  find_package(gz-plugin2 REQUIRED COMPONENTS register)
  set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})

  find_package(gz-sim7 REQUIRED)
  set(GZ_SIM_VER ${gz-sim7_VERSION_MAJOR})

  target_link_libraries(leo_gz_differential_plugin
    gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
    gz-plugin${GZ_PLUGIN_VER}::gz-plugin${GZ_PLUGIN_VER}
  )
elseif(DEFINED ENV{GZ_VERSION} AND "$ENV{GZ_VERSION}" STREQUAL "harmonic")
  find_package(gz-plugin2 REQUIRED COMPONENTS register)
  set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})

  find_package(gz-sim8 REQUIRED)
  set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})

  target_link_libraries(leo_gz_differential_plugin
    gz-sim${GZ_SIM_VER}::gz-sim${GZ_SIM_VER}
    gz-plugin${GZ_PLUGIN_VER}::gz-plugin${GZ_PLUGIN_VER}
  )

  message(STATUS "Compiling against Gazebo Harmonic")
else()
  #If GZ_VERSION is not set default to Fortress
  find_package(ignition-plugin1 REQUIRED COMPONENTS register)
  set(IGN_PLUGIN_VER ${ignition-plugin1_VERSION_MAJOR})

  find_package(ignition-gazebo6 REQUIRED)
  set(IGN_GAZEBO_VER ${ignition-gazebo6_VERSION_MAJOR})

  add_definitions(-DUSE_IGNITION)

  target_link_libraries(leo_gz_differential_plugin
    ignition-gazebo${IGN_GAZEBO_VER}::ignition-gazebo${IGN_GAZEBO_VER}
    ignition-plugin${IGN_PLUGIN_VER}::ignition-plugin${IGN_PLUGIN_VER}
  )
endif()

set_property(TARGET leo_gz_differential_plugin PROPERTY CXX_STANDARD 17)

install(
  TARGETS
    leo_gz_differential_plugin
  DESTINATION lib
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/gz_sim_system_plugin_path.sh")
ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/hooks/gz_sim_system_plugin_path.dsv")

ament_package()
