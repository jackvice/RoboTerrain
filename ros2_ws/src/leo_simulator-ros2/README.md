# Packages for simulating Leo Rover in ROS2 and Gazebo

ROS 2 version | Gazebo version | Branch | Binaries hosted at
-- | -- | -- | --
Humble | Fortress | [humble](https://github.com/LeoRover/leo_simulator-ros2/tree/humble) | https://packages.ros.org
Humble | Garden | [humble](https://github.com/LeoRover/leo_simulator-ros2/tree/humble) | only from source
Humble | Harmonic | [humble](https://github.com/LeoRover/leo_simulator-ros2/tree/humble) | only from source
Iron | Fortress | [iron](https://github.com/LeoRover/leo_simulator-ros2/tree/iron) | https://packages.ros.org
Iron | Garden | [iron](https://github.com/LeoRover/leo_simulator-ros2/tree/iron) | only from source
Iron | Harmonic | [iron](https://github.com/LeoRover/leo_simulator-ros2/tree/iron) | only from source

## Packages
* `leo_simulator` - Metapackage which provides all other packages.
* `leo_gz_bringup` - Launch files for starting simulation and adding Leo Rover inside a simulated world.
* `leo_gz_plugins` - Gazebo plugins for simulated Leo Rover.
* `leo_gz_worlds` - Custom simulation worlds.

## Install

This branch supports ROS Humble and Iron. See above for other ROS versions.

### Binaries

Humble and Iron binaries are available for Gazebo Fortress. They are hosted at https://packages.ros.org.

1. Make sure you've [installed ROS](https://docs.ros.org/en/iron/Installation.html) from binary repositories. 

1. Install `ros-<distro>-leo-simulator` package. On Ubuntu with ROS Iron:
   ```
   sudo apt install ros-iron-leo-simulator
   ```

### From source

1. Make sure you've [installed ROS](https://docs.ros.org/en/iron/Installation.html) from binary repositories. 
1. (skip this step if using Fortress) [Install Gazebo](https://gazebosim.org/docs/garden/install_ubuntu) version you'd like to use from http://packages.osrfoundation.org repositories.
1. (skip this step if using Fortress) Install [ros_gz](https://github.com/gazebosim/ros_gz) package for ROS and Gazebo version you'd like to use. For example:
   ```
   sudo apt install ros-iron-ros-gzgarden
   ```
1. (skip this step if using Fortress) Set the `GZ_VERSION` environment variable to the Gazebo version you'd like to compile against. For example:
   ```
   export GZ_VERSION=garden
   ```
1. Setup a [colcon workspace](https://docs.ros.org/en/iron/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html):
   ```
   mkdir -p ~/ws/src
   ```
1. Clone this repository into the workspace:
   ```
   cd ~/ws/src
   git clone https://github.com/LeoRover/leo_simulator-ros2 -b <distro>
   ```
1. Install dependencies using [rosdep](https://docs.ros.org/en/humble/Tutorials/Intermediate/Rosdep.html#how-do-i-use-the-rosdep-tool):
   ```
   cd ~/ws
   sudo rosdep init
   rosdep update
   rosdep install --from-paths src -y --ignore-src -r
   ```
1. Build the project and source the workspace:
   ```
   colcon build --symlink-install
   source install/setup.bash
   ```
### Run Simulation
Run a simulation world with leo rover:

```
ros2 launch leo_gz_bringup leo_gz.launch.py
```

Launch agruments:
* `sim_world` (default: `leo_empty.sdf`) - The Gazebo world to use. Refer to the [leo_gz_worlds](https://github.com/LeoRover/leo_simulator-ros2/tree/iron/leo_gz_worlds) package for available worlds.
* `robot_ns` (default: `""`) - Robot namespace
    
Example:

```
ros2 launch leo_gz_bringup leo_gz.launch.py sim_world:=marsyard2021.sdf robot_ns:=leo1
```

Add another leo rover to an already running gazebo world:

```
ros2 launch leo_gz_bringup spawn_robot.launch.py robot_ns:=leo2
```
