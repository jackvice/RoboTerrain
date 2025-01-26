# Off-Road Mobile Robot Simulation and Reinforcement Learning Framework

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

![Rover Zero climbing terrain](assets/Rover_climb.png)
<div align="center">
  <p float="left">
    <img src="assets/d657b628-86e6-4ec8-bb6c-914f8d173218.png" width="250" />
    <img src="assets/moon2.jpg" width="220" />
    <img src="assets/solar-panels-1024x620.png" width="250" />
  </p>
</div>

This repository provides a comprehensive framework for off-road mobile robot research, integrating ROS 2 Humble, Gazebo Fortress, and PyTorch with Stable Baselines3 (SB3) and a Gymnasium interface for reinforcement learning. It includes the models for the Rover Zero, Clearpath Husky and Leo Rover platforms and as well as mutiple outdoor Gazebo environments to include industrial inspection, agricultural and construction sites. Autonmated metrics logging includes Success Rate (SC), Total Collisions (TC), Mean time to Traverse (MTT), Travers Rate (TR), and Velocify Over Rough terrain (VOR).  This framework serves as a foundation for developing and testing mobile robot tasks on unstructured terrain.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Clone the Repository](#clone-the-repository)
  - [Build the Docker Image](#build-the-docker-image)
  - [Run the Docker Container](#run-the-docker-container)
- [Using the Framework](#using-the-framework)
  - [Launching the Simulation](#launching-the-simulation)
  - [Running the PPO PointNav RL Agent](#running-the-ppo-pointnav-rl-agent)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This project aims to facilitate off-road mobile robot research by providing a ready-to-use simulation environment and tools for developing reinforcement learning agents. It leverages:

- **ROS 2 Humble**: The latest ROS 2 distribution for robotics development.
- **Gazebo Fortress**: A powerful simulation environment compatible with ROS 2 Humble.
- **PyTorch and Stable Baselines3**: For implementing PPO (Proximal Policy Optimization) RL agents.

The framework is designed to simulate the Rover Zero robot, but it can be extended to other platforms.

## Features

- **Off Road Robot Simulation**: Models of the Rover Zero, Leo Rover and Clearpath Husky platforms in Gazebo Fortress.
- **Reinforcement Learning Integration**: Placeholder and structure for implementing SB3 PPO PointNav RL agents.
- **ROS 2 Workspace**: Includes necessary ROS 2 packages and custom code.
- **Extensibility**: Modular design allows for integration of additional robots and sensors.

## Prerequisites

- **NVIDIA Container Toolkit**: Required if you plan to use GPU acceleration.
  - Installation instructions: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **X11 Server**: For GUI applications like Gazebo (usually available on Linux systems).

## Getting Started
### Installation
Main dependencies: 

* [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
* [Gazebo Fortress](https://gazebosim.org/docs/latest/ros_installation/)
* [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

### Clone the Repository

```bash
git clone git@github.com:jackvice/RoboTerrain.git

cd RoboTerrain/ros2_ws/

colon build
```

## Using the Framework

### Launching the Simulation



```bash
# Source ROS 2 and workspace setup files
source /opt/ros/humble/setup.bash
source /home/ros2_ws/install/setup.bash

# Launch the Rover Zero simulation
ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

```

Gazebo should open on your host machine, displaying the Rover Zero robot in the simulation environment.

### Running the SAC PointNav RL Agent


1. **Running the baseline SAC agent:** The agent can be started in 'predict' mode or 'train' mode


   ```bash
   cd ros2_ws/src/sb3/

   # run the position bridge node for the Inspection world:
   python ign_ros2_pose_topic.py inspect rover_zero4wd

   # to run the policy
   python sb3_SAC.py --mode predict --load True --checkpoint_name checkpoints/sac_pointnav.zip

   # to training the agent
   python sb3_SAC.py --mode train --load False

   # TensorBoard
   tensorboard --logdir tboard_logs/
   ```

2. **Changing World Model**: Select between Maze, Inspection, Rubicon or Island worlds.

   ```bash
   # open the gazebo launch file and uncommnet the desired world line in the 'DeclareLaunchArgument()' function (line 24):
   nano RoboTerrain/ros2_ws/src/roverrobotics_ros2/roverrobotics_gazebo/launch/4wd_rover_gazebo.launch.py

   # Rebuild the workspace:
   cd RoboTerrain/ros2_ws/
   colcon build

   # Launch the simualtion:
   ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py 

   ```

4. **Running the metrics logger **: The metrics logger node writes metrics log files to ```RoboTerrain/metrics_analyzer/data/metric_logs```

   ```bash
   ros2 run rover_metrics metrics_node
   
   ```

5. **Spawn Dynamic Actors**: The dynamic actor default trajectories are for Inspection world.
   ```bash
   cd RoboTerrain/ros2_ws/src/dynamic_obstacles

   # First actor
   python spawn.py

   # Second actor
   python spawn2.py
   
   ```

5. **Graph Navigation Metrics**: Pass log file names to the metrics grapher.

   ```bash
   cd RoboTerrain/metrics_analyzer/
   python cli_main.py data/metric_logs/log_filename1.csv data/metric_logs/log_filename3.csv data/metric_logs/log_filename3.csv 
   
   ```


**Note**: Ensure that all ROS 2 nodes and the simulation are running before starting the training script.


## Repository Structure

```
project_root/
├── metrics_analyzer
├── ros2_ws/
│   ├── src/
│   │   ├── roverrobotics_ros2/       # Rover Robotics packages
│   │   ├── sb3/                # Your SAC and PPO agent code
│   │   ├── dynamic_obstacles/                # Your SAC and PPO agent code
│   │   └── rover_metrics/      # Additional ROS 2 packagse
│   ├── install/
│   ├── build/
│   └── log/
├── LICENSE
├── assets
└── README.md
```

# Robot Navigation Metrics Analysis

This system provides tools for logging and visualizing robot navigation metrics in ROS2, particularly useful for analyzing robot performance over rough terrain.

## System Components

1. **Metrics Node** (`metrics_node.py`): 
   - ROS2 node that collects real-time metrics
   - Subscribes to key topics: `/scan`, `/imu/data`, `/rover/pose_array`, `/odometry/wheels`
   - Logs data to timestamped CSV files

2. **Metrics Analyzer** (`cli_main.py`):
   - Command-line tool for analyzing and visualizing the collected metrics
   - Supports multiple data files for trial comparison
   - Generates publication-ready plots

## Usage Instructions

### 1. Data Collection

First, ensure ROS2 and your robot simulation/hardware are running. Then:

Launch the metrics logging node
ros2 run rover_metrics metrics_node

This will create a CSV file in `/metric_logs` with timestamp and metrics including:
- Total Collisions
- Obstacle Clearance
- Current Velocity
- IMU Acceleration Magnitude
- Rough Terrain Detection

### 2. Data Visualization

After collecting data, use the metrics analyzer to visualize results:

Basic usage (plot all metrics)
python cli_main.py path/to/metrics_log.csv

Compare velocity and IMU data from multiple trials
python cli_main.py trial1.csv trial2.csv -m CV IM -p time_series

Available metrics flags:
TC: Total Collisions
CS: Current Collision Status
SM: Smoothness Metric
OC: Obstacle Clearance
DT: Distance Traveled
CV: Current Velocity
IM: IMU Acceleration Magnitude
RT: Is Rough Terrain

### 3. Output

The visualization tool will create:
- Individual metric plots
- Combined comparison plots for multiple trials
- Special visualization for velocity vs. terrain roughness

Output files are saved in:
output/combined_metrics/

## Examples

### Single Trial Analysis
python cli_main.py metrics_log_20250105_190453.csv -m CV IM -p time_series

### Multiple Trial Comparison
python cli_main.py \
    metrics_log_20250105_190453.csv \
    metrics_log_20250105_190759.csv \
    -m CV IM -p time_series


- **ros2_ws/**: ROS 2 workspace containing all source code.
  - **src/**: Source directory for ROS 2 packages and PPO agent.
  - **install/**, **build/**, **log/**: Generated after building the workspace.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a new branch for your feature or bug fix.
2. **Ensure code quality** by following best practices and including docstrings.
3. **Submit a pull request** with a detailed description of your changes.



## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION


```


## Third-Party Licenses

This repository integrates components from several open-source projects. The licensing details for these components are as follows:

- **Gazebo Fuel Terrain Models**: Apache 2.0 License. See [gazebosim.org](https://app.gazebosim.org/dashboard).
- **Rover Zero Robot Model**: Apache 2.0 License. See [Rover Robotics Repository](https://github.com/roverrobotics).
- **Leo Rover Robot Model**: MIT License. See [Leo Rover Repository](https://github.com/LeoRover/leo_robot).
- **Clearpath Robotics Terrain and Husky Models**: BSD License. See [Clearpath Robotics Repositories](https://github.com/clearpathrobotics).

Each license allows for reuse and distribution with proper attribution and adherence to their terms.


For the full license text, please refer to the [LICENSE](LICENSE) file included in the repository.

---


