
# Terminal 1: Gazebo (includes static TF now)
ros2 launch roverrobotics_gazebo Leo_rover_depth.launch.py
ros2 launch roverrobotics_gazebo Leo_rover_lidar.launch.py

# Terminal 2: Nav2
ros2 launch roverrobotics_gazebo leo_nav2_lidar_launch.py

# Terminal 3: RViz2
cd /rover_metrics
python island_lidar_metrics_collector.py


rviz2
rviz2 --ros-args -p use_sim_time:=true



############# Nav2

Open RViz2: rviz2
Set Fixed Frame to odom
Add displays:

Map → topic: /local_costmap/costmap
Map → topic: /global_costmap/costmap
Add PointCloud2 → topic: /depth_camera/points
pointcloud → topic: /local_costmap/costmap
Path → topic: /plan


Click the 2D Goal Pose button in the toolbar
Click and drag on the map to set a goal position and orientation


#################  Statistical Analysis ########################

python statistical_analysis.py /path/to/metrics_data/island --output island_results.csv
python csv_to_latex.py island_results.csv --output island_table.tex


#################  Roboterrain and Attention: ########################

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch roverrobotics_gazebo Leo_rover_gazebo.launch.py
# active vision
ros2 launch roverrobotics_gazebo Leo_rover_fisheye.launch.py
ros2 launch roverrobotics_gazebo Leo_rover_fisheye.launch.py headless:=true

ros2 launch roverrobotics_gazebo Leo_rover_depth.launch.py


#export ROS_DOMAIN_ID=0


#REAL time factor 1.40
#Step size 0.005


#python3 ign_ros2_pose_topic.py inspect leo_rover

#python3 ign_ros2_Nav2_topics.py moon leo_rover --ros-args -p use_sim_time:=true


cd attent/inference
python ros2_mem_share.py
python fisheye_ros2_mem_share.py

# ~/src/attention
conda activate attent 
python inference.py --attention_mode ./model_output/checkpoint_epoch_1000.pkl 

# dynamic obstacles
# inspect
python spawn.py --trajectory_file trajectories/inspect_linear.sdf --world_name inspect --actor_name linear
ros2 run ros_gz_bridge parameter_bridge /linear_actor/pose@geometry_msgs/msg/Pose[gz.msgs.Pose

python spawn_float.py --trajectory_file trajectories/inspect_corner_triangle.sdf --world_name inspect --actor_name triangle

# constuct
python spawn.py --trajectory_file trajectories/construction_upper.sdf --actor_name upper --world_name default

# island/moon
python spawn_float.py --trajectory_file trajectories/flat_triangle_traject.sdf --world_name moon --actor_name triangle


############## Dreamerv3 commands ########################
conda activate jaxRos
FILTERED_LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E '^/opt/ros' | tr '\n' ':' | sed 's/:$//')

env LD_LIBRARY_PATH="$FILTERED_LD_LIBRARY_PATH" CUDA_HOME="" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python dreamerv3/main.py --configs leorover --logdir ./logdir/dreamer/{timestamp}

env LD_LIBRARY_PATH="$FILTERED_LD_LIBRARY_PATH" CUDA_HOME="" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python dreamerv3/main.py --configs leorover --script eval_only --logdir ./logdir/dreamer/0831T1151_working --run.eval_eps 100 --run.eval_envs 1 --run.from_checkpoint ./logdir/dreamer/0831T1151_working/ckpt/2025

# Active Vision
env LD_LIBRARY_PATH="$FILTERED_LD_LIBRARY_PATH" CUDA_HOME="" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform python dreamerv3/main.py --configs leorover --logdir ./logdir/dreamer/20251119T212857


python plot_RL_metrics.py  0831T1151_working/metrics.jsonl --window 30



############## ROS2 commands ########################
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/leo1/cmd_vel

# to see what 96x96 view looks like.
ros2 launch launch/crop_decimate.launch.py

ros2 run rqt_image_view rqt_image_view


ros2 daemon start

############## new world setup ########################

# new Actor trajectory
ros2 bag record /rover/pose_array
python trajectory-from-bag.py > construction_upper.sdf





############## OLD SB3 ########################
python sb3_SAC.py --mode train --load False --world inspect --vision True

python sb3_SAC.py \
  --mode train \
  --load True \
  --world island \
  --vision True \
  --checkpoint_name checkpoints/sac_island_20250804_1954_1000000_steps.zip \
  --normalize_stats checkpoints/sac_island_20250804_1954_1000000_steps_normalize.pkl


python spawn.py --trajectory_file trajectories/inspect_linear.py --actor_name linear_actor --world_name inspect
/dynamic_obstacles$ python spawn.py trajectory_file_name actor_name walk_name world_name
/dynamic_obstacles$ python spawn_publisher.py trajectory_file_name actor_name walk_name world_name


# Leo Rover

ros2 launch leo_gz_bringup leo_gz.launch.py sim_world:=marsyard2020.sdf robot_ns:=leo1

python ign_ros2_pose_topic.py leo_marsyard leo_rover_leo1

python dreamerv3/leo_main.py --logdir ~/logdir/{timestamp} --configs leorover --run.train_ratio 64

python dreamerv3/leo_main.py --logdir ~/logdir/20250315T131727 --configs leorover



ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/leo1/cmd_vel

# to see what 96x96 view looks like.
ros2 launch launch/crop_decimate.launch.py

ros2 run rqt_image_view rqt_image_view


# launch gazeob
ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

#pose converter:
python3 ign_ros2_pose_topic.py inspect rover_zero4wd

# agent
conda activate sb3
python sb3_SAC.py --load False
python sb3_SAC.py --load True --checkpoint_name checkpoints/ppo_rover_model_20241115_1152_900000_steps.zip
python sb3_SAC.py --mode predict --load True --checkpoint_name checkpoints/sac_baseline_pointnav.zip --normalize_stats checkpoints/vec_normalize_20250201_1453_final.pkl

# tensorboard
~/src/RoboTerrain/ros2_ws/src/sb3$ tensorboard --logdir tboard_logs/


# metrics logger in ~/rover_workspace/rover_metrics/
ros2 run rover_metrics metrics_node



# Claude prompt
How will you go about writing this program? I prefer a functional programming style with explicit variable typing.
Please ask me any questions that you have or clarifications that you need. If you have any suggestions please let me know.
Do not write the program yet. Let us work on getting all issues resolved and then write the program.



# Dream Rover commands:

python dreamerv3/main.py --configs rover --logdir ~/logdir/r_camera_test

python dreamerv3/main.py --configs rover --logdir ~/logdir/rover

python dreamerv3/main.py --configs turtlebot --logdir ~/logdir/turtlebot

python dreamerv3/main.py --logdir ~/logdir/{timestamp} --configs dmc_proprio



################ ROS2

ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py


ign fuel download -u https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Jersey%20Barrier


#rviz2:  'Fixed Frame' set to lidar_link
ros2 run rviz2 rviz2

ros2 run rqt_image_view rqt_image_view

ros2 run teleop_twist_keyboard teleop_twist_keyboard

ros2 control list_controllers

#Open an sdf world. for headless use '-s' for server mode
ign gazebo -v 4 simplecave3.sdf -s

# to open an sdf, first start gazebo with empty then run the ign command
#ign gazebo empty.sdf
#ign service -s /world/empty/create --reqtype ignition.msgs.EntityFactory --reptype ignition.msgs.Boolean --timeout 10000 --req 'sdf_filename: "/home/jack/worlds/harmonic/h_terrain/model.sdf"'




ros2 run rqt_image_view rqt_image_view

Fix contollers:
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{start_controllers: ['joint_state_broadcaster'], stop_controllers: [], strictness: 1, start_asap: false, timeout: {sec: 5, nanosec: 0}}"

ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController "{start_controllers: ['diffdrive_controller'], stop_controllers: [], strictness: 1, start_asap: false, timeout: {sec: 5, nanosec: 0}}"

ros2 run teleop_twist_keyboard teleop_twist_keyboard

# Files
# robot urdf files: changed lidar from 640 to 64
/opt/ros/humble/share/turtlebot4_description/urdf/

#world file with update rate:
/opt/ros/humble/share/nav2_simple_commander/warehouse.world



colcon build --symlink-install

rozer map is 20 x 20 meters

# list ign topics
ign topic -l
ign topic -t <topic_name> -e # echo
ign topic -t /world/maze/dynamic_pose/info -i #info

sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update

sudo apt install ignition-edifice
sudo apt install ignition-edifice-fuel-tools


export IGN_GAZEBO_RESOURCE_PATH=$HOME/src/RoboTerrain/models:$IGN_GAZEBO_RESOURCE_PATH



# Show sizes of all directories in current folder, sorted:
du -h --max-depth=1 | sort -h
du -sh ./*
