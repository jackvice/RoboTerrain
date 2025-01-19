# launch gazeob
ros2 launch roverrobotics_gazebo 4wd_rover_gazebo.launch.py

#pose converter:
python3 ign_ros2_pose_topic.py inspect rover_zero4wd

# agent
conda activate sb3
python sb3_ppo.py --load False
python sb3_ppo.py --load True --checkpoint_name checkpoints/ppo_rover_model_20241115_1152_900000_steps.zip

# tensorboard
~/src/RoboTerrain/ros2_ws/src/sb3$ tensorboar --logdir tboard_logs/

# Claude prompt
How will you go about writing this program?
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

# metrics logger in ~/rover_workspace/rover_metrics/
ros2 run rover_metrics metrics_node


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
