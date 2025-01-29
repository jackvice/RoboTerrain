"""
ROS2 Gymnasium Environment for Rover Navigation Training
----------------------------------------------------
A custom OpenAI Gym/Gymnasium environment for training a rover in navigation tasks using 
ROS2 and reinforcement learning. Supports different world environments including 
inspection zones, mazes, and island terrains.

Features:
- Continuous action space: [linear velocity, desired heading]
- Multi-modal observation space including:
 * LIDAR scans (32 points, 12m range)
 * Robot pose (x, y, z)
 * IMU data (roll, pitch, yaw)
 * Target information (distance, relative angle)
 * Current velocities
- Configurable reward function balancing:
 * Distance to goal progress
 * Heading alignment
 * Collision avoidance
 * Anti-stuck mechanisms
- Safety features:
 * Flip detection
 * Stuck detection
 * Out-of-bounds detection
 * Collision monitoring

Dependencies:
- ROS2 
- Gymnasium
- NumPy
- transforms3d
- cv_bridge (for vision extension)
- rclpy

Usage:
   Instantiate with world name ('inspect', 'maze', 'island') and other optional params.
   Used as a standard Gym environment with step(), reset(), close() methods.

Author: Jack Vice
Last Updated: 01/26/25
"""

import gymnasium as gym
import numpy as np
import rclpy
import subprocess
import time
import math
import os
from geometry_msgs.msg import Twist, Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from transforms3d.euler import quat2euler
from gymnasium import spaces
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from sensor_msgs.msg import Image
import cv2
from collections import deque
from time import strftime


class RoverEnv(gym.Env):
    """Custom Environment that follows gymnasium interface"""
    metadata = {'render_modes': ['human']}
    def __init__(self, size=(64, 64), length=2000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', camera_topic='/camera/image_raw', world_n='inspect',
                 connection_check_timeout=30, lidar_points=32, max_lidar_range=12.0):

        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')


        self.current_image = np.zeros((64, 64), dtype=np.float32)  # grayscale image buffer

        # Initialize these in __init__
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0


        self.log_name = "episode_log" + world_n + '_' + strftime("%H_%M") + '.csv'
        
        # Initialize environment parameters
        self.pose_node = None
        self.lidar_points = lidar_points
        self.max_lidar_range = max_lidar_range
        self.lidar_data = np.zeros(self.lidar_points, dtype=np.float32)
        self._length = length
        self.world_name = world_n
        self._step = 0
        self._received_scan = False
        self.first = False
        self.total_steps = 0
        self.last_speed = 0.0
        self.last_heading = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.current_yaw = 0.0
        self.rover_position = (0, 0, 0)
        self.min_raw_lidar = 100
        
        # Stuck detection parameters
        self.position_history = []
        self.stuck_threshold = 0.001  # Minimum distance the robot should move
        self.stuck_window = 600    # Number of steps to check for being stuck
        self.stuck_penilty = -25.0

        # Collision detection parameters
        self.collision_history = []
        self.collision_window = 15  # Number of steps to check for persistent collision
        self.collision_count = 0
        
        # PID control parameters for heading
        self.Kp = 2.0  # Proportional gain
        self.Ki = 0.0  # Integral gain
        self.Kd = 0.1  # Derivative gain
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_angular_velocity = 7.0
        
        # Cooldown mechanism
        self.cooldown_steps = 20
        self.steps_since_correction = self.cooldown_steps
        self.corrective_speed = 0.0
        self.corrective_heading = 0.0
        
        # Flip detection parameters
        self.flip_threshold = math.pi / 3  # 60 degrees in radians
        self.is_flipped = False
        self.initial_position = None
        self.initial_orientation = None

        self.last_pose = None
        
        # Ground truth pose
        self.current_pose = Pose()
        self.current_pose.position.x = 0.0
        self.current_pose.position.y = 0.0
        self.current_pose.position.z = 0.0
        self.current_pose.orientation.x = 0.0
        self.current_pose.orientation.y = 0.0
        self.current_pose.orientation.z = 0.0
        self.current_pose.orientation.w = 1.0

        self.yaw_history = deque(maxlen=200)
        
        # Add these as class variables in your environment's __init__
        self.down_facing_training_steps = 200000  # Duration of temporary training
        self.heading_steps = 0  # Add this to track when training began
        
        #self.rand_goal_x_range = (-7, 7) #(-23, -14) 
        #self.rand_goal_y_range = (-7, 7)# (-28, -19) 
        #self.rand_x_range =  (-7, 7) #(-23, -14) #x(-5.4, -1) # moon y(-9.3, -0.5) moon, x(-3.5, 2.5) 
        #self.rand_y_range =  (-7, 7) #(-28, -19) 
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.previous_distance = None

        self.world_pose_path = '/world/' + self.world_name + '/set_pose'
        print('world is', self.world_name)

        self.general_penility = -20 # flip, stuck
        if self.world_name == 'inspect':
            # Navigation parameters previous
            self.rand_goal_x_range = (-27, -19) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_goal_y_range = (-27, -19) # -27,-19 for inspection
            self.rand_x_range = (-27, -19) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_y_range = (-27, -19) # -27,-19 for inspection
            self.too_far_away_low_x = -29 #for inspection
            self.too_far_away_high_x = -13 #for inspection
            self.too_far_away_low_y = -29 # for inspection
            self.too_far_away_high_y = -17  # 29 for inspection
            self.too_far_away_penilty = -10 # -25.0
        elif self.world_name == 'moon': # moon is island
            # Navigation parameters previous
            self.rand_goal_x_range = (-4, 4) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_goal_y_range = (-4, 4) # -27,-19 for inspection
            self.rand_x_range = (-4, 4) #x(-5.4, -1) # moon y(-9.3, -0.5) # moon,  x(-3.5, 2.5) 
            self.rand_y_range = (-4, 4) # -27,-19 for inspection
            self.too_far_away_low_x = -20 #for inspection
            self.too_far_away_high_x = 10 #for inspection
            self.too_far_away_low_y = -20 # for inspection
            self.too_far_away_high_y = 20  # 29 for inspection
            self.too_far_away_penilty = -10 # -25.0
        else: ###### world_name = 'maze' use as default
            self.rand_goal_x_range = (-4, 4)
            self.rand_goal_y_range = (-4, 4)
            self.rand_x_range = (-4, 4) 
            self.rand_y_range = (-4, 4)
            self.too_far_away_low_x = -30 #for inspection
            self.too_far_away_high_x = 30 #for inspection
            self.too_far_away_low_y = -30 # for inspection
            self.too_far_away_high_y = 30  # 29 for inspection
            self.too_far_away_penilty = -10 # -25.0
            
        self.goal_reward = 100
        
        self.last_time = time.time()
        # Add at the end of your existing __init__ 
        self.heading_log = []  # To store headings
        self.heading_log_file = "initial_headings.csv"
        self.heading_log_created = False
        
        # Define action space
        # [speed, desired_heading]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi]),  # [min_speed, min_heading]
            high=np.array([1.0, np.pi]),   # [max_speed, max_heading]
            dtype=np.float32
        )

        self.episode_log_path = '/home/jack/src/RoboTerrain/metrics_analyzer/data/episode_logs/'
        os.makedirs(self.episode_log_path, exist_ok=True)
        self.episode_number = 0

        # Observation space remains the same
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(
                low=0,
                high=max_lidar_range,
                shape=(lidar_points,),
                dtype=np.float32
            ),
            'pose': spaces.Box(
                low=np.array([-30.0, -30.0, -10.0]),
                high=np.array([30.0, 30.0, 10.0]),
                dtype=np.float32
            ),
            'imu': spaces.Box(
                low=np.array([-np.pi, -np.pi, -np.pi]),
                high=np.array([np.pi, np.pi, np.pi]),
                dtype=np.float32
            ),
            'target': spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([100, np.pi]),
                shape=(2,),
                dtype=np.float32
            ),
            'velocities': spaces.Box(
                low=np.array([-10.0, -10.0]),
                high=np.array([10.0, 10.0]),
                shape=(2,),
                dtype=np.float32
            ),
        })

        # Check robot connection
        self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        if not self._robot_connected:
            self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")
        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10)
        
        self.lidar_subscriber = self.node.create_subscription(
            LaserScan,
            scan_topic,
            self.lidar_callback,
            10)
        
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pose_array_subscriber = self.node.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            qos_profile
        )

        self.bridge = CvBridge()

        # Add this in __init__ with your other subscribers
        self.odom_subscriber = self.node.create_subscription(
            Odometry,
            '/odometry/wheels',  
            self.odom_callback,
            10)



    def heading_controller(self, desired_heading, current_heading):
        """
        PID controller given a desired *absolute* heading,
        but the 'desired_heading' here is computed from
        (current_heading + relative_heading_command).
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Compute the heading error in [-pi, pi]
        error = math.atan2(
            math.sin(desired_heading - current_heading),
            math.cos(desired_heading - current_heading)
        )
        
        # PID update
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt
        self.last_error = error
        
        control = (
            self.Kp * error
            + self.Ki * self.integral_error
            + self.Kd * derivative_error
        )

        return np.clip(control, -self.max_angular_velocity, self.max_angular_velocity)


    def too_far_away(self):
        if ( self.current_pose.position.x < self.too_far_away_low_x or
             self.current_pose.position.x > self.too_far_away_high_x or
             self.current_pose.position.y < self.too_far_away_low_y or
             self.current_pose.position.y > self.too_far_away_high_y):
            print('too far, x, y is', self.current_pose.position.x,
                  self.current_pose.position.y, ', episode done. ************** reward',
                  self.too_far_away_penilty)
            return True
        else:
            return False


    def step(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1

        if self.collision_count > self.stuck_window:
            self.collision_count = 0
            print('stuck in collision, ending episode')
            return self.get_observation(), self.general_penility, True, False, {}  

        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', episode done')
            if self._step > 500:
                print('Robot flipped on its own')
                return self.get_observation(), self.general_penility, True, False, {}
            else:
                return self.get_observation(), 0, True, False, {} 
            

        # Update position history
        self.position_history.append((self.current_pose.position.x, self.current_pose.position.y))
        if len(self.position_history) > self.stuck_window:
            self.position_history.pop(0)

        # Check if robot is stuck - do this every step once we have enough history
        if len(self.position_history) >= self.stuck_window:
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 +
                                       (end_pos[1] - start_pos[1])**2)

            if distance_moved < self.stuck_threshold:
                print('Robot is stuck, has moved only', distance_moved,
                      'meters in', self.stuck_window, 'steps, resetting')
                return self.get_observation(), self.stuck_penilty, True, False, {}

        if self.too_far_away():
            return self.get_observation(), self.too_far_away_penilty, True, False, {}
                
        # Wait for new sensor data
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        
        self._received_scan = False


        # action = [speed, desired_relative_heading]
        speed = float(action[0])
        relative_heading_command = float(action[1])
    
        # Instead of passing (desired_relative_heading, current_yaw) directly,
        # we compute the *new desired absolute heading*:
        desired_heading = self.current_yaw + relative_heading_command
        
        angular_velocity = self.heading_controller(desired_heading, self.current_yaw)
        
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = angular_velocity
        self.publisher.publish(twist)
        self.last_speed = speed
        
        # Calculate reward and components
        reward = self.task_reward()

        # Check if episode is done
        self._step += 1
        done = (self._step >= self._length)
        
        # Get observation
        observation = self.get_observation()


        if self.total_steps % 1000 == 0:
            temp_obs_target = observation["target"]
            print(
                f"current pose x,y: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                f"Speed: {speed:.2f}, Heading: {math.degrees(self.current_yaw):.1f}°, "
            )
            print(
                f"current target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                f"distance and angle to target: ({temp_obs_target[0]:.3f}, {temp_obs_target[1]:.3f}), "
                f"Final Reward: {reward:.3f}"
            )
        
        if self.total_steps % 1000 == 0:
            print('Observation: Pose:', observation['pose'],
                  ', IMU:', observation['imu'],
                  ', target:', observation['target'],
                  )
        
        info = {
            'steps': self._step,
            'total_steps': self.total_steps,
            'reward': reward
        }
        return observation, reward, done, False, info


    def get_observation(self):
        return {
            'lidar': self.lidar_data,
            'pose': self.rover_position,
            'imu': np.array([self.current_pitch, self.current_roll, self.current_yaw],
                            dtype=np.float32),
            'target': self.get_target_info(),
            'velocities': np.array([self.current_linear_velocity, self.current_angular_velocity],
                                   dtype=np.float32)
        }    


    def update_target_pos(self):
        print('###################################################### GOAL ACHIVED!')
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        print(f'\nNew target x,y: {self.target_positions_x:.2f}, {self.target_positions_y:.2f}')
        self.previous_distance = None
        timestamp = time.time()
        #with open(f'{self.episode_log_path}/{self.log_name}', 'a') as f:
        #    f.write(f"{timestamp},goal_reached,{self.episode_number-1},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
        #    f.write(f"{timestamp},episode_start,{self.episode_number},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
        self.episode_number += 1
        return


    def task_reward(self):
        """
        Reward function that accounts for robot dynamics and gradual acceleration
        """
        # Constants
        final_reward_multiplier = 1.5
        collision_threshold = 0.3
        collision_penalty = -0.5
        success_distance = 0.5
        distance_delta_scale = 0.3
        heading_tolerance = math.pi/4  # 45 degrees

        # Get current state info
        distance_heading_info = self.get_target_info()
        current_distance = distance_heading_info[0]
        heading_diff = distance_heading_info[1]

        # Initialize previous distance if needed
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Check for goal achievement
        if current_distance < success_distance:
            self.update_target_pos()
            return self.goal_reward

        # Check for collisions
        min_distance = np.min(self.lidar_data[np.isfinite(self.lidar_data)])
        if min_distance < collision_threshold:
            timestamp = time.time()
            #with open(f'{self.episode_log_path}//{self.log_name}', 'a') as f:
            #    f.write(f"{timestamp},Collision,{self.episode_number-1},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
            print('Collision!')
            return collision_penalty
        
        # Calculate distance change (positive means got closer, negative means got further)
        distance_delta = self.previous_distance - current_distance

        # Calculate reward components
        distance_reward = 0.0
        heading_reward = 0.0
        

        # Heading component - reward facing towards target even if not moving much
        # This helps during acceleration phases
        heading_alignment = 1.0 - (abs(heading_diff) / math.pi)  # 1.0 when perfect, 0.0 when opposite
        heading_reward = 0.01 * heading_alignment  # 0.01 per step when perfect (30.0 over 3000 steps)
        # Heading component with new alignment calculation
        # Convert heading difference to range [-π, π]
        heading_diff = math.atan2(math.sin(heading_diff), math.cos(heading_diff))
        abs_heading_diff = abs(heading_diff)

        if abs_heading_diff <= math.pi/2:
            # From 0 to 90 degrees: scale from 1 to 0
            heading_alignment = 1.0 - (2 * abs_heading_diff / math.pi)
        else:
            # From 90 to 180 degrees: scale from 0 to -1
            heading_alignment = -2 * (abs_heading_diff - math.pi/2) / math.pi
        heading_reward = 0.01 * heading_alignment  # 0.01 per step when perfect (30.0 over 3000 steps)

        # Distance component - reward any progress towards goal
        if abs(distance_delta) > 0.001 and heading_reward > 0.002:  # Only reward meaningful movement with good heading
            distance_reward = distance_delta * distance_delta_scale
        else:
            distance_reward = -0.03
            
        # Combine rewards
        #reward = ((distance_reward + heading_reward) * final_reward_multiplier) + (self.current_linear_velocity * 0.0025)
        reward = (distance_reward * final_reward_multiplier) + (self.current_linear_velocity * 0.0025)
        # Debug logging
        if self.total_steps % 1000 == 0:
            print(f"Distance: {current_distance:.3f}, Previous Distance: {self.previous_distance:.3f}, "
                  f"distance_delta: {distance_delta:.3f}, Heading diff: {math.degrees(heading_diff):.1f}°, "
                  f"Speed: {self.last_speed:.3f}, Current vel: {self.current_linear_velocity:.3f}, "
                  f"Distance reward: {distance_reward:.3f}, Heading reward: {heading_reward:.3f}, "
                  f"Total reward: {reward:.3f}")

        self.previous_distance = current_distance
        return reward
    
    
    def get_target_info(self):
        """Calculate distance and azimuth to current target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target_x = self.target_positions_x
        target_y = self.target_positions_y 
    
        # Calculate distance
        distance = math.sqrt(
            (current_x - target_x)**2 + 
            (current_y - target_y)**2
        )
    
        # Calculate azimuth (relative angle to target)
        target_heading = math.atan2(target_y - current_y, target_x - current_x)
        relative_angle = math.atan2(math.sin(target_heading - self.current_yaw), 
                               math.cos(target_heading - self.current_yaw)
                                    )

        return np.array([distance, relative_angle], dtype=np.float32)

    def is_robot_flipped(self):
        """Detect if robot has flipped in any direction past 85 degrees"""
        FLIP_THRESHOLD = 1.48  # ~85 degrees in radians
        
        # Check both roll and pitch angles
        if abs(self.current_roll) > FLIP_THRESHOLD:
            print('flipped')
            return 'roll_left' if self.current_roll > 0 else 'roll_right'
        elif abs(self.current_pitch) > FLIP_THRESHOLD:
            print('flipped')
            return 'pitch_forward' if self.current_pitch < 0 else 'pitch_backward'
        
        return False
        

    def reset(self, seed=None, options=None):
        print('################'+ self.world_name + ' Environment Reset')
        print('')
        twist = Twist()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        """Reset the environment to its initial state"""
        super().reset(seed=seed)
        self.collision_history = []  # Clear collision history on reset
        x_insert = np.random.uniform(*self.rand_x_range)
        y_insert = np.random.uniform(*self.rand_y_range)
        
        if self.world_name == 'inspect':
            z_insert = 5.5 # for inspection
            if x_insert < -24.5 and y_insert < -24.5: #inspection
                z_insert = 6.5 
        else:
            z_insert = .75 # for maze and default

        ##  Random Yaw
        final_yaw = np.random.uniform(-np.pi, np.pi)
        print(f"Generated heading: {math.degrees(final_yaw)}°")
        # Normalize to [-pi, pi] range
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))

        # make final yaw point down cause normal action distribution pointing up.
        #final_yaw = np.pi + np.random.uniform(-0.2, 0.2)
        
        quat_w = np.cos(final_yaw / 2)
        quat_z = np.sin(final_yaw / 2)

        # Print the full reset command
        reset_cmd_str = ('name: "rover_zero4wd", ' +
                        f'position: {{x: {x_insert}, y: {y_insert}, z: {z_insert}}}, ' +
                        f'orientation: {{x: 0, y: 0, z: {quat_z}, w: {quat_w}}}')
        #print("\nFull reset command:")
        #print(reset_cmd_str)
        
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', self.world_pose_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "rover_zero4wd", position: {x: ' + str(x_insert) +
                ',y: '+ str(y_insert) +
                ', z: '+ str(z_insert) + '}, orientation: {x: 0, y: 0, z: ' +
                str(quat_z) + ', w: ' + str(quat_w) + '}'
            ]
            result = subprocess.run(reset_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed to reset robot pose: {result.stderr}")
        except Exception as e:
            print(f"Error executing reset command: {str(e)}")

        # Reset internal state
        self._step = 0
        self.last_linear_velocity = 0.0
        self.steps_since_correction = self.cooldown_steps
        self.is_flipped = False
        # Reset PointNav-specific variables
        self.target_positions_x = np.random.uniform(*self.rand_goal_x_range)
        self.target_positions_y = np.random.uniform(*self.rand_goal_y_range)
        print(f'\nNew target x,y: {self.target_positions_x:.2f}, {self.target_positions_y:.2f}')
        self.previous_distance = None
        # Add a small delay to ensure the robot has time to reset
        for _ in range(100):  # Increased from 3 to 5 to allow more time for pose reset
            rclpy.spin_once(self.node, timeout_sec=0.1)
        time.sleep(1.0)        
        observation = self.get_observation()
        # Normal operation
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        
        self.publisher.publish(twist)
        timestamp = time.time()
        print(timestamp)
        #with open(f'{self.episode_log_path}//{self.log_name}', 'a') as f:
        #    f.write(f"{timestamp},episode_start,{self.episode_number},x={x_insert:.2f},y={y_insert:.2f}\n")

        self.episode_number += 1
        return observation, {}
    

    def render(self):
        """Render the environment (optional)"""
        pass


    def close(self):
        """Clean up resources"""
        self.node.destroy_node()
        rclpy.shutdown()
        

    def pose_array_callback(self, msg):
        """Callback for processing pose array messages"""
        if msg.poses:  # Check if we have any poses
            self.last_pose = self.current_pose if hasattr(self, 'current_pose') else None
            self.current_pose = msg.poses[0]  # Take the first pose
            
            # UPDATE - Store position as numpy array
            self.rover_position = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ], dtype=np.float32)
        

    def lidar_callback(self, msg):
        """Process LIDAR data with error checking and downsampling."""

        # Convert to numpy array
        try:
            lidar_data = np.array(msg.ranges, dtype=np.float32)
        except Exception as e:
            print(f"Error converting LIDAR data to numpy array: {e}")
            return

        # Check for invalid values before processing
        #if np.any(np.isneginf(lidar_data)):
        #    print(f"WARNING: Found {np.sum(np.isneginf(lidar_data))} negative infinity values")
            
        if np.any(np.isnan(lidar_data)):
            print(f"WARNING: Found {np.sum(np.isnan(lidar_data))} NaN values")
            
        #if np.any(lidar_data < 0):
        #    print(f"WARNING: Found {np.sum(lidar_data < 0)} negative values")
            #print("Negative values:", lidar_data[lidar_data < 0])

        # Add Gaussian noise
        #gaussian_noise = np.random.normal(0, 0.05, size=lidar_data.shape)  # 0.1m standard deviation
        #lidar_data = lidar_data + gaussian_noise
    
        # Add random dropouts (set some measurements to max_range)
        #dropout_mask = np.random.random(lidar_data.shape) < 0.05  # 5% chance of dropout
        #lidar_data[dropout_mask] = self.max_lidar_range

        # Replace inf values with max_lidar_range
        inf_mask = np.isinf(lidar_data)
        if np.any(inf_mask):
            #print(f"INFO: Replaced {np.sum(inf_mask)} infinity values with max_lidar_range")
            lidar_data[inf_mask] = self.max_lidar_range

        # Replace any remaining invalid values (NaN, negative) with max_range
        invalid_mask = np.logical_or(np.isnan(lidar_data), lidar_data < 0)
        if np.any(invalid_mask):
            print(f"INFO: Replaced {np.sum(invalid_mask)} invalid values with max_lidar_range")
            lidar_data[invalid_mask] = self.max_lidar_range

        # Clip values to valid range
        lidar_data = np.clip(lidar_data, 0, self.max_lidar_range)

        # Verify we have enough data points for downsampling
        expected_points = self.lidar_points * (len(lidar_data) // self.lidar_points)
        if expected_points == 0:
            print(f"ERROR: Not enough LIDAR points for downsampling. Got {len(lidar_data)} points")
            return

        # Downsample by taking minimum value in each segment
        try:
            segment_size = len(lidar_data) // self.lidar_points
            reshaped_data = lidar_data[:segment_size * self.lidar_points].reshape(self.lidar_points,
                                                                                  segment_size)
            self.lidar_data = np.min(reshaped_data, axis=1)
            
            # Verify downsampled data
            if len(self.lidar_data) != self.lidar_points:
                print(f"ERROR: Downsampled has wrong size. Expected {self.lidar_points}, got {len(self.lidar_data)}")
                return
                
            if np.any(np.isnan(self.lidar_data)) or np.any(np.isinf(self.lidar_data)):
                print("ERROR: Downsampled data contains invalid values")
                print("NaN count:", np.sum(np.isnan(self.lidar_data)))
                print("Inf count:", np.sum(np.isinf(self.lidar_data)))
                return
                
        except Exception as e:
            print(f"Error during downsampling: {e}")
            return

        self._received_scan = True


    def imu_callback(self, msg):
        try:
            quat = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y,
                             msg.orientation.z])
            norm = np.linalg.norm(quat)
            if norm == 0:
                raise ValueError("Received a zero-length quaternion")
            quat_normalized = quat / norm
            roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
            self.current_pitch = pitch
            self.current_roll = roll
            self.current_yaw = yaw
        except Exception as e:
            self.node.get_logger().error(f"Error processing IMU data: {e}")


    # Add this callback
    def odom_callback(self, msg):
        """Process odometry data for velocities"""
        self.current_linear_velocity = msg.twist.twist.linear.x
        self.current_angular_velocity = msg.twist.twist.angular.z

    def _check_robot_connection(self, timeout):
        start_time = time.time()
        while not self._received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                return False
            if self._received_scan:
                return True
        return False


    
