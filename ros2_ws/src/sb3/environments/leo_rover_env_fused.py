import gymnasium as gym
import numpy as np
import rclpy
import subprocess
import time
import math
import os
import struct
from multiprocessing import shared_memory
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
from typing import Optional
import numpy.typing as npt
from datetime import datetime
from time import perf_counter

# Type definitions
ObservationArray = npt.NDArray[np.float32]  # [H, W, 3]


def save_fused_image_channels(fused_image: np.ndarray, output_dir: str = './out_images') -> None:
    """
    Save each channel of fused image as separate PNG files for debugging.
    
    Args:
        fused_image: Fused observation array [H, W, 3] with values in [0,1]
        output_dir: Directory to save images (default: './out_images')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [0,1] to [0,255] and ensure uint8
    fused_image_uint8 = (fused_image * 255).astype(np.uint8)
    
    # Extract and save each channel
    now = datetime.now()
    time_string = now.strftime("%M_%S")
    check_black = fused_image_uint8[:, :, 1]
    #if np.sum(check_black) == 0: # if no person don't bother writing to file
    #    return 
    for i in range(3): #(3) for depth
        channel = fused_image_uint8[:, :, i]
        filename = f"channel_{time_string}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, channel)
    
    print(f"Saved fused image channels to {output_dir}/channel_[1-3].png")

class RoverEnvFused(gym.Env):
    """Custom Environment that follows gymnasium interface with fused vision observations"""
    metadata = {'render_modes': ['human']}
    
    def __init__(self, size=(96, 96), length=20000, scan_topic='/scan', imu_topic='/imu/data',
                 cmd_vel_topic='/cmd_vel', world_n='inspect',
                 connection_check_timeout=30, lidar_points=32, max_lidar_range=12.0,
                 rl_obs_name='rl_observation'):

        super().__init__()
        
        # Initialize ROS2 node and publishers/subscribers
        rclpy.init()
        self.bridge = CvBridge()
        self.node = rclpy.create_node('turtlebot_controller')

        # Fused observation parameters
        self.rl_obs_height, self.rl_obs_width = 96, 96
        self.rl_obs_name = rl_obs_name
        self.current_fused_obs = np.zeros((self.rl_obs_height, self.rl_obs_width, 3), dtype=np.float32)

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
        if world_n == 'island':
            self.world_name = 'moon'
        else:
            self.world_name = world_n
        self._step = 0
        self._received_fused_obs = False
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
        self.stuck_threshold = 0.0001  # Minimum distance the robot should move
        self.stuck_window = 5000 #600 for one minute Number of steps to check for being stuck
        self.stuck_penalty = -25.0

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
        self.flip_threshold = 1.48 #85 degrees in radians #math.pi / 3  # 60 degrees in radians
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
        
        self.target_positions_x = 0
        self.target_positions_y = 0
        self.previous_distance = None

        self.world_pose_path = '/world/' + self.world_name + '/set_pose'
        print('world is', self.world_name)

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

        self.observation_space = spaces.Dict({
            'fused_image': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.rl_obs_height, self.rl_obs_width, 3),
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

        
        # Setup shared memory for fused observations
        try:

            self.rl_obs_shm = shared_memory.SharedMemory(name=self.rl_obs_name)
            # Remove from resource_tracker to avoid shutdown warning
            import multiprocessing.resource_tracker
            multiprocessing.resource_tracker.unregister(self.rl_obs_shm._name, "shared_memory")

            print(f"Successfully attached to RL observation shared memory: {self.rl_obs_name}")
        except FileNotFoundError:
            print(f"Error: Could not find RL observation shared memory '{self.rl_obs_name}'. "
                  "Make sure the inference pipeline is running.")
            exit(1)
        except Exception as e:
            print(f"Error attaching to RL observation shared memory: {e}")
            exit(1)

        # Check robot connection - but using lidar since we still need it for rewards
        #self._robot_connected = self._check_robot_connection(timeout=connection_check_timeout)
        #if not self._robot_connected:
        #    self.node.get_logger().warn("No actual robot detected. Running in simulation mode.")
            
        # Initialize publishers and subscribers
        self.publisher = self.node.create_publisher(
            Twist,
            cmd_vel_topic,
            10)
        
        """
        # Keep IMU subscriber for reward calculation  
        self.imu_subscriber = self.node.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            10)
        """
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Keep pose subscriber for reward calculation
        self.pose_array_subscriber = self.node.create_subscription(
            PoseArray,
            '/rover/pose_array',
            self.pose_array_callback,
            qos_profile
        )

        # Keep odometry subscriber for reward calculation
        self.odom_subscriber = self.node.create_subscription(
            Odometry,
            '/odometry/wheels',  
            self.odom_callback,
            10)

    def get_fused_observation(self) -> np.ndarray:
        """Block until a new frame arrives, then return the fused observation from shared memory."""
        buf = self.rl_obs_shm.buf
        shape = (self.rl_obs_height, self.rl_obs_width, 3)
    
        # Calculate memory layout constants  
        header_size = 8 + 4  # timestamp (8 bytes) + valid flag (4 bytes)
        expected_data_size = self.rl_obs_height * self.rl_obs_width * 3 * 4  # float32 = 4 bytes per element
    
        while True:
            # Read timestamp
            ts = struct.unpack_from('<d', buf, 0)[0]  # float64 at offset 0
            
            # Read valid flag
            valid_flag = struct.unpack_from('<L', buf, 8)[0]  # uint32 at offset 8
            
            # Initialize internal timestamp once
            if not hasattr(self, "_last_obs_ts"):
                self._last_obs_ts = ts

            # Wait until new timestamp AND data is valid
            if ts != self._last_obs_ts and valid_flag == 1:
                self._last_obs_ts = ts
            
                # Read observation data as float32 starting after header
                observation_bytes = bytes(buf[header_size:header_size + expected_data_size])
                observation = np.frombuffer(observation_bytes, dtype=np.float32)
                observation = observation.reshape(shape)
            
                # Convert from [0,1] float32 to [0,255] uint8 if needed by your RL agent
                # If your RL agent expects float32 in [0,1], just return observation.copy()
                # If your RL agent expects uint8 in [0,255], uncomment the next line:
                # observation = (observation * 255).astype(np.uint8)
            
                return observation.copy()

            time.sleep(0.001)

        
    def get_fused_observation_old(self) -> Optional[ObservationArray]:
        """
        Read fused observation from shared memory.
        
        Returns:
            Observation array [H, W, 3] if valid and recent, None otherwise
        """
        # Calculate expected memory size
        header_size = 8 + 4  # timestamp + valid flag
        expected_data_size = self.rl_obs_height * self.rl_obs_width * 3 * 4  # float32
        total_size = header_size + expected_data_size
        
        if len(self.rl_obs_shm.buf) < total_size:
            raise ValueError(f"Shared memory too small: need {total_size}, got {len(self.rl_obs_shm.buf)}")
        
        # Get buffer view
        buf = self.rl_obs_shm.buf
        
        # Read validity flag first
        valid_flag = struct.unpack_from('<L', buf, 8)[0]  # uint32 at offset 8
        
        if valid_flag != 1:
            return None  # Data is not valid
        
        # Read timestamp
        timestamp = struct.unpack_from('<d', buf, 0)[0]  # float64 at offset 0
        current_time = time.time()
        
        # For blocking behavior, we don't check age here - just validity
        
        # Read observation data
        observation_bytes = bytes(buf[header_size:header_size + expected_data_size])
        observation = np.frombuffer(observation_bytes, dtype=np.float32)
        observation = observation.reshape((self.rl_obs_height, self.rl_obs_width, 3))
        struct.pack_into('<L', buf, 8, 0)  # Set valid flag to 0      
        return observation

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
        t0 = perf_counter()
        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', episode done')
            if self._step > 500:
                print('Robot flipped on its own')
                return self.get_observation(), self.stuck_penalty, True, False, {}
            else:
                return self.get_observation(), 0, True, False, {} 
        
        if self.collision_count > self.stuck_window:
            self.collision_count = 0
            print('stuck in collision, ending episode')
            return self.get_observation(), -1 * self.goal_reward, True, False, {}  

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

            if distance_moved < 0.0: #self.stuck_threshold:
                print('Robot is stuck, has moved only', distance_moved,
                      'meters in', self.stuck_window, 'steps, resetting')
                return self.get_observation(), self.stuck_penalty, True, False, {}

        if self.too_far_away():
            print('Too far away, resetting.')
            return self.get_observation(), self.too_far_away_penilty, True, False, {}
                
        # Wait for new fused observation data - block indefinitely
        t1 = perf_counter()

        fused_obs = self.get_fused_observation()
        
        # FIX: Store the new observation in the instance variable
        self.current_fused_obs = fused_obs

        t_fused = perf_counter()
        self._received_fused_obs = False

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
        if self._step >= self._length:
            print(f"Episode length limit reached: {self._step} >= {self._length}")
            #done = True #(self._step >= self._length)
        #else:
        done = (self._step >= self._length)
        # Get observation
        observation = self.get_observation()

        rclpy.spin_once(self.node, timeout_sec=0.001)  # 1ms
        
        t_obs = perf_counter()

        if self.total_steps % 10000 == 0:
            temp_obs_target = self.get_target_info()
            print(
                f"current pose x,y: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                f"Speed: {speed:.2f}, Heading: {math.degrees(self.current_yaw):.1f}°, "
            )
            print(
                f"current target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                f"distance and angle to target: ({temp_obs_target[0]:.3f}, {temp_obs_target[1]:.3f}), "
                f"Final Reward: {reward:.3f}"
            )
        

        info = {
            'steps': self._step,
            'total_steps': self.total_steps,
            'reward': reward
        }

        t_end_step = perf_counter()
        
        if self.total_steps % 5000 == 0:
            print(f"time_ms t1-t0:{(t1-t0)*1000}, t_fused:{(t_fused-t1)*1000}, t_obs:{(t_obs-t_fused)*1000}, t_total:{(t_end_step-t0)*1000}")
        if self.total_steps % 50000 == 0:
            save_fused_image_channels(observation['fused_image'])
            #print('Observation: Fused image shape:', observation['fused_image'].shape,
            #      ', Fused image range: [', np.min(observation['fused_image']), 
            #      ', ', np.max(observation['fused_image']), ']')
            
        return observation, reward, done, False, info
        
    def step_old(self, action):
        """Execute one time step within the environment"""
        self.total_steps += 1
        t0 = perf_counter()
        flip_status = self.is_robot_flipped()
        if flip_status:
            print('Robot flipped', flip_status, ', episode done')
            if self._step > 500:
                print('Robot flipped on its own')
                return self.get_observation(), self.stuck_penalty, True, False, {}
            else:
                return self.get_observation(), 0, True, False, {} 
        
        if self.collision_count > self.stuck_window:
            self.collision_count = 0
            print('stuck in collision, ending episode')
            return self.get_observation(), -1 * self.goal_reward, True, False, {}  


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

            if distance_moved < 0.0: #self.stuck_threshold:
                print('Robot is stuck, has moved only', distance_moved,
                      'meters in', self.stuck_window, 'steps, resetting')
                return self.get_observation(), self.stuck_penalty, True, False, {}

        if self.too_far_away():
            print('Too far away, resetting.')
            return self.get_observation(), self.too_far_away_penilty, True, False, {}
                
        # Wait for new fused observation data - block indefinitely
        t1 = perf_counter()

        fused_obs = self.get_fused_observation()

        t_fused = perf_counter()
        self._received_fused_obs = False

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
        if self._step >= self._length:
            print(f"Episode length limit reached: {self._step} >= {self._length}")
            #done = True #(self._step >= self._length)
        #else:
        done = (self._step >= self._length)
        # Get observation
        observation = self.get_observation()

        t_obs = perf_counter()

        if self.total_steps % 10000 == 0:
            temp_obs_target = self.get_target_info()
            print(
                f"current pose x,y: ({self.current_pose.position.x:.2f}, {self.current_pose.position.y:.2f}), "
                f"Speed: {speed:.2f}, Heading: {math.degrees(self.current_yaw):.1f}°, "
            )
            print(
                f"current target x,y: ({self.target_positions_x:.2f}, {self.target_positions_y:.2f}), "
                f"distance and angle to target: ({temp_obs_target[0]:.3f}, {temp_obs_target[1]:.3f}), "
                f"Final Reward: {reward:.3f}"
            )
        

        info = {
            'steps': self._step,
            'total_steps': self.total_steps,
            'reward': reward
        }

        t_end_step = perf_counter()
        
        if self.total_steps % 5000 == 0:
            print(f"time_ms t1-t0:{(t1-t0)*1000}, t_fused:{(t_fused-t1)*1000}, t_obs:{(t_obs-t_fused)*1000}, t_total:{(t_end_step-t0)*1000}")
        if self.total_steps % 5000 == 0:
            save_fused_image_channels(observation['fused_image'])
            #print('Observation: Fused image shape:', observation['fused_image'].shape,
            #      ', Fused image range: [', np.min(observation['fused_image']), 
            #      ', ', np.max(observation['fused_image']), ']')
            
        return observation, reward, done, False, info

    def get_observation(self):
        return {
            'fused_image': self.current_fused_obs,  
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
        with open(f'{self.episode_log_path}/{self.log_name}', 'a') as f:
            f.write(f"{timestamp},goal_reached,{self.episode_number-1},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
            f.write(f"{timestamp},episode_start,{self.episode_number},x={self.current_pose.position.x:.2f},y={self.current_pose.position.y:.2f}\n")
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
        distance_delta_scale = 0.5
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
        if abs(distance_delta) > 0.001 and heading_reward > 0.004:  # Only reward meaningful movement with good heading
            distance_reward = distance_delta * distance_delta_scale
            distance_reward += heading_reward
        else:
            distance_reward = -0.03 #(-1.1 * distance_delta * distance_delta_scale)
            
        # Combine rewards
        reward = (distance_reward * final_reward_multiplier) + (self.current_linear_velocity * 0.0025)

        # Debug logging
        if self.total_steps % 10000 == 0:
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
        
        # Check both roll and pitch angles
        if abs(self.current_roll) > self.flip_threshold:
            print('flipped')
            return 'roll_left' if self.current_roll > 0 else 'roll_right'
        elif abs(self.current_pitch) > self.flip_threshold:
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
            z_insert = 5 # for inspection
            if x_insert < -24.5 and y_insert < -24.5: #inspection
                z_insert = 6.5 
        else:
            z_insert = .75 # for maze and default

        ##  Random Yaw
        final_yaw = np.random.uniform(-np.pi, np.pi)
        print(f"Generated heading: {math.degrees(final_yaw)}°")
        # Normalize to [-pi, pi] range
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        quat_w = np.cos(final_yaw / 2)
        quat_z = np.sin(final_yaw / 2)

        # Print the full reset command
        reset_cmd_str = ('name: "leo_rover", ' +
                        f'position: {{x: {x_insert}, y: {y_insert}, z: {z_insert}}}, ' +
                        f'orientation: {{x: 0, y: 0, z: {quat_z}, w: {quat_w}}}')
        
        # Reset robot pose using ign service
        try:
            reset_cmd = [
                'ign', 'service', '-s', self.world_pose_path,
                '--reqtype', 'ignition.msgs.Pose',
                '--reptype', 'ignition.msgs.Boolean',
                '--timeout', '2000',
                '--req', 'name: "leo_rover", position: {x: ' + str(x_insert) +
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
        
        with open(f'{self.episode_log_path}//{self.log_name}', 'a') as f:
            f.write(f"{timestamp},episode_start,{self.episode_number},x={x_insert:.2f},y={y_insert:.2f}\n")

        self.episode_number += 1
        return observation, {}
    
    def render(self):
        """Render the environment (optional)"""
        pass

    def close(self):
        """Clean up resources"""
        try:
            self.rl_obs_shm.close()
            print("Closed RL observation shared memory")
        except Exception as e:
            print(f"Error closing RL observation shared memory: {e}")
        
        self.node.destroy_node()
        rclpy.shutdown()


    def pose_array_callback(self, msg):
        """Callback for processing pose array messages and extracting orientation"""
        if msg.poses:  # Check if we have any poses
            self.last_pose = self.current_pose if hasattr(self, 'current_pose') else None
            self.current_pose = msg.poses[0]  # Take the first pose
            
            # UPDATE - Store position as numpy array
            self.rover_position = np.array([
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_pose.position.z
            ], dtype=np.float32)
            
            # Extract orientation (yaw, pitch, roll) from quaternion
            try:
                quat = np.array([
                    self.current_pose.orientation.w, 
                    self.current_pose.orientation.x, 
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z
                ])
                norm = np.linalg.norm(quat)
                if norm == 0:
                    raise ValueError("Received a zero-length quaternion")
                quat_normalized = quat / norm
                roll, pitch, yaw = quat2euler(quat_normalized, axes='sxyz')
                self.current_pitch = pitch
                self.current_roll = roll
                self.current_yaw = yaw
            except Exception as e:
                self.node.get_logger().error(f"Error processing pose orientation data: {e}")
        
    def pose_array_callback_old(self, msg):
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
            
    """
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
    """
    
    # Add this callback
    def odom_callback(self, msg):
        """Process odometry data for velocities"""
        self.current_linear_velocity = msg.twist.twist.linear.x
        self.current_angular_velocity = msg.twist.twist.angular.z

    def _check_robot_connection(self, timeout):
        start_time = time.time()
        # Check for lidar data since we still need it for rewards
        received_scan = False
        while not received_scan:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if len(self.lidar_data) > 0 and np.any(self.lidar_data > 0):
                received_scan = True
            if time.time() - start_time > timeout:
                return False
        return True
